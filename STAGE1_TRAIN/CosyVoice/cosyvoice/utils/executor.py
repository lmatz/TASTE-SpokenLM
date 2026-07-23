# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import json
import os
import torch
import time
import datetime
import tqdm
import torch.distributed as dist

from collections import defaultdict
from contextlib import nullcontext
from cosyvoice.utils.common import IGNORE_ID
from cosyvoice.utils.checkpoint import (
    check_checkpoint_sync_worker,
    enqueue_stage_best_sync,
    restore_rng_state,
    save_training_checkpoint,
)
from cosyvoice.utils.train_utils import update_parameter_and_lr, log_per_step, log_per_save, batch_forward, batch_backward, save_model, cosyvoice_join, set_dataloader_seed, check_model_finite
from cosyvoice.utils.training_metrics import StepMetricAccumulator


class Executor:

    def __init__(self):
        self.step = 0
        self.epoch = 0
        self.rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.device = torch.device(
            'cuda:{}'.format(local_rank) if torch.cuda.is_available() else 'cpu')
        self.cv_best_score = 0.0
        self.accumulation_count = 0
        self.accumulated_audio_seconds = 0.0
        self.accumulated_compute_seconds = 0.0
        self.accumulated_dataloader_wait_seconds = 0.0
        self.step_metrics = StepMetricAccumulator()

    def append_metric(self, info_dict, payload):
        metrics_path = info_dict.get('metrics_jsonl')
        if self.rank != 0 or not metrics_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(metrics_path)), exist_ok=True)
        payload = {
            'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **payload,
        }
        with open(metrics_path, 'a') as metrics_file:
            metrics_file.write(json.dumps(payload, sort_keys=True) + '\n')
            metrics_file.flush()
            os.fsync(metrics_file.fileno())

    def record_sample_trace(self, batch_dict, info_dict):
        trace_directory = info_dict.get('sample_trace_dir')
        if not trace_directory:
            return
        os.makedirs(trace_directory, exist_ok=True)
        trace_path = os.path.join(
            trace_directory, 'rank_{:05d}.jsonl'.format(self.rank))
        payload = {
            'epoch': self.epoch,
            'batch_idx': info_dict['batch_idx'],
            'optimizer_step_before_batch': self.step,
            'utts': list(batch_dict.get('utts', [])),
        }
        with open(trace_path, 'a') as trace_file:
            trace_file.write(json.dumps(payload, sort_keys=True) + '\n')

    @staticmethod
    def batch_audio_seconds(batch_dict):
        if batch_dict.get('audio_feat_len') is not None:
            return float(batch_dict['audio_feat_len'].sum().item()) / 100.0
        if batch_dict.get('speech_feat_len') is not None:
            return float(batch_dict['speech_feat_len'].sum().item()) * 256.0 / 22050.0
        if batch_dict.get('speech_token_len') is not None:
            return float(batch_dict['speech_token_len'].sum().item()) / 25.0
        raise KeyError('batch has no supported audio-length tensor')

    def record_optimizer_step_metric(self, info_dict):
        audio_seconds = torch.tensor(
            self.accumulated_audio_seconds,
            dtype=torch.float64,
            device=self.device)
        timing_seconds = torch.tensor([
            self.accumulated_dataloader_wait_seconds,
            self.accumulated_compute_seconds,
        ], dtype=torch.float64, device=self.device)
        if dist.is_initialized():
            dist.all_reduce(audio_seconds, op=dist.ReduceOp.SUM)
            dist.all_reduce(timing_seconds, op=dist.ReduceOp.MAX)
        step_seconds = float(timing_seconds.sum().item())
        global_audio_seconds = float(audio_seconds.item())
        learning_metrics = self.step_metrics.reduce(self.device)
        self.append_metric(info_dict, {
            'event': 'optimizer_step',
            'epoch': self.epoch,
            'batch_idx': info_dict['batch_idx'],
            'optimizer_step': self.step,
            'global_audio_seconds': global_audio_seconds,
            'dataloader_wait_seconds_max': float(timing_seconds[0].item()),
            'compute_seconds_max': float(timing_seconds[1].item()),
            'step_seconds_max': step_seconds,
            'audio_seconds_per_second': (
                global_audio_seconds / step_seconds if step_seconds else None),
            'learning_rate': float(info_dict['lr']),
            'gradient_norm': float(torch.as_tensor(info_dict['grad_norm']).item()),
            **learning_metrics,
        })
        self.accumulated_audio_seconds = 0.0
        self.accumulated_compute_seconds = 0.0
        self.accumulated_dataloader_wait_seconds = 0.0
        self.step_metrics.reset()

    def record_distributed_duration(self, info_dict, event, duration, **payload):
        duration_tensor = torch.tensor(
            duration, dtype=torch.float64, device=self.device)
        if dist.is_initialized():
            dist.all_reduce(duration_tensor, op=dist.ReduceOp.MAX)
        self.append_metric(info_dict, {
            'event': event,
            'epoch': self.epoch,
            'optimizer_step': self.step,
            'duration_seconds_max': float(duration_tensor.item()),
            **payload,
        })

    def train_one_epoc(self,
                       model,
                       optimizer,
                       scheduler,
                       train_data_loader,
                       cv_data_loader,
                       writer,
                       info_dict,
                       group_join,
                       start_batch_idx=0,
                       resume_rng_state=None):
        ''' Train one epoch
        '''

        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {} rank {}'.format(self.epoch, lr, self.rank))
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(info_dict['accum_grad']))
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        model.train()
        # model_context = model.join if info_dict['train_engine'] == 'torch_ddp' else nullcontext
        # with model_context():
        train_iterator = iter(train_data_loader)
        for skipped_batch_idx in range(start_batch_idx):
            try:
                next(train_iterator)
            except StopIteration as error:
                raise RuntimeError(
                    'resume cursor batch {} exceeds epoch {} length at skipped batch {}'.format(
                        start_batch_idx, self.epoch, skipped_batch_idx)) from error
        if resume_rng_state is not None:
            restore_rng_state(resume_rng_state)
            logging.info(
                '[Rank %s] restored RNG state after rebuilding epoch %s through batch %s',
                self.rank, self.epoch, start_batch_idx)

        previous_batch_end = time.perf_counter()
        progress = tqdm.tqdm(
            train_iterator, position=self.rank, desc=f"[Rank {self.rank}] Training...")
        for batch_idx, batch_dict in enumerate(progress, start=start_batch_idx):
            batch_ready = time.perf_counter()
            self.accumulated_dataloader_wait_seconds += batch_ready - previous_batch_end
            self.accumulated_audio_seconds += self.batch_audio_seconds(batch_dict)
            info_dict["tag"] = "TRAIN"
            info_dict["step"] = self.step
            info_dict["epoch"] = self.epoch
            info_dict["batch_idx"] = batch_idx
            info_dict["accumulation_boundary"] = (
                self.accumulation_count + 1 == info_dict["accum_grad"])
            
            if cosyvoice_join(group_join, info_dict):
                break
            self.record_sample_trace(batch_dict, info_dict)

            # Disable gradient synchronizations across DDP processes.
            # Within this context, gradients will be accumulated on module
            # variables, which will later be synchronized.
            if info_dict['train_engine'] == 'torch_ddp' and not info_dict["accumulation_boundary"]:
                context = model.no_sync
            # Used for single gpu training and DDP gradient synchronization
            # processes.
            else:
                context = nullcontext

            with context():
                info_dict = batch_forward(model, batch_dict, info_dict)
                self.step_metrics.update(info_dict['loss_dict'])
                info_dict = batch_backward(model, info_dict)

            info_dict = update_parameter_and_lr(model, optimizer, scheduler, info_dict)
            if info_dict['optimizer_step']:
                self.step += 1
                self.accumulation_count = 0
                info_dict['step'] = self.step - 1
            else:
                self.accumulation_count += 1
            if (info_dict['optimizer_step'] and
                    info_dict['finite_parameter_check_per_step'] > 0 and
                    self.step % info_dict['finite_parameter_check_per_step'] == 0):
                check_model_finite(model)
            self.accumulated_compute_seconds += time.perf_counter() - batch_ready
            if info_dict['optimizer_step']:
                self.record_optimizer_step_metric(info_dict)
                if info_dict.get('checkpoint_sync_mode') == 'queued':
                    check_checkpoint_sync_worker(
                        info_dict.get('checkpoint_sync_queue_dir'),
                        info_dict.get('checkpoint_sync_worker_pid'))
            log_per_step(writer, info_dict)
            # NOTE specify save_per_step in cosyvoice.yaml if you want to enable step save
            if info_dict['optimizer_step'] and info_dict['save_per_step'] > 0 and self.step % info_dict['save_per_step'] == 0:
                # dist.barrier(group=group_join)
                cv_started = time.perf_counter()
                cv_metrics = self.cv(
                    model, cv_data_loader, writer, info_dict,
                    on_batch_end=False, group_join=group_join)
                self.record_distributed_duration(
                    info_dict,
                    'validation',
                    time.perf_counter() - cv_started,
                    on_batch_end=False,
                    metrics=cv_metrics)
                logging.info(f"[Rank {self.rank}] waiting after CV...")
                dist.barrier()
                logging.info(f"[Rank {self.rank}] FINISHED waiting after CV, will continue training...")
                model.train()
            full_checkpoint_due = (
                info_dict['optimizer_step'] and
                info_dict['full_checkpoint_per_step'] > 0 and
                self.step % info_dict['full_checkpoint_per_step'] == 0)
            if full_checkpoint_due:
                self.save_resume_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    info_dict,
                    next_epoch=self.epoch,
                    next_batch_idx=batch_idx + 1)
            max_steps_reached = (
                info_dict['optimizer_step'] and
                info_dict['max_optimizer_steps'] > 0 and
                self.step >= info_dict['max_optimizer_steps'])
            if max_steps_reached:
                save_model(model, 'checkpoint_last', info_dict)
                if not full_checkpoint_due:
                    self.save_resume_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        info_dict,
                        next_epoch=self.epoch,
                        next_batch_idx=batch_idx + 1)
                logging.info(
                    '[Rank %s] reached max_optimizer_steps=%s',
                    self.rank, info_dict['max_optimizer_steps'])
                return True
            previous_batch_end = time.perf_counter()
        
        logging.info(f"[Rank {self.rank}] has finished epoch {self.epoch} training. Break other workers'")
        dist.barrier()
        logging.info(f"[Rank {self.rank}] all ranks are done. Conduct CV after epoch now.")
        cv_started = time.perf_counter()
        cv_metrics = self.cv(
            model, cv_data_loader, writer, info_dict,
            on_batch_end=True, group_join=group_join)
        self.record_distributed_duration(
            info_dict,
            'validation',
            time.perf_counter() - cv_started,
            on_batch_end=True,
            metrics=cv_metrics)
        dist.barrier()
        if self.accumulation_count == 0:
            self.save_resume_checkpoint(
                model,
                optimizer,
                scheduler,
                info_dict,
                next_epoch=self.epoch + 1,
                next_batch_idx=0,
                model_name='resume_epoch_{:04d}_step_{:08d}'.format(
                    self.epoch + 1, self.step))
        else:
            logging.warning(
                '[Rank %s] epoch %s ended with %s/%s accumulated microbatches; '
                'carrying gradients into the next epoch and deferring full checkpoint',
                self.rank,
                self.epoch,
                self.accumulation_count,
                info_dict['accum_grad'])
        return False

    def save_resume_checkpoint(self,
                               model,
                               optimizer,
                               scheduler,
                               info_dict,
                               next_epoch,
                               next_batch_idx,
                               model_name=None):
        if model_name is None:
            model_name = 'resume_step_{:08d}'.format(self.step)
        checkpoint_path = os.path.join(
            info_dict['model_dir'], '{}.pt'.format(model_name))
        checkpoint_started = time.perf_counter()
        save_training_checkpoint(
            model,
            optimizer,
            scheduler,
            self,
            checkpoint_path,
            next_epoch,
            next_batch_idx,
            info_dict['train_engine'],
            sync_uri=info_dict.get('checkpoint_sync_uri'),
            sync_mode=info_dict.get('checkpoint_sync_mode', 'synchronous'),
            sync_queue_dir=info_dict.get('checkpoint_sync_queue_dir'),
            sync_worker_pid=info_dict.get('checkpoint_sync_worker_pid'),
            sync_queue_wait_seconds=info_dict.get(
                'checkpoint_sync_queue_wait_seconds', 600),
            resume_contract_sha256=info_dict.get('resume_contract_sha256'))
        self.record_distributed_duration(
            info_dict,
            'checkpoint',
            time.perf_counter() - checkpoint_started,
            checkpoint=os.path.basename(checkpoint_path),
            sync_uri=info_dict.get('checkpoint_sync_uri'))
        logging.info(
            '[Rank %s] saved full-state checkpoint %s with next cursor %s:%s',
            self.rank, checkpoint_path, next_epoch, next_batch_idx)

    # @torch.inference_mode()
    def cv(self, model, cv_data_loader, writer, info_dict, on_batch_end=True, group_join=None):
        ''' Cross validation on
        '''
        logging.info('Epoch {} Step {} on_batch_end {} CV rank {}'.format(self.epoch, self.step, on_batch_end, self.rank))
        set_dataloader_seed(
            cv_data_loader,
            info_dict['seed'] + 50_000 + self.epoch * 100_000 + self.step * 100 + self.rank)
        model.eval()
        total_num_utts = 0
        cv_metrics = StepMetricAccumulator()
        arrows_coverage_dict = defaultdict(lambda: 0)
        with torch.inference_mode():
            for batch_idx, batch_dict in enumerate(tqdm.tqdm(cv_data_loader, position=self.rank, desc=f"[Rank {self.rank}] Validating...")):
                info_dict["tag"] = "CV"
                info_dict["step"] = self.step
                info_dict["epoch"] = self.epoch
                info_dict["batch_idx"] = batch_idx

                num_utts = len(batch_dict["utts"])
                if num_utts > 0:
                    _arrow_name = batch_dict["utts"][0].split("__")[0]
                    arrows_coverage_dict[_arrow_name] += 1
                total_num_utts += num_utts

                info_dict = batch_forward(model, batch_dict, info_dict)
                cv_metrics.update(info_dict['loss_dict'])
                log_per_step(None, info_dict)
        
        logging.info(f"[Rank {self.rank}] finished partial cv. waiting other process to gather metrics.")
        for k, v in arrows_coverage_dict.items():
            logging.info(f"[Rank {self.rank}] has {v} items in {k}.")
        total_num_batches = batch_idx + 1
        logging.info(f"[Rank {self.rank}] total_num_utts={total_num_utts}, total_num_batches={total_num_batches}, average CV batch size is {(total_num_utts / total_num_batches):.2f}")
        total_loss_dict = cv_metrics.reduce(self.device)
        info_dict['loss_dict'] = total_loss_dict
        cur_cv_score = total_loss_dict.get('acc', 0.0)
        if self.rank == 0:
            if cur_cv_score > self.cv_best_score:
                self.cv_best_score = cur_cv_score
                logging.info(f"[Rank {self.rank}] CV New best score: {self.cv_best_score}, will save new best ckpt.")
                best_model_name = 'checkpoint_best'
                save_model(model, best_model_name, info_dict)
                if info_dict.get('durable_best_checkpoint'):
                    best_checkpoint_path = os.path.join(
                        info_dict['model_dir'], best_model_name + '.pt')
                    best_metadata_path = os.path.join(
                        info_dict['model_dir'], best_model_name + '.yaml')
                    if info_dict.get('checkpoint_sync_mode') != 'queued':
                        raise RuntimeError(
                            'durable_best_checkpoint currently requires queued sync mode')
                    enqueue_stage_best_sync(
                        best_checkpoint_path,
                        best_metadata_path,
                        info_dict.get('checkpoint_sync_uri'),
                        self.cv_best_score,
                        self.epoch,
                        self.step,
                        info_dict.get('checkpoint_sync_queue_dir'),
                        info_dict.get('checkpoint_sync_worker_pid'),
                        info_dict.get('checkpoint_sync_queue_wait_seconds', 600))
            log_per_save(writer, info_dict)
        model_name = 'epoch_{}_whole'.format(self.epoch) if on_batch_end else 'epoch_{}_step_{}'.format(self.epoch, self.step)
        save_model(model, model_name, info_dict)
        logging.info(f"[Rank {self.rank}] Finished CV.")
        return total_loss_dict
