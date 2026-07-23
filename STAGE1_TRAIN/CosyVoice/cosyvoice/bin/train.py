# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
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

from __future__ import print_function
import argparse
import datetime
import json
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from copy import deepcopy
import torch
import os
import shutil
import torch.distributed as dist
import deepspeed

from hyperpyyaml import load_hyperpyyaml

from torch.distributed.elastic.multiprocessing.errors import record

from cosyvoice.utils.executor import Executor
from cosyvoice.utils.checkpoint import (
    load_training_checkpoint,
    parameter_inventory,
    seed_everything,
    torch_load,
    unwrap_model,
)
from cosyvoice.utils.train_utils import (
    init_distributed,
    init_dataset_and_dataloader,
    init_optimizer_and_scheduler,
    init_summarywriter, save_model,
    wrap_cuda_model, check_modify_and_save_config, set_dataloader_seed)


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--train_engine',
                        default='torch_ddp',
                        choices=['torch_ddp', 'deepspeed'],
                        help='Engine for paralleled training')
    parser.add_argument('--model', required=True, help='model which will be trained')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        '--checkpoint', help='model-only checkpoint used to initialize a new stage')
    checkpoint_group.add_argument(
        '--resume', help='full-state checkpoint used to resume the same stage')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--deepspeed.save_states',
                        dest='save_states',
                        default='model_only',
                        choices=['model_only', 'model+optimizer'],
                        help='save model/optimizer states')
    parser.add_argument('--timeout',
                        default=30,
                        type=int,
                        help='timeout (in seconds) of cosyvoice_join.')
    parser.add_argument('--seed',
                        default=777,
                        type=int,
                        help='base seed for model and dataloader RNGs')
    parser.add_argument('--full_checkpoint_per_step',
                        default=0,
                        type=int,
                        help='save a resumable checkpoint every N optimizer steps')
    parser.add_argument('--max_optimizer_steps',
                        default=0,
                        type=int,
                        help='stop after N optimizer steps; 0 disables the cap')
    parser.add_argument('--checkpoint_sync_uri',
                        help='S3 prefix for synchronous, checksum-verified full checkpoints')
    parser.add_argument('--checkpoint_sync_mode',
                        default='synchronous',
                        choices=['synchronous', 'queued'],
                        help='publish checkpoints inline or through a supervised queue worker')
    parser.add_argument('--checkpoint_sync_queue_dir',
                        help='local queue directory used by queued checkpoint sync')
    parser.add_argument('--checkpoint_sync_worker_pid',
                        type=int,
                        help='PID of the supervised queued checkpoint sync worker')
    parser.add_argument('--checkpoint_sync_queue_wait_seconds',
                        default=600,
                        type=int,
                        help='maximum wait for a prior queued upload to finish')
    parser.add_argument('--durable_best_checkpoint',
                        action='store_true',
                        help='publish every improved checkpoint_best through the sync queue')
    parser.add_argument('--resume_contract_sha256',
                        help='exact run contract required by full-state resume checkpoints')
    parser.add_argument('--resume_best_checkpoint',
                        help='verified model-only best checkpoint restored from durable storage')
    parser.add_argument('--resume_best_score',
                        type=float,
                        help='score associated with --resume_best_checkpoint')
    parser.add_argument('--finite_parameter_check_per_step',
                        default=0,
                        type=int,
                        help='scan model state for NaN/Inf every N optimizer steps')
    parser.add_argument('--metrics_jsonl',
                        help='rank-zero structured training/validation/checkpoint metrics')
    parser.add_argument('--sample_trace_dir',
                        help='optional per-rank JSONL trace of batch utterance keys')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


@record
def main():
    args = get_args()
    if args.checkpoint_sync_mode == 'queued':
        if not args.checkpoint_sync_uri:
            raise ValueError('queued checkpoint sync requires --checkpoint_sync_uri')
        if not args.checkpoint_sync_queue_dir:
            raise ValueError('queued checkpoint sync requires --checkpoint_sync_queue_dir')
        if not args.checkpoint_sync_worker_pid:
            raise ValueError('queued checkpoint sync requires --checkpoint_sync_worker_pid')
    if args.durable_best_checkpoint and args.checkpoint_sync_mode != 'queued':
        raise ValueError('--durable_best_checkpoint requires queued checkpoint sync')
    if (args.resume_best_checkpoint is None) != (args.resume_best_score is None):
        raise ValueError(
            '--resume_best_checkpoint and --resume_best_score must be supplied together')
    # logging.basicConfig(level=logging.DEBUG,
    #                     format='%(asctime)s %(levelname)s %(message)s')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    override_dict = {k: None for k in ['llm', 'flow', 'hift'] if k != args.model}
    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)
    configs['train_conf'].update(vars(args))

    # Init env for ddp
    _, _, rank = init_distributed(args)
    seed_everything(args.seed + rank)

    # Get dataset & dataloader
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
        init_dataset_and_dataloader(args, configs)

    # Do some sanity checks and save config to arsg.model_dir
    configs = check_modify_and_save_config(args, configs)

    # Tensorboard summary
    writer = init_summarywriter(args)

    # load checkpoint
    model = configs[args.model]
    if args.checkpoint is not None:
        model.load_state_dict(torch_load(args.checkpoint, map_location='cpu'))

    # Dispatch model from cpu to gpu
    model = wrap_cuda_model(args, model)

    # Get optimizer & scheduler
    model, optimizer, scheduler = init_optimizer_and_scheduler(args, configs, model)

    # Get executor
    executor = Executor()

    info_dict = deepcopy(configs['train_conf'])
    start_epoch = 0
    start_batch_idx = 0
    resume_rng_state = None
    if args.resume is not None:
        resume_state = load_training_checkpoint(
            model,
            optimizer,
            scheduler,
            executor,
            args.resume,
            args.train_engine,
            expected_resume_contract_sha256=args.resume_contract_sha256)
        start_epoch = resume_state['next_epoch']
        start_batch_idx = resume_state['next_batch_idx']
        resume_rng_state = resume_state['rng_state']
        logging.info(
            'Resuming from %s at optimizer step %s, epoch %s, batch %s',
            args.resume, executor.step, start_epoch, start_batch_idx)
        if args.durable_best_checkpoint and executor.cv_best_score > 0:
            if args.resume_best_checkpoint is None:
                raise ValueError(
                    'resume checkpoint has a prior best score but no durable best model')
            if args.resume_best_score < executor.cv_best_score:
                raise ValueError(
                    'durable best score is older than the resume checkpoint best score')
        if args.resume_best_checkpoint is not None:
            if not os.path.isfile(args.resume_best_checkpoint):
                raise FileNotFoundError(args.resume_best_checkpoint)
            executor.cv_best_score = max(
                executor.cv_best_score, args.resume_best_score)
            if rank == 0:
                shutil.copyfile(
                    args.resume_best_checkpoint,
                    os.path.join(args.model_dir, 'checkpoint_best.pt'))
                logging.info(
                    'Restored durable best model with score %s',
                    args.resume_best_score)
            dist.barrier()
    else:
        save_model(model, 'init', info_dict)

    inventory = parameter_inventory(model)
    logging.info(
        'Parameter inventory: %s trainable tensors, %s frozen tensors',
        len(inventory['trainable']), len(inventory['frozen']))
    if rank == 0:
        named_parameters = dict(unwrap_model(model).named_parameters())
        inventory_payload = {
            'trainable': inventory['trainable'],
            'frozen': inventory['frozen'],
            'trainable_numel': sum(
                named_parameters[name].numel() for name in inventory['trainable']),
            'frozen_numel': sum(
                named_parameters[name].numel() for name in inventory['frozen']),
        }
        with open(os.path.join(args.model_dir, 'parameter_inventory.json'), 'w') as inventory_file:
            json.dump(inventory_payload, inventory_file, indent=2, sort_keys=True)
            inventory_file.write('\n')

    # Copy the config file to the exp dir
    src_fpath = args.config
    tgt_fpath = os.path.join(args.model_dir, "config.yaml")
    if rank == 0:
        shutil.copyfile(src_fpath, tgt_fpath)
    dist.barrier()

    if info_dict['max_optimizer_steps'] > 0 and executor.step >= info_dict['max_optimizer_steps']:
        logging.info(
            'Current optimizer step %s already reached max_optimizer_steps=%s',
            executor.step, info_dict['max_optimizer_steps'])
        return
    if start_epoch >= info_dict['max_epoch']:
        logging.info('Checkpoint cursor is already past max_epoch=%s', info_dict['max_epoch'])
        return

    # Start training loop
    for epoch in range(start_epoch, info_dict['max_epoch']):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        cv_dataset.set_epoch(epoch)
        set_dataloader_seed(
            train_data_loader, args.seed + epoch * 100_000 + rank)
        set_dataloader_seed(
            cv_data_loader, args.seed + 50_000 + epoch * 100_000 + rank)
        dist.barrier()
        group_join = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))
        # group_join = dist.new_group(backend=args.dist_backend, timeout=datetime.timedelta(seconds=args.timeout))
        stop_requested = executor.train_one_epoc(
            model,
            optimizer,
            scheduler,
            train_data_loader,
            cv_data_loader,
            writer,
            info_dict,
            group_join,
            start_batch_idx=start_batch_idx if epoch == start_epoch else 0,
            resume_rng_state=resume_rng_state if epoch == start_epoch else None)
        logging.info(f"Finished epoch {epoch} training. Try to destroy process group")
        dist.destroy_process_group(group_join)
        del group_join
        start_batch_idx = 0
        resume_rng_state = None
        if stop_requested:
            break

if __name__ == '__main__':
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
