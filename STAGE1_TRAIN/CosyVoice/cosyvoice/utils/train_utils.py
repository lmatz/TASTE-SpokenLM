# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2023 Horizon Inc. (authors: Xingchen Song)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import nullcontext
import logging
import os
import random
import torch
import json
import re
import datetime
import yaml
import numpy as np

import deepspeed
import torch.optim as optim
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live

from cosyvoice.dataset.dataset import Dataset
from cosyvoice.utils.checkpoint import unwrap_model
from cosyvoice.utils.scheduler import WarmupLR, NoamHoldAnnealing, ConstantLR


def init_distributed(args):
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    logging.info('training on multiple gpus, this gpu {}'.format(local_rank) +
                 ', rank {}, world_size {}'.format(rank, world_size))
    if args.train_engine == 'torch_ddp':
        torch.cuda.set_device(local_rank)
        dist.init_process_group(args.dist_backend)
    else:
        deepspeed.init_distributed(dist_backend=args.dist_backend)
    return world_size, local_rank, rank


def init_dataset_and_dataloader(args, configs):
    train_dataset = Dataset(args.train_data, data_pipeline=configs['data_pipeline'], mode='train', shuffle=True, partition=True)
    # cv_dataset = Dataset(args.cv_data, data_pipeline=configs['data_pipeline'], mode='train', shuffle=False, partition=False)
    cv_dataset = Dataset(args.cv_data, data_pipeline=configs['data_pipeline'], mode='train', shuffle=False, partition=True) # set to true for speeding up validation

    # do not use persistent_workers=True, as whisper tokenizer opens tiktoken file each time when the for loop starts
    rank = int(os.environ.get('RANK', 0))
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed + rank)
    cv_generator = torch.Generator()
    cv_generator.manual_seed(args.seed + 10_000 + rank)
    loader_kwargs = {
        'batch_size': None,
        'pin_memory': args.pin_memory,
        'num_workers': args.num_workers,
        'worker_init_fn': seed_worker,
    }
    if args.num_workers > 0:
        loader_kwargs['prefetch_factor'] = args.prefetch
    train_data_loader = DataLoader(
        train_dataset, generator=train_generator, **loader_kwargs)
    cv_data_loader = DataLoader(
        cv_dataset, generator=cv_generator, **loader_kwargs)
    return train_dataset, cv_dataset, train_data_loader, cv_data_loader


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def set_dataloader_seed(data_loader, seed):
    if data_loader.generator is None:
        raise RuntimeError('dataloader has no dedicated generator')
    data_loader.generator.manual_seed(seed)



def check_modify_and_save_config(args, configs):
    if args.train_engine == "torch_ddp":
        configs['train_conf']["dtype"] = 'fp32'
    else:
        with open(args.deepspeed_config, 'r') as fin:
            ds_configs = json.load(fin)
        if "fp16" in ds_configs and ds_configs["fp16"]["enabled"]:
            configs['train_conf']["dtype"] = "fp16"
        elif "bf16" in ds_configs and ds_configs["bf16"]["enabled"]:
            configs['train_conf']["dtype"] = "bf16"
        else:
            configs['train_conf']["dtype"] = "fp32"
        assert ds_configs["train_micro_batch_size_per_gpu"] == 1
        # if use deepspeed, override ddp config
        configs['train_conf']['save_per_step'] = int(configs['train_conf']['save_per_step'] * configs['train_conf']['accum_grad'] / ds_configs["gradient_accumulation_steps"])
        configs['train_conf']['accum_grad'] = ds_configs["gradient_accumulation_steps"]
        configs['train_conf']['grad_clip'] = ds_configs["gradient_clipping"]
        configs['train_conf']['log_interval'] = ds_configs["steps_per_print"]
    return configs


def wrap_cuda_model(args, model):
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if args.train_engine == "torch_ddp":  # native pytorch ddp
        assert (torch.cuda.is_available())
        model.cuda()
        # print(model.device)
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        # model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
    else:
        if int(os.environ.get('RANK', 0)) == 0:
            logging.info("Estimating model states memory needs (zero2)...")
            estimate_zero2_model_states_mem_needs_all_live(
                model,
                num_gpus_per_node=local_world_size,
                num_nodes=world_size // local_world_size)
    return model


def init_optimizer_and_scheduler(args, configs, model):
    if configs['train_conf']['optim'] == 'adam':
        optimizer = optim.Adam(model.parameters(), **configs['train_conf']['optim_conf'])
    elif configs['train_conf']['optim'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), **configs['train_conf']['optim_conf'])
    else:
        raise ValueError("unknown optimizer: " + configs['train_conf'])

    if configs['train_conf']['scheduler'] == 'warmuplr':
        scheduler_type = WarmupLR
        scheduler = WarmupLR(optimizer, **configs['train_conf']['scheduler_conf'])
    elif configs['train_conf']['scheduler'] == 'NoamHoldAnnealing':
        scheduler_type = NoamHoldAnnealing
        scheduler = NoamHoldAnnealing(optimizer, **configs['train_conf']['scheduler_conf'])
    elif configs['train_conf']['scheduler'] == 'constantlr':
        scheduler_type = ConstantLR
        scheduler = ConstantLR(optimizer)
    else:
        raise ValueError("unknown scheduler: " + configs['train_conf'])

    # use deepspeed optimizer for speedup
    if args.train_engine == "deepspeed":
        def scheduler(opt):
            return scheduler_type(opt, **configs['train_conf']['scheduler_conf'])
        model, optimizer, _, scheduler = deepspeed.initialize(
            args=args,
            model=model,
            optimizer=None,
            lr_scheduler=scheduler,
            model_parameters=model.parameters())

    return model, optimizer, scheduler


def init_summarywriter(args):
    writer = None
    if int(os.environ.get('RANK', 0)) == 0:
        os.makedirs(args.model_dir, exist_ok=True)
        writer = SummaryWriter(args.tensorboard_dir)
    return writer


def save_model(model, model_name, info_dict):
    rank = int(os.environ.get('RANK', 0))
    model_dir = info_dict["model_dir"]
    save_model_path = os.path.join(model_dir, '{}.pt'.format(model_name))

    if info_dict["train_engine"] == "torch_ddp":
        if rank == 0:
            torch.save(unwrap_model(model).state_dict(), save_model_path)
    else:
        with torch.no_grad():
            model.save_checkpoint(save_dir=model_dir,
                                  tag=model_name,
                                  client_state=info_dict)
    if rank == 0:
        info_path = re.sub('.pt$', '.yaml', save_model_path)
        info_dict['save_time'] = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        with open(info_path, 'w') as fout:
            data = yaml.dump(info_dict)
            fout.write(data)
        logging.info('[Rank {}] Checkpoint: save to checkpoint {}'.format(rank, save_model_path))


def cosyvoice_join(group_join, info_dict):
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))

    if info_dict["batch_idx"] != 0:
        # we try to join all rank in both ddp and deepspeed mode, in case different rank has different lr
        try:
            dist.monitored_barrier(group=group_join,
                                   timeout=group_join.options._timeout)
            return False
        except RuntimeError as e:
            logging.info("Detected uneven workload distribution: {}\n".format(e) +
                         "Break current worker to manually join all workers, " +
                         "world_size {}, current rank {}, current local_rank {}\n".
                         format(world_size, rank, local_rank))
            return True
    else:
        return False


def batch_forward(model, batch, info_dict):
    device = int(os.environ.get('LOCAL_RANK', 0))

    dtype = info_dict["dtype"]
    if dtype == "fp16":
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    else:  # fp32
        dtype = torch.float32

    if info_dict['train_engine'] == 'torch_ddp':
        # autocast = nullcontext()
        autocast = torch.cuda.amp.autocast(enabled=True, dtype=dtype, cache_enabled=False)
    else:
        autocast = torch.cuda.amp.autocast(enabled=True, dtype=dtype, cache_enabled=False)

    with autocast:
        info_dict['loss_dict'] = model(batch, device)
    for name, value in info_dict['loss_dict'].items():
        if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
            raise FloatingPointError(
                'non-finite {} value at epoch {} batch {} step {}'.format(
                    name,
                    info_dict.get('epoch', 0),
                    info_dict.get('batch_idx', 0),
                    info_dict.get('step', 0)))
    return info_dict


def batch_backward(model, info_dict):
    if info_dict["train_engine"] == "deepspeed":
        scaled_loss = model.backward(info_dict['loss_dict']['loss'])
    else:
        scaled_loss = info_dict['loss_dict']['loss'] / info_dict['accum_grad']
        scaled_loss.backward()

    info_dict['loss_dict']['loss'] = scaled_loss
    return info_dict


def update_parameter_and_lr(model, optimizer, scheduler, info_dict):
    grad_norm = 0.0
    optimizer_step = False
    if info_dict['train_engine'] == "deepspeed":
        info_dict["is_gradient_accumulation_boundary"] = model.is_gradient_accumulation_boundary()
        model.step()
        grad_norm = model.get_global_grad_norm()
        optimizer_step = info_dict["is_gradient_accumulation_boundary"]
        if optimizer_step and not torch.isfinite(torch.as_tensor(grad_norm)).all():
            raise FloatingPointError('non-finite deepspeed gradient norm')
    elif info_dict["accumulation_boundary"]:
        grad_norm = clip_grad_norm_(model.parameters(), info_dict['grad_clip'])
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(
                'non-finite gradient norm at epoch {} batch {} step {}'.format(
                    info_dict.get('epoch', 0),
                    info_dict.get('batch_idx', 0),
                    info_dict.get('step', 0)))
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        optimizer_step = True
    info_dict["lr"] = optimizer.param_groups[0]['lr']
    info_dict["grad_norm"] = grad_norm
    info_dict["optimizer_step"] = optimizer_step
    return info_dict


def check_model_finite(model):
    target_model = unwrap_model(model)
    nonfinite = []
    for name, value in target_model.state_dict().items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            nonfinite.append(name)
    if nonfinite:
        raise FloatingPointError(
            'non-finite model state tensors: {}'.format(', '.join(nonfinite[:20])))


def log_per_step(writer, info_dict):
    tag = info_dict["tag"]
    epoch = info_dict.get('epoch', 0)
    step = info_dict["step"]
    batch_idx = info_dict["batch_idx"]
    loss_dict = info_dict['loss_dict']
    rank = int(os.environ.get('RANK', 0))

    # only rank 0 write to tensorboard to avoid multi-process write
    if writer is not None:
        if (info_dict['train_engine'] == 'deepspeed' and info_dict['is_gradient_accumulation_boundary'] is True) or \
           (info_dict['train_engine'] == 'torch_ddp' and info_dict['optimizer_step']):
            for k in ['epoch', 'lr', 'grad_norm']:
                writer.add_scalar('{}/{}'.format(tag, k), info_dict[k], step + 1)
            for k, v in loss_dict.items():
                writer.add_scalar('{}/{}'.format(tag, k), v, step + 1)

    # TRAIN & CV, Shell log (stdout)
    if (info_dict['batch_idx'] + 1) % info_dict['log_interval'] == 0:
        log_str = '{} Batch {}/{} '.format(tag, epoch, batch_idx + 1)
        for name, value in loss_dict.items():
            log_str += '{} {:.6f} '.format(name, value)
        if tag == "TRAIN":
            log_str += 'lr {:.8f} grad_norm {:.6f}'.format(
                info_dict["lr"], info_dict['grad_norm'])
        log_str += ' rank {}'.format(rank)
        logging.info(log_str)


def log_per_save(writer, info_dict):
    tag = info_dict["tag"]
    epoch = info_dict["epoch"]
    step = info_dict["step"]
    loss_dict = info_dict["loss_dict"]
    lr = info_dict['lr']
    rank = int(os.environ.get('RANK', 0))
    logging.info(
        'Epoch {} Step {} CV info lr {} {} rank {}'.format(
            epoch, step + 1, lr, rank, ' '.join(['{}_{}'.format(k, v) for k, v in loss_dict.items()])))

    if writer is not None:
        for k in ['epoch', 'lr']:
            writer.add_scalar('{}/{}'.format(tag, k), info_dict[k], step + 1)
        for k, v in loss_dict.items():
            writer.add_scalar('{}/{}'.format(tag, k), v, step + 1)
