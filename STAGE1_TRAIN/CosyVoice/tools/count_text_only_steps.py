#!/usr/bin/env python3

import argparse
import json
import random

import torch
import torch.distributed as dist
from hyperpyyaml import load_hyperpyyaml
from torch.utils.data import DataLoader

from cosyvoice.dataset.dataset import Dataset
from cosyvoice.utils.scheduler import WarmupLR


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data_list', required=True)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--prefetch', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1986)
    parser.add_argument('--output')
    return parser.parse_args()


def load_config(config_path):
    overrides = {key: None for key in ('llm', 'flow', 'hift')}
    with open(config_path) as config_file:
        return load_hyperpyyaml(config_file, overrides=overrides)


def simulate_lr(config, total_steps):
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam(
        [parameter], **config['train_conf']['optim_conf'])
    scheduler = WarmupLR(
        optimizer, **config['train_conf']['scheduler_conf'])
    learning_rates = []
    for _ in range(total_steps):
        learning_rates.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
    peak_lr = max(learning_rates)
    peak_step = learning_rates.index(peak_lr) + 1
    return {
        'peak_lr': peak_lr,
        'peak_optimizer_step': peak_step,
        'final_lr': learning_rates[-1],
    }


def main():
    args = get_args()
    dist.init_process_group('gloo')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    config = load_config(args.config)
    epochs = args.epochs or config['train_conf']['max_epoch']
    accum_grad = config['train_conf']['accum_grad']
    warmup_steps = config['train_conf']['scheduler_conf']['warmup_steps']

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = Dataset(
        args.data_list,
        data_pipeline=config['data_pipeline'],
        mode='train',
        shuffle=True,
        partition=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch,
        pin_memory=False,
    )

    epoch_reports = []
    total_effective_optimizer_steps = 0
    for epoch in range(epochs):
        dataset.set_epoch(epoch)

        local_microbatches = 0
        local_rows = 0
        for batch in loader:
            local_microbatches += 1
            local_rows += len(batch['utts'])

        local_counts = torch.tensor(
            [local_microbatches, local_rows], dtype=torch.int64)
        gathered = [
            torch.zeros_like(local_counts) for _ in range(world_size)
        ]
        dist.all_gather(gathered, local_counts)
        rank_microbatches = [int(item[0]) for item in gathered]
        rank_rows = [int(item[1]) for item in gathered]
        effective_microbatches = min(rank_microbatches)
        effective_optimizer_steps = effective_microbatches // accum_grad
        total_effective_optimizer_steps += effective_optimizer_steps
        epoch_reports.append({
            'epoch': epoch,
            'rank_microbatches': rank_microbatches,
            'rank_rows': rank_rows,
            'effective_microbatches': effective_microbatches,
            'effective_optimizer_steps': effective_optimizer_steps,
        })

    if rank == 0:
        if total_effective_optimizer_steps <= 0:
            raise AssertionError('no effective optimizer steps')
        report = {
            'status': 'PASS',
            'world_size': world_size,
            'epochs': epochs,
            'num_workers_per_rank': args.num_workers,
            'prefetch_per_worker': args.prefetch,
            'accum_grad': accum_grad,
            'warmup_steps': warmup_steps,
            'total_effective_optimizer_steps':
                total_effective_optimizer_steps,
            'warmup_fraction':
                warmup_steps / total_effective_optimizer_steps,
            'warmup_completes':
                total_effective_optimizer_steps > warmup_steps,
            'lr_schedule': simulate_lr(
                config, total_effective_optimizer_steps),
            'epoch_reports': epoch_reports,
        }
        serialized = json.dumps(report, indent=2, sort_keys=True) + '\n'
        if args.output:
            with open(args.output, 'w') as output_file:
                output_file.write(serialized)
        print(serialized, end='')

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
