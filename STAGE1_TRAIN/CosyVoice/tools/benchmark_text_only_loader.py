#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import random
import statistics
import time

import torch
import torch.distributed as dist
from hyperpyyaml import load_hyperpyyaml
from torch.utils.data import DataLoader

from cosyvoice.dataset.dataset import Dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data_list', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_batches', type=int, default=200)
    parser.add_argument('--warmup_batches', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--prefetch', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1986)
    parser.add_argument('--compute_sleep_seconds', type=float, default=0.0)
    return parser.parse_args()


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def distribution(values):
    return {
        'count': len(values),
        'mean': statistics.fmean(values) if values else None,
        'p50': percentile(values, 0.50),
        'p95': percentile(values, 0.95),
        'p99': percentile(values, 0.99),
        'max': max(values) if values else None,
    }


def load_pipeline(config_path):
    overrides = {key: None for key in ('llm', 'flow', 'hift')}
    with open(config_path) as config_file:
        return load_hyperpyyaml(
            config_file, overrides=overrides)['data_pipeline']


def main():
    args = get_args()
    dist.init_process_group('gloo')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = Dataset(
        args.data_list,
        data_pipeline=load_pipeline(args.config),
        mode='train',
        shuffle=True,
        partition=True,
    )
    dataset.set_epoch(0)
    generator = torch.Generator()
    generator.manual_seed(args.seed + rank)
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch,
        generator=generator,
        pin_memory=False,
    )

    iterator = iter(loader)
    fetch_seconds = []
    global_fetch_max_seconds = []
    batch_rows = []
    key_hasher = hashlib.sha256()
    started = time.monotonic()
    for batch_index in range(args.num_batches):
        fetch_started = time.monotonic()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        fetch_elapsed = time.monotonic() - fetch_started

        rank_fetch = torch.tensor(fetch_elapsed, dtype=torch.float64)
        dist.all_reduce(rank_fetch, op=dist.ReduceOp.MAX)
        if args.compute_sleep_seconds > 0:
            time.sleep(args.compute_sleep_seconds)

        for utt in batch['utts']:
            key_hasher.update(utt.encode())
            key_hasher.update(b'\0')
        if batch_index >= args.warmup_batches:
            fetch_seconds.append(fetch_elapsed)
            global_fetch_max_seconds.append(rank_fetch.item())
            batch_rows.append(len(batch['utts']))

    elapsed = time.monotonic() - started
    os.makedirs(args.output_dir, exist_ok=True)
    rank_report = {
        'rank': rank,
        'world_size': world_size,
        'config': os.path.abspath(args.config),
        'data_list': os.path.abspath(args.data_list),
        'num_workers': args.num_workers,
        'prefetch': args.prefetch,
        'requested_batches': args.num_batches,
        'warmup_batches': args.warmup_batches,
        'measured_batches': len(fetch_seconds),
        'elapsed_seconds': elapsed,
        'fetch_seconds': distribution(fetch_seconds),
        'global_fetch_max_seconds': distribution(global_fetch_max_seconds),
        'batch_rows': distribution(batch_rows),
        'key_sha256': key_hasher.hexdigest(),
    }
    rank_path = os.path.join(args.output_dir, f'rank-{rank}.json')
    with open(rank_path, 'w') as report_file:
        json.dump(rank_report, report_file, indent=2, sort_keys=True)
        report_file.write('\n')

    dist.barrier()
    if rank == 0:
        reports = []
        for report_rank in range(world_size):
            with open(os.path.join(
                    args.output_dir, f'rank-{report_rank}.json')) as report_file:
                reports.append(json.load(report_file))
        aggregate = {
            'world_size': world_size,
            'num_workers_per_rank': args.num_workers,
            'prefetch_per_worker': args.prefetch,
            'rank_fetch_p99_seconds': distribution([
                report['fetch_seconds']['p99'] for report in reports
            ]),
            'rank_fetch_max_seconds': distribution([
                report['fetch_seconds']['max'] for report in reports
            ]),
            'rank_elapsed_seconds': distribution([
                report['elapsed_seconds'] for report in reports
            ]),
            'rank_reports': reports,
        }
        with open(os.path.join(
                args.output_dir, 'aggregate.json'), 'w') as report_file:
            json.dump(aggregate, report_file, indent=2, sort_keys=True)
            report_file.write('\n')
        print(json.dumps(aggregate, sort_keys=True))

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
