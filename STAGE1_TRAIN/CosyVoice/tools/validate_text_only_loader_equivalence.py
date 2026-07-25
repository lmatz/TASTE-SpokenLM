#!/usr/bin/env python3

import argparse
import random

import torch
from hyperpyyaml import load_hyperpyyaml
from torch.utils.data import DataLoader

from cosyvoice.dataset.dataset import Dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_config', required=True)
    parser.add_argument('--optimized_config', required=True)
    parser.add_argument('--data_list', required=True)
    parser.add_argument('--num_batches', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--prefetch', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1986)
    return parser.parse_args()


def load_pipeline(config_path):
    overrides = {key: None for key in ('llm', 'flow', 'hift')}
    with open(config_path) as config_file:
        return load_hyperpyyaml(
            config_file, overrides=overrides)['data_pipeline']


def make_loader(config_path, args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = Dataset(
        args.data_list,
        data_pipeline=load_pipeline(config_path),
        mode='train',
        shuffle=True,
        partition=True,
    )
    dataset.set_epoch(0)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader_kwargs = {
        'batch_size': None,
        'num_workers': args.num_workers,
        'generator': generator,
    }
    if args.num_workers > 0:
        loader_kwargs['prefetch_factor'] = args.prefetch
    return DataLoader(dataset, **loader_kwargs)


def assert_tensor_equal(name, baseline, optimized):
    if baseline.dtype != optimized.dtype:
        raise AssertionError(
            f'{name} dtype mismatch: {baseline.dtype} != {optimized.dtype}')
    if baseline.shape != optimized.shape:
        raise AssertionError(
            f'{name} shape mismatch: {baseline.shape} != {optimized.shape}')
    if not torch.equal(baseline, optimized):
        raise AssertionError(f'{name} value mismatch')


def main():
    args = get_args()
    baseline_loader = make_loader(args.baseline_config, args)
    optimized_loader = make_loader(args.optimized_config, args)
    baseline_iterator = iter(baseline_loader)
    optimized_iterator = iter(optimized_loader)

    compared_batches = 0
    compared_rows = 0
    for batch_index in range(args.num_batches):
        try:
            baseline = next(baseline_iterator)
            optimized = next(optimized_iterator)
        except StopIteration:
            break

        if baseline['utts'] != optimized['utts']:
            raise AssertionError(f'batch {batch_index} utt order mismatch')
        for name in (
                'speech_token',
                'speech_token_len',
                'speech_feat_len',
                'text_token',
                'text_token_len',
                'utt_embedding',
                'spk_embedding',
                'embedding'):
            assert_tensor_equal(
                f'batch {batch_index} {name}', baseline[name], optimized[name])
        if baseline['text'] != optimized['text']:
            raise AssertionError(f'batch {batch_index} text mismatch')
        if baseline['speech_feat'].shape[:2] != optimized['speech_feat'].shape[:2]:
            raise AssertionError(
                f'batch {batch_index} speech feature frame mismatch: '
                f'{baseline["speech_feat"].shape} != '
                f'{optimized["speech_feat"].shape}')
        if baseline['speech_feat'].shape[2] != 80:
            raise AssertionError('baseline mel width is not 80')
        if optimized['speech_feat'].shape[2] != 1:
            raise AssertionError('optimized placeholder width is not 1')

        compared_batches += 1
        compared_rows += len(baseline['utts'])

    if compared_batches == 0:
        raise AssertionError('no batches compared')
    print({
        'status': 'PASS',
        'compared_batches': compared_batches,
        'compared_rows': compared_rows,
        'num_workers': args.num_workers,
        'prefetch': args.prefetch if args.num_workers > 0 else None,
    })


if __name__ == '__main__':
    main()
