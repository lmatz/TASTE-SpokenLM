#!/usr/bin/env python3

import argparse
import random

import torch
from hyperpyyaml import load_hyperpyyaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from cosyvoice.dataset.dataset import Dataset
from cosyvoice.utils.scheduler import WarmupLR


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_config', required=True)
    parser.add_argument('--optimized_config', required=True)
    parser.add_argument('--data_list', required=True)
    parser.add_argument('--seed', type=int, default=1986)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path) as config_file:
        return load_hyperpyyaml(
            config_file, overrides={'flow': None, 'hift': None})


def load_batches(config, data_list, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    dataset = Dataset(
        data_list,
        data_pipeline=config['data_pipeline'],
        mode='train',
        shuffle=True,
        partition=True,
    )
    dataset.set_epoch(0)
    loader = DataLoader(dataset, batch_size=None, num_workers=0)
    iterator = iter(loader)
    return [next(iterator), next(iterator)]


def assert_batch_equal(baseline, optimized, batch_index):
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
        if not torch.equal(baseline[name], optimized[name]):
            raise AssertionError(f'batch {batch_index} {name} mismatch')


def train_one_step(model, batches, config, seed, device):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), **config['train_conf']['optim_conf'])
    scheduler = WarmupLR(
        optimizer, **config['train_conf']['scheduler_conf'])
    optimizer.zero_grad()

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    metrics = []
    accum_grad = config['train_conf']['accum_grad']
    for batch in batches:
        loss_dict = model(batch, device)
        (loss_dict['loss'] / accum_grad).backward()
        metrics.append({
            'loss': loss_dict['loss'].detach().cpu(),
            'acc': loss_dict['acc'].detach().cpu(),
            'len': loss_dict['len'].detach().cpu(),
        })

    grad_norm = clip_grad_norm_(
        model.parameters(), config['train_conf']['grad_clip'])
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    return metrics, grad_norm.detach().cpu(), model


def main():
    args = get_args()
    baseline_config = load_config(args.baseline_config)
    optimized_config = load_config(args.optimized_config)
    baseline_batches = load_batches(
        baseline_config, args.data_list, args.seed)
    optimized_batches = load_batches(
        optimized_config, args.data_list, args.seed)
    for batch_index, (baseline, optimized) in enumerate(zip(
            baseline_batches, optimized_batches)):
        assert_batch_equal(baseline, optimized, batch_index)

    baseline_model = baseline_config['llm']
    optimized_model = optimized_config['llm']
    baseline_state = baseline_model.state_dict()
    optimized_state = optimized_model.state_dict()
    if baseline_state.keys() != optimized_state.keys():
        raise AssertionError('initial model keys mismatch')
    for name in baseline_state:
        if not torch.equal(baseline_state[name], optimized_state[name]):
            raise AssertionError(f'initial model tensor mismatch: {name}')

    device = torch.device(args.device)
    baseline_metrics, baseline_grad_norm, baseline_model = train_one_step(
        baseline_model, baseline_batches, baseline_config, args.seed, device)
    optimized_metrics, optimized_grad_norm, optimized_model = train_one_step(
        optimized_model, optimized_batches, optimized_config, args.seed, device)

    for batch_index, (baseline, optimized) in enumerate(zip(
            baseline_metrics, optimized_metrics)):
        for name in ('loss', 'acc', 'len'):
            if not torch.equal(baseline[name], optimized[name]):
                raise AssertionError(
                    f'batch {batch_index} {name} mismatch: '
                    f'{baseline[name]} != {optimized[name]}')
    if not torch.equal(baseline_grad_norm, optimized_grad_norm):
        raise AssertionError(
            f'grad norm mismatch: {baseline_grad_norm} != '
            f'{optimized_grad_norm}')
    for (baseline_name, baseline_parameter), (
            optimized_name, optimized_parameter) in zip(
                baseline_model.named_parameters(),
                optimized_model.named_parameters()):
        if baseline_name != optimized_name:
            raise AssertionError('post-step parameter names mismatch')
        if not torch.equal(
                baseline_parameter.detach(),
                optimized_parameter.detach()):
            raise AssertionError(
                f'post-step parameter mismatch: {baseline_name}')

    print({
        'status': 'PASS',
        'microbatches': len(baseline_batches),
        'rows': sum(len(batch['utts']) for batch in baseline_batches),
        'losses': [
            float(metric['loss']) for metric in baseline_metrics
        ],
        'accuracies': [
            float(metric['acc']) for metric in baseline_metrics
        ],
        'valid_lengths': [
            int(metric['len']) for metric in baseline_metrics
        ],
        'preclip_grad_norm': float(baseline_grad_norm),
        'post_step_parameters_exact': True,
    })


if __name__ == '__main__':
    main()
