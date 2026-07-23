#!/usr/bin/env python3

import argparse
import json
import os
import pathlib

import torch
import torch.distributed as dist

from cosyvoice.utils.runtime_contract import (
    enforce_runtime_contract,
    runtime_payload,
)
from cosyvoice.utils.scheduler import WarmupLR


class TinyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.trainable = torch.nn.Linear(3, 2)
        self.frozen = torch.nn.Linear(2, 1)
        for parameter in self.frozen.parameters():
            parameter.requires_grad = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    dist.init_process_group(backend='gloo')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError('expected exactly two validation ranks')

    output_dir = pathlib.Path(args.output_dir).resolve()
    config_path = output_dir / 'config.yaml'
    expected_path = output_dir / 'expected.json'
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            'train_conf:\n'
            '    optim: adam\n'
            '    optim_conf:\n'
            '        lr: 0.0016\n')
    dist.barrier()

    torch.manual_seed(20260723)
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0016)
    scheduler = WarmupLR(optimizer, warmup_steps=20000)
    runtime_args = argparse.Namespace(
        config=str(config_path),
        model='llm',
        train_engine='torch_ddp',
        seed=20260723,
        model_dir=str(output_dir),
        expected_runtime_contract=str(expected_path),
        runtime_contract_dir=str(output_dir),
        local_rank=int(os.environ['LOCAL_RANK']),
    )
    configs = {
        'train_conf': {
            'optim': 'adam',
            'optim_conf': {'lr': 0.0016},
            'local_rank': int(os.environ['LOCAL_RANK']),
        },
    }
    payload = runtime_payload(
        runtime_args,
        configs,
        model,
        optimizer,
        scheduler,
        rank,
        world_size,
    )
    if rank == 0:
        expected_path.write_text(json.dumps({
            'format': 'cosyvoice-expected-runtime-contract-v1',
            'assertions': {
                'config': {
                    'sha256': payload['invariant']['config']['sha256'],
                },
                'model': payload['invariant']['model'],
                'optimizer': {
                    'class': 'torch.optim.adam.Adam',
                    'defaults': {
                        'lr': 0.0016,
                        'betas': [0.9, 0.999],
                        'eps': 1e-08,
                        'weight_decay': 0,
                        'amsgrad': False,
                    },
                },
                'scheduler': {
                    'class': 'cosyvoice.utils.scheduler.WarmupLR',
                    'warmup_steps': 20000,
                },
                'runtime': {
                    'world_size': 2,
                    'distributed_backend': 'gloo',
                    'python_path': payload[
                        'invariant']['runtime']['python_path'],
                    'pythonpath_environment': payload[
                        'invariant']['runtime']['pythonpath_environment'],
                    'matcha_module_file': payload[
                        'invariant']['runtime']['matcha_module_file'],
                },
            },
        }))
    dist.barrier()

    enforce_runtime_contract(
        runtime_args,
        configs,
        model,
        optimizer,
        scheduler,
        expected_path,
        output_dir,
    )
    if rank == 0:
        aggregate = json.loads(
            (output_dir / 'runtime-contract-aggregate.json').read_text())
        if not aggregate['pass'] or aggregate['ranks'] != [0, 1]:
            raise RuntimeError('two-rank runtime contract did not pass')
        print('two-rank runtime contract validation: PASS')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
