#!/usr/bin/env python3

import argparse
import json
import pathlib
import tempfile

import torch

from cosyvoice.utils.runtime_contract import (
    compare_expected,
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


def make_args(config_path, output_dir, expected_path):
    return argparse.Namespace(
        config=str(config_path),
        model='llm',
        train_engine='torch_ddp',
        seed=20260723,
        model_dir=str(output_dir),
        expected_runtime_contract=str(expected_path),
        runtime_contract_dir=str(output_dir),
    )


def main():
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = pathlib.Path(temporary_directory)
        config_path = root / 'config.yaml'
        config_path.write_text(
            'train_conf:\n'
            '    optim: adam\n'
            '    optim_conf:\n'
            '        lr: 0.0016\n'
            '    scheduler: warmuplr\n'
            '    scheduler_conf:\n'
            '        warmup_steps: 20000\n')
        expected_path = root / 'expected.json'
        output_dir = root / 'contracts'
        model = TinyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0016)
        scheduler = WarmupLR(optimizer, warmup_steps=20000)
        configs = {
            'train_conf': {
                'optim': 'adam',
                'optim_conf': {'lr': 0.0016},
                'scheduler': 'warmuplr',
                'scheduler_conf': {'warmup_steps': 20000},
            },
        }
        args = make_args(config_path, output_dir, expected_path)
        payload = runtime_payload(
            args, configs, model, optimizer, scheduler, rank=0, world_size=1)
        assertions = {
            'config': {
                'sha256': payload['invariant']['config']['sha256'],
            },
            'model': {
                'unwrapped_class': (
                    '__main__.TinyModel'),
                'parameters': payload[
                    'invariant']['model']['parameters'],
            },
            'optimizer': {
                'class': 'torch.optim.adam.Adam',
                'defaults': {
                    'lr': 0.0016,
                    'betas': [0.9, 0.999],
                    'eps': 1e-08,
                    'weight_decay': 0,
                    'amsgrad': False,
                },
                'param_groups': [{
                    'index': 0,
                    'parameter_tensor_count': 4,
                    'trainable_parameter_tensor_count': 2,
                    'frozen_parameter_tensor_count': 2,
                }],
            },
            'scheduler': {
                'class': 'cosyvoice.utils.scheduler.WarmupLR',
                'warmup_steps': 20000,
            },
            'runtime': {
                'world_size': 1,
                'python_path': payload['invariant']['runtime']['python_path'],
                'pythonpath_environment': payload[
                    'invariant']['runtime']['pythonpath_environment'],
                'matcha_module_file': payload[
                    'invariant']['runtime']['matcha_module_file'],
            },
        }
        expected_path.write_text(json.dumps({
            'format': 'cosyvoice-expected-runtime-contract-v1',
            'assertions': assertions,
        }))
        enforce_runtime_contract(
            args,
            configs,
            model,
            optimizer,
            scheduler,
            expected_path,
            output_dir,
        )
        aggregate = json.loads(
            (output_dir / 'runtime-contract-aggregate.json').read_text())
        if not aggregate['pass'] or aggregate['ranks'] != [0]:
            raise RuntimeError('runtime contract aggregate did not pass')

        broken = json.loads(json.dumps(assertions))
        broken['optimizer']['defaults']['lr'] = 0.001
        try:
            compare_expected(payload['invariant'], broken)
        except RuntimeError as error:
            if 'optimizer.defaults.lr' not in str(error):
                raise
        else:
            raise RuntimeError('mismatched optimizer was not rejected')
    print('runtime contract validation: PASS')


if __name__ == '__main__':
    main()
