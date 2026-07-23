#!/usr/bin/env python3

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import tempfile

import torch
from hyperpyyaml import load_hyperpyyaml

from cosyvoice.utils.checkpoint import parameter_inventory, seed_everything


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as input_file:
        while True:
            chunk = input_file.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_state_sha256(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b'\0')
        digest.update(str(tensor.dtype).encode())
        digest.update(b'\0')
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(b'\0')
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def name_list_sha256(names):
    return hashlib.sha256(
        ('\n'.join(names) + '\n').encode()).hexdigest()


def atomic_torch_save(payload, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix='.{}.tmp.'.format(destination.name),
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary_path = pathlib.Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        with temporary_path.open('rb') as output_file:
            os.fsync(output_file.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', default='llm')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()

    config_path = pathlib.Path(args.config).resolve()
    checkpoint_path = pathlib.Path(args.checkpoint).resolve()
    report_path = pathlib.Path(args.report).resolve()
    override_dict = {
        name: None
        for name in ['llm', 'flow', 'hift']
        if name != args.model
    }

    seed_everything(args.seed)
    with config_path.open() as config_file:
        configs = load_hyperpyyaml(
            config_file,
            overrides=override_dict,
        )
    model = configs[args.model]
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    inventory = parameter_inventory(model)
    named_parameters = dict(model.named_parameters())
    trainable_names = inventory['trainable']
    frozen_names = inventory['frozen']
    atomic_torch_save(state, checkpoint_path)
    report = {
        'format': 'cosyvoice_scratch_model_checkpoint_v1',
        'generated_at_utc': datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        'initialization': 'random-from-config',
        'seed': args.seed,
        'seed_applied_before_config_load': True,
        'config': {
            'path': str(config_path),
            'sha256': sha256_file(config_path),
        },
        'model': args.model,
        'checkpoint': {
            'path': str(checkpoint_path),
            'bytes': checkpoint_path.stat().st_size,
            'sha256': sha256_file(checkpoint_path),
            'canonical_tensor_sha256': canonical_state_sha256(state),
            'tensor_count': len(state),
        },
        'parameters': {
            'trainable_tensor_count': len(trainable_names),
            'frozen_tensor_count': len(frozen_names),
            'trainable_numel': sum(
                named_parameters[name].numel()
                for name in trainable_names
            ),
            'frozen_numel': sum(
                named_parameters[name].numel()
                for name in frozen_names
            ),
            'trainable_names_sha256': name_list_sha256(trainable_names),
            'frozen_names_sha256': name_list_sha256(frozen_names),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
