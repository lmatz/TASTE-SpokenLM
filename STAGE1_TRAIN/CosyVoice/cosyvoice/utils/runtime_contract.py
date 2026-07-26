import hashlib
import importlib.util
import json
import logging
import os
import pathlib
import sys
import tempfile

import torch
import torch.distributed as dist

from cosyvoice.utils.checkpoint import (
    parameter_inventory,
    unwrap_model,
)


def qualified_name(value):
    value_type = value if isinstance(value, type) else type(value)
    return '{}.{}'.format(value_type.__module__, value_type.__qualname__)


def normalize_json(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [normalize_json(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): normalize_json(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return str(value)


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as input_file:
        while True:
            chunk = input_file.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def temporary_python_path_sha256(path):
    digest = hashlib.sha256()
    for source_path in sorted(
            candidate for candidate in path.rglob('*')
            if candidate.is_file() and '__pycache__' not in candidate.parts):
        relative_path = str(source_path.relative_to(path))
        digest.update(relative_path.encode())
        digest.update(b'\0')
        digest.update(source_path.read_bytes())
        digest.update(b'\0')
    return digest.hexdigest()


def canonical_python_path_entry(path):
    resolved_path = pathlib.Path(path).resolve()
    generated_module = resolved_path / '_remote_module_non_scriptable.py'
    if (resolved_path.parent == pathlib.Path(tempfile.gettempdir()).resolve()
            and generated_module.is_file()):
        return 'torch-distributed-jit-sha256:{}'.format(
            temporary_python_path_sha256(resolved_path))
    return str(resolved_path)


def name_list_sha256(names):
    return sha256_bytes(
        ('\n'.join(names) + '\n').encode())


def module_origin(module_name):
    specification = importlib.util.find_spec(module_name)
    if specification is None or specification.origin is None:
        return None
    return str(pathlib.Path(specification.origin).resolve())


def optimizer_payload(optimizer):
    groups = []
    for index, group in enumerate(optimizer.param_groups):
        parameters = group['params']
        groups.append({
            'index': index,
            'hyperparameters': normalize_json({
                key: value
                for key, value in group.items()
                if key != 'params'
            }),
            'parameter_tensor_count': len(parameters),
            'parameter_numel': sum(
                parameter.numel() for parameter in parameters),
            'trainable_parameter_tensor_count': sum(
                int(parameter.requires_grad) for parameter in parameters),
            'trainable_parameter_numel': sum(
                parameter.numel()
                for parameter in parameters
                if parameter.requires_grad),
            'frozen_parameter_tensor_count': sum(
                int(not parameter.requires_grad) for parameter in parameters),
            'frozen_parameter_numel': sum(
                parameter.numel()
                for parameter in parameters
                if not parameter.requires_grad),
        })
    return {
        'class': qualified_name(optimizer),
        'defaults': normalize_json(optimizer.defaults),
        'param_groups': groups,
        'state_entry_count': len(optimizer.state),
    }


def scheduler_payload(scheduler):
    return {
        'class': qualified_name(scheduler),
        'state': normalize_json(scheduler.state_dict()),
        'warmup_steps': normalize_json(
            getattr(scheduler, 'warmup_steps', None)),
    }


def model_payload(model):
    unwrapped = unwrap_model(model)
    inventory = parameter_inventory(unwrapped)
    named_parameters = dict(unwrapped.named_parameters())
    trainable_names = inventory['trainable']
    frozen_names = inventory['frozen']
    return {
        'wrapper_class': qualified_name(model),
        'unwrapped_class': qualified_name(unwrapped),
        'state_tensor_count': len(unwrapped.state_dict()),
        'parameters': {
            'trainable_tensor_count': len(trainable_names),
            'frozen_tensor_count': len(frozen_names),
            'trainable_numel': sum(
                named_parameters[name].numel()
                for name in trainable_names),
            'frozen_numel': sum(
                named_parameters[name].numel()
                for name in frozen_names),
            'trainable_names_sha256': name_list_sha256(trainable_names),
            'frozen_names_sha256': name_list_sha256(frozen_names),
        },
    }


def runtime_payload(args, configs, model, optimizer, scheduler, rank, world_size):
    config_path = pathlib.Path(args.config).resolve()
    config_bytes = config_path.read_bytes()
    arguments = vars(args).copy()
    effective_train_conf = configs['train_conf'].copy()
    rank_arguments = {}
    for key in ['local_rank']:
        if key in arguments:
            rank_arguments[key] = arguments.pop(key)
        effective_train_conf.pop(key, None)
    invariant = {
        'config': {
            'path': str(config_path),
            'bytes': len(config_bytes),
            'sha256': sha256_bytes(config_bytes),
            'yaml': config_bytes.decode(),
        },
        'arguments': normalize_json(arguments),
        'effective_train_conf': normalize_json(effective_train_conf),
        'model': model_payload(model),
        'optimizer': optimizer_payload(optimizer),
        'scheduler': scheduler_payload(scheduler),
        'runtime': {
            'python_version': '{}.{}.{}'.format(
                *tuple(__import__('sys').version_info[:3])),
            'torch_version': torch.__version__,
            'torch_cuda_version': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count(),
            'distributed_backend': (
                str(dist.get_backend()) if dist.is_initialized() else None),
            'world_size': world_size,
            'python_path': [
                canonical_python_path_entry(path) for path in sys.path
            ],
            'pythonpath_environment': os.environ.get('PYTHONPATH'),
            'matcha_module_file': module_origin('matcha'),
        },
    }
    invariant_json = json.dumps(
        invariant, sort_keys=True, separators=(',', ':'))
    return {
        'format': 'cosyvoice-runtime-contract-v1',
        'rank': rank,
        'local_rank': int(os.environ.get('LOCAL_RANK', rank)),
        'rank_arguments': normalize_json(rank_arguments),
        'cuda_current_device': (
            torch.cuda.current_device()
            if torch.cuda.is_available() else None),
        'cuda_device_name': (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available() else None),
        'invariant_sha256': sha256_bytes(invariant_json.encode()),
        'invariant': invariant,
    }


def compare_expected(actual, expected, path='runtime'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise RuntimeError(
                '{}: expected object, got {}'.format(path, type(actual).__name__))
        for key, expected_value in expected.items():
            if key not in actual:
                raise RuntimeError(
                    '{}: missing expected key {}'.format(path, key))
            compare_expected(
                actual[key], expected_value, '{}.{}'.format(path, key))
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            raise RuntimeError(
                '{}: expected list, got {}'.format(path, type(actual).__name__))
        if len(actual) != len(expected):
            raise RuntimeError(
                '{}: expected {} items, got {}'.format(
                    path, len(expected), len(actual)))
        for index, expected_value in enumerate(expected):
            compare_expected(
                actual[index], expected_value, '{}[{}]'.format(path, index))
        return
    if actual != expected:
        raise RuntimeError(
            '{}: expected {!r}, got {!r}'.format(path, expected, actual))


def atomic_json_dump(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix='.{}.tmp.'.format(path.name),
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, 'w') as output_file:
            json.dump(payload, output_file, indent=2, sort_keys=True)
            output_file.write('\n')
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def enforce_runtime_contract(
        args, configs, model, optimizer, scheduler,
        expected_path, output_dir):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    expected = json.loads(pathlib.Path(expected_path).read_text())
    payload = runtime_payload(
        args, configs, model, optimizer, scheduler, rank, world_size)
    compare_expected(payload['invariant'], expected['assertions'])

    output_root = pathlib.Path(output_dir)
    rank_path = output_root / 'runtime-contract-rank-{:04d}.json'.format(rank)
    atomic_json_dump(payload, rank_path)
    logging.info(
        'Runtime contract rank %s PASS invariant_sha256=%s payload=%s',
        rank,
        payload['invariant_sha256'],
        json.dumps(payload, sort_keys=True))
    if rank == 0:
        logging.info(
            'Resolved runtime config YAML BEGIN sha256=%s bytes=%s\n%s\n'
            'Resolved runtime config YAML END',
            payload['invariant']['config']['sha256'],
            payload['invariant']['config']['bytes'],
            payload['invariant']['config']['yaml'])

    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        rank_payloads = [
            json.loads(
                (output_root / 'runtime-contract-rank-{:04d}.json'.format(
                    current_rank)).read_text())
            for current_rank in range(world_size)
        ]
        invariant_hashes = {
            item['invariant_sha256'] for item in rank_payloads
        }
        observed_ranks = {
            item['rank'] for item in rank_payloads
        }
        expected_ranks = set(range(world_size))
        if invariant_hashes != {payload['invariant_sha256']}:
            raise RuntimeError(
                'runtime contract invariants differ across ranks: {}'.format(
                    sorted(invariant_hashes)))
        if observed_ranks != expected_ranks:
            raise RuntimeError(
                'runtime contract rank set mismatch: expected {}, got {}'.format(
                    sorted(expected_ranks), sorted(observed_ranks)))
        aggregate = {
            'format': 'cosyvoice-runtime-contract-aggregate-v1',
            'world_size': world_size,
            'ranks': sorted(observed_ranks),
            'invariant_sha256': payload['invariant_sha256'],
            'expected_contract': str(pathlib.Path(expected_path).resolve()),
            'rank_contracts': [
                str(
                    (output_root / 'runtime-contract-rank-{:04d}.json'.format(
                        current_rank)).resolve())
                for current_rank in range(world_size)
            ],
            'pass': True,
        }
        atomic_json_dump(
            aggregate, output_root / 'runtime-contract-aggregate.json')
        logging.info(
            'Runtime contract aggregate PASS: %s',
            json.dumps(aggregate, sort_keys=True))
    if dist.is_initialized():
        dist.barrier()
    return payload
