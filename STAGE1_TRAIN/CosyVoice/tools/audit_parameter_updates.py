#!/usr/bin/env python3

import argparse
import collections
import json

import torch

from cosyvoice.utils.checkpoint import FULL_CHECKPOINT_FORMAT, torch_load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial', required=True)
    parser.add_argument('--final', required=True)
    parser.add_argument('--full_checkpoint', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()

    initial = torch_load(args.initial, map_location='cpu')
    final = torch_load(args.final, map_location='cpu')
    full_checkpoint = torch_load(args.full_checkpoint, map_location='cpu')
    if full_checkpoint.get('format') != FULL_CHECKPOINT_FORMAT:
        raise ValueError('full_checkpoint is not a supported full-state checkpoint')
    if set(initial) != set(final):
        raise ValueError('initial and final model state keys differ')

    inventory = full_checkpoint['parameter_inventory']
    changed_trainable = []
    unchanged_trainable = []
    changed_frozen = []
    changed_by_module = collections.Counter()
    for name in inventory['trainable']:
        if torch.equal(initial[name].cpu(), final[name].cpu()):
            unchanged_trainable.append(name)
        else:
            changed_trainable.append(name)
            changed_by_module['.'.join(name.split('.')[:2])] += 1
    for name in inventory['frozen']:
        if not torch.equal(initial[name].cpu(), final[name].cpu()):
            changed_frozen.append(name)

    result = {
        'pass': bool(changed_trainable) and not changed_frozen,
        'trainable_count': len(inventory['trainable']),
        'frozen_count': len(inventory['frozen']),
        'changed_trainable_count': len(changed_trainable),
        'unchanged_trainable_count': len(unchanged_trainable),
        'changed_frozen_count': len(changed_frozen),
        'changed_trainable_by_module': dict(sorted(changed_by_module.items())),
        'unchanged_trainable': unchanged_trainable,
        'changed_frozen': changed_frozen,
    }
    output = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, 'w') as output_file:
            output_file.write(output)
            output_file.write('\n')
    print(output)
    if not result['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
