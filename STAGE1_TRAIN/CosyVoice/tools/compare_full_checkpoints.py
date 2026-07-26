#!/usr/bin/env python3

import argparse
import json

import torch

from cosyvoice.utils.checkpoint import FULL_CHECKPOINT_FORMAT, torch_load


def compare_tensor_dict(left, right, atol, rtol):
    if set(left) != set(right):
        return {
            'pass': False,
            'missing_left': sorted(set(right) - set(left)),
            'missing_right': sorted(set(left) - set(right)),
        }
    mismatches = []
    max_abs_diff = 0.0
    for name in sorted(left):
        left_value = left[name].detach().cpu()
        right_value = right[name].detach().cpu()
        if left_value.shape != right_value.shape or left_value.dtype != right_value.dtype:
            mismatches.append({
                'name': name,
                'left_shape': list(left_value.shape),
                'right_shape': list(right_value.shape),
                'left_dtype': str(left_value.dtype),
                'right_dtype': str(right_value.dtype),
            })
            continue
        if left_value.is_floating_point():
            difference = (left_value - right_value).abs()
            tensor_max = float(difference.max()) if difference.numel() else 0.0
            max_abs_diff = max(max_abs_diff, tensor_max)
            matches = torch.allclose(left_value, right_value, atol=atol, rtol=rtol)
        else:
            tensor_max = 0.0
            matches = torch.equal(left_value, right_value)
        if not matches:
            mismatches.append({'name': name, 'max_abs_diff': tensor_max})
    return {
        'pass': not mismatches,
        'tensor_count': len(left),
        'mismatch_count': len(mismatches),
        'max_abs_diff': max_abs_diff,
        'mismatches': mismatches[:50],
    }


def compare_nested(left, right, atol, rtol, path='root'):
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            return [{'path': path, 'reason': 'tensor/type mismatch'}]
        if left.shape != right.shape or left.dtype != right.dtype:
            return [{
                'path': path,
                'reason': 'tensor shape/dtype mismatch',
                'left_shape': list(left.shape),
                'right_shape': list(right.shape),
                'left_dtype': str(left.dtype),
                'right_dtype': str(right.dtype),
            }]
        if left.is_floating_point():
            matches = torch.allclose(
                left.detach().cpu(), right.detach().cpu(), atol=atol, rtol=rtol)
        else:
            matches = torch.equal(left.detach().cpu(), right.detach().cpu())
        return [] if matches else [{'path': path, 'reason': 'tensor value mismatch'}]
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return [{'path': path, 'reason': 'mapping/type mismatch'}]
        if set(left) != set(right):
            return [{
                'path': path,
                'reason': 'mapping keys mismatch',
                'missing_left': sorted(set(right) - set(left), key=str),
                'missing_right': sorted(set(left) - set(right), key=str),
            }]
        mismatches = []
        for key in sorted(left, key=str):
            mismatches.extend(compare_nested(
                left[key], right[key], atol, rtol, f'{path}.{key}'))
            if len(mismatches) >= 50:
                break
        return mismatches
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            return [{'path': path, 'reason': 'sequence type/length mismatch'}]
        mismatches = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            mismatches.extend(compare_nested(
                left_item, right_item, atol, rtol, f'{path}[{index}]'))
            if len(mismatches) >= 50:
                break
        return mismatches
    return [] if left == right else [{'path': path, 'reason': 'value mismatch'}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', required=True)
    parser.add_argument('--right', required=True)
    parser.add_argument('--atol', type=float, default=0.0)
    parser.add_argument('--rtol', type=float, default=0.0)
    parser.add_argument('--output')
    args = parser.parse_args()

    left = torch_load(args.left, map_location='cpu')
    right = torch_load(args.right, map_location='cpu')
    if left.get('format') != FULL_CHECKPOINT_FORMAT:
        raise ValueError('left input is not a full-state checkpoint')
    if right.get('format') != FULL_CHECKPOINT_FORMAT:
        raise ValueError('right input is not a full-state checkpoint')

    model_result = compare_tensor_dict(
        left['model_state_dict'], right['model_state_dict'], args.atol, args.rtol)
    optimizer_mismatches = compare_nested(
        left['optimizer_state_dict'], right['optimizer_state_dict'],
        args.atol, args.rtol, 'optimizer')
    scheduler_mismatches = compare_nested(
        left['scheduler_state_dict'], right['scheduler_state_dict'],
        args.atol, args.rtol, 'scheduler')
    result = {
        'pass': model_result['pass'],
        'left': args.left,
        'right': args.right,
        'atol': args.atol,
        'rtol': args.rtol,
        'model': model_result,
        'left_cursor': left['cursor'],
        'right_cursor': right['cursor'],
        'left_executor': left['executor_state'],
        'right_executor': right['executor_state'],
        'optimizer_state_equal': not optimizer_mismatches,
        'optimizer_mismatches': optimizer_mismatches[:50],
        'scheduler_state_equal': not scheduler_mismatches,
        'scheduler_mismatches': scheduler_mismatches[:50],
        'parameter_inventory_equal': (
            left['parameter_inventory'] == right['parameter_inventory']),
    }
    result['pass'] = (
        result['pass'] and
        result['optimizer_state_equal'] and
        result['scheduler_state_equal'] and
        result['parameter_inventory_equal'] and
        result['left_cursor'] == result['right_cursor'] and
        result['left_executor'] == result['right_executor'])
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
