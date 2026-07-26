#!/usr/bin/env python3

import argparse
import json
import pathlib


def load_rank_traces(directory, world_size):
    traces = {}
    for rank in range(world_size):
        path = pathlib.Path(directory) / f'rank_{rank:05d}.jsonl'
        if not path.is_file():
            raise FileNotFoundError(path)
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if not records:
            raise RuntimeError(f'{path}: empty sample trace')
        traces[rank] = records
    return traces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--continuous', required=True)
    parser.add_argument('--resumed', required=True)
    parser.add_argument('--world_size', type=int, required=True)
    parser.add_argument('--output')
    args = parser.parse_args()

    continuous = load_rank_traces(args.continuous, args.world_size)
    resumed = load_rank_traces(args.resumed, args.world_size)
    resume_suffix_matches = True
    mismatched_ranks = []
    continuous_keys = []
    rank_record_counts = {}
    rank_utterance_counts = {}
    for rank in range(args.world_size):
        continuous_records = continuous[rank]
        resumed_records = resumed[rank]
        rank_record_counts[str(rank)] = len(continuous_records)
        rank_utterance_counts[str(rank)] = sum(
            len(record['utts']) for record in continuous_records)
        continuous_keys.extend(
            key for record in continuous_records for key in record['utts'])
        if len(resumed_records) > len(continuous_records):
            resume_suffix_matches = False
            mismatched_ranks.append(rank)
            continue
        suffix = continuous_records[-len(resumed_records):]
        if suffix != resumed_records:
            resume_suffix_matches = False
            mismatched_ranks.append(rank)

    duplicate_count = len(continuous_keys) - len(set(continuous_keys))
    record_counts_equal = len(set(rank_record_counts.values())) == 1
    result = {
        'pass': (
            resume_suffix_matches and
            duplicate_count == 0 and
            record_counts_equal),
        'world_size': args.world_size,
        'resume_suffix_matches': resume_suffix_matches,
        'mismatched_ranks': mismatched_ranks,
        'continuous_record_counts': rank_record_counts,
        'continuous_utterance_counts': rank_utterance_counts,
        'continuous_utterance_count': len(continuous_keys),
        'continuous_unique_utterance_count': len(set(continuous_keys)),
        'continuous_duplicate_count': duplicate_count,
        'record_counts_equal_across_ranks': record_counts_equal,
    }
    output = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        pathlib.Path(args.output).write_text(output + '\n')
    print(output)
    if not result['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
