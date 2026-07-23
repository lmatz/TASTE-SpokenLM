#!/usr/bin/env python3

import argparse
import gzip
import hashlib
import heapq
import json
import os
import pathlib
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from hyperpyyaml import load_hyperpyyaml
from torch.utils.data import DataLoader

from cosyvoice.dataset.dataset import Dataset
from cosyvoice.utils.train_utils import seed_worker


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--expected_rows", type=int, required=True)
    parser.add_argument("--expected_shards", type=int, required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        while True:
            chunk = input_file.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data_pipeline(config_path):
    with open(config_path, "r") as config_file:
        configs = load_hyperpyyaml(
            config_file,
            overrides={"llm": None, "flow": None, "hift": None},
        )
    return configs["data_pipeline"]


def batch_metrics(batch, batch_index):
    batch_size = len(batch["utts"])
    audio_lengths = batch["audio_feat_len"].tolist()
    speech_lengths = batch["speech_token_len"].tolist()
    text_lengths = batch["text_token_len"].tolist()
    return {
        "batch_index": batch_index,
        "batch_size": batch_size,
        "utts": list(batch["utts"]),
        "audio_feat_shape": list(batch["audio_feat"].shape),
        "audio_feat_length_min": min(audio_lengths),
        "audio_feat_length_max": max(audio_lengths),
        "audio_feat_length_sum": sum(audio_lengths),
        "speech_token_length_max": max(speech_lengths),
        "speech_token_length_sum": sum(speech_lengths),
        "text_token_length_max": max(text_lengths),
        "text_token_length_sum": sum(text_lengths),
        "words_index_count": len(batch.get("words_index", [])),
        "eager_encoder_attention_bytes": (
            batch_size * 20 * 1500 * 1500 * 4
        ),
    }


def clone_batch(batch):
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value.clone()
        elif isinstance(value, list):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def update_top(heap, score, sequence, metrics, batch, top_k):
    entry = (score, sequence, metrics, clone_batch(batch))
    if len(heap) < top_k:
        heapq.heappush(heap, entry)
    elif entry[:2] > heap[0][:2]:
        heapq.heapreplace(heap, entry)


def write_rank_outputs(output_dir, rank, candidates, all_utts, summary):
    rank_dir = output_dir / "rank_{:05d}".format(rank)
    rank_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries = []
    ordered = sorted(
        candidates.values(),
        key=lambda entry: (
            -entry[2]["batch_size"],
            -entry[2]["speech_token_length_sum"],
            entry[2]["batch_index"],
        ),
    )
    for candidate_index, entry in enumerate(ordered):
        _, _, metrics, batch = entry
        batch_path = rank_dir / "candidate_{:03d}.pt".format(candidate_index)
        torch.save(batch, batch_path)
        manifest_entries.append(
            {
                "path": str(batch_path),
                "bytes": batch_path.stat().st_size,
                "sha256": sha256_file(batch_path),
                "metrics": metrics,
            }
        )
    utt_path = rank_dir / "utts.txt.gz"
    with gzip.open(utt_path, "wt") as output_file:
        for utt in sorted(all_utts):
            output_file.write(utt + "\n")
    summary["candidate_batches"] = manifest_entries
    summary["utt_list"] = {
        "path": str(utt_path),
        "bytes": utt_path.stat().st_size,
        "sha256": sha256_file(utt_path),
    }
    summary_path = rank_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def aggregate_outputs(output_dir, world_size, args):
    summaries = []
    all_utts = set()
    duplicate_utts = []
    source_shards = set()
    for rank in range(world_size):
        rank_dir = output_dir / "rank_{:05d}".format(rank)
        summary = json.loads((rank_dir / "summary.json").read_text())
        summaries.append(summary)
        with gzip.open(rank_dir / "utts.txt.gz", "rt") as input_file:
            for line in input_file:
                utt = line.rstrip("\n")
                if utt in all_utts:
                    duplicate_utts.append(utt)
                all_utts.add(utt)
                source_shards.add(utt.split("__", 1)[0])
    total_batches = sum(summary["batch_count"] for summary in summaries)
    total_rows = sum(summary["row_count"] for summary in summaries)
    aggregate = {
        "format": "taste-stage1-full-dynamic-batch-scan-v1",
        "pass": bool(
            total_rows == args.expected_rows
            and len(all_utts) == args.expected_rows
            and not duplicate_utts
            and len(source_shards) == args.expected_shards
        ),
        "world_size": world_size,
        "num_workers_per_rank": args.num_workers,
        "max_frames_in_batch": 16000,
        "expected_rows": args.expected_rows,
        "expected_shards": args.expected_shards,
        "row_count": total_rows,
        "unique_utt_count": len(all_utts),
        "duplicate_utt_count": len(duplicate_utts),
        "duplicate_utt_examples": duplicate_utts[:20],
        "source_shard_count": len(source_shards),
        "batch_count": total_batches,
        "rank_batch_counts": [summary["batch_count"] for summary in summaries],
        "rank_row_counts": [summary["row_count"] for summary in summaries],
        "max_batch_size": max(summary["max_batch_size"] for summary in summaries),
        "max_speech_token_length_sum": max(
            summary["max_speech_token_length_sum"] for summary in summaries
        ),
        "max_speech_token_length": max(
            summary["max_speech_token_length"] for summary in summaries
        ),
        "ranks": summaries,
        "config": os.path.abspath(args.config),
        "config_sha256": sha256_file(args.config),
        "train_data": os.path.abspath(args.train_data),
        "train_data_sha256": sha256_file(args.train_data),
    }
    output_path = output_dir / "aggregate-report.json"
    output_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    if not aggregate["pass"]:
        raise SystemExit(1)


def main():
    args = parse_args()
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    seed_everything(args.seed + rank)
    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_pipeline = load_data_pipeline(args.config)
    dataset = Dataset(
        args.train_data,
        data_pipeline=data_pipeline,
        mode="train",
        shuffle=True,
        partition=True,
    )
    dataset.set_epoch(0)
    generator = torch.Generator()
    generator.manual_seed(args.seed + rank)
    loader = DataLoader(
        dataset,
        batch_size=None,
        pin_memory=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    top_by_batch_size = []
    top_by_speech_sum = []
    top_by_speech_max = []
    all_utts = []
    batch_count = 0
    row_count = 0
    max_batch_size = 0
    max_speech_sum = 0
    max_speech_length = 0
    start_time = time.monotonic()
    for batch_index, batch in enumerate(loader):
        metrics = batch_metrics(batch, batch_index)
        batch_count += 1
        row_count += metrics["batch_size"]
        max_batch_size = max(max_batch_size, metrics["batch_size"])
        max_speech_sum = max(
            max_speech_sum, metrics["speech_token_length_sum"]
        )
        max_speech_length = max(
            max_speech_length, metrics["speech_token_length_max"]
        )
        all_utts.extend(metrics["utts"])
        update_top(
            top_by_batch_size,
            metrics["batch_size"],
            batch_index,
            metrics,
            batch,
            args.top_k,
        )
        update_top(
            top_by_speech_sum,
            metrics["speech_token_length_sum"],
            batch_index,
            metrics,
            batch,
            args.top_k,
        )
        update_top(
            top_by_speech_max,
            metrics["speech_token_length_max"],
            batch_index,
            metrics,
            batch,
            args.top_k,
        )
    candidates = {}
    for entry in top_by_batch_size + top_by_speech_sum + top_by_speech_max:
        candidates[entry[2]["batch_index"]] = entry
    summary = {
        "rank": rank,
        "world_size": world_size,
        "batch_count": batch_count,
        "row_count": row_count,
        "max_batch_size": max_batch_size,
        "max_speech_token_length_sum": max_speech_sum,
        "max_speech_token_length": max_speech_length,
        "elapsed_seconds": time.monotonic() - start_time,
    }
    write_rank_outputs(output_dir, rank, candidates, all_utts, summary)
    dist.barrier()
    if rank == 0:
        aggregate_outputs(output_dir, world_size, args)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
