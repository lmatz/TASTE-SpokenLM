#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import pathlib
import platform
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as functional
from hyperpyyaml import load_hyperpyyaml

from cosyvoice.audio.customized_whisper.modeling_whisper import (
    WhisperAttention,
    WhisperSdpaAttention,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--optimizer_steps", type=int, default=32)
    parser.add_argument("--accum_grad", type=int, default=2)
    parser.add_argument("--min_free_gib", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260723)
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
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, "r") as config_file:
        return load_hyperpyyaml(
            config_file,
            overrides={"flow": None, "hift": None},
        )


def load_candidates(candidate_root, rank):
    rank_dir = candidate_root / "rank_{:05d}".format(rank)
    summary = json.loads((rank_dir / "summary.json").read_text())
    candidates = []
    candidate_metadata = []
    for entry in summary["candidate_batches"]:
        path = pathlib.Path(entry["path"])
        if not path.is_absolute():
            path = rank_dir / path.name
        if path.stat().st_size != entry["bytes"]:
            raise RuntimeError("candidate size mismatch: {}".format(path))
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError("candidate SHA256 mismatch: {}".format(path))
        candidates.append(torch.load(path, map_location="cpu"))
        candidate_metadata.append(entry)
    if not candidates:
        raise RuntimeError("rank {} has no stress candidates".format(rank))
    return candidates, candidate_metadata


def memory_snapshot(device):
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
    }


def distributed_extrema(value, operation, device):
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=operation)
    return float(tensor.item())


def main():
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed_everything(args.seed + rank)
    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_root = pathlib.Path(args.candidate_root).resolve()
    candidates, candidate_metadata = load_candidates(candidate_root, rank)

    configs = load_config(args.config)
    if configs["train_conf"]["accum_grad"] != args.accum_grad:
        raise RuntimeError("config accumulation does not match stress gate")
    model = configs["llm"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint, load_partial_list=[])
    model.cuda(local_rank)
    model.train()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), **configs["train_conf"]["optim_conf"]
    )
    optimizer.zero_grad(set_to_none=True)

    counters = {
        "sdpa_function_calls": 0,
        "eager_forward_calls": 0,
        "encoder_sdpa_module_calls": 0,
        "decoder_eager_module_calls": 0,
        "unexpected_encoder_eager_module_calls": 0,
        "unexpected_decoder_sdpa_module_calls": 0,
    }
    hooks = []
    original_sdpa = functional.scaled_dot_product_attention
    original_eager_forward = WhisperAttention.forward

    def counted_sdpa(*function_args, **function_kwargs):
        counters["sdpa_function_calls"] += 1
        return original_sdpa(*function_args, **function_kwargs)

    def counted_eager_forward(self, *forward_args, **forward_kwargs):
        counters["eager_forward_calls"] += 1
        return original_eager_forward(self, *forward_args, **forward_kwargs)

    def count_sdpa_module(module, module_inputs, module_output):
        del module_inputs, module_output
        if module.is_decoder:
            counters["unexpected_decoder_sdpa_module_calls"] += 1
        else:
            counters["encoder_sdpa_module_calls"] += 1

    def count_eager_module(module, module_inputs, module_output):
        del module_inputs, module_output
        if module.is_decoder:
            counters["decoder_eager_module_calls"] += 1
        else:
            counters["unexpected_encoder_eager_module_calls"] += 1

    functional.scaled_dot_product_attention = counted_sdpa
    WhisperAttention.forward = counted_eager_forward
    for module in model.modules():
        if isinstance(module, WhisperSdpaAttention):
            hooks.append(module.register_forward_hook(count_sdpa_module))
        elif isinstance(module, WhisperAttention):
            hooks.append(module.register_forward_hook(count_eager_module))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    minimum_free_bytes = None
    maximum_allocated_bytes = 0
    maximum_reserved_bytes = 0
    step_records = []
    start_time = time.monotonic()
    try:
        for optimizer_step in range(1, args.optimizer_steps + 1):
            step_start = time.monotonic()
            losses = []
            rank_batch_sizes = []
            for accumulation_index in range(args.accum_grad):
                candidate_index = (
                    (optimizer_step - 1) * args.accum_grad + accumulation_index
                ) % len(candidates)
                batch = candidates[candidate_index]
                rank_batch_sizes.append(len(batch["utts"]))
                with torch.cuda.amp.autocast(
                    enabled=True, dtype=torch.float32, cache_enabled=False
                ):
                    outputs = model(batch, local_rank)
                    loss = outputs["loss"] / args.accum_grad
                if not torch.isfinite(loss).all():
                    raise FloatingPointError(
                        "non-finite loss at optimizer step {} accumulation {}".format(
                            optimizer_step, accumulation_index
                        )
                    )
                loss.backward()
                losses.append(float(loss.detach().item()))
                memory = memory_snapshot(device)
                minimum_free_bytes = (
                    memory["free_bytes"]
                    if minimum_free_bytes is None
                    else min(minimum_free_bytes, memory["free_bytes"])
                )
                maximum_allocated_bytes = max(
                    maximum_allocated_bytes, memory["max_allocated_bytes"]
                )
                maximum_reserved_bytes = max(
                    maximum_reserved_bytes, memory["max_reserved_bytes"]
                )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), configs["train_conf"]["grad_clip"]
            )
            if not torch.isfinite(torch.as_tensor(gradient_norm)).all():
                raise FloatingPointError(
                    "non-finite gradient norm at optimizer step {}".format(
                        optimizer_step
                    )
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            memory = memory_snapshot(device)
            minimum_free_bytes = min(minimum_free_bytes, memory["free_bytes"])
            maximum_allocated_bytes = max(
                maximum_allocated_bytes, memory["max_allocated_bytes"]
            )
            maximum_reserved_bytes = max(
                maximum_reserved_bytes, memory["max_reserved_bytes"]
            )
            global_min_free = distributed_extrema(
                minimum_free_bytes, dist.ReduceOp.MIN, device
            )
            global_max_allocated = distributed_extrema(
                maximum_allocated_bytes, dist.ReduceOp.MAX, device
            )
            global_max_reserved = distributed_extrema(
                maximum_reserved_bytes, dist.ReduceOp.MAX, device
            )
            global_max_batch = distributed_extrema(
                max(rank_batch_sizes), dist.ReduceOp.MAX, device
            )
            global_min_batch = distributed_extrema(
                min(rank_batch_sizes), dist.ReduceOp.MIN, device
            )
            if rank == 0:
                step_records.append(
                    {
                        "optimizer_step": optimizer_step,
                        "loss_sum_rank0": sum(losses),
                        "gradient_norm_rank0": float(
                            torch.as_tensor(gradient_norm).item()
                        ),
                        "step_seconds": time.monotonic() - step_start,
                        "global_min_free_bytes_so_far": global_min_free,
                        "global_max_allocated_bytes_so_far": global_max_allocated,
                        "global_max_reserved_bytes_so_far": global_max_reserved,
                        "global_batch_size_min": int(global_min_batch),
                        "global_batch_size_max": int(global_max_batch),
                    }
                )
    finally:
        for hook in hooks:
            hook.remove()
        functional.scaled_dot_product_attention = original_sdpa
        WhisperAttention.forward = original_eager_forward

    rank_summary = {
        "rank": rank,
        "candidate_count": len(candidates),
        "candidate_metadata": candidate_metadata,
        "minimum_free_bytes": minimum_free_bytes,
        "maximum_allocated_bytes": maximum_allocated_bytes,
        "maximum_reserved_bytes": maximum_reserved_bytes,
        "sdpa_function_calls": counters["sdpa_function_calls"],
        "eager_forward_calls": counters["eager_forward_calls"],
        "encoder_sdpa_module_calls": counters["encoder_sdpa_module_calls"],
        "decoder_eager_module_calls": counters["decoder_eager_module_calls"],
        "unexpected_encoder_eager_module_calls": counters[
            "unexpected_encoder_eager_module_calls"
        ],
        "unexpected_decoder_sdpa_module_calls": counters[
            "unexpected_decoder_sdpa_module_calls"
        ],
    }
    rank_path = output_dir / "rank_{:05d}.json".format(rank)
    rank_path.write_text(json.dumps(rank_summary, indent=2, sort_keys=True) + "\n")
    dist.barrier()
    if rank == 0:
        ranks = [
            json.loads(
                (output_dir / "rank_{:05d}.json".format(other_rank)).read_text()
            )
            for other_rank in range(world_size)
        ]
        min_free_bytes = min(item["minimum_free_bytes"] for item in ranks)
        total_memory = torch.cuda.get_device_properties(local_rank).total_memory
        report = {
            "format": "taste-stage1-encoder-sdpa-full-distribution-stress-v2",
            "pass": bool(
                len(step_records) == args.optimizer_steps
                and min_free_bytes >= args.min_free_gib * 1024 ** 3
                and all(item["sdpa_function_calls"] > 0 for item in ranks)
                and all(
                    item["sdpa_function_calls"]
                    == item["encoder_sdpa_module_calls"]
                    for item in ranks
                )
                and all(item["decoder_eager_module_calls"] > 0 for item in ranks)
                and all(
                    item["eager_forward_calls"]
                    == item["decoder_eager_module_calls"]
                    for item in ranks
                )
                and all(
                    item["unexpected_encoder_eager_module_calls"] == 0
                    for item in ranks
                )
                and all(
                    item["unexpected_decoder_sdpa_module_calls"] == 0
                    for item in ranks
                )
            ),
            "attention_contract": {
                "encoder": "sdpa",
                "decoder": "eager",
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(local_rank),
                "total_memory_bytes": total_memory,
            },
            "world_size": world_size,
            "optimizer_steps": args.optimizer_steps,
            "accum_grad": args.accum_grad,
            "microbatches": args.optimizer_steps * args.accum_grad,
            "minimum_free_bytes": min_free_bytes,
            "minimum_free_gib": min_free_bytes / 1024 ** 3,
            "required_minimum_free_gib": args.min_free_gib,
            "maximum_allocated_bytes": max(
                item["maximum_allocated_bytes"] for item in ranks
            ),
            "maximum_reserved_bytes": max(
                item["maximum_reserved_bytes"] for item in ranks
            ),
            "elapsed_seconds": time.monotonic() - start_time,
            "config": os.path.abspath(args.config),
            "config_sha256": sha256_file(args.config),
            "checkpoint": os.path.abspath(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "candidate_root": str(candidate_root),
            "ranks": ranks,
            "steps": step_records,
        }
        output_path = output_dir / "aggregate-report.json"
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["pass"]:
            raise SystemExit(1)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
