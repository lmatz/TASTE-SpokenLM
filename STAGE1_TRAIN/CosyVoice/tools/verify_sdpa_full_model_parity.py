#!/usr/bin/env python3

import argparse
import contextlib
import gc
import hashlib
import json
import os
import pathlib
import platform
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as functional
from hyperpyyaml import load_hyperpyyaml
from torch.utils.data import DataLoader

from cosyvoice.audio.customized_whisper.modeling_whisper import (
    WhisperAttention,
    WhisperSdpaAttention,
)
from cosyvoice.dataset.dataset import Dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eager_config", required=True)
    parser.add_argument("--sdpa_config", required=True)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state_path", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--loss_atol", type=float, default=1e-4)
    parser.add_argument("--logit_max_atol", type=float, default=1e-3)
    parser.add_argument("--logit_mean_atol", type=float, default=1e-4)
    parser.add_argument("--gradient_max_atol", type=float, default=2e-3)
    parser.add_argument("--gradient_mean_atol", type=float, default=1e-5)
    parser.add_argument("--gradient_norm_rtol", type=float, default=1e-4)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        while True:
            chunk = input_file.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path):
    with open(path, "r") as config_file:
        return load_hyperpyyaml(
            config_file,
            overrides={"flow": None, "hift": None},
        )


def slice_batch(batch, batch_size):
    original_size = len(batch["utts"])
    if batch_size > original_size:
        raise ValueError(
            "requested batch size {} exceeds real batch size {}".format(
                batch_size, original_size
            )
        )
    result = {}
    for key, value in batch.items():
        if key == "words_index":
            result[key] = [entry for entry in value if entry[0] < batch_size]
        elif torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == original_size:
            result[key] = value[:batch_size].clone()
        elif isinstance(value, list) and len(value) == original_size:
            result[key] = value[:batch_size]
        else:
            result[key] = value
    return result, original_size


def load_real_batch(configs, train_data, batch_size):
    dataset = Dataset(
        train_data,
        data_pipeline=configs["data_pipeline"],
        mode="train",
        shuffle=False,
        partition=False,
    )
    dataset.set_epoch(0)
    loader = DataLoader(dataset, batch_size=None, num_workers=0)
    batch = next(iter(loader))
    sliced, original_size = slice_batch(batch, batch_size)
    metadata = {
        "source_batch_size": original_size,
        "parity_batch_size": batch_size,
        "utts": list(sliced["utts"]),
        "audio_feat_shape": list(sliced["audio_feat"].shape),
        "audio_feat_lengths": sliced["audio_feat_len"].tolist(),
        "text_token_lengths": sliced["text_token_len"].tolist(),
        "speech_token_lengths": sliced["speech_token_len"].tolist(),
        "words_index_count": len(sliced.get("words_index", [])),
    }
    return sliced, metadata


def cuda_memory():
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def run_model(model, batch, implementation):
    model.cuda()
    model.eval()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    counters = {
        "sdpa_function_calls": 0,
        "encoder_sdpa_module_calls": 0,
        "decoder_eager_module_calls": 0,
        "unexpected_encoder_eager_module_calls": 0,
        "unexpected_decoder_sdpa_module_calls": 0,
        "eager_forward_calls_during_sdpa": 0,
        "sdpa_inputs": [],
    }
    hooks = []
    original_sdpa = functional.scaled_dot_product_attention
    original_eager_forward = WhisperAttention.forward

    def counted_sdpa(query, key, value, *function_args, **function_kwargs):
        counters["sdpa_function_calls"] += 1
        if len(counters["sdpa_inputs"]) < 64:
            counters["sdpa_inputs"].append(
                {
                    "query_shape": list(query.shape),
                    "key_shape": list(key.shape),
                    "value_shape": list(value.shape),
                    "dtype": str(query.dtype),
                    "is_causal": bool(function_kwargs.get("is_causal", False)),
                    "has_mask": function_kwargs.get("attn_mask") is not None,
                }
            )
        return original_sdpa(query, key, value, *function_args, **function_kwargs)

    def counted_eager_forward(self, *forward_args, **forward_kwargs):
        counters["eager_forward_calls_during_sdpa"] += 1
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

    if implementation == "sdpa":
        functional.scaled_dot_product_attention = counted_sdpa
        WhisperAttention.forward = counted_eager_forward
        for module in model.modules():
            if isinstance(module, WhisperSdpaAttention):
                hooks.append(module.register_forward_hook(count_sdpa_module))
            elif isinstance(module, WhisperAttention):
                hooks.append(module.register_forward_hook(count_eager_module))

    profiler_activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        profiler_activities.append(torch.profiler.ProfilerActivity.CUDA)
    profiler_context = (
        torch.profiler.profile(activities=profiler_activities)
        if implementation == "sdpa"
        else contextlib.nullcontext()
    )

    try:
        with profiler_context as profiler:
            with torch.cuda.amp.autocast(
                enabled=True, dtype=torch.float32, cache_enabled=False
            ):
                outputs = model(batch, 0, return_logits=True)
                loss = outputs["loss"]
            loss.backward()
            torch.cuda.synchronize()
        profiler_keys = []
        if implementation == "sdpa":
            profiler_keys = sorted(
                event.key
                for event in profiler.key_averages()
                if "scaled_dot_product" in event.key.lower()
                or "flash" in event.key.lower()
                or "efficient_attention" in event.key.lower()
            )
        result = {
            "loss": outputs["loss"].detach().float().cpu(),
            "acc": outputs["acc"].detach().float().cpu(),
            "len": outputs["len"].detach().cpu(),
            "logits": outputs["logits"].detach().float().cpu(),
            "gradients": {
                name: (
                    None
                    if parameter.grad is None
                    else parameter.grad.detach().float().cpu()
                )
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            },
        }
        gradient_tensors = [
            gradient
            for gradient in result["gradients"].values()
            if gradient is not None
        ]
        metadata = {
            "implementation": implementation,
            "attention_class_counts": class_counts(model),
            "counters": counters,
            "profiler_attention_keys": profiler_keys,
            "cuda_memory": cuda_memory(),
            "gradients": {
                "trainable_parameter_count": len(result["gradients"]),
                "present_count": len(gradient_tensors),
                "none_count": sum(
                    gradient is None for gradient in result["gradients"].values()
                ),
                "elements": sum(gradient.numel() for gradient in gradient_tensors),
                "nonfinite_elements": sum(
                    int((~torch.isfinite(gradient)).sum().item())
                    for gradient in gradient_tensors
                ),
            },
        }
        return result, metadata
    finally:
        for hook in hooks:
            hook.remove()
        functional.scaled_dot_product_attention = original_sdpa
        WhisperAttention.forward = original_eager_forward


def class_counts(model):
    counts = {}
    for module in model.modules():
        name = module.__class__.__name__
        if "Attention" in name:
            counts[name] = counts.get(name, 0) + 1
    return counts


def compare_outputs(eager, sdpa, args):
    eager_logits = eager["logits"]
    sdpa_logits = sdpa["logits"]
    if eager_logits.shape != sdpa_logits.shape:
        raise RuntimeError(
            "logit shapes differ: {} versus {}".format(
                tuple(eager_logits.shape), tuple(sdpa_logits.shape)
            )
        )
    difference = (eager_logits - sdpa_logits).abs()
    eager_tokens = eager_logits.argmax(dim=-1)
    sdpa_tokens = sdpa_logits.argmax(dim=-1)
    token_mismatches = int((eager_tokens != sdpa_tokens).sum().item())
    token_count = eager_tokens.numel()
    loss_abs = float((eager["loss"] - sdpa["loss"]).abs().item())
    acc_abs = float((eager["acc"] - sdpa["acc"]).abs().item())
    lengths_equal = bool(torch.equal(eager["len"], sdpa["len"]))
    report = {
        "loss": {
            "eager": float(eager["loss"].item()),
            "sdpa": float(sdpa["loss"].item()),
            "absolute_difference": loss_abs,
            "atol": args.loss_atol,
        },
        "accuracy": {
            "eager": float(eager["acc"].item()),
            "sdpa": float(sdpa["acc"].item()),
            "absolute_difference": acc_abs,
        },
        "valid_length": {
            "eager": int(eager["len"].item()),
            "sdpa": int(sdpa["len"].item()),
            "equal": lengths_equal,
        },
        "logits": {
            "shape": list(eager_logits.shape),
            "max_absolute_difference": float(difference.max().item()),
            "mean_absolute_difference": float(difference.mean().item()),
            "max_relative_to_eager_amplitude": float(
                difference.max().item()
                / max(float(eager_logits.abs().max().item()), 1e-12)
            ),
            "max_atol": args.logit_max_atol,
            "mean_atol": args.logit_mean_atol,
        },
        "argmax_tokens": {
            "mismatches": token_mismatches,
            "count": token_count,
            "mismatch_rate": token_mismatches / token_count,
        },
    }
    report["pass"] = bool(
        loss_abs <= args.loss_atol
        and acc_abs == 0.0
        and lengths_equal
        and report["logits"]["max_absolute_difference"] <= args.logit_max_atol
        and report["logits"]["mean_absolute_difference"] <= args.logit_mean_atol
        and token_mismatches == 0
    )
    return report


def compare_gradients(eager, sdpa, args):
    eager_gradients = eager["gradients"]
    sdpa_gradients = sdpa["gradients"]
    eager_names = set(eager_gradients)
    sdpa_names = set(sdpa_gradients)
    missing_from_sdpa = sorted(eager_names - sdpa_names)
    missing_from_eager = sorted(sdpa_names - eager_names)
    none_mismatches = []
    shape_mismatches = []
    tensor_reports = []
    total_elements = 0
    total_absolute_difference = 0.0
    eager_squared_norm = 0.0
    sdpa_squared_norm = 0.0
    max_absolute_difference = 0.0
    nonfinite_difference_elements = 0

    for name in sorted(eager_names & sdpa_names):
        eager_gradient = eager_gradients[name]
        sdpa_gradient = sdpa_gradients[name]
        if (eager_gradient is None) != (sdpa_gradient is None):
            none_mismatches.append(name)
            continue
        if eager_gradient is None:
            continue
        if eager_gradient.shape != sdpa_gradient.shape:
            shape_mismatches.append(
                {
                    "name": name,
                    "eager": list(eager_gradient.shape),
                    "sdpa": list(sdpa_gradient.shape),
                }
            )
            continue
        difference = (eager_gradient - sdpa_gradient).abs()
        tensor_max = float(difference.max().item())
        tensor_sum = float(difference.double().sum().item())
        elements = difference.numel()
        total_elements += elements
        total_absolute_difference += tensor_sum
        eager_squared_norm += float(eager_gradient.double().square().sum().item())
        sdpa_squared_norm += float(sdpa_gradient.double().square().sum().item())
        max_absolute_difference = max(max_absolute_difference, tensor_max)
        nonfinite_difference_elements += int((~torch.isfinite(difference)).sum().item())
        tensor_reports.append(
            {
                "name": name,
                "elements": elements,
                "max_absolute_difference": tensor_max,
                "mean_absolute_difference": tensor_sum / elements,
            }
        )

    eager_norm = eager_squared_norm ** 0.5
    sdpa_norm = sdpa_squared_norm ** 0.5
    norm_absolute_difference = abs(eager_norm - sdpa_norm)
    norm_relative_difference = norm_absolute_difference / max(eager_norm, 1e-12)
    mean_absolute_difference = (
        total_absolute_difference / total_elements if total_elements else float("inf")
    )
    tensor_reports.sort(
        key=lambda item: item["max_absolute_difference"], reverse=True
    )
    report = {
        "parameter_names": {
            "eager_count": len(eager_names),
            "sdpa_count": len(sdpa_names),
            "missing_from_sdpa": missing_from_sdpa,
            "missing_from_eager": missing_from_eager,
        },
        "none_mismatches": none_mismatches,
        "shape_mismatches": shape_mismatches,
        "compared_tensors": len(tensor_reports),
        "compared_elements": total_elements,
        "max_absolute_difference": max_absolute_difference,
        "mean_absolute_difference": mean_absolute_difference,
        "nonfinite_difference_elements": nonfinite_difference_elements,
        "global_norm": {
            "eager": eager_norm,
            "sdpa": sdpa_norm,
            "absolute_difference": norm_absolute_difference,
            "relative_difference": norm_relative_difference,
        },
        "tolerances": {
            "max_atol": args.gradient_max_atol,
            "mean_atol": args.gradient_mean_atol,
            "norm_rtol": args.gradient_norm_rtol,
        },
        "largest_tensor_differences": tensor_reports[:20],
    }
    report["pass"] = bool(
        not missing_from_sdpa
        and not missing_from_eager
        and not none_mismatches
        and not shape_mismatches
        and total_elements > 0
        and nonfinite_difference_elements == 0
        and max_absolute_difference <= args.gradient_max_atol
        and mean_absolute_difference <= args.gradient_mean_atol
        and norm_relative_difference <= args.gradient_norm_rtol
    )
    return report


def environment_report():
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": {
            "name": properties.name,
            "total_memory": properties.total_memory,
            "major": properties.major,
            "minor": properties.minor,
        },
        "sdpa_backend_flags": {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
        },
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    seed_everything(args.seed)
    eager_configs = load_config(args.eager_config)
    batch, batch_metadata = load_real_batch(
        eager_configs, args.train_data, args.batch_size
    )
    eager_model = eager_configs["llm"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    eager_model.load_state_dict(checkpoint)
    state_path = pathlib.Path(args.state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(eager_model.state_dict(), state_path)
    state_sha256 = sha256_file(state_path)
    eager_outputs, eager_metadata = run_model(eager_model, batch, "eager")
    del eager_model, eager_configs, checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    seed_everything(args.seed)
    sdpa_configs = load_config(args.sdpa_config)
    sdpa_model = sdpa_configs["llm"]
    full_state = torch.load(state_path, map_location="cpu")
    sdpa_model.load_state_dict(full_state, load_partial_list=[])
    sdpa_outputs, sdpa_metadata = run_model(sdpa_model, batch, "sdpa")
    comparison = compare_outputs(eager_outputs, sdpa_outputs, args)
    gradient_comparison = compare_gradients(eager_outputs, sdpa_outputs, args)
    expected_attention_paths = bool(
        sdpa_metadata["counters"]["sdpa_function_calls"] > 0
        and sdpa_metadata["counters"]["encoder_sdpa_module_calls"]
        == sdpa_metadata["counters"]["sdpa_function_calls"]
        and sdpa_metadata["counters"]["decoder_eager_module_calls"] > 0
        and sdpa_metadata["counters"]["decoder_eager_module_calls"]
        == sdpa_metadata["counters"]["eager_forward_calls_during_sdpa"]
        and sdpa_metadata["counters"]["unexpected_encoder_eager_module_calls"] == 0
        and sdpa_metadata["counters"]["unexpected_decoder_sdpa_module_calls"] == 0
    )
    profiler_keys = sdpa_metadata["profiler_attention_keys"]
    fused_backend_observed = any(
        "flash" in key.lower() or "efficient" in key.lower()
        for key in profiler_keys
    )
    report = {
        "format": "taste-stage1-encoder-sdpa-full-model-parity-v2",
        "environment": environment_report(),
        "inputs": {
            "eager_config": os.path.abspath(args.eager_config),
            "eager_config_sha256": sha256_file(args.eager_config),
            "sdpa_config": os.path.abspath(args.sdpa_config),
            "sdpa_config_sha256": sha256_file(args.sdpa_config),
            "train_data": os.path.abspath(args.train_data),
            "train_data_sha256": sha256_file(args.train_data),
            "source_checkpoint": os.path.abspath(args.checkpoint),
            "source_checkpoint_sha256": sha256_file(args.checkpoint),
            "canonical_full_state": str(state_path.resolve()),
            "canonical_full_state_sha256": state_sha256,
        },
        "batch": batch_metadata,
        "eager": eager_metadata,
        "sdpa": sdpa_metadata,
        "comparison": comparison,
        "gradient_comparison": gradient_comparison,
        "expected_attention_paths": expected_attention_paths,
        "attention_contract": {
            "encoder": "sdpa",
            "decoder": "eager",
        },
        "fused_or_efficient_backend_observed": fused_backend_observed,
    }
    report["pass"] = bool(
        comparison["pass"]
        and gradient_comparison["pass"]
        and expected_attention_paths
        and fused_backend_observed
    )
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
