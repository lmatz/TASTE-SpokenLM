import math
from collections import defaultdict
from typing import Dict

import torch
import torch.distributed as dist


MEAN_METRIC_WEIGHTS = {
    'loss': 'len',
    'ce_loss': 'len',
    'quantization_loss_raw': 'len',
    'quantization_loss_weighted': 'len',
    'acc': 'len',
    'audio_repr_variance_mean': 'audio_repr_count',
    'audio_repr_pairwise_cosine_mean': 'audio_repr_count',
    'audio_repr_effective_rank_ratio': 'audio_repr_count',
    'audio_repr_norm_mean': 'audio_repr_count',
    'audio_segment_count_mean': 'utterance_count',
    'audio_empty_segment_fraction': 'utterance_count',
    'audio_to_text_segment_ratio': 'text_token_count',
    'audio_fusion_weight': 'utterance_count',
    'text_fusion_weight': 'utterance_count',
}
MIN_METRICS = ('audio_segment_count_min',)
MAX_METRICS = ('audio_segment_count_max',)
COUNTER_METRICS = (
    'len',
    'audio_repr_count',
    'utterance_count',
    'text_token_count',
)


def _scalar(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError('training metrics must be scalar tensors')
        return float(value.detach().item())
    return float(value)


class StepMetricAccumulator:

    def __init__(self):
        self.reset()

    def reset(self):
        self.numerators = defaultdict(float)
        self.denominators = defaultdict(float)
        self.minimums = {name: math.inf for name in MIN_METRICS}
        self.maximums = {name: -math.inf for name in MAX_METRICS}
        self.counters = defaultdict(float)

    def update(self, metrics):
        scalar_metrics = {
            name: _scalar(value) for name, value in metrics.items()
            if name in MEAN_METRIC_WEIGHTS
            or name in MIN_METRICS
            or name in MAX_METRICS
            or name in COUNTER_METRICS
        }
        for name in COUNTER_METRICS:
            if name in scalar_metrics:
                self.counters[name] += scalar_metrics[name]
        for name, weight_name in MEAN_METRIC_WEIGHTS.items():
            if name not in scalar_metrics:
                continue
            if weight_name not in scalar_metrics:
                raise KeyError(
                    '{} requires metric weight {}'.format(name, weight_name))
            weight = scalar_metrics[weight_name]
            self.numerators[name] += scalar_metrics[name] * weight
            self.denominators[name] += weight
        for name in MIN_METRICS:
            if name in scalar_metrics:
                self.minimums[name] = min(
                    self.minimums[name], scalar_metrics[name])
        for name in MAX_METRICS:
            if name in scalar_metrics:
                self.maximums[name] = max(
                    self.maximums[name], scalar_metrics[name])

    def reduce(self, device):
        mean_names = list(MEAN_METRIC_WEIGHTS)
        mean_values = []
        for name in mean_names:
            mean_values.extend([
                self.numerators[name],
                self.denominators[name],
            ])
        mean_tensor = torch.tensor(
            mean_values, dtype=torch.float64, device=device)
        counter_tensor = torch.tensor(
            [self.counters[name] for name in COUNTER_METRICS],
            dtype=torch.float64,
            device=device)
        minimum_tensor = torch.tensor(
            [self.minimums[name] for name in MIN_METRICS],
            dtype=torch.float64,
            device=device)
        maximum_tensor = torch.tensor(
            [self.maximums[name] for name in MAX_METRICS],
            dtype=torch.float64,
            device=device)

        if dist.is_initialized():
            dist.all_reduce(mean_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(counter_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(minimum_tensor, op=dist.ReduceOp.MIN)
            dist.all_reduce(maximum_tensor, op=dist.ReduceOp.MAX)

        result = {}
        for index, name in enumerate(mean_names):
            numerator = float(mean_tensor[index * 2].item())
            denominator = float(mean_tensor[index * 2 + 1].item())
            if denominator > 0:
                result[name] = numerator / denominator
        for index, name in enumerate(COUNTER_METRICS):
            value = float(counter_tensor[index].item())
            if value > 0:
                result[name] = value
        for index, name in enumerate(MIN_METRICS):
            value = float(minimum_tensor[index].item())
            if math.isfinite(value):
                result[name] = value
        for index, name in enumerate(MAX_METRICS):
            value = float(maximum_tensor[index].item())
            if math.isfinite(value):
                result[name] = value
        return result
