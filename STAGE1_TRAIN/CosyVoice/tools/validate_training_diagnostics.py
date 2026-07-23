#!/usr/bin/env python3

import math
import os

import torch
import torch.distributed as dist

from cosyvoice.utils.diagnostic_metrics import summarize_audio_representations
from cosyvoice.utils.training_metrics import StepMetricAccumulator


def assert_close(actual, expected, name, tolerance=1e-7):
    if not math.isclose(
            float(actual), float(expected), rel_tol=tolerance,
            abs_tol=tolerance):
        raise AssertionError(
            '{}: expected {}, got {}'.format(name, expected, actual))


def validate_representation_summary():
    features = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0], [99.0, 99.0]],
        [[1.0, 0.0], [99.0, 99.0], [99.0, 99.0]],
    ], requires_grad=True)
    lengths = torch.tensor([2, 1])
    text_lengths = torch.tensor([4, 2])
    fusion_logits = torch.tensor([0.0, math.log(3.0)], requires_grad=True)
    metrics = summarize_audio_representations(
        features, lengths, text_lengths, fusion_logits)

    assert_close(metrics['audio_repr_variance_mean'], 2.0 / 9.0, 'variance')
    assert_close(
        metrics['audio_repr_pairwise_cosine_mean'], 1.0 / 3.0,
        'pairwise cosine')
    assert_close(
        metrics['audio_repr_effective_rank_ratio'], 1.0,
        'effective rank ratio')
    assert_close(metrics['audio_repr_norm_mean'], 1.0, 'norm')
    assert_close(metrics['audio_segment_count_mean'], 1.5, 'segment mean')
    assert_close(metrics['audio_segment_count_min'], 1.0, 'segment min')
    assert_close(metrics['audio_segment_count_max'], 2.0, 'segment max')
    assert_close(metrics['audio_empty_segment_fraction'], 0.0, 'empty fraction')
    assert_close(metrics['audio_to_text_segment_ratio'], 0.5, 'segment ratio')
    assert_close(metrics['audio_fusion_weight'], 0.25, 'audio fusion weight')
    assert_close(metrics['text_fusion_weight'], 0.75, 'text fusion weight')
    if metrics['audio_repr_count'].item() != 3:
        raise AssertionError('unexpected representation count')
    if any(value.requires_grad for value in metrics.values()):
        raise AssertionError('diagnostic metrics must not retain gradients')

    collapsed = summarize_audio_representations(
        torch.ones(2, 3, 4),
        torch.tensor([3, 3]),
        torch.tensor([3, 3]))
    assert_close(collapsed['audio_repr_variance_mean'], 0.0, 'collapsed variance')
    assert_close(
        collapsed['audio_repr_pairwise_cosine_mean'], 1.0,
        'collapsed cosine')
    assert_close(
        collapsed['audio_repr_effective_rank_ratio'], 0.0,
        'collapsed effective rank')


def validate_step_aggregation():
    accumulator = StepMetricAccumulator()
    accumulator.update({
        'loss': torch.tensor(2.0),
        'ce_loss': torch.tensor(1.5),
        'quantization_loss_raw': torch.tensor(0.5),
        'quantization_loss_weighted': torch.tensor(0.5),
        'acc': torch.tensor(0.25),
        'len': torch.tensor(2),
        'audio_repr_variance_mean': torch.tensor(0.1),
        'audio_repr_pairwise_cosine_mean': torch.tensor(0.2),
        'audio_repr_effective_rank_ratio': torch.tensor(0.3),
        'audio_repr_norm_mean': torch.tensor(0.4),
        'audio_segment_count_mean': torch.tensor(2.0),
        'audio_segment_count_min': torch.tensor(1.0),
        'audio_segment_count_max': torch.tensor(3.0),
        'audio_empty_segment_fraction': torch.tensor(0.0),
        'audio_to_text_segment_ratio': torch.tensor(0.5),
        'audio_fusion_weight': torch.tensor(0.6),
        'text_fusion_weight': torch.tensor(0.4),
        'audio_repr_count': torch.tensor(4),
        'utterance_count': torch.tensor(2),
        'text_token_count': torch.tensor(8),
    })
    accumulator.update({
        'loss': torch.tensor(4.0),
        'ce_loss': torch.tensor(3.0),
        'quantization_loss_raw': torch.tensor(1.0),
        'quantization_loss_weighted': torch.tensor(1.0),
        'acc': torch.tensor(0.5),
        'len': torch.tensor(6),
        'audio_repr_variance_mean': torch.tensor(0.3),
        'audio_repr_pairwise_cosine_mean': torch.tensor(0.4),
        'audio_repr_effective_rank_ratio': torch.tensor(0.5),
        'audio_repr_norm_mean': torch.tensor(0.6),
        'audio_segment_count_mean': torch.tensor(4.0),
        'audio_segment_count_min': torch.tensor(2.0),
        'audio_segment_count_max': torch.tensor(7.0),
        'audio_empty_segment_fraction': torch.tensor(0.5),
        'audio_to_text_segment_ratio': torch.tensor(0.25),
        'audio_fusion_weight': torch.tensor(0.2),
        'text_fusion_weight': torch.tensor(0.8),
        'audio_repr_count': torch.tensor(12),
        'utterance_count': torch.tensor(4),
        'text_token_count': torch.tensor(16),
    })
    result = accumulator.reduce(torch.device('cpu'))

    assert_close(result['loss'], 3.5, 'weighted loss')
    assert_close(result['ce_loss'], 2.625, 'weighted ce loss')
    assert_close(result['audio_repr_variance_mean'], 0.25, 'weighted variance')
    assert_close(
        result['audio_segment_count_mean'], 10.0 / 3.0,
        'weighted segment count')
    assert_close(
        result['audio_to_text_segment_ratio'], 1.0 / 3.0,
        'weighted segment ratio')
    assert_close(result['audio_segment_count_min'], 1.0, 'global segment min')
    assert_close(result['audio_segment_count_max'], 7.0, 'global segment max')
    assert_close(result['len'], 8.0, 'speech token count')
    assert_close(result['audio_repr_count'], 16.0, 'audio representation count')


def validate_distributed_step_aggregation():
    if int(os.environ.get('WORLD_SIZE', '1')) == 1:
        return
    dist.init_process_group(backend='gloo')
    try:
        rank = dist.get_rank()
        accumulator = StepMetricAccumulator()
        accumulator.update({
            'loss': torch.tensor(float(rank + 1)),
            'ce_loss': torch.tensor(float(rank + 1)),
            'quantization_loss_raw': torch.tensor(0.0),
            'quantization_loss_weighted': torch.tensor(0.0),
            'acc': torch.tensor(float(rank) / 2.0),
            'len': torch.tensor(rank + 1),
            'audio_segment_count_min': torch.tensor(rank + 2),
            'audio_segment_count_max': torch.tensor(rank + 4),
        })
        result = accumulator.reduce(torch.device('cpu'))
        assert_close(result['loss'], 5.0 / 3.0, 'distributed weighted loss')
        assert_close(result['len'], 3.0, 'distributed token count')
        assert_close(
            result['audio_segment_count_min'], 2.0,
            'distributed segment min')
        assert_close(
            result['audio_segment_count_max'], 5.0,
            'distributed segment max')
    finally:
        dist.destroy_process_group()


def main():
    validate_representation_summary()
    validate_step_aggregation()
    validate_distributed_step_aggregation()
    print('training diagnostics validation: PASS')


if __name__ == '__main__':
    main()
