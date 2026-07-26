from typing import Dict, Optional

import torch
import torch.nn.functional as F


@torch.no_grad()
def summarize_audio_representations(
        features: torch.Tensor,
        lengths: torch.Tensor,
        text_lengths: torch.Tensor,
        fusion_logits: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    if features.dim() != 3:
        raise ValueError(
            'audio representations must have shape [batch, time, dimension]')
    if lengths.dim() != 1 or lengths.size(0) != features.size(0):
        raise ValueError('audio representation lengths must match the batch')
    if text_lengths.dim() != 1 or text_lengths.size(0) != features.size(0):
        raise ValueError('text lengths must match the batch')

    detached = features.detach().float()
    lengths = lengths.detach().to(device=features.device, dtype=torch.long)
    text_lengths = text_lengths.detach().to(
        device=features.device, dtype=torch.long)
    max_length = features.size(1)
    if torch.any(lengths < 0) or torch.any(lengths > max_length):
        raise ValueError('audio representation lengths are out of range')

    mask = (
        torch.arange(max_length, device=features.device).unsqueeze(0)
        < lengths.unsqueeze(1))
    mask_float = mask.unsqueeze(-1).to(detached.dtype)
    representation_count = mask.sum()
    safe_count = representation_count.clamp_min(1).to(detached.dtype)

    feature_sum = (detached * mask_float).sum(dim=(0, 1))
    feature_square_sum = (detached.square() * mask_float).sum(dim=(0, 1))
    feature_mean = feature_sum / safe_count
    feature_variance = (
        feature_square_sum / safe_count - feature_mean.square()).clamp_min(0.0)
    variance_sum = feature_variance.sum()
    variance_effective_rank = (
        variance_sum.square()
        / feature_variance.square().sum().clamp_min(
            torch.finfo(detached.dtype).eps))
    variance_effective_rank_ratio = (
        variance_effective_rank / max(1, features.size(-1)))
    variance_effective_rank_ratio = torch.where(
        variance_sum > 0,
        variance_effective_rank_ratio,
        torch.zeros_like(variance_effective_rank_ratio)).clamp(0.0, 1.0)

    normalized = F.normalize(detached, dim=-1)
    normalized_sum = (normalized * mask_float).sum(dim=(0, 1))
    count_float = representation_count.to(detached.dtype)
    pairwise_cosine = torch.where(
        representation_count > 1,
        (
            normalized_sum.square().sum() - count_float
        ) / (count_float * (count_float - 1.0)),
        torch.zeros((), dtype=detached.dtype, device=features.device))
    pairwise_cosine = pairwise_cosine.clamp(-1.0, 1.0)

    feature_norm_mean = (
        detached.norm(dim=-1) * mask.to(detached.dtype)).sum() / safe_count
    utterance_count = torch.tensor(
        features.size(0), dtype=torch.long, device=features.device)
    safe_utterance_count = utterance_count.clamp_min(1).to(detached.dtype)
    text_token_count = text_lengths.sum()
    safe_text_token_count = text_token_count.clamp_min(1).to(detached.dtype)

    metrics = {
        'audio_repr_variance_mean': feature_variance.mean(),
        'audio_repr_pairwise_cosine_mean': pairwise_cosine,
        'audio_repr_effective_rank_ratio': variance_effective_rank_ratio,
        'audio_repr_norm_mean': feature_norm_mean,
        'audio_segment_count_mean': lengths.sum().to(
            detached.dtype) / safe_utterance_count,
        'audio_segment_count_min': lengths.min().to(detached.dtype),
        'audio_segment_count_max': lengths.max().to(detached.dtype),
        'audio_empty_segment_fraction': (
            lengths == 0).sum().to(detached.dtype) / safe_utterance_count,
        'audio_to_text_segment_ratio': lengths.sum().to(
            detached.dtype) / safe_text_token_count,
        'audio_repr_count': representation_count,
        'utterance_count': utterance_count,
        'text_token_count': text_token_count,
    }

    if fusion_logits is not None:
        if fusion_logits.numel() != 2:
            raise ValueError('fusion logits must contain audio and text weights')
        fusion_weights = F.softmax(
            fusion_logits.detach().float().reshape(2), dim=0)
        metrics['audio_fusion_weight'] = fusion_weights[0]
        metrics['text_fusion_weight'] = fusion_weights[1]

    return metrics
