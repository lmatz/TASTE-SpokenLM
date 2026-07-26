"""Seekable Stage1 resume (task #34) — FOUNDATION primitives only.

This package currently provides ONLY the reviewed, self-contained building
blocks: deterministic canonical ordering, seed derivation, and model-input /
identity hashing (this module) plus the decode-free header/STFT length +
ordered acceptance core (``row_index``). It does NOT implement the batch-plan
compiler, offset index, or runtime cursor seek, so it does NOT provide O(1)
resume and NOTHING here activates seekable resume. Those are task #34
follow-ups; production activation stays gated on all contract gates + explicit
``switch_checkpoint`` approval.

Frozen contract: CONTRACT.md v1.3, sha256
21d414b507b082e39e45919d7ea1634aad1409d9ca8546bd9badabef0634b9b8.
"""
from .canonical import canonical_json_bytes, canonical_json_sha256, sha256_hex
from .seed import (
    MAGIC,
    SHARD_ORDER_STREAM,
    ROW_BUFFER_STREAM,
    SHARD_ORDER_GLOBAL_RANK,
    derive_seed,
    shard_order_seed,
    row_buffer_seed,
)
from .hashing import (
    MODEL_INPUT_MAGIC,
    IDENTITY_PRECHECK_MAGIC,
    model_input_hash,
    identity_precheck_hash,
)

__all__ = [
    "canonical_json_bytes", "canonical_json_sha256", "sha256_hex",
    "MAGIC", "SHARD_ORDER_STREAM", "ROW_BUFFER_STREAM", "SHARD_ORDER_GLOBAL_RANK",
    "derive_seed", "shard_order_seed", "row_buffer_seed",
    "MODEL_INPUT_MAGIC", "IDENTITY_PRECHECK_MAGIC",
    "model_input_hash", "identity_precheck_hash",
]
