"""Seekable Stage1 resume (task #34): immutable batch-plan + O(1) cursor seek.

Targets CONTRACT.md v1.2, sha256
bc54a7a6a09e7ab7b362699c6b16d35a56deb28c03ebac625653579041dfac43.
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
