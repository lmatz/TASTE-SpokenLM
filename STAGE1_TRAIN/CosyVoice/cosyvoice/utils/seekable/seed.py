"""Deterministic seed derivation for seekable plan ordering (task #34).

Implements CONTRACT.md v1 section 4 (Seed derivation) VERBATIM. The function
body below is the normative code from the contract and must not be "improved".

Forbidden for plan ordering (contract sect. 4): Python ``hash()``, NumPy
``default_rng``, and implicit DataLoader worker seeds. The compiler uses
``random.Random(derive_seed(...)).shuffle`` (Mersenne Twister; officially
cross-version reproducible). Worker id / num_workers / prefetch / pin-memory do
NOT enter seed derivation.
"""

import hashlib
import struct

MAGIC = b"TASTE_STAGE1_SEEDED_PLAN_V1\x00"

# Stream names (contract sect. 4).
SHARD_ORDER_STREAM = "shard-order"
ROW_BUFFER_STREAM = "row-buffer"

# For the shard-order stream, global_rank is fixed to 2**64 - 1 (contract sect. 4);
# the shard index list is shuffled globally BEFORE rank slicing.
SHARD_ORDER_GLOBAL_RANK = 2 ** 64 - 1


def derive_seed(stream: str, base_seed: int, epoch: int, global_rank: int) -> int:
    """Contract sect. 4 normative derivation (verbatim)."""
    stream_bytes = stream.encode("ascii")
    assert 0 <= base_seed < 2 ** 64
    assert 0 <= epoch < 2 ** 64
    assert 0 <= global_rank < 2 ** 64
    assert len(stream_bytes) < 2 ** 16
    payload = (
        MAGIC
        + struct.pack(">H", len(stream_bytes))
        + stream_bytes
        + struct.pack(">QQQ", base_seed, epoch, global_rank)
    )
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def shard_order_seed(base_seed: int, epoch: int) -> int:
    """Seed for the global shard-index shuffle (before rank slicing)."""
    return derive_seed(SHARD_ORDER_STREAM, base_seed, epoch, SHARD_ORDER_GLOBAL_RANK)


def row_buffer_seed(base_seed: int, epoch: int, global_rank: int) -> int:
    """Seed for a rank's bounded row-buffer shuffle."""
    return derive_seed(ROW_BUFFER_STREAM, base_seed, epoch, global_rank)
