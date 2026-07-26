"""Canonical JSON + digest helpers for the seekable-resume contract (task #34).

Implements CONTRACT.md v1 section 3 (Canonical JSON). SHA256 of CONTRACT.md v1
that this module targets: 5dda734324f7dbf051594385d42a1b48db8537b17c6d1d51586a14b996352511

Rules (contract sect. 3):
- json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
  separators=(",", ":")).encode("utf-8")
- No trailing newline in bytes used for a digest.
- Integers only for seed/cursor/shape/size/count fields (enforced by callers).
- Hashes are lowercase 64-hex SHA-256.
- Timestamps and filesystem paths are excluded from *semantic* hashes (callers'
  responsibility; this module only serializes what it is given).
"""

import hashlib
import json


def canonical_json_bytes(value) -> bytes:
    """Canonical, digest-stable JSON encoding per contract sect. 3.

    allow_nan=False makes NaN/Infinity raise (contract forbids them in payloads).
    """
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """Lowercase 64-char SHA-256 hex (contract sect. 3)."""
    return hashlib.sha256(data).hexdigest()


def canonical_json_sha256(value) -> str:
    """SHA-256 hex of the canonical JSON encoding of ``value``."""
    return sha256_hex(canonical_json_bytes(value))
