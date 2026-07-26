"""Batch hashing for the seekable-resume oracle (task #34), CONTRACT.md v1 sect. 9.

Two hashes per batch:
  * identity/order PRECHECK  -- cheap fail-fast (domain tag + ordered utt keys +
    ordered row ids).
  * authoritative MODEL-INPUT hash -- over the exact tensors the text-only model
    consumes, in a fixed field order, with dtype/shape/endianness bound in.

The model-input digest begins with b"TASTE_STAGE1_MODEL_INPUT_V1\\x00" and hashes,
IN THIS EXACT ORDER (contract sect. 9): text_token, text_token_len, speech_token,
speech_token_len, embedding, then the guards ordered `utts` and `speech_feat_len`.
`speech_feat` values are NOT hashed (the text-only model does not consume them);
`speech_feat_len` IS hashed because it determines sorting and batch formation.

Per-tensor-field encoding (contract sect. 9, items 1-6):
  1. field-name length (uint16 BE) + UTF-8 field name
  2. dtype-name length + ASCII canonical dtype name
  3. ndim (uint16 BE)
  4. each shape dim (uint64 BE)
  5. payload byte length (uint64 BE)
  6. CPU-contiguous C-order value bytes, explicitly little-endian

Encoding widths / framing (contract v1.2 sect. 9, SHA
bc54a7a6a09e7ab7b362699c6b16d35a56deb28c03ebac625653579041dfac43):
  (A) dtype-name length prefix width  = uint16 BE   [item 2, now normative]
  (B) string-list element count width = uint64 BE   [utts / string lists]
  (C) canonical dtype name = EXPLICIT FIXED MAPPING (contract v1.2, NOT str(dtype)):
      bool/uint8/int8/int16/int32/int64/float16/bfloat16/float32/float64.
      Any dtype outside this v1 mapping is forbidden -> abort. Framework aliases
      such as ``torch.long`` canonicalize to the storage dtype name (``int64``)
      because ``tensor.dtype`` already normalizes aliases to the storage dtype.
  (D) identity-precheck row-id encoding = source_shard_sha256 (32 raw bytes)
      || zero_based_row_index (uint64 BE); precheck prefix
      b"TASTE_STAGE1_IDENTITY_PRECHECK_V1\\x00".
"""

import struct

import numpy as np
import torch

MODEL_INPUT_MAGIC = b"TASTE_STAGE1_MODEL_INPUT_V1\x00"
IDENTITY_PRECHECK_MAGIC = b"TASTE_STAGE1_IDENTITY_PRECHECK_V1\x00"

# Contract sect. 9 field order for the text-only Stage1 v1 authoritative hash,
# followed by the guard fields.
MODEL_INPUT_FIELDS = (
    "text_token",
    "text_token_len",
    "speech_token",
    "speech_token_len",
    "embedding",
)
GUARD_TENSOR_FIELDS = ("speech_feat_len",)  # utts (string list) handled separately

_DTYPE_NAME = {
    torch.float64: "float64",
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}


def _canonical_dtype_name(dtype: torch.dtype) -> str:
    """Choice (C): canonical ASCII dtype name from the explicit fixed mapping.

    ``dtype`` is ``tensor.dtype``, which already normalizes framework aliases
    (``torch.long`` -> ``torch.int64``) to the storage dtype, so the mapping keys
    are the canonical storage dtypes. Any dtype outside the v1 mapping is
    forbidden by contract v1.2 sect. 9 and aborts rather than guessing a name.
    """
    if dtype not in _DTYPE_NAME:
        raise ValueError(
            "dtype forbidden by seekable contract v1.2 model-input hash: {}".format(dtype))
    return _DTYPE_NAME[dtype]


def _le_bytes_from_numpy(a: "np.ndarray") -> bytes:
    a = np.ascontiguousarray(a)
    if a.dtype.byteorder not in ("<", "|"):  # big or native-on-BE host
        a = a.astype(a.dtype.newbyteorder("<"))
    return a.tobytes(order="C")


def _little_endian_c_bytes(t: torch.Tensor) -> bytes:
    """CPU-contiguous, C-order, explicitly little-endian value bytes (item 6).

    Forces LE regardless of host endianness and forbids NaN/Inf in float payloads
    (contract sect. 9). bfloat16 has no numpy dtype, so its raw 16-bit storage is
    viewed as uint16 and LE-canonicalized (the bf16 bit pattern is preserved).
    """
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        if not bool(torch.isfinite(t).all()):
            raise ValueError("NaN/Inf forbidden in bfloat16 model-input payload")
        # reinterpret the 2-byte bf16 storage as uint16 (same element size)
        return _le_bytes_from_numpy(t.view(torch.uint16).numpy())
    a = t.numpy()
    if a.dtype.kind == "f" and not np.all(np.isfinite(a)):
        raise ValueError("NaN/Inf forbidden in float model-input payload")
    return _le_bytes_from_numpy(a)


def _encode_tensor_field(name: str, t: torch.Tensor) -> bytes:
    name_b = name.encode("utf-8")
    dtype_b = _canonical_dtype_name(t.dtype).encode("ascii")
    payload = _little_endian_c_bytes(t)
    out = bytearray()
    out += struct.pack(">H", len(name_b)) + name_b            # item 1
    out += struct.pack(">H", len(dtype_b)) + dtype_b          # item 2 (choice A: uint16 BE)
    out += struct.pack(">H", t.dim())                         # item 3
    for dim in t.shape:                                       # item 4
        out += struct.pack(">Q", int(dim))
    out += struct.pack(">Q", len(payload))                    # item 5
    out += payload                                            # item 6
    return bytes(out)


def _encode_string_list_field(name: str, items) -> bytes:
    """String-list field (e.g. utts): field name, then count, then per element
    a uint64 BE byte length + UTF-8 bytes (contract sect. 9)."""
    name_b = name.encode("utf-8")
    out = bytearray()
    out += struct.pack(">H", len(name_b)) + name_b
    out += struct.pack(">Q", len(items))                      # choice B: uint64 BE count
    for s in items:
        b = s.encode("utf-8")
        out += struct.pack(">Q", len(b)) + b
    return bytes(out)


def _require_tensor(name: str, v) -> torch.Tensor:
    """Fail-closed: the oracle hashes the ACTUAL collated tensor objects; a list
    or ndarray silently coerced to a tensor would let runtime type drift hash
    successfully instead of aborting before forward (contract sect. 9)."""
    if not isinstance(v, torch.Tensor):
        raise TypeError(
            "model-input field {!r} must be a torch.Tensor, got {}".format(
                name, type(v).__name__))
    return v


def _require_str_sequence(name: str, v):
    """Fail-closed: ``utts`` must be a sequence of str, not a bare string (which
    would iterate into characters) and not contain non-str elements."""
    if isinstance(v, (str, bytes, bytearray)):
        raise TypeError("{!r} must be a sequence of str, not a bare string".format(name))
    items = list(v)
    for x in items:
        if not isinstance(x, str):
            raise TypeError("{!r} must contain only str, got {}".format(name, type(x).__name__))
    return items


def model_input_hash(batch: dict) -> str:
    """Authoritative model-input SHA-256 (contract sect. 9). ``batch`` is the
    collated dict the model consumes plus the ``utts``/``speech_feat_len`` guards.
    All five model fields and ``speech_feat_len`` must be real ``torch.Tensor``;
    ``utts`` must be a strict sequence of str."""
    import hashlib
    h = hashlib.sha256()
    h.update(MODEL_INPUT_MAGIC)
    for field in MODEL_INPUT_FIELDS:
        h.update(_encode_tensor_field(field, _require_tensor(field, batch[field])))
    h.update(_encode_string_list_field("utts", _require_str_sequence("utts", batch["utts"])))
    for field in GUARD_TENSOR_FIELDS:
        h.update(_encode_tensor_field(field, _require_tensor(field, batch[field])))
    return h.hexdigest()


_HEX64_CHARS = frozenset("0123456789abcdef")


def _require_sha256_hex(v) -> bytes:
    """Fail-closed: exactly lowercase [0-9a-f]{64}; reject short/whitespace/upper."""
    if not isinstance(v, str) or len(v) != 64 or any(c not in _HEX64_CHARS for c in v):
        raise ValueError("shard sha256 must be exactly lowercase 64-hex: {!r}".format(v))
    return bytes.fromhex(v)


def _require_row_index(v) -> int:
    """Fail-closed: a real non-bool integer in [0, 2**64) (reject float/str/bool)."""
    if isinstance(v, bool) or not isinstance(v, int) or not (0 <= v < 2 ** 64):
        raise ValueError("row_index must be a non-bool int in [0, 2**64): {!r}".format(v))
    return v


def identity_precheck_hash(utts, row_ids) -> str:
    """Cheap identity/order precheck (contract sect. 9), fail-closed.

    ``row_ids`` is an ordered iterable of (source_shard_sha256_hex, row_index).
    Choice (D): row id encoded as 32 raw sha256 bytes + uint64 BE row index.
    Requires len(utts) == len(row_ids), each utt a str, each shard sha exactly
    lowercase 64-hex, each row_index a non-bool int in [0, 2**64).
    """
    import hashlib
    utts = _require_str_sequence("utts", utts)
    row_ids = list(row_ids)
    if len(utts) != len(row_ids):
        raise ValueError(
            "identity precheck: len(utts)={} != len(row_ids)={}".format(len(utts), len(row_ids)))
    h = hashlib.sha256()
    h.update(IDENTITY_PRECHECK_MAGIC)
    h.update(struct.pack(">Q", len(utts)))
    for u in utts:
        b = u.encode("utf-8")
        h.update(struct.pack(">Q", len(b)) + b)
    h.update(struct.pack(">Q", len(row_ids)))
    for shard_sha256_hex, row_index in row_ids:
        h.update(_require_sha256_hex(shard_sha256_hex))
        h.update(struct.pack(">Q", _require_row_index(row_index)))
    return h.hexdigest()
