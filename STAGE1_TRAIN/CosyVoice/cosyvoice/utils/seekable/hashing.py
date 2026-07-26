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

>>> IMPLEMENTER CHOICES flagged for Codex gate1 review (contract leaves the width
>>> of a few prefixes unspecified; these are pinned here and locked by golden
>>> vectors -- confirm or adjust before production):
>>>   (A) dtype-name length prefix width  = uint16 BE   [item 2]
>>>   (B) string-list element count width = uint64 BE   [utts / string lists]
>>>   (C) canonical dtype name = str(torch.dtype) with the leading "torch."
>>>       stripped, e.g. torch.int64 -> "int64", torch.bfloat16 -> "bfloat16".
>>>   (D) identity-precheck row-id encoding = source_shard_sha256 (32 raw bytes)
>>>       || zero_based_row_index (uint64 BE); precheck prefix
>>>       b"TASTE_STAGE1_IDENTITY_PRECHECK_V1\\x00".
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
    """Choice (C): canonical ASCII dtype name."""
    if dtype not in _DTYPE_NAME:
        raise ValueError("unsupported dtype for model-input hash: {}".format(dtype))
    return _DTYPE_NAME[dtype]


def _little_endian_c_bytes(t: torch.Tensor) -> bytes:
    """CPU-contiguous, C-order, explicitly little-endian value bytes (item 6).

    Forces LE regardless of host endianness via numpy byteorder, and forbids NaN
    in float payloads (contract sect. 9).
    """
    a = t.detach().cpu().contiguous().numpy()
    if a.dtype.kind == "f" and not np.all(np.isfinite(a)):
        raise ValueError("NaN/Inf forbidden in float model-input payload")
    a = np.ascontiguousarray(a)
    if a.dtype.byteorder not in ("<", "|"):  # big or native-on-BE host
        a = a.astype(a.dtype.newbyteorder("<"))
    return a.tobytes(order="C")


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


def _as_tensor(v) -> torch.Tensor:
    return v if isinstance(v, torch.Tensor) else torch.as_tensor(v)


def model_input_hash(batch: dict) -> str:
    """Authoritative model-input SHA-256 (contract sect. 9). ``batch`` is the
    collated dict the model consumes plus the ``utts``/``speech_feat_len`` guards."""
    import hashlib
    h = hashlib.sha256()
    h.update(MODEL_INPUT_MAGIC)
    for field in MODEL_INPUT_FIELDS:
        h.update(_encode_tensor_field(field, _as_tensor(batch[field])))
    h.update(_encode_string_list_field("utts", list(batch["utts"])))
    for field in GUARD_TENSOR_FIELDS:
        h.update(_encode_tensor_field(field, _as_tensor(batch[field])))
    return h.hexdigest()


def identity_precheck_hash(utts, row_ids) -> str:
    """Cheap identity/order precheck (contract sect. 9).

    ``row_ids`` is an ordered iterable of (source_shard_sha256_hex, row_index).
    Choice (D): row id encoded as 32 raw sha256 bytes + uint64 BE row index.
    """
    import hashlib
    h = hashlib.sha256()
    h.update(IDENTITY_PRECHECK_MAGIC)
    utts = list(utts)
    h.update(struct.pack(">Q", len(utts)))
    for u in utts:
        b = u.encode("utf-8")
        h.update(struct.pack(">Q", len(b)) + b)
    row_ids = list(row_ids)
    h.update(struct.pack(">Q", len(row_ids)))
    for shard_sha256_hex, row_index in row_ids:
        h.update(bytes.fromhex(shard_sha256_hex))
        h.update(struct.pack(">Q", int(row_index)))
    return h.hexdigest()
