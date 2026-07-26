#!/usr/bin/env python3
"""Gate-1 tests for seekable-resume deterministic primitives (task #34).

Covers CONTRACT.md v1.2 sect. 3 (canonical JSON), sect. 4 (seed derivation), and
sect. 9 (identity precheck + authoritative model-input hash). Includes:
  * fixed GOLDEN VECTORS that pin the byte-exact encodings,
  * cross-process reproducibility (recompute in a fresh interpreter),
  * negative cases: dtype / shape / field-order / endianness / single-byte tamper,
  * seed domain separation (shard-order vs row-buffer) and forbidden-primitive
    determinism.

Run: PYTHONPATH=<CosyVoice> python3 tools/test_seekable_primitives.py
Golden vectors are the reference for Codex's gate-1 sign-off; if any implementer
choice (see hashing.py IMPLEMENTER CHOICES A-D) is adjusted, regenerate these.
"""

import struct
import subprocess
import sys

import torch

from cosyvoice.utils.seekable import (
    canonical_json_bytes, canonical_json_sha256,
    derive_seed, shard_order_seed, row_buffer_seed,
    SHARD_ORDER_STREAM, ROW_BUFFER_STREAM,
    model_input_hash, identity_precheck_hash,
)

# ---------------------------------------------------------------- golden vectors
GOLDEN_CJSON_VALUE = {"b": 1, "a": [2, 3], "z": "x", "nested": {"k2": 2, "k1": 1}}
GOLDEN_CJSON_BYTES = b'{"a":[2,3],"b":1,"nested":{"k1":1,"k2":2},"z":"x"}'
GOLDEN_CJSON_SHA = "650959b5f8a861c0b7875aad77a049d036923a7fcd701e7e9e3925fc5897a569"

GOLDEN_SHARD_ORDER_SEED = 6970483981335531404922005407878698961452659582210266836051821019476787781752
GOLDEN_ROW_BUFFER_SEED = 102551367972260946208989426023777979394896806194243746302112000315044413538292

GOLDEN_IDENTITY_UTTS = ["shardA__k0", "shardB__k1"]
GOLDEN_IDENTITY_ROWIDS = [("aa" * 32, 0), ("bb" * 32, 5)]
GOLDEN_IDENTITY_HASH = "9d306513e5a326ea9b9a4d8e47e9f12e17ed115a2819f88fab19107a99fb9981"

GOLDEN_MODEL_INPUT_HASH = "6ec9d309091a3b9c07ae6cf32622a8d65de93b3910a72af56d5b42acb6400f24"


def golden_batch():
    return {
        "text_token": torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.int64),
        "text_token_len": torch.tensor([3, 2], dtype=torch.int32),
        "speech_token": torch.tensor([[10, 11], [12, 0]], dtype=torch.int64),
        "speech_token_len": torch.tensor([2, 1], dtype=torch.int32),
        "embedding": torch.tensor([[0.5, -0.25], [1.0, 0.0]], dtype=torch.float32),
        "utts": ["shardA__k0", "shardB__k1"],
        "speech_feat_len": torch.tensor([7, 4], dtype=torch.int32),
    }


_RESULTS = []


def check(name, cond):
    _RESULTS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)


# ------------------------------------------------------------------------- tests
def test_canonical_json():
    check("cjson.bytes", canonical_json_bytes(GOLDEN_CJSON_VALUE) == GOLDEN_CJSON_BYTES)
    check("cjson.sha", canonical_json_sha256(GOLDEN_CJSON_VALUE) == GOLDEN_CJSON_SHA)
    # key order irrelevant; content-equal dict -> identical bytes
    other = {"z": "x", "nested": {"k1": 1, "k2": 2}, "a": [2, 3], "b": 1}
    check("cjson.keyorder_invariant",
          canonical_json_bytes(other) == GOLDEN_CJSON_BYTES)
    # NaN forbidden
    try:
        canonical_json_bytes({"x": float("nan")})
        check("cjson.nan_forbidden", False)
    except ValueError:
        check("cjson.nan_forbidden", True)


def test_seed():
    check("seed.shard_order_golden", shard_order_seed(12345, 0) == GOLDEN_SHARD_ORDER_SEED)
    check("seed.row_buffer_golden", row_buffer_seed(12345, 0, 3) == GOLDEN_ROW_BUFFER_SEED)
    # deterministic across repeated calls
    check("seed.deterministic",
          shard_order_seed(12345, 0) == shard_order_seed(12345, 0))
    # domain separation: same (base,epoch,rank) but different stream -> different
    s1 = derive_seed(SHARD_ORDER_STREAM, 7, 1, 2)
    s2 = derive_seed(ROW_BUFFER_STREAM, 7, 1, 2)
    check("seed.stream_separation", s1 != s2)
    # epoch/rank sensitivity
    check("seed.epoch_sensitive", row_buffer_seed(12345, 0, 3) != row_buffer_seed(12345, 1, 3))
    check("seed.rank_sensitive", row_buffer_seed(12345, 0, 3) != row_buffer_seed(12345, 0, 4))


def test_identity_hash():
    check("identity.golden",
          identity_precheck_hash(GOLDEN_IDENTITY_UTTS, GOLDEN_IDENTITY_ROWIDS) == GOLDEN_IDENTITY_HASH)
    # order sensitivity
    check("identity.utt_order",
          identity_precheck_hash(list(reversed(GOLDEN_IDENTITY_UTTS)), GOLDEN_IDENTITY_ROWIDS)
          != GOLDEN_IDENTITY_HASH)
    check("identity.rowid_order",
          identity_precheck_hash(GOLDEN_IDENTITY_UTTS, list(reversed(GOLDEN_IDENTITY_ROWIDS)))
          != GOLDEN_IDENTITY_HASH)


def test_model_input_hash():
    check("model_input.golden", model_input_hash(golden_batch()) == GOLDEN_MODEL_INPUT_HASH)
    check("model_input.deterministic",
          model_input_hash(golden_batch()) == model_input_hash(golden_batch()))


def test_model_input_negatives():
    base = model_input_hash(golden_batch())
    # dtype tamper: int32 len -> int64 len
    b = golden_batch(); b["text_token_len"] = b["text_token_len"].to(torch.int64)
    check("neg.dtype", model_input_hash(b) != base)
    # shape tamper: pad an extra column
    b = golden_batch(); b["text_token"] = torch.nn.functional.pad(b["text_token"], (0, 1))
    check("neg.shape", model_input_hash(b) != base)
    # field-order/value swap: swap text_token & speech_token values (same shapes)
    b = golden_batch(); b["text_token"], b["speech_token"] = (
        torch.tensor([[10, 11, 0], [12, 0, 0]], dtype=torch.int64),
        torch.tensor([[1, 2], [4, 5]], dtype=torch.int64))
    check("neg.field_content", model_input_hash(b) != base)
    # single-value (single-byte-region) tamper
    b = golden_batch(); b["text_token"][0, 0] = 99
    check("neg.single_value", model_input_hash(b) != base)
    # embedding endianness/value tamper: change one float bit-pattern
    b = golden_batch(); b["embedding"][0, 0] = 0.5000001
    check("neg.embedding_value", model_input_hash(b) != base)
    # NaN forbidden in float payload
    b = golden_batch(); b["embedding"][0, 0] = float("nan")
    try:
        model_input_hash(b); check("neg.nan_forbidden", False)
    except ValueError:
        check("neg.nan_forbidden", True)
    # utts guard tamper
    b = golden_batch(); b["utts"] = ["shardA__k0", "shardB__kX"]
    check("neg.utts_guard", model_input_hash(b) != base)
    # speech_feat_len guard tamper (same identity, different sort-determining len)
    b = golden_batch(); b["speech_feat_len"] = torch.tensor([8, 4], dtype=torch.int32)
    check("neg.speech_feat_len_guard", model_input_hash(b) != base)


def test_dtype_mapping():
    """Contract v1.2 sect. 9 (choice C): explicit fixed dtype mapping, alias
    canonicalization to storage dtype, unlisted dtype aborts."""
    from cosyvoice.utils.seekable.hashing import _canonical_dtype_name
    # alias: torch.long IS torch.int64 -> "int64"; torch.float IS float32.
    check("dtype.alias_long", _canonical_dtype_name(torch.long) == "int64")
    check("dtype.alias_float", _canonical_dtype_name(torch.float) == "float32")
    # alias batch equivalence: same values, torch.long vs torch.int64 -> same hash
    b_long = golden_batch()
    b_long["text_token"] = b_long["text_token"].to(torch.long)
    b_long["speech_token"] = b_long["speech_token"].to(torch.long)
    check("dtype.alias_batch_equiv", model_input_hash(b_long) == GOLDEN_MODEL_INPUT_HASH)
    # unlisted dtype -> abort (not str(dtype) fallback)
    b = golden_batch(); b["embedding"] = b["embedding"].to(torch.complex64)
    try:
        model_input_hash(b); check("dtype.unlisted_abort", False)
    except ValueError:
        check("dtype.unlisted_abort", True)


def test_cross_process():
    """Recompute all golden hashes in a fresh interpreter; must be byte-identical
    (guards against PYTHONHASHSEED / process-salted nondeterminism)."""
    code = (
        "import torch,sys;"
        "sys.path.insert(0, %r);"
        "from cosyvoice.utils.seekable import canonical_json_sha256, shard_order_seed, "
        "row_buffer_seed, identity_precheck_hash, model_input_hash;"
        "import tools.test_seekable_primitives as T;"
        "print(canonical_json_sha256(T.GOLDEN_CJSON_VALUE));"
        "print(shard_order_seed(12345,0));"
        "print(row_buffer_seed(12345,0,3));"
        "print(identity_precheck_hash(T.GOLDEN_IDENTITY_UTTS, T.GOLDEN_IDENTITY_ROWIDS));"
        "print(model_input_hash(T.golden_batch()))"
    ) % sys.path[0]
    # run with a deliberately different PYTHONHASHSEED to catch hash()-salting
    import os
    env = dict(os.environ, PYTHONHASHSEED="12345")
    out = subprocess.check_output([sys.executable, "-c", code], env=env).decode().split()
    check("xproc.cjson", out[0] == GOLDEN_CJSON_SHA)
    check("xproc.shard_seed", out[1] == str(GOLDEN_SHARD_ORDER_SEED))
    check("xproc.row_seed", out[2] == str(GOLDEN_ROW_BUFFER_SEED))
    check("xproc.identity", out[3] == GOLDEN_IDENTITY_HASH)
    check("xproc.model_input", out[4] == GOLDEN_MODEL_INPUT_HASH)


def main():
    test_canonical_json()
    test_seed()
    test_identity_hash()
    test_model_input_hash()
    test_model_input_negatives()
    test_dtype_mapping()
    test_cross_process()
    n = len(_RESULTS); passed = sum(1 for _, ok in _RESULTS if ok)
    print("\n%d/%d passed" % (passed, n))
    failed = [name for name, ok in _RESULTS if not ok]
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)
    print("GATE1_PRIMITIVES_PASS")


if __name__ == "__main__":
    main()
