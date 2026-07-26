#!/usr/bin/env python3
"""Handwritten known-answer tests for row_index_v1 §6 core (task #34).

CONTRACT.md v1.3 (sha 21d414b5...) sect. 6: decode-free header/STFT length +
ordered acceptance/fatal. Synthetic (source_frames, source_sample_rate) headers;
no real audio needed. This is the INDEPENDENT ordered-reason evaluator — it must
replicate production's EXACT integer expressions (esp. the resample ceil), which
is why it is handwritten and not shared with the compiler.

Run: PYTHONPATH=<CosyVoice> python3 tools/test_seekable_row_index.py
"""

import sys

from cosyvoice.utils.seekable import row_index as R

_RESULTS = []


def check(name, cond):
    _RESULTS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)


def rejects_fatal(name, fn):
    try:
        fn(); check(name, False)
    except R.FatalRowError:
        check(name, True)


def test_length_golden():
    # (source_frames, sr) -> speech_feat_len
    for fr, sr, exp in [
        (22050, 22050, 86),
        (16000, 16000, 86),   # resampled -> 22050
        (44100, 44100, 86),   # resampled -> 22050
        (1000, 22050, 3),
        (385, 22050, 1),      # resampled=385 > pad=384 -> feat_len 1
    ]:
        check("len.%d@%d==%d" % (fr, sr, exp), R.speech_feat_len(fr, sr) == exp)


def test_resample_ceil_integer_exact():
    """Contract §6 / production text_only_audio_lengths uses the INTEGER ceil
    ``(fr*22050 + sr - 1)//sr``, NOT math.ceil(float). Verify integer semantics
    at a divisibility boundary (no +1) and a non-divisible case (+1)."""
    # divisible: 44100*22050 divisible by 44100 -> exact quotient 22050, no +1
    check("ceil.divisible_no_plus1", R.resampled_frames(44100, 44100) == 22050)
    # non-divisible: 44101*22050 / 44100 -> ceil = 22051 (+1)
    check("ceil.nondivisible_plus1", R.resampled_frames(44101, 44100) == 22051)
    # matches the exact integer formula across a range (never float math.ceil)
    ok = all(
        R.resampled_frames(fr, sr) == (fr * 22050 + sr - 1) // sr
        for sr in (8000, 11025, 16000, 24000, 44100, 48000)
        for fr in (401, 1000, 44099, 44100, 44101, 1_000_000, 4_999_999)
    )
    check("ceil.matches_integer_formula", ok)
    # identity at 22050
    check("ceil.identity_at_22050", R.resampled_frames(12345, 22050) == 12345)


def test_fatal():
    rejects_fatal("fatal.reflect_pad_384", lambda: R.speech_feat_len(384, 22050))
    rejects_fatal("fatal.nonpos_frames", lambda: R.speech_feat_len(0, 22050))
    rejects_fatal("fatal.nonpos_sr", lambda: R.speech_feat_len(22050, 0))
    rejects_fatal("fatal.classify_reflect_pad",
                  lambda: R.classify_row(384, 22050, 50, 10))
    # reflect-pad is FATAL, not an ordinary reject
    try:
        R.classify_row(384, 22050, 50, 10)
        check("fatal.reflect_pad_is_fatal_not_reject", False)
    except R.FatalRowError as e:
        check("fatal.reflect_pad_is_fatal_not_reject",
              e.reason == R.FATAL_REFLECT_PAD_VIOLATION)


def test_acceptance():
    C = R.classify_row
    check("acc.accept", C(22050, 22050, 50, 10) == (True, None))
    check("rej.sr", C(8000, 8000, 50, 10) == (False, R.REASON_SAMPLE_RATE_BELOW_MIN))
    check("rej.dur", C(9040500, 22050, 50, 10) == (False, R.REASON_DURATION_OUT_OF_RANGE))
    check("rej.textlen_low", C(22050, 22050, 0, 10) == (False, R.REASON_TEXT_TOKEN_LEN_OUT_OF_RANGE))
    check("rej.textlen_high", C(22050, 22050, 201, 10) == (False, R.REASON_TEXT_TOKEN_LEN_OUT_OF_RANGE))
    check("rej.speech0", C(22050, 22050, 50, 0) == (False, R.REASON_SPEECH_TOKEN_LEN_ZERO))
    check("rej.ratio_high", C(22050, 22050, 150, 1) == (False, R.REASON_TEXT_RATIO_OUT_OF_RANGE))
    check("rej.ratio_low", C(9031680, 22050, 1, 1) == (False, R.REASON_TEXT_RATIO_OUT_OF_RANGE))


def test_boundaries():
    C = R.classify_row
    # rule2 duration inclusive [0, 40960]: fr=9031680@22050 -> dur=40960.0 accept
    check("bnd.dur_40960_accept", C(9031680, 22050, 50, 1)[0] is True)
    check("bnd.dur_over_reject", C(9031681, 22050, 50, 1) == (False, R.REASON_DURATION_OUT_OF_RANGE))
    # rule3 text_token_len inclusive [1, 200]
    check("bnd.textlen_1_accept", C(22050, 22050, 1, 10)[0] is True)
    # text_len=200 needs dur>=200 so ratio<=1: fr=44100@22050 -> dur=200, ratio=1.0
    check("bnd.textlen_200_accept", C(44100, 22050, 200, 10)[0] is True)
    # rule5 ratio inclusive [0.0005, 1.0]: dur=100, text=100 -> ratio 1.0 accept
    check("bnd.ratio_1.0_accept", C(22050, 22050, 100, 1)[0] is True)
    # ratio 0.0005 exact: text=1, dur=2000 (fr=441000@22050) -> accept
    check("bnd.ratio_0.0005_accept", C(441000, 22050, 1, 1)[0] is True)


def test_fatal_precedes_ordinary():
    """Contract §6 / production order: text_only_audio_lengths (fatal, raises)
    runs BEFORE the 5 acceptance filters (prepare_text_only_batching:29 has no
    try/except, so reflect-pad propagates = abort; the sr filter is at :36).
    A row that is BOTH reflect-pad-fatal AND sr<16000 must ABORT (reflect-pad),
    NOT reject-sr — otherwise seekable compile would abort on a row production
    merely filters, or filter a row production aborts on."""
    # sr=8000 (<16000) AND frames=100 -> resampled=276 <= 384 (reflect-pad)
    try:
        R.classify_row(100, 8000, 50, 10)
        check("fatal_before_sr_filter", False)
    except R.FatalRowError as e:
        check("fatal_before_sr_filter", e.reason == R.FATAL_REFLECT_PAD_VIOLATION)


def test_precedence():
    """Multi-rejection records the FIRST failing rule in order."""
    C = R.classify_row
    # sr<16000 AND textlen0 AND speech0 -> records sr (rule 1)
    check("prec.sr_first", C(8000, 8000, 0, 0) == (False, R.REASON_SAMPLE_RATE_BELOW_MIN))
    # dur out AND textlen out -> records dur (rule 2 before rule 3)
    check("prec.dur_before_textlen", C(9040500, 22050, 0, 0) == (False, R.REASON_DURATION_OUT_OF_RANGE))
    # textlen out AND speech0 -> records textlen (rule 3 before rule 4)
    check("prec.textlen_before_speech", C(22050, 22050, 0, 0) == (False, R.REASON_TEXT_TOKEN_LEN_OUT_OF_RANGE))


def main():
    test_length_golden()
    test_resample_ceil_integer_exact()
    test_fatal()
    test_acceptance()
    test_boundaries()
    test_fatal_precedes_ordinary()
    test_precedence()
    n = len(_RESULTS); passed = sum(1 for _, ok in _RESULTS if ok)
    print("\n%d/%d passed" % (passed, n))
    failed = [name for name, ok in _RESULTS if not ok]
    if failed:
        print("FAILED:", ", ".join(failed)); sys.exit(1)
    print("ROW_INDEX_SECT6_PASS")


if __name__ == "__main__":
    main()
