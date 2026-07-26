"""row_index_v1 length + acceptance core (task #34), CONTRACT.md v1.3 sect. 6.

Contract v1.3 SHA256:
21d414b507b082e39e45919d7ea1634aad1409d9ca8546bd9badabef0634b9b8

`speech_feat_len` is a decode-free audio-HEADER/STFT function (NOT a speech-token
pure function and NOT a container-duration approximation). The authoritative
frozen text-only path reads encoded-audio header metadata with
``soundfile.info(BytesIO(audio_data))`` and applies the exact frozen formula
below (target_sample_rate=22050, n_fft=1024, hop=256).

Two decision classes (contract sect. 6):
  * FATAL (aborts compilation, NOT an ordinary filtered row): empty audio bytes,
    nonpositive header values, reflect-pad violation (resampled_frames <= 384),
    nonpositive feature length, malformed embeddings, or any
    tokenizer/header/transform exception.
  * ORDINARY acceptance (records the FIRST failing reason, in order): the five
    filters in ``ACCEPTANCE_RULES``.

This module owns ONLY the header-formula + ordered-acceptance logic so it can be
unit-tested with synthetic ``soundfile.info`` headers. The full row_index build
runs the real frozen ``parquet_opener(decode_audio=False) -> tokenize_by_words
-> prepare_text_only_batching -> parse_embedding`` per row; the >=10k production-
output oracle (separate source-pinned process) is the length ground truth, never
a full waveform decode.
"""

TARGET_SAMPLE_RATE = 22050
N_FFT = 1024
HOP_SIZE = 256
REFLECT_PAD = (N_FFT - HOP_SIZE) // 2  # 384

# Ordered rejection reason codes (contract sect. 6, precedence order).
REASON_SAMPLE_RATE_BELOW_MIN = "sample_rate_below_16000"
REASON_DURATION_OUT_OF_RANGE = "duration_frames_out_of_range"
REASON_TEXT_TOKEN_LEN_OUT_OF_RANGE = "text_token_len_out_of_range"
REASON_SPEECH_TOKEN_LEN_ZERO = "speech_token_len_zero"
REASON_TEXT_RATIO_OUT_OF_RANGE = "text_token_ratio_out_of_range"

# Fatal reason codes.
FATAL_EMPTY_AUDIO = "empty_audio"
FATAL_NONPOSITIVE_HEADER = "nonpositive_header"
FATAL_REFLECT_PAD_VIOLATION = "reflect_pad_violation"
FATAL_NONPOSITIVE_FEATURE_LENGTH = "nonpositive_feature_length"

MIN_SAMPLE_RATE = 16000
DURATION_FRAMES_MAX = 40960
TEXT_TOKEN_LEN_MIN = 1
TEXT_TOKEN_LEN_MAX = 200
TEXT_RATIO_MIN = 0.0005
TEXT_RATIO_MAX = 1.0


class FatalRowError(Exception):
    """A condition that aborts compilation rather than filtering a row."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__("{}: {}".format(reason, detail) if detail else reason)
        self.reason = reason


def resampled_frames(source_frames: int, source_sample_rate: int) -> int:
    """Frozen resample-length rule (contract sect. 6 / production
    text_only_audio_lengths): identity at 22050, else the EXACT INTEGER ceil
    ``(source_frames*22050 + sr - 1) // sr``.

    This replicates production's exact integer expression verbatim. It must NOT
    be substituted with ``math.ceil(source_frames*22050/sr)``: float division
    differs from the integer ceil by 1 at divisibility/float boundaries.
    """
    if source_sample_rate == TARGET_SAMPLE_RATE:
        return source_frames
    return (
        source_frames * TARGET_SAMPLE_RATE + source_sample_rate - 1
    ) // source_sample_rate


def speech_feat_len(source_frames: int, source_sample_rate: int) -> int:
    """Exact decode-free placeholder height (contract sect. 6).

    Raises FatalRowError on nonpositive header, reflect-pad violation
    (resampled_frames <= 384), or nonpositive feature length. The placeholder is
    ``torch.zeros((speech_feat_len, 1), dtype=torch.float32)``.
    """
    if source_frames <= 0 or source_sample_rate <= 0:
        raise FatalRowError(FATAL_NONPOSITIVE_HEADER,
                            "frames={} sr={}".format(source_frames, source_sample_rate))
    rf = resampled_frames(source_frames, source_sample_rate)
    if rf <= REFLECT_PAD:
        raise FatalRowError(FATAL_REFLECT_PAD_VIOLATION,
                            "resampled_frames={} <= {}".format(rf, REFLECT_PAD))
    padded_frames = rf + 2 * REFLECT_PAD
    feat_len = 1 + (padded_frames - N_FFT) // HOP_SIZE  # floor division
    if feat_len <= 0:
        raise FatalRowError(FATAL_NONPOSITIVE_FEATURE_LENGTH, "feat_len={}".format(feat_len))
    return feat_len


def duration_frames(source_frames: int, source_sample_rate: int) -> float:
    """duration_frames = source_frames / source_sample_rate * 100 (contract sect. 6)."""
    return source_frames / source_sample_rate * 100


def classify_row(source_frames: int, source_sample_rate: int,
                 text_token_len: int, speech_token_len: int):
    """Ordered acceptance decision (contract sect. 6).

    Returns (accepted: bool, reason: str|None). Fatal conditions raise
    FatalRowError. The FIRST failing rule (in precedence order) is recorded.
    ``speech_feat_len`` is computed first so its fatal conditions are checked
    before ordinary filtering.
    """
    # Fatal header/length checks first (raise, not reject).
    _ = speech_feat_len(source_frames, source_sample_rate)
    dur = duration_frames(source_frames, source_sample_rate)
    # Ordered ordinary acceptance (record first failing reason).
    if source_sample_rate < MIN_SAMPLE_RATE:
        return False, REASON_SAMPLE_RATE_BELOW_MIN
    if not (0 <= dur <= DURATION_FRAMES_MAX):
        return False, REASON_DURATION_OUT_OF_RANGE
    if not (TEXT_TOKEN_LEN_MIN <= text_token_len <= TEXT_TOKEN_LEN_MAX):
        return False, REASON_TEXT_TOKEN_LEN_OUT_OF_RANGE
    if speech_token_len == 0:
        return False, REASON_SPEECH_TOKEN_LEN_ZERO
    if dur != 0:
        ratio = text_token_len / dur
        if not (TEXT_RATIO_MIN <= ratio <= TEXT_RATIO_MAX):
            return False, REASON_TEXT_RATIO_OUT_OF_RANGE
    return True, None
