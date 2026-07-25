# Text-only loader fast path

The text-only baseline consumes `text_token`, `speech_token`, and the selected
speaker embedding. `TransformerLM.forward` does not consume `speech_feat`.
Before this change, the shared data pipeline still decoded every embedded MP3,
resampled it to 22.05 kHz, and computed an 80-bin mel spectrogram. Those values
were discarded by the model; only their frame count affected filtering,
sorting, dynamic batch boundaries, and padding.

The text-only config now:

1. reads the encoded MP3 bytes without decoding the waveform;
2. reads frame count and sample rate from the audio header with SoundFile;
3. derives torchaudio's resampled length and Matcha's STFT frame count with the
   same integer formulas;
4. applies the original filter conditions;
5. emits a one-column zero placeholder with the exact original frame count.

The placeholder retains the existing dataset contract while avoiding waveform
decode, resampling, STFT, and mel projection. Other configs keep
`decode_audio=True` by default and are unchanged. The text-only launcher keeps
two workers per rank to preserve shard assignment and raises only the prefetch
depth from 8 to 32.

## Invariants

- model architecture, initialization, seed, optimizer, peak learning rate,
  warmup, epochs, accumulation, clipping, and dynamic batch size are unchanged;
- accepted utterances, key order, per-rank partitioning, frame lengths, sort
  order, dynamic batch boundaries, and padded batch order must be exact;
- the first accumulated optimizer step must have exact loss, accuracy, valid
  length, pre-clip gradient norm, and post-step parameters;
- every rank's benchmark key SHA-256 must match the decode baseline.

## Validation

Create an untouched reference config from the pinned base commit:

```bash
git show 43d44b891d164b996a0e3a61e8fb99bfee9740b8:\
STAGE1_TRAIN/CosyVoice/examples/emilia/taste/conf/text-only_baseline.yaml \
  > /tmp/text-only-baseline.decode-reference.yaml
```

From `STAGE1_TRAIN/CosyVoice`, run:

```bash
python tools/validate_text_only_frame_lengths.py \
  --data_list /path/to/frozen-two-shard.data.list \
  --num_samples 128

python tools/validate_text_only_loader_equivalence.py \
  --baseline_config /tmp/text-only-baseline.decode-reference.yaml \
  --optimized_config examples/emilia/taste/conf/text-only_baseline.yaml \
  --data_list /path/to/frozen-two-shard.data.list \
  --num_batches 200 --num_workers 2 --prefetch 8

python tools/validate_text_only_step_equivalence.py \
  --baseline_config /tmp/text-only-baseline.decode-reference.yaml \
  --optimized_config examples/emilia/taste/conf/text-only_baseline.yaml \
  --data_list /path/to/frozen-two-shard.data.list --device cuda:0
```

Run the eight-rank loader gate once with the decode reference and once with the
optimized config. Use a cold page cache for each run when measuring latency.

```bash
torchrun --standalone --nproc_per_node=8 \
  tools/benchmark_text_only_loader.py \
  --config /tmp/text-only-baseline.decode-reference.yaml \
  --data_list /path/to/train.data.list \
  --output_dir /tmp/text-loader-baseline \
  --num_batches 300 --warmup_batches 50 \
  --num_workers 2 --prefetch 8 --compute_sleep_seconds 0.25

torchrun --standalone --nproc_per_node=8 \
  tools/benchmark_text_only_loader.py \
  --config examples/emilia/taste/conf/text-only_baseline.yaml \
  --data_list /path/to/train.data.list \
  --output_dir /tmp/text-loader-optimized \
  --num_batches 300 --warmup_batches 50 \
  --num_workers 2 --prefetch 32 --compute_sleep_seconds 0.25
```

Finally, count the effective optimizer steps across the exact training
topology. The report proves whether the 10,000-step warmup reaches peak
`0.0002` and enters the decay phase.

```bash
torchrun --standalone --nproc_per_node=8 \
  tools/count_text_only_steps.py \
  --config examples/emilia/taste/conf/text-only_baseline.yaml \
  --data_list /path/to/train.data.list \
  --num_workers 2 --prefetch 32 \
  --output /tmp/text-only-step-count.json
```
