#!/usr/bin/env python3

import argparse
import json
import random
from io import BytesIO

import torchaudio
from datasets import Audio, Dataset
from matcha.utils.audio import mel_spectrogram

from cosyvoice.dataset.processor import text_only_audio_lengths
from cosyvoice.utils.file_utils import read_lists


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_list', required=True)
    parser.add_argument('--num_samples', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1986)
    parser.add_argument('--target_sample_rate', type=int, default=22050)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--hop_size', type=int, default=256)
    parser.add_argument('--num_mels', type=int, default=80)
    parser.add_argument('--fmin', type=int, default=0)
    parser.add_argument('--fmax', type=int, default=8000)
    return parser.parse_args()


def sample_locations(data_list, count, seed):
    paths = read_lists(data_list)
    if not paths:
        raise ValueError('data list is empty')
    rng = random.Random(seed)
    path_order = list(paths)
    rng.shuffle(path_order)
    locations = []
    for sample_index in range(count):
        path = path_order[sample_index % len(path_order)]
        dataset = Dataset.from_file(path)
        if len(dataset) == 0:
            raise ValueError(f'empty dataset: {path}')
        locations.append((path, rng.randrange(len(dataset))))
    return locations


def main():
    args = get_args()
    if args.num_samples <= 0:
        raise ValueError('num_samples must be positive')

    datasets = {}
    checked = []
    for path, row_index in sample_locations(
            args.data_list, args.num_samples, args.seed):
        if path not in datasets:
            datasets[path] = Dataset.from_file(path).cast_column(
                'mp3', Audio(decode=False))
        sample = datasets[path][row_index]
        audio_data = sample['mp3']['bytes']
        predicted = text_only_audio_lengths(
            audio_data,
            target_sample_rate=args.target_sample_rate,
            n_fft=args.n_fft,
            hop_size=args.hop_size)

        waveform, source_sample_rate = torchaudio.load(BytesIO(audio_data))
        if waveform.shape[-1] != predicted['source_frames']:
            raise AssertionError(
                f'{path}:{row_index} decoded source frames '
                f'{waveform.shape[-1]} != header {predicted["source_frames"]}')
        if source_sample_rate != predicted['source_sample_rate']:
            raise AssertionError(
                f'{path}:{row_index} decoded sample rate '
                f'{source_sample_rate} != header '
                f'{predicted["source_sample_rate"]}')

        if source_sample_rate != args.target_sample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=source_sample_rate,
                new_freq=args.target_sample_rate)(waveform)
        if waveform.shape[-1] != predicted['resampled_frames']:
            raise AssertionError(
                f'{path}:{row_index} actual resampled frames '
                f'{waveform.shape[-1]} != predicted '
                f'{predicted["resampled_frames"]}')

        feature = mel_spectrogram(
            waveform,
            n_fft=args.n_fft,
            num_mels=args.num_mels,
            sampling_rate=args.target_sample_rate,
            hop_size=args.hop_size,
            win_size=args.n_fft,
            fmin=args.fmin,
            fmax=args.fmax,
            center=False)
        actual_feature_frames = feature.shape[-1]
        if actual_feature_frames != predicted['feature_frames']:
            raise AssertionError(
                f'{path}:{row_index} actual feature frames '
                f'{actual_feature_frames} != predicted '
                f'{predicted["feature_frames"]}')
        checked.append({
            'path': path,
            'row_index': row_index,
            **predicted,
        })

    report = {
        'status': 'PASS',
        'checked_samples': len(checked),
        'source_sample_rates': sorted({
            item['source_sample_rate'] for item in checked
        }),
        'source_frame_remainders': len({
            item['source_frames'] % item['source_sample_rate']
            for item in checked
        }),
        'min_source_frames': min(
            item['source_frames'] for item in checked),
        'max_source_frames': max(
            item['source_frames'] for item in checked),
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
