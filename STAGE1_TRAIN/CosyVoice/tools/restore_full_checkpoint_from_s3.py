#!/usr/bin/env python3

import argparse
import hashlib
import json
import pathlib
import subprocess

from cosyvoice.utils.checkpoint import FULL_CHECKPOINT_FORMAT, torch_load


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as input_file:
        while True:
            chunk = input_file.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download(uri, destination, allow_missing=False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([
        'aws', 's3', 'cp', uri, str(destination), '--only-show-errors',
    ], capture_output=True, text=True)
    if result.returncode == 0:
        return True
    if allow_missing and ('404' in result.stderr or 'Not Found' in result.stderr):
        return False
    raise subprocess.CalledProcessError(
        result.returncode, result.args, output=result.stdout, stderr=result.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sync_uri', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--expected_step', type=int)
    parser.add_argument('--allow_missing', action='store_true')
    args = parser.parse_args()

    sync_uri = args.sync_uri.rstrip('/')
    output_directory = pathlib.Path(args.output_dir)
    latest_path = output_directory / 'latest_resume.json'
    found = download(
        f'{sync_uri}/latest_resume.json',
        latest_path,
        allow_missing=args.allow_missing)
    if not found:
        report = {
            'format': 'cosyvoice-full-checkpoint-restore-v1',
            'status': 'missing',
            'sync_uri': sync_uri,
        }
        report_path = output_directory / 'restore-report.json'
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    latest = json.loads(latest_path.read_text())
    if latest.get('format') != FULL_CHECKPOINT_FORMAT:
        raise ValueError('latest pointer has unsupported format')
    if args.expected_step is not None and latest.get('step') != args.expected_step:
        raise ValueError(
            f'expected step {args.expected_step}, got {latest.get("step")}')

    manifest_path = output_directory / latest['manifest']
    download(f'{sync_uri}/{latest["manifest"]}', manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('format') != FULL_CHECKPOINT_FORMAT:
        raise ValueError('checkpoint manifest has unsupported format')
    if manifest.get('checkpoint') != latest.get('checkpoint'):
        raise ValueError('latest pointer and manifest checkpoint names differ')
    if manifest.get('step') != latest.get('step'):
        raise ValueError('latest pointer and manifest steps differ')

    for record in manifest['objects']:
        destination = output_directory / record['name']
        download(record['s3_uri'], destination)
        if destination.stat().st_size != record['size']:
            raise ValueError(f'{record["name"]}: size mismatch')
        digest = sha256_file(destination)
        if digest != record['sha256']:
            raise ValueError(f'{record["name"]}: SHA256 mismatch')

    checkpoint_path = output_directory / manifest['checkpoint']
    checkpoint = torch_load(checkpoint_path, map_location='cpu')
    if checkpoint.get('format') != FULL_CHECKPOINT_FORMAT:
        raise ValueError('downloaded checkpoint has unsupported format')
    if checkpoint['executor_state']['step'] != manifest['step']:
        raise ValueError('downloaded checkpoint and manifest steps differ')
    report = {
        'format': 'cosyvoice-full-checkpoint-restore-v1',
        'status': 'restored',
        'sync_uri': sync_uri,
        'checkpoint_path': str(checkpoint_path),
        'step': manifest['step'],
        'object_count': len(manifest['objects']),
        'total_bytes': sum(record['size'] for record in manifest['objects']),
        'objects': manifest['objects'],
    }
    report_path = output_directory / 'restore-report.json'
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
