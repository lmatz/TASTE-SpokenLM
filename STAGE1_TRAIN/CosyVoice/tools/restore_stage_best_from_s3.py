#!/usr/bin/env python3

import argparse
import hashlib
import json
import pathlib
import subprocess

from cosyvoice.utils.checkpoint import STAGE_BEST_FORMAT


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
    parser.add_argument('--allow_missing', action='store_true')
    args = parser.parse_args()

    sync_uri = args.sync_uri.rstrip('/')
    output_directory = pathlib.Path(args.output_dir)
    pointer_path = output_directory / 'latest_best.json'
    found = download(
        f'{sync_uri}/latest_best.json',
        pointer_path,
        allow_missing=args.allow_missing)
    if not found:
        report = {
            'format': 'cosyvoice-stage-best-restore-v1',
            'status': 'missing',
            'sync_uri': sync_uri,
        }
        report_path = output_directory / 'restore-best-report.json'
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    pointer = json.loads(pointer_path.read_text())
    if pointer.get('format') != STAGE_BEST_FORMAT:
        raise ValueError('latest best pointer has unsupported format')
    manifest_path = output_directory / 'stage-best.manifest.json'
    download(f'{sync_uri}/{pointer["manifest"]}', manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('format') != STAGE_BEST_FORMAT:
        raise ValueError('stage best manifest has unsupported format')
    if float(manifest['score']) != float(pointer['score']):
        raise ValueError('stage best pointer and manifest scores differ')

    restored = []
    for record in manifest['objects']:
        if record['role'] == 'model':
            destination = output_directory / 'checkpoint_best.pt'
        elif record['role'] == 'metadata':
            destination = output_directory / 'checkpoint_best.yaml'
        else:
            raise ValueError(f'unsupported stage best object role: {record["role"]}')
        download(record['s3_uri'], destination)
        if destination.stat().st_size != record['size']:
            raise ValueError(f'{record["role"]}: size mismatch')
        if sha256_file(destination) != record['sha256']:
            raise ValueError(f'{record["role"]}: SHA256 mismatch')
        restored.append({**record, 'local_path': str(destination)})

    if not any(record['role'] == 'model' for record in restored):
        raise ValueError('stage best manifest has no model object')
    report = {
        'format': 'cosyvoice-stage-best-restore-v1',
        'status': 'restored',
        'sync_uri': sync_uri,
        'score': float(pointer['score']),
        'epoch': pointer['epoch'],
        'step': pointer['step'],
        'objects': restored,
    }
    report_path = output_directory / 'restore-best-report.json'
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
