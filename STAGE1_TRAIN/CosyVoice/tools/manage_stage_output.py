#!/usr/bin/env python3

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import tempfile

from cosyvoice.utils.checkpoint import STAGE_BEST_FORMAT


STAGE_OUTPUT_FORMAT = 'cosyvoice_stage_output_v1'


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as input_file:
        while True:
            chunk = input_file.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def upload_and_verify(path, uri):
    digest = sha256_file(path)
    size = path.stat().st_size
    subprocess.run([
        'aws', 's3', 'cp', str(path), uri,
        '--only-show-errors', '--sse', 'AES256',
    ], check=True)
    bucket, key = uri[5:].split('/', 1)
    head = json.loads(subprocess.run([
        'aws', 's3api', 'head-object',
        '--bucket', bucket,
        '--key', key,
        '--output', 'json',
    ], check=True, capture_output=True, text=True).stdout)
    if head.get('ContentLength') != size:
        raise ValueError(f'{uri}: uploaded size mismatch')
    if head.get('ServerSideEncryption') != 'AES256':
        raise ValueError(f'{uri}: uploaded object is not AES256 encrypted')
    remote_digest = hashlib.sha256()
    process = subprocess.Popen(
        ['aws', 's3', 'cp', uri, '-', '--only-show-errors'],
        stdout=subprocess.PIPE)
    try:
        while True:
            chunk = process.stdout.read(8 * 1024 * 1024)
            if not chunk:
                break
            remote_digest.update(chunk)
    finally:
        process.stdout.close()
    if process.wait() != 0:
        raise RuntimeError(f'{uri}: failed to re-download uploaded object')
    if remote_digest.hexdigest() != digest:
        raise ValueError(f'{uri}: uploaded SHA256 mismatch')
    return {
        's3_uri': uri,
        'size': size,
        'sha256': digest,
    }


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


def load_verified_json(uri, destination, expected_sha256=None):
    download(uri, destination)
    digest = sha256_file(destination)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f'{uri}: JSON SHA256 mismatch')
    return json.loads(destination.read_text()), digest


def load_stage_best(sync_uri, temporary_directory):
    pointer, _ = load_verified_json(
        f'{sync_uri}/latest_best.json',
        temporary_directory / 'latest_best.json')
    if pointer.get('format') != STAGE_BEST_FORMAT:
        raise ValueError('latest stage-best pointer has unsupported format')
    manifest, manifest_sha256 = load_verified_json(
        f'{sync_uri}/{pointer["manifest"]}',
        temporary_directory / 'stage-best.manifest.json')
    if manifest.get('format') != STAGE_BEST_FORMAT:
        raise ValueError('stage-best manifest has unsupported format')
    if float(pointer['score']) != float(manifest['score']):
        raise ValueError('stage-best pointer and manifest scores differ')
    model_records = [
        record for record in manifest['objects'] if record.get('role') == 'model'
    ]
    if len(model_records) != 1:
        raise ValueError('stage-best manifest must have exactly one model object')
    return pointer, manifest, manifest_sha256, model_records[0]


def artifact_paths(stage_root):
    candidates = [
        stage_root / 'train.log',
        stage_root / 'model' / 'command.sh',
        stage_root / 'model' / 'config.yaml',
        stage_root / 'model' / 'start-time-utc.txt',
        stage_root / 'model' / 'end-time-utc.txt',
        stage_root / 'model' / 'metrics.jsonl',
        stage_root / 'model' / 'parameter_inventory.json',
        stage_root / 'checkpoint-sync-metrics.jsonl',
        stage_root / 'checkpoint-sync-health.json',
    ]
    return [path for path in candidates if path.is_file()]


def publish(args):
    stage_root = pathlib.Path(args.stage_root).resolve()
    output_uri = args.stage_output_uri.rstrip('/')
    sync_uri = args.checkpoint_sync_uri.rstrip('/')
    if not (stage_root / 'model' / 'end-time-utc.txt').is_file():
        raise ValueError('stage end-time marker is missing')
    local_best = stage_root / 'model' / 'checkpoint_best.pt'
    if not local_best.is_file():
        raise ValueError('local checkpoint_best.pt is missing')

    with tempfile.TemporaryDirectory() as temporary_directory_name:
        temporary_directory = pathlib.Path(temporary_directory_name)
        best_pointer, best_manifest, best_manifest_sha256, model_record = \
            load_stage_best(sync_uri, temporary_directory)
        if local_best.stat().st_size != model_record['size']:
            raise ValueError('local and durable best-model sizes differ')
        if sha256_file(local_best) != model_record['sha256']:
            raise ValueError('local and durable best-model SHA256 values differ')

        files = []
        for path in artifact_paths(stage_root):
            relative_path = path.relative_to(stage_root).as_posix()
            record = upload_and_verify(
                path, f'{output_uri}/files/{relative_path}')
            files.append({**record, 'relative_path': relative_path})

        manifest = {
            'format': STAGE_OUTPUT_FORMAT,
            'stage_name': args.stage_name,
            'schedule_id': args.schedule_id,
            'resume_contract_sha256': args.resume_contract_sha256,
            'max_epoch': args.max_epoch,
            'completed_at_utc': datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
            'checkpoint_sync_uri': sync_uri,
            'best': {
                'pointer': best_pointer,
                'manifest_sha256': best_manifest_sha256,
                'manifest': best_manifest,
            },
            'files': files,
        }
        manifest_path = temporary_directory / 'stage-output-manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        manifest_record = upload_and_verify(
            manifest_path, f'{output_uri}/stage-output-manifest.json')
        marker = {
            'format': STAGE_OUTPUT_FORMAT,
            'stage_name': args.stage_name,
            'schedule_id': args.schedule_id,
            'resume_contract_sha256': args.resume_contract_sha256,
            'manifest': manifest_record,
            'published_at_utc': datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
        }
        marker_path = temporary_directory / '_COMPLETE'
        marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + '\n')
        marker_record = upload_and_verify(marker_path, f'{output_uri}/_COMPLETE')
    print(json.dumps({
        'status': 'published',
        'stage_name': args.stage_name,
        'manifest': manifest_record,
        'marker': marker_record,
        'best_model': model_record,
    }, indent=2, sort_keys=True))


def restore(args):
    output_directory = pathlib.Path(args.output_dir).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    output_uri = args.stage_output_uri.rstrip('/')
    marker_path = output_directory / '_COMPLETE'
    if not download(
            f'{output_uri}/_COMPLETE',
            marker_path,
            allow_missing=args.allow_missing):
        report = {
            'format': STAGE_OUTPUT_FORMAT,
            'status': 'missing',
            'stage_output_uri': output_uri,
        }
        (output_directory / 'restore-stage-output-report.json').write_text(
            json.dumps(report, indent=2, sort_keys=True) + '\n')
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    marker = json.loads(marker_path.read_text())
    if marker.get('format') != STAGE_OUTPUT_FORMAT:
        raise ValueError('stage output marker has unsupported format')
    if marker.get('resume_contract_sha256') != args.resume_contract_sha256:
        raise ValueError('stage output resume contract does not match')
    manifest_record = marker['manifest']
    manifest_path = output_directory / 'stage-output-manifest.json'
    manifest, _ = load_verified_json(
        manifest_record['s3_uri'],
        manifest_path,
        expected_sha256=manifest_record['sha256'])
    if manifest.get('format') != STAGE_OUTPUT_FORMAT:
        raise ValueError('stage output manifest has unsupported format')
    if manifest.get('resume_contract_sha256') != args.resume_contract_sha256:
        raise ValueError('stage output manifest resume contract does not match')
    model_records = [
        record for record in manifest['best']['manifest']['objects']
        if record.get('role') == 'model'
    ]
    if len(model_records) != 1:
        raise ValueError('stage output has no unique best-model record')
    model_record = model_records[0]
    checkpoint_path = output_directory / 'checkpoint_best.pt'
    download(model_record['s3_uri'], checkpoint_path)
    if checkpoint_path.stat().st_size != model_record['size']:
        raise ValueError('restored stage output model size mismatch')
    if sha256_file(checkpoint_path) != model_record['sha256']:
        raise ValueError('restored stage output model SHA256 mismatch')
    report = {
        'format': STAGE_OUTPUT_FORMAT,
        'status': 'complete',
        'stage_output_uri': output_uri,
        'stage_name': manifest['stage_name'],
        'checkpoint_path': str(checkpoint_path),
        'best_score': manifest['best']['pointer']['score'],
        'model': model_record,
    }
    (output_directory / 'restore-stage-output-report.json').write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    publish_parser = subparsers.add_parser('publish')
    publish_parser.add_argument('--stage_name', required=True)
    publish_parser.add_argument('--stage_root', required=True)
    publish_parser.add_argument('--checkpoint_sync_uri', required=True)
    publish_parser.add_argument('--stage_output_uri', required=True)
    publish_parser.add_argument('--resume_contract_sha256', required=True)
    publish_parser.add_argument('--schedule_id', required=True)
    publish_parser.add_argument('--max_epoch', type=int, required=True)
    publish_parser.set_defaults(handler=publish)

    restore_parser = subparsers.add_parser('restore')
    restore_parser.add_argument('--stage_output_uri', required=True)
    restore_parser.add_argument('--resume_contract_sha256', required=True)
    restore_parser.add_argument('--output_dir', required=True)
    restore_parser.add_argument('--allow_missing', action='store_true')
    restore_parser.set_defaults(handler=restore)
    args = parser.parse_args()
    args.handler(args)


if __name__ == '__main__':
    main()
