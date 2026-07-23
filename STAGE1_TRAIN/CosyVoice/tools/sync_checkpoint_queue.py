#!/usr/bin/env python3

import argparse
import datetime
import json
import os
import pathlib
import time
import traceback

from cosyvoice.utils.checkpoint import process_sync_request


def atomic_json_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name('.{}.tmp.{}'.format(path.name, os.getpid()))
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    os.replace(str(temporary_path), str(path))


def append_metric(path, payload):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as metrics_file:
        metrics_file.write(json.dumps(payload, sort_keys=True) + '\n')
        metrics_file.flush()
        os.fsync(metrics_file.fileno())


def cleanup_snapshots(request):
    if request.get('kind') != 'stage_best':
        return
    for record in request.get('objects', []):
        pathlib.Path(record['local_path']).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue_dir', required=True)
    parser.add_argument('--stop_file', required=True)
    parser.add_argument('--health_file', required=True)
    parser.add_argument('--metrics_jsonl')
    parser.add_argument('--poll_seconds', type=float, default=1.0)
    args = parser.parse_args()

    queue_directory = pathlib.Path(args.queue_dir)
    stop_path = pathlib.Path(args.stop_file)
    health_path = pathlib.Path(args.health_file)
    metrics_path = pathlib.Path(args.metrics_jsonl) if args.metrics_jsonl else None
    failure_path = queue_directory / 'sync-worker.failed.json'
    queue_directory.mkdir(parents=True, exist_ok=True)
    failure_path.unlink(missing_ok=True)
    for processing_path in sorted(queue_directory.glob('*.processing.json')):
        ready_path = processing_path.with_name(
            processing_path.name.replace('.processing.json', '.ready.json'))
        if ready_path.exists():
            raise RuntimeError(
                'cannot recover stale request because ready file exists: {}'.format(
                    ready_path))
        os.replace(str(processing_path), str(ready_path))
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    atomic_json_write(health_path, {
        'format': 'cosyvoice-checkpoint-sync-health-v1',
        'pid': os.getpid(),
        'started_at_utc': started_at,
        'status': 'running',
    })

    try:
        while True:
            ready_paths = sorted(queue_directory.glob('*.ready.json'))
            processing_paths = sorted(queue_directory.glob('*.processing.json'))
            if processing_paths:
                raise RuntimeError(
                    'stale processing request found: {}'.format(processing_paths[0]))
            if not ready_paths:
                if stop_path.exists():
                    break
                time.sleep(args.poll_seconds)
                continue

            ready_path = ready_paths[0]
            processing_path = ready_path.with_name(
                ready_path.name.replace('.ready.json', '.processing.json'))
            os.replace(str(ready_path), str(processing_path))
            request = json.loads(processing_path.read_text())
            sync_started = time.perf_counter()
            result = process_sync_request(request)
            duration = time.perf_counter() - sync_started
            completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            done_path = processing_path.with_name(
                processing_path.name.replace('.processing.json', '.done.json'))
            atomic_json_write(done_path, {
                'format': 'cosyvoice-checkpoint-sync-result-v1',
                'request': request,
                'result': result,
                'duration_seconds': duration,
                'completed_at_utc': completed_at,
            })
            processing_path.unlink()
            cleanup_snapshots(request)
            append_metric(metrics_path, {
                'event': 'checkpoint_sync',
                'kind': request['kind'],
                'step': request.get('step'),
                'epoch': request.get('epoch'),
                'duration_seconds': duration,
                'completed_at_utc': completed_at,
            })
            atomic_json_write(health_path, {
                'format': 'cosyvoice-checkpoint-sync-health-v1',
                'pid': os.getpid(),
                'started_at_utc': started_at,
                'status': 'running',
                'last_publish_at_utc': completed_at,
                'last_publish_kind': request['kind'],
                'last_publish_step': request.get('step'),
            })

        completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_json_write(health_path, {
            'format': 'cosyvoice-checkpoint-sync-health-v1',
            'pid': os.getpid(),
            'started_at_utc': started_at,
            'status': 'complete',
            'completed_at_utc': completed_at,
        })
    except BaseException as error:
        failure = {
            'format': 'cosyvoice-checkpoint-sync-failure-v1',
            'failed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'error_type': type(error).__name__,
            'error': str(error),
            'traceback': traceback.format_exc(),
        }
        atomic_json_write(failure_path, failure)
        atomic_json_write(health_path, {
            'format': 'cosyvoice-checkpoint-sync-health-v1',
            'pid': os.getpid(),
            'started_at_utc': started_at,
            'status': 'failed',
            'failure': str(failure_path),
        })
        raise


if __name__ == '__main__':
    main()
