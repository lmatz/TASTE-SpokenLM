#!/usr/bin/env python3

import json
import os
import pathlib
import stat
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import torch

from cosyvoice.utils.checkpoint import (
    enqueue_stage_best_sync,
    save_training_checkpoint,
)


FAKE_AWS = r'''#!/usr/bin/env python3
import json
import os
import pathlib
import shutil
import sys

root = pathlib.Path(os.environ['FAKE_S3_ROOT'])
args = sys.argv[1:]

def s3_path(uri):
    if not uri.startswith('s3://'):
        raise ValueError(uri)
    return root / uri[5:]

if args[:2] == ['s3', 'cp']:
    source, destination = args[2:4]
    if source.startswith('s3://'):
        source_path = s3_path(source)
        if not source_path.is_file():
            print('404 Not Found', file=sys.stderr)
            raise SystemExit(1)
        if destination == '-':
            sys.stdout.buffer.write(source_path.read_bytes())
        else:
            destination_path = pathlib.Path(destination)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, destination_path)
    else:
        destination_path = s3_path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination_path)
elif args[:2] == ['s3api', 'head-object']:
    bucket = args[args.index('--bucket') + 1]
    key = args[args.index('--key') + 1]
    path = root / bucket / key
    if not path.is_file():
        print('404 Not Found', file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps({
        'ContentLength': path.stat().st_size,
        'ServerSideEncryption': 'AES256',
    }))
else:
    raise SystemExit('unsupported fake aws invocation: ' + repr(args))
'''


def wait_for_path(path, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise TimeoutError(path)


def main():
    cosyvoice_root = pathlib.Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as temporary_directory_name:
        temporary_directory = pathlib.Path(temporary_directory_name)
        fake_bin = temporary_directory / 'bin'
        fake_bin.mkdir()
        fake_aws = fake_bin / 'aws'
        fake_aws.write_text(FAKE_AWS)
        fake_aws.chmod(fake_aws.stat().st_mode | stat.S_IXUSR)
        fake_s3 = temporary_directory / 's3'
        queue_directory = temporary_directory / 'queue'
        queue_directory.mkdir()
        stop_path = queue_directory / 'stop'
        health_path = queue_directory / 'health.json'
        metrics_path = queue_directory / 'metrics.jsonl'
        environment = os.environ.copy()
        environment['FAKE_S3_ROOT'] = str(fake_s3)
        environment['PATH'] = '{}:{}'.format(fake_bin, environment['PATH'])
        environment['PYTHONPATH'] = '{}:{}'.format(
            cosyvoice_root, environment.get('PYTHONPATH', ''))
        worker = subprocess.Popen([
            sys.executable,
            str(cosyvoice_root / 'tools' / 'sync_checkpoint_queue.py'),
            '--queue_dir', str(queue_directory),
            '--stop_file', str(stop_path),
            '--health_file', str(health_path),
            '--metrics_jsonl', str(metrics_path),
            '--poll_seconds', '0.05',
        ], cwd=str(cosyvoice_root), env=environment)
        wait_for_path(health_path)

        model = torch.nn.Linear(4, 3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        executor = SimpleNamespace(
            step=7,
            epoch=1,
            cv_best_score=0.5,
            accumulation_count=0)
        checkpoint_path = temporary_directory / 'model' / 'resume_step_00000007.pt'
        save_training_checkpoint(
            model,
            optimizer,
            scheduler,
            executor,
            str(checkpoint_path),
            next_epoch=1,
            next_batch_idx=19,
            train_engine='torch_ddp',
            sync_uri='s3://test-bucket/checkpoints/stage',
            sync_mode='queued',
            sync_queue_dir=str(queue_directory),
            sync_worker_pid=worker.pid,
            sync_queue_wait_seconds=10,
            resume_contract_sha256='c' * 64)

        best_path = temporary_directory / 'model' / 'checkpoint_best.pt'
        best_metadata_path = temporary_directory / 'model' / 'checkpoint_best.yaml'
        torch.save(model.state_dict(), best_path)
        best_metadata_path.write_text('score: 0.75\n')
        enqueue_stage_best_sync(
            str(best_path),
            str(best_metadata_path),
            's3://test-bucket/checkpoints/stage',
            score=0.75,
            epoch=1,
            step=7,
            sync_queue_dir=str(queue_directory),
            sync_worker_pid=worker.pid,
            sync_queue_wait_seconds=10)
        stop_path.touch()
        if worker.wait(timeout=20) != 0:
            raise RuntimeError('checkpoint sync worker failed')

        latest_resume = json.loads(
            (fake_s3 / 'test-bucket/checkpoints/stage/latest_resume.json').read_text())
        latest_best = json.loads(
            (fake_s3 / 'test-bucket/checkpoints/stage/latest_best.json').read_text())
        assert latest_resume['step'] == 7
        assert latest_resume['next_batch_idx'] == 19
        assert latest_best['score'] == 0.75

        restore_root = temporary_directory / 'restore'
        subprocess.run([
            sys.executable,
            str(cosyvoice_root / 'tools' / 'restore_full_checkpoint_from_s3.py'),
            '--sync_uri', 's3://test-bucket/checkpoints/stage',
            '--output_dir', str(restore_root / 'full'),
            '--expected_step', '7',
        ], check=True, cwd=str(cosyvoice_root), env=environment,
           stdout=subprocess.DEVNULL)
        subprocess.run([
            sys.executable,
            str(cosyvoice_root / 'tools' / 'restore_stage_best_from_s3.py'),
            '--sync_uri', 's3://test-bucket/checkpoints/stage',
            '--output_dir', str(restore_root / 'best'),
        ], check=True, cwd=str(cosyvoice_root), env=environment,
           stdout=subprocess.DEVNULL)
        assert (restore_root / 'full' / checkpoint_path.name).is_file()
        assert (restore_root / 'best' / 'checkpoint_best.pt').is_file()
        assert torch.equal(
            torch.load(best_path, map_location='cpu')['weight'],
            torch.load(
                restore_root / 'best' / 'checkpoint_best.pt',
                map_location='cpu')['weight'])

        stage_root = temporary_directory / 'stage'
        stage_model_root = stage_root / 'model'
        stage_model_root.mkdir(parents=True)
        for source, destination_name in [
                (best_path, 'checkpoint_best.pt'),
                (best_metadata_path, 'checkpoint_best.yaml')]:
            (stage_model_root / destination_name).write_bytes(source.read_bytes())
        (stage_model_root / 'end-time-utc.txt').write_text('2026-07-23T00:00:00Z\n')
        (stage_model_root / 'start-time-utc.txt').write_text('2026-07-22T23:00:00Z\n')
        (stage_model_root / 'metrics.jsonl').write_text('{"event":"test"}\n')
        (stage_root / 'train.log').write_text('test stage log\n')
        subprocess.run([
            sys.executable,
            str(cosyvoice_root / 'tools' / 'manage_stage_output.py'),
            'publish',
            '--stage_name', 'taste-no-vq',
            '--stage_root', str(stage_root),
            '--checkpoint_sync_uri', 's3://test-bucket/checkpoints/stage',
            '--stage_output_uri', 's3://test-bucket/artifacts/stage',
            '--resume_contract_sha256', 'c' * 64,
            '--schedule_id', 'test-schedule',
            '--max_epoch', '2',
        ], check=True, cwd=str(cosyvoice_root), env=environment,
           stdout=subprocess.DEVNULL)
        restored_output = restore_root / 'stage-output'
        subprocess.run([
            sys.executable,
            str(cosyvoice_root / 'tools' / 'manage_stage_output.py'),
            'restore',
            '--stage_output_uri', 's3://test-bucket/artifacts/stage',
            '--resume_contract_sha256', 'c' * 64,
            '--output_dir', str(restored_output),
        ], check=True, cwd=str(cosyvoice_root), env=environment,
           stdout=subprocess.DEVNULL)
        assert torch.equal(
            torch.load(best_path, map_location='cpu')['weight'],
            torch.load(
                restored_output / 'checkpoint_best.pt',
                map_location='cpu')['weight'])

    print('async checkpoint sync validation: PASS')


if __name__ == '__main__':
    main()
