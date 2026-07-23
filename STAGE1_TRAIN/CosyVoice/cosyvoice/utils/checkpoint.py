import datetime
import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist


FULL_CHECKPOINT_FORMAT = 'cosyvoice_full_state_v1'
CHECKPOINT_SYNC_REQUEST_FORMAT = 'cosyvoice_checkpoint_sync_request_v1'
STAGE_BEST_FORMAT = 'cosyvoice_stage_best_v1'


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def parameter_inventory(model):
    trainable = []
    frozen = []
    for name, parameter in unwrap_model(model).named_parameters():
        if parameter.requires_grad:
            trainable.append(name)
        else:
            frozen.append(name)
    return {'trainable': trainable, 'frozen': frozen}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state()
    return state


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if 'torch_cuda' in state:
        if not torch.cuda.is_available():
            raise RuntimeError('resume checkpoint contains CUDA RNG state but CUDA is unavailable')
        torch.cuda.set_rng_state(state['torch_cuda'])


def torch_load(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _atomic_torch_save(payload, path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix='.{}.tmp.'.format(os.path.basename(path)), dir=directory)
    os.close(file_descriptor)
    try:
        torch.save(payload, temporary_path)
        with open(temporary_path, 'rb') as checkpoint_file:
            os.fsync(checkpoint_file.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(directory)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _atomic_json_save(payload, path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix='.{}.tmp.'.format(os.path.basename(path)), dir=directory, text=True)
    try:
        with os.fdopen(file_descriptor, 'w') as output_file:
            json.dump(payload, output_file, indent=2, sort_keys=True)
            output_file.write('\n')
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(directory)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _fsync_directory(directory):
    directory_descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _rank_sidecar_path(checkpoint_path, rank):
    stem, extension = os.path.splitext(checkpoint_path)
    return '{}.rank{:05d}.rng{}'.format(stem, rank, extension)


def save_training_checkpoint(model,
                             optimizer,
                             scheduler,
                             executor,
                             checkpoint_path,
                             next_epoch,
                             next_batch_idx,
                             train_engine,
                             sync_uri=None,
                             sync_mode='synchronous',
                             sync_queue_dir=None,
                             sync_worker_pid=None,
                             sync_queue_wait_seconds=600,
                             resume_contract_sha256=None):
    if train_engine != 'torch_ddp':
        raise NotImplementedError('full-state checkpoints currently support torch_ddp only')

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if getattr(executor, 'accumulation_count', 0) != 0:
        raise RuntimeError('full-state checkpoints require an optimizer-step boundary')
    for name, value in unwrap_model(model).state_dict().items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise FloatingPointError('non-finite model state tensor: {}'.format(name))
    sidecar_path = _rank_sidecar_path(checkpoint_path, rank)
    _atomic_torch_save({
        'format': FULL_CHECKPOINT_FORMAT,
        'rank': rank,
        'world_size': world_size,
        'rng_state': capture_rng_state(),
    }, sidecar_path)

    if dist.is_initialized():
        dist.barrier()

    inventory = parameter_inventory(model)
    if rank == 0:
        payload = {
            'format': FULL_CHECKPOINT_FORMAT,
            'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'world_size': world_size,
            'train_engine': train_engine,
            'model_state_dict': unwrap_model(model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'executor_state': {
                'step': executor.step,
                'epoch': executor.epoch,
                'cv_best_score': executor.cv_best_score,
                'accumulation_count': getattr(executor, 'accumulation_count', 0),
            },
            'resume_contract_sha256': resume_contract_sha256,
            'cursor': {
                'next_epoch': next_epoch,
                'next_batch_idx': next_batch_idx,
            },
            'parameter_inventory': inventory,
            'rank_sidecars': [
                os.path.basename(_rank_sidecar_path(checkpoint_path, sidecar_rank))
                for sidecar_rank in range(world_size)
            ],
        }
        _atomic_torch_save(payload, checkpoint_path)
        _atomic_json_save({
            'format': FULL_CHECKPOINT_FORMAT,
            'checkpoint': os.path.basename(checkpoint_path),
            'rank_sidecars': payload['rank_sidecars'],
            'world_size': world_size,
            'step': executor.step,
            'next_epoch': next_epoch,
            'next_batch_idx': next_batch_idx,
        }, os.path.join(os.path.dirname(checkpoint_path), 'latest_resume.json'))
        if sync_uri:
            if sync_mode == 'synchronous':
                _sync_checkpoint_to_s3(
                    checkpoint_path,
                    payload['rank_sidecars'],
                    sync_uri,
                    executor.step,
                    next_epoch,
                    next_batch_idx)
            elif sync_mode == 'queued':
                _enqueue_checkpoint_sync(
                    checkpoint_path,
                    payload['rank_sidecars'],
                    sync_uri,
                    executor.step,
                    next_epoch,
                    next_batch_idx,
                    sync_queue_dir,
                    sync_worker_pid,
                    sync_queue_wait_seconds)
            else:
                raise ValueError('unsupported checkpoint sync mode: {}'.format(sync_mode))

    if dist.is_initialized():
        dist.barrier()
    return checkpoint_path


def _sync_checkpoint_to_s3(checkpoint_path,
                           rank_sidecars,
                           sync_uri,
                           step,
                           next_epoch,
                           next_batch_idx,
                           expected_objects=None):
    if not sync_uri.startswith('s3://'):
        raise ValueError('checkpoint_sync_uri must be an s3:// URI')
    sync_uri = sync_uri.rstrip('/')
    checkpoint_directory = os.path.dirname(checkpoint_path)
    if expected_objects is None:
        local_paths = [checkpoint_path] + [
            os.path.join(checkpoint_directory, sidecar_name)
            for sidecar_name in rank_sidecars
        ]
        expected_objects = [
            _local_object_record(local_path) for local_path in local_paths
        ]
    object_records = []
    for expected_object in expected_objects:
        local_path = expected_object['local_path']
        if not os.path.isfile(local_path):
            raise FileNotFoundError(local_path)
        if os.path.getsize(local_path) != expected_object['size']:
            raise RuntimeError('queued checkpoint size changed: {}'.format(local_path))
        object_name = expected_object['name']
        remote_uri = '{}/{}'.format(sync_uri, object_name)
        verification = _upload_and_verify(
            local_path,
            remote_uri,
            expected_size=expected_object['size'],
            expected_sha256=expected_object['sha256'])
        object_records.append({
            'name': object_name,
            'size': verification['size'],
            'sha256': verification['sha256'],
            's3_uri': remote_uri,
        })

    checkpoint_name = os.path.basename(checkpoint_path)
    manifest = {
        'format': FULL_CHECKPOINT_FORMAT,
        'checkpoint': checkpoint_name,
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'step': step,
        'next_epoch': next_epoch,
        'next_batch_idx': next_batch_idx,
        'objects': object_records,
    }
    manifest_path = os.path.join(
        checkpoint_directory, '{}.manifest.json'.format(checkpoint_name))
    _atomic_json_save(manifest, manifest_path)
    manifest_uri = '{}/{}'.format(sync_uri, os.path.basename(manifest_path))
    _upload_and_verify(manifest_path, manifest_uri)

    latest_path = os.path.join(checkpoint_directory, 'latest_resume.json')
    _atomic_json_save({
        'format': FULL_CHECKPOINT_FORMAT,
        'checkpoint': checkpoint_name,
        'manifest': os.path.basename(manifest_path),
        'step': step,
        'next_epoch': next_epoch,
        'next_batch_idx': next_batch_idx,
        'sync_uri': sync_uri,
    }, latest_path)
    _upload_and_verify(
        latest_path, '{}/latest_resume.json'.format(sync_uri))
    return manifest


def _enqueue_checkpoint_sync(checkpoint_path,
                             rank_sidecars,
                             sync_uri,
                             step,
                             next_epoch,
                             next_batch_idx,
                             sync_queue_dir,
                             sync_worker_pid,
                             sync_queue_wait_seconds):
    if not sync_queue_dir:
        raise ValueError('queued checkpoint sync requires sync_queue_dir')
    if not sync_worker_pid:
        raise ValueError('queued checkpoint sync requires sync_worker_pid')
    _wait_for_sync_queue(
        sync_queue_dir,
        sync_worker_pid,
        sync_queue_wait_seconds,
        allow_single_stage_best=True)
    checkpoint_directory = os.path.dirname(checkpoint_path)
    local_paths = [checkpoint_path] + [
        os.path.join(checkpoint_directory, sidecar_name)
        for sidecar_name in rank_sidecars
    ]
    request = {
        'format': CHECKPOINT_SYNC_REQUEST_FORMAT,
        'kind': 'full_checkpoint',
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'checkpoint_path': checkpoint_path,
        'rank_sidecars': rank_sidecars,
        'sync_uri': sync_uri,
        'step': step,
        'next_epoch': next_epoch,
        'next_batch_idx': next_batch_idx,
        'objects': [_local_object_record(path) for path in local_paths],
    }
    _write_sync_request(sync_queue_dir, request, os.path.basename(checkpoint_path))


def enqueue_stage_best_sync(checkpoint_path,
                            metadata_path,
                            sync_uri,
                            score,
                            epoch,
                            step,
                            sync_queue_dir,
                            sync_worker_pid,
                            sync_queue_wait_seconds=600):
    if not sync_uri:
        raise ValueError('durable stage best requires checkpoint_sync_uri')
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    _wait_for_sync_queue(
        sync_queue_dir, sync_worker_pid, sync_queue_wait_seconds)
    snapshot_directory = os.path.join(sync_queue_dir, 'snapshots')
    os.makedirs(snapshot_directory, exist_ok=True)
    request_id = '{:020d}-best-epoch{:04d}-step{:08d}'.format(
        time.time_ns(), epoch, step)
    snapshot_path = os.path.join(snapshot_directory, request_id + '.pt')
    _atomic_copy(checkpoint_path, snapshot_path)
    snapshot_metadata_path = None
    if metadata_path and os.path.isfile(metadata_path):
        snapshot_metadata_path = os.path.join(snapshot_directory, request_id + '.yaml')
        _atomic_copy(metadata_path, snapshot_metadata_path)
    local_paths = [snapshot_path]
    if snapshot_metadata_path:
        local_paths.append(snapshot_metadata_path)
    request = {
        'format': CHECKPOINT_SYNC_REQUEST_FORMAT,
        'kind': 'stage_best',
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'sync_uri': sync_uri,
        'score': float(score),
        'epoch': int(epoch),
        'step': int(step),
        'objects': [_local_object_record(path) for path in local_paths],
    }
    _write_sync_request(sync_queue_dir, request, request_id)


def process_sync_request(request):
    if request.get('format') != CHECKPOINT_SYNC_REQUEST_FORMAT:
        raise ValueError('unsupported checkpoint sync request format')
    if request.get('kind') == 'full_checkpoint':
        return _sync_checkpoint_to_s3(
            request['checkpoint_path'],
            request['rank_sidecars'],
            request['sync_uri'],
            request['step'],
            request['next_epoch'],
            request['next_batch_idx'],
            expected_objects=request['objects'])
    if request.get('kind') == 'stage_best':
        return _sync_stage_best_to_s3(request)
    raise ValueError('unsupported checkpoint sync request kind: {}'.format(
        request.get('kind')))


def _sync_stage_best_to_s3(request):
    sync_uri = request['sync_uri'].rstrip('/')
    score = float(request['score'])
    model_object = request['objects'][0]
    digest = model_object['sha256']
    existing = _read_s3_json_optional('{}/latest_best.json'.format(sync_uri))
    if existing is not None:
        if existing.get('format') != STAGE_BEST_FORMAT:
            raise ValueError('existing latest_best pointer has unsupported format')
        existing_score = float(existing['score'])
        if existing_score > score:
            return existing
        if existing_score == score:
            if existing.get('model_sha256') != digest:
                raise ValueError(
                    'equal stage-best scores have different model SHA256 values')
            return existing

    object_records = []
    for index, expected_object in enumerate(request['objects']):
        suffix = '.pt' if index == 0 else '.yaml'
        object_name = 'stage-best/{}{}'.format(expected_object['sha256'], suffix)
        remote_uri = '{}/{}'.format(sync_uri, object_name)
        verification = _upload_and_verify(
            expected_object['local_path'],
            remote_uri,
            expected_size=expected_object['size'],
            expected_sha256=expected_object['sha256'])
        object_records.append({
            'name': object_name,
            'role': 'model' if index == 0 else 'metadata',
            'size': verification['size'],
            'sha256': verification['sha256'],
            's3_uri': remote_uri,
        })
    manifest = {
        'format': STAGE_BEST_FORMAT,
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'score': score,
        'epoch': request['epoch'],
        'step': request['step'],
        'objects': object_records,
    }
    manifest_name = 'stage-best/{}.manifest.json'.format(digest)
    with tempfile.TemporaryDirectory() as temporary_directory:
        manifest_path = os.path.join(temporary_directory, 'manifest.json')
        _atomic_json_save(manifest, manifest_path)
        _upload_and_verify(
            manifest_path, '{}/{}'.format(sync_uri, manifest_name))
        pointer = {
            'format': STAGE_BEST_FORMAT,
            'manifest': manifest_name,
            'score': score,
            'epoch': request['epoch'],
            'step': request['step'],
            'sync_uri': sync_uri,
            'model_sha256': digest,
        }
        pointer_path = os.path.join(temporary_directory, 'latest_best.json')
        _atomic_json_save(pointer, pointer_path)
        _upload_and_verify(
            pointer_path, '{}/latest_best.json'.format(sync_uri))
    return pointer


def check_checkpoint_sync_worker(sync_queue_dir, sync_worker_pid):
    if not sync_queue_dir or not sync_worker_pid:
        raise RuntimeError('queued checkpoint sync worker is not configured')
    failure_path = os.path.join(sync_queue_dir, 'sync-worker.failed.json')
    if os.path.isfile(failure_path):
        with open(failure_path) as failure_file:
            failure = failure_file.read().strip()
        raise RuntimeError('checkpoint sync worker failed: {}'.format(failure))
    try:
        os.kill(int(sync_worker_pid), 0)
    except OSError as error:
        raise RuntimeError('checkpoint sync worker is not alive') from error
    proc_stat_path = '/proc/{}/stat'.format(int(sync_worker_pid))
    if os.path.isfile(proc_stat_path):
        with open(proc_stat_path) as proc_stat_file:
            fields = proc_stat_file.read().split()
        if len(fields) > 2 and fields[2] == 'Z':
            raise RuntimeError('checkpoint sync worker is a zombie process')


def _wait_for_sync_queue(sync_queue_dir,
                         sync_worker_pid,
                         timeout_seconds,
                         allow_single_stage_best=False):
    os.makedirs(sync_queue_dir, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    while True:
        check_checkpoint_sync_worker(sync_queue_dir, sync_worker_pid)
        pending = [
            name for name in os.listdir(sync_queue_dir)
            if name.endswith('.ready.json') or name.endswith('.processing.json')
        ]
        if not pending:
            return
        if allow_single_stage_best and len(pending) == 1:
            pending_path = os.path.join(sync_queue_dir, pending[0])
            try:
                with open(pending_path) as pending_file:
                    pending_request = json.load(pending_file)
            except FileNotFoundError:
                continue
            if pending_request.get('kind') == 'stage_best':
                return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                'checkpoint sync queue did not drain within {} seconds'.format(
                    timeout_seconds))
        time.sleep(1)


def _write_sync_request(sync_queue_dir, request, label):
    request_name = '{:020d}-{}.ready.json'.format(
        time.time_ns(), label.replace(os.sep, '_'))
    _atomic_json_save(request, os.path.join(sync_queue_dir, request_name))


def _local_object_record(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return {
        'local_path': os.path.abspath(path),
        'name': os.path.basename(path),
        'size': os.path.getsize(path),
        'sha256': _sha256_file(path),
    }


def _atomic_copy(source, destination):
    directory = os.path.dirname(os.path.abspath(destination))
    os.makedirs(directory, exist_ok=True)
    temporary_path = '{}.tmp.{}'.format(destination, os.getpid())
    try:
        shutil.copyfile(source, temporary_path)
        with open(temporary_path, 'rb') as copied_file:
            os.fsync(copied_file.fileno())
        os.replace(temporary_path, destination)
        _fsync_directory(directory)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _read_s3_json_optional(uri):
    result = subprocess.run(
        ['aws', 's3', 'cp', uri, '-', '--only-show-errors'],
        capture_output=True,
        text=True)
    if result.returncode != 0:
        if '404' in result.stderr or 'Not Found' in result.stderr:
            return None
        raise subprocess.CalledProcessError(
            result.returncode, result.args, output=result.stdout, stderr=result.stderr)
    return json.loads(result.stdout)


def _upload_and_verify(local_path,
                       remote_uri,
                       expected_size=None,
                       expected_sha256=None):
    size = os.path.getsize(local_path)
    if expected_size is not None and size != expected_size:
        raise RuntimeError('local upload size changed for {}'.format(local_path))
    digest = expected_sha256 or _sha256_file(local_path)
    subprocess.run([
        'aws', 's3', 'cp', local_path, remote_uri,
        '--only-show-errors', '--sse', 'AES256',
    ], check=True)
    head = _head_s3_object(remote_uri)
    if head.get('ContentLength') != size:
        raise RuntimeError('S3 checkpoint size verification failed for {}'.format(remote_uri))
    if head.get('ServerSideEncryption') != 'AES256':
        raise RuntimeError('S3 checkpoint encryption verification failed for {}'.format(remote_uri))
    remote_digest = _sha256_s3_object(remote_uri)
    if remote_digest != digest:
        raise RuntimeError('S3 checkpoint SHA256 verification failed for {}'.format(remote_uri))
    return {'size': size, 'sha256': digest}


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as input_file:
        while True:
            chunk = input_file.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _head_s3_object(uri):
    bucket_and_key = uri[5:].split('/', 1)
    if len(bucket_and_key) != 2 or not all(bucket_and_key):
        raise ValueError('invalid S3 URI: {}'.format(uri))
    output = subprocess.run([
        'aws', 's3api', 'head-object',
        '--bucket', bucket_and_key[0],
        '--key', bucket_and_key[1],
        '--output', 'json',
    ], check=True, capture_output=True, text=True)
    return json.loads(output.stdout)


def _sha256_s3_object(uri):
    digest = hashlib.sha256()
    process = subprocess.Popen(
        ['aws', 's3', 'cp', uri, '-', '--only-show-errors'],
        stdout=subprocess.PIPE)
    try:
        while True:
            chunk = process.stdout.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        process.stdout.close()
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, process.args)
    return digest.hexdigest()


def load_training_checkpoint(model,
                             optimizer,
                             scheduler,
                             executor,
                             checkpoint_path,
                             train_engine,
                             expected_resume_contract_sha256=None):
    if train_engine != 'torch_ddp':
        raise NotImplementedError('full-state checkpoints currently support torch_ddp only')

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    checkpoint = torch_load(checkpoint_path, map_location='cpu')
    if checkpoint.get('format') != FULL_CHECKPOINT_FORMAT:
        raise ValueError('not a supported full-state checkpoint: {}'.format(checkpoint_path))
    if checkpoint.get('train_engine') != train_engine:
        raise ValueError('checkpoint train engine does not match current run')
    if checkpoint.get('world_size') != world_size:
        raise ValueError(
            'checkpoint world_size {} does not match current world_size {}'.format(
                checkpoint.get('world_size'), world_size))
    if expected_resume_contract_sha256 is not None:
        actual_contract = checkpoint.get('resume_contract_sha256')
        if actual_contract != expected_resume_contract_sha256:
            raise ValueError(
                'checkpoint resume contract {} does not match current contract {}'.format(
                    actual_contract, expected_resume_contract_sha256))

    current_inventory = parameter_inventory(model)
    if checkpoint.get('parameter_inventory') != current_inventory:
        raise ValueError('checkpoint trainable/frozen parameter inventory does not match current model')

    torch.nn.Module.load_state_dict(
        unwrap_model(model), checkpoint['model_state_dict'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    executor.step = checkpoint['executor_state']['step']
    executor.epoch = checkpoint['executor_state']['epoch']
    executor.cv_best_score = checkpoint['executor_state']['cv_best_score']
    executor.accumulation_count = checkpoint['executor_state'].get('accumulation_count', 0)
    if executor.accumulation_count != 0:
        raise ValueError('resume checkpoint was not saved at an optimizer-step boundary')

    sidecar_path = _rank_sidecar_path(checkpoint_path, rank)
    sidecar = torch_load(sidecar_path, map_location='cpu')
    if sidecar.get('format') != FULL_CHECKPOINT_FORMAT:
        raise ValueError('invalid RNG sidecar format: {}'.format(sidecar_path))
    if sidecar.get('rank') != rank or sidecar.get('world_size') != world_size:
        raise ValueError('RNG sidecar rank/world_size does not match current run')

    return {
        'next_epoch': checkpoint['cursor']['next_epoch'],
        'next_batch_idx': checkpoint['cursor']['next_batch_idx'],
        'rng_state': sidecar['rng_state'],
        'parameter_inventory': current_inventory,
    }
