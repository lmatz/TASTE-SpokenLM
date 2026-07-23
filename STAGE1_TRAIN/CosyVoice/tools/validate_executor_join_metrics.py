#!/usr/bin/env python3

import math
from unittest import mock

import torch

from cosyvoice.utils.executor import Executor


def assert_close(actual, expected, name, tolerance=1e-7):
    if not math.isclose(
            float(actual), float(expected), rel_tol=tolerance,
            abs_tol=tolerance):
        raise AssertionError(
            '{}: expected {}, got {}'.format(name, expected, actual))


def validate_uneven_join_does_not_count_dropped_batch():
    executor = Executor()
    executor.cv = lambda *args, **kwargs: {}
    executor.record_distributed_duration = lambda *args, **kwargs: None
    executor.save_resume_checkpoint = lambda *args, **kwargs: None

    class DummyModel:

        @staticmethod
        def train():
            pass

    info_dict = {
        'accum_grad': 2,
        'train_engine': 'torch_ddp',
    }
    dropped_batch = {
        'audio_feat_len': torch.tensor([100]),
        'utts': ['dropped'],
    }
    with (
            mock.patch(
                'cosyvoice.utils.executor.cosyvoice_join',
                return_value=True),
            mock.patch('cosyvoice.utils.executor.dist.barrier')):
        executor.train_one_epoc(
            DummyModel(),
            optimizer=None,
            scheduler=None,
            train_data_loader=[dropped_batch],
            cv_data_loader=[],
            writer=None,
            info_dict=info_dict,
            group_join=None,
        )
    assert_close(
        executor.accumulated_audio_seconds, 0.0,
        'dropped batch audio seconds')
    assert_close(
        executor.accumulated_dataloader_wait_seconds, 0.0,
        'dropped batch dataloader wait')


def main():
    validate_uneven_join_does_not_count_dropped_batch()
    print('executor uneven-join metrics validation: PASS')


if __name__ == '__main__':
    main()
