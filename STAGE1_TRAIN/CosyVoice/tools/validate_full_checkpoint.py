#!/usr/bin/env python3

import json
import os
import random
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from cosyvoice.utils.checkpoint import (
    load_training_checkpoint,
    restore_rng_state,
    save_training_checkpoint,
    seed_everything,
)


def main():
    seed_everything(20260723)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 3),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 2),
    )
    model[0].weight.requires_grad = False
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    executor = SimpleNamespace(step=0, epoch=2, cv_best_score=0.75)

    inputs = torch.randn(5, 4)
    loss = model(inputs).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    executor.step = 1
    saved_model = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    with tempfile.TemporaryDirectory() as temporary_directory:
        checkpoint_path = os.path.join(temporary_directory, 'resume_step_00000001.pt')
        save_training_checkpoint(
            model,
            optimizer,
            scheduler,
            executor,
            checkpoint_path,
            next_epoch=2,
            next_batch_idx=17,
            train_engine='torch_ddp',
            resume_contract_sha256='a' * 64)

        expected_random = (
            random.random(),
            float(np.random.random()),
            torch.rand(4),
        )
        for parameter in model.parameters():
            parameter.data.add_(100)
        executor.step = 99
        executor.epoch = 99
        executor.cv_best_score = -1

        resume_state = load_training_checkpoint(
            model,
            optimizer,
            scheduler,
            executor,
            checkpoint_path,
            train_engine='torch_ddp',
            expected_resume_contract_sha256='a' * 64)
        restore_rng_state(resume_state['rng_state'])
        actual_random = (
            random.random(),
            float(np.random.random()),
            torch.rand(4),
        )

        assert resume_state['next_epoch'] == 2
        assert resume_state['next_batch_idx'] == 17
        assert executor.step == 1
        assert executor.epoch == 2
        assert executor.cv_best_score == 0.75
        assert expected_random[0] == actual_random[0]
        assert expected_random[1] == actual_random[1]
        assert torch.equal(expected_random[2], actual_random[2])
        for name, value in model.state_dict().items():
            assert torch.equal(value, saved_model[name]), name

        sidecar_path = os.path.join(
            temporary_directory, 'resume_step_00000001.rank00000.rng.pt')
        assert os.path.isfile(sidecar_path)
        with open(os.path.join(temporary_directory, 'latest_resume.json')) as pointer_file:
            pointer = json.load(pointer_file)
        assert pointer['checkpoint'] == os.path.basename(checkpoint_path)
        assert pointer['rank_sidecars'] == [os.path.basename(sidecar_path)]
        assert pointer['next_epoch'] == 2
        assert pointer['next_batch_idx'] == 17

        model[0].weight.requires_grad = True
        try:
            load_training_checkpoint(
                model,
                optimizer,
                scheduler,
                executor,
                checkpoint_path,
                train_engine='torch_ddp')
        except ValueError as error:
            assert 'parameter inventory' in str(error)
        else:
            raise AssertionError('parameter inventory mismatch was not rejected')

        model[0].weight.requires_grad = False
        try:
            load_training_checkpoint(
                model,
                optimizer,
                scheduler,
                executor,
                checkpoint_path,
                train_engine='torch_ddp',
                expected_resume_contract_sha256='b' * 64)
        except ValueError as error:
            assert 'resume contract' in str(error)
        else:
            raise AssertionError('resume contract mismatch was not rejected')

        executor.accumulation_count = 1
        try:
            save_training_checkpoint(
                model,
                optimizer,
                scheduler,
                executor,
                os.path.join(temporary_directory, 'invalid_partial.pt'),
                next_epoch=3,
                next_batch_idx=0,
                train_engine='torch_ddp')
        except RuntimeError as error:
            assert 'optimizer-step boundary' in str(error)
        else:
            raise AssertionError('partial-gradient checkpoint was not rejected')

    print('full checkpoint validation: PASS')


if __name__ == '__main__':
    main()
