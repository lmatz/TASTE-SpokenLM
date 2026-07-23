#!/usr/bin/env python3

import argparse
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

from cosyvoice.utils.checkpoint import (
    load_training_checkpoint,
    restore_rng_state,
    save_training_checkpoint,
    seed_everything,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    dist.init_process_group('gloo')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    try:
        seed_everything(20260723)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 3),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 2),
        )
        model[0].weight.requires_grad = False
        model = torch.nn.parallel.DistributedDataParallel(model)
        optimizer = torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        executor = SimpleNamespace(step=0, epoch=4, cv_best_score=0.5)

        seed_everything(20260723 + rank)
        inputs = torch.randn(5, 4)
        loss = model(inputs).square().mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        executor.step = 1
        saved_model = {
            name: value.detach().clone() for name, value in model.module.state_dict().items()
        }

        checkpoint_path = os.path.join(args.output_dir, 'resume_step_00000001.pt')
        save_training_checkpoint(
            model,
            optimizer,
            scheduler,
            executor,
            checkpoint_path,
            next_epoch=4,
            next_batch_idx=9,
            train_engine='torch_ddp')
        expected_random = (
            random.random(),
            float(np.random.random()),
            torch.rand(4),
        )
        for parameter in model.parameters():
            parameter.data.add_(100)

        resume_state = load_training_checkpoint(
            model,
            optimizer,
            scheduler,
            executor,
            checkpoint_path,
            train_engine='torch_ddp')
        restore_rng_state(resume_state['rng_state'])
        actual_random = (
            random.random(),
            float(np.random.random()),
            torch.rand(4),
        )

        assert resume_state['next_epoch'] == 4
        assert resume_state['next_batch_idx'] == 9
        assert expected_random[0] == actual_random[0]
        assert expected_random[1] == actual_random[1]
        assert torch.equal(expected_random[2], actual_random[2])
        for name, value in model.module.state_dict().items():
            assert torch.equal(value, saved_model[name]), name

        dist.barrier()
        if rank == 0:
            for sidecar_rank in range(world_size):
                sidecar_path = os.path.join(
                    args.output_dir,
                    'resume_step_00000001.rank{:05d}.rng.pt'.format(sidecar_rank))
                assert os.path.isfile(sidecar_path)
            print('distributed full checkpoint validation: PASS')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
