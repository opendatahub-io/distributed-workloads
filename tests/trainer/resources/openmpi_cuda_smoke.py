#!/usr/bin/env python3
"""
Lightweight OpenMPI + CUDA smoke workload for Trainer v2 Kueue tests.

The launcher invokes this script through mpirun. Each rank joins the MPI
process group, performs a CUDA all-reduce, waits briefly so the test can
observe both launcher and worker pods running, and then exits cleanly.
"""

import os
import time

import torch
import torch.distributed as dist


def setup():
    if not dist.is_mpi_available():
        raise RuntimeError("PyTorch MPI backend is not available in this image")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the openmpi-cuda smoke test")

    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
    dist.init_process_group(backend="mpi")
    torch.cuda.set_device(local_rank)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, device


def main():
    rank, world_size, device = setup()

    if rank == 0:
        print(f"MPI world_size={world_size}", flush=True)

    tensor = torch.tensor([rank + 1], device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    expected = world_size * (world_size + 1) / 2
    if float(tensor.item()) != float(expected):
        raise RuntimeError(
            f"unexpected allreduce result: got {tensor.item()}, want {expected}"
        )

    print(f"rank {rank}: allreduce_result={tensor.item()}", flush=True)

    dist.barrier()

    hold_seconds = float(os.environ.get("MPI_TEST_HOLD_SECONDS", "15"))
    time.sleep(hold_seconds)

    dist.barrier()

    if rank == 0:
        print("MPI CUDA allreduce succeeded", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
