#!/usr/bin/env python3

"""Verify two-rank PyTorch MPI communication on CPU or CUDA tensors."""

import argparse
import os
import threading

import torch
import torch.distributed as dist


def _tensor(values, device):
    return torch.tensor(values, dtype=torch.float32, device=device)


def _expect(operation, tensor, expected, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    actual = tensor.tolist()
    if actual != expected:
        raise AssertionError(f"{operation}: got {actual}, expected {expected}")


def run_collectives(device_mode, hold_until_stopped=False):
    if device_mode not in ("cpu", "cuda"):
        raise ValueError(f"Unsupported device mode: {device_mode}")

    rank = os.environ.get("OMPI_COMM_WORLD_RANK", "unknown")
    operation = "setup"
    try:
        if not dist.is_mpi_available():
            raise RuntimeError("PyTorch was built without the MPI backend")

        if device_mode == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA mode requires an available CUDA device")
            local_rank_value = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
            if local_rank_value is None:
                raise RuntimeError("OMPI_COMM_WORLD_LOCAL_RANK is required in CUDA mode")
            local_rank = int(local_rank_value)
            if local_rank < 0 or local_rank >= torch.cuda.device_count():
                raise RuntimeError(f"OpenMPI local rank {local_rank} has no CUDA device")
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            _tensor([0], device)
            torch.cuda.synchronize(device)
        else:
            device = torch.device("cpu")

        dist.init_process_group(backend="mpi")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != 2:
            raise RuntimeError(f"Expected exactly two MPI ranks, got {world_size}")

        operation = "send_recv"
        point_to_point = _tensor([7] if rank == 0 else [0], device)
        if rank == 0:
            dist.send(point_to_point, dst=1)
        else:
            dist.recv(point_to_point, src=0)
            _expect(operation, point_to_point, [7.0], device)

        operation = "broadcast"
        broadcast_value = _tensor([7] if rank == 0 else [0], device)
        dist.broadcast(broadcast_value, src=0)
        _expect(operation, broadcast_value, [7.0], device)

        operation = "reduce"
        reduce_value = _tensor([rank + 1], device)
        dist.reduce(reduce_value, dst=0, op=dist.ReduceOp.SUM)
        if rank == 0:
            _expect(operation, reduce_value, [3.0], device)

        operation = "all_reduce"
        all_reduce_value = _tensor([rank + 1], device)
        dist.all_reduce(all_reduce_value, op=dist.ReduceOp.SUM)
        _expect(operation, all_reduce_value, [3.0], device)

        operation = "gather"
        gather_output = [_tensor([0], device) for _ in range(2)] if rank == 0 else None
        dist.gather(_tensor([rank + 1], device), gather_list=gather_output, dst=0)
        if rank == 0:
            for source_rank, value in enumerate(gather_output):
                _expect(operation, value, [float(source_rank + 1)], device)

        operation = "all_gather"
        all_gather_output = [_tensor([0], device) for _ in range(2)]
        dist.all_gather(all_gather_output, _tensor([rank + 1], device))
        for source_rank, value in enumerate(all_gather_output):
            _expect(operation, value, [float(source_rank + 1)], device)

        operation = "scatter"
        scatter_output = _tensor([0], device)
        scatter_input = [_tensor([10], device), _tensor([20], device)] if rank == 0 else None
        dist.scatter(scatter_output, scatter_list=scatter_input, src=0)
        _expect(operation, scatter_output, [float(10 * (rank + 1))], device)

        operation = "all_to_all"
        all_to_all_input = [
            _tensor([rank * 10 + destination], device) for destination in range(2)
        ]
        all_to_all_output = [_tensor([0], device) for _ in range(2)]
        dist.all_to_all(all_to_all_output, all_to_all_input)
        for source_rank, value in enumerate(all_to_all_output):
            _expect(operation, value, [float(source_rank * 10 + rank)], device)

        operation = "barrier"
        dist.barrier()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        operation = "reduce_scatter"
        reduce_scatter_input = (
            [_tensor([1], device), _tensor([2], device)]
            if rank == 0
            else [_tensor([10], device), _tensor([20], device)]
        )
        reduce_scatter_output = _tensor([0], device)
        dist.reduce_scatter(
            reduce_scatter_output, reduce_scatter_input, op=dist.ReduceOp.SUM
        )
        _expect(operation, reduce_scatter_output, [11.0 if rank == 0 else 22.0], device)

        print(
            f"MPI COLLECTIVES PASSED device={device_mode} rank={rank} world_size={world_size}",
            flush=True,
        )
        if hold_until_stopped:
            print(f"MPI COLLECTIVES HOLDING rank={rank}", flush=True)
            threading.Event().wait()
    except Exception as exc:
        print(
            f"MPI COLLECTIVES FAILED device={device_mode} rank={rank} "
            f"operation={operation} error={exc}",
            flush=True,
        )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--hold-until-stopped", action="store_true")
    arguments = parser.parse_args()
    run_collectives(arguments.device, arguments.hold_until_stopped)
