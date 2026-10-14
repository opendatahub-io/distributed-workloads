#!/usr/bin/env python3

"""Train a two-qubit H2 variational circuit across two MPI ranks."""

import argparse
import os

import torch
import torch.distributed as dist


H2_HAMILTONIAN = (
    ("II", -1.052373245772859),
    ("IZ", 0.39793742484318045),
    ("ZI", -0.39793742484318045),
    ("ZZ", -0.01128010425623538),
    ("XX", 0.18093119978423156),
)
TRAINING_STEPS = 25


def _device(device_mode):
    if device_mode == "cpu":
        return torch.device("cpu")
    if device_mode != "cuda":
        raise ValueError(f"Unsupported device mode: {device_mode}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA mode requires an available CUDA device")

    local_rank_value = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
    if local_rank_value is None:
        raise RuntimeError("OMPI_COMM_WORLD_LOCAL_RANK is required in CUDA mode")
    local_rank = int(local_rank_value)
    if local_rank < 0 or local_rank >= torch.cuda.device_count():
        raise RuntimeError(f"OpenMPI local rank {local_rank} has no CUDA device")
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def _local_hamiltonian(rank, device):
    identity = torch.eye(2, dtype=torch.complex64, device=device)
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
    pauli = {"I": identity, "X": x, "Z": z}
    return [
        (coefficient, torch.kron(pauli[operator[0]], pauli[operator[1]]))
        for index, (operator, coefficient) in enumerate(H2_HAMILTONIAN)
        if index % 2 == rank
    ]


def _state(theta):
    # Starting from |01>, a Y rotation and CNOT produce this two-qubit state.
    zero = torch.zeros_like(theta)
    return torch.stack((zero, torch.cos(theta / 2), torch.sin(theta / 2), zero)).to(
        torch.complex64
    )


def _energy(theta, hamiltonian):
    state = _state(theta)
    return sum(
        coefficient * torch.vdot(state, matrix @ state).real
        for coefficient, matrix in hamiltonian
    )


def run_vqe(device_mode):
    rank = os.environ.get("OMPI_COMM_WORLD_RANK", "unknown")
    operation = "setup"
    try:
        if not dist.is_mpi_available():
            raise RuntimeError("PyTorch was built without the MPI backend")

        device = _device(device_mode)
        dist.init_process_group(backend="mpi")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != 2:
            raise RuntimeError(f"Expected exactly two MPI ranks, got {world_size}")

        hamiltonian = _local_hamiltonian(rank, device)
        theta = torch.nn.Parameter(torch.tensor(0.2, device=device))
        optimizer = torch.optim.SGD((theta,), lr=0.1)
        initial_energy = None

        for step in range(TRAINING_STEPS + 1):
            operation = f"energy_step_{step}"
            optimizer.zero_grad()
            local_energy = _energy(theta, hamiltonian)
            global_energy = local_energy.detach().clone()
            dist.all_reduce(global_energy, op=dist.ReduceOp.SUM)
            if step == 0:
                initial_energy = global_energy.item()
            if step % 5 == 0:
                print(
                    f"MPI VQE PROGRESS device={device_mode} rank={rank} "
                    f"step={step} energy={global_energy.item():.6f}",
                    flush=True,
                )
            if step == TRAINING_STEPS:
                final_energy = global_energy.item()
                break

            local_energy.backward()
            operation = f"gradient_step_{step}"
            if step == 0 and theta.grad.abs().item() < 1e-4:
                raise AssertionError(f"Rank {rank} produced no local VQE gradient")
            # Each rank owns different Hamiltonian terms, so sum their gradients.
            dist.all_reduce(theta.grad, op=dist.ReduceOp.SUM)
            optimizer.step()

        if not final_energy < initial_energy - 0.01 or final_energy >= -1.85:
            raise AssertionError(
                f"VQE energy did not converge: initial={initial_energy:.6f}, "
                f"final={final_energy:.6f}"
            )

        print(
            f"MPI VQE PASSED device={device_mode} rank={rank} world_size={world_size} "
            f"steps={TRAINING_STEPS} initial_energy={initial_energy:.6f} "
            f"final_energy={final_energy:.6f} final_theta={theta.item():.6f}",
            flush=True,
        )
    except Exception as exc:
        print(
            f"MPI VQE FAILED device={device_mode} rank={rank} "
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
    arguments = parser.parse_args()
    run_vqe(arguments.device)
