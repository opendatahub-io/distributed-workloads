#!/usr/bin/env python3
"""
Lightweight OpenMPI + CUDA smoke workload for Trainer v2 Kueue tests.

The launcher invokes this script through mpirun. OpenMPI is the process
launcher; GPU collectives use NCCL (midstream OpenMPI is not CUDA-aware).

The MPI hostfile is only mounted on the launcher pod. Workers resolve the
rendezvous address from OMPI_MCA_orte_hnp_uri (tcp://launcher-ip:...).
"""

import os
import re
import time

import torch
import torch.distributed as dist


def _master_addr_from_ompi():
    # Prefer the HNP TCP address so launcher and workers agree on the same IP.
    hnp = os.environ.get("OMPI_MCA_orte_hnp_uri", "")
    match = re.search(r"tcp://([^/\s\]]+)", hnp)
    if match:
        hostport = match.group(1)
        if hostport.startswith("["):
            return hostport[1 : hostport.index("]")]
        return hostport.rsplit(":", 1)[0]

    hostfile = os.environ.get("OMPI_MCA_orte_default_hostfile", "/etc/mpi/hostfile")
    if os.path.isfile(hostfile):
        with open(hostfile, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                return line.split()[0]

    raise RuntimeError(
        "unable to resolve MASTER_ADDR: OMPI_MCA_orte_hnp_uri and hostfile unavailable"
    )


def _export_ompi_env_for_torch():
    if "RANK" not in os.environ and "OMPI_COMM_WORLD_RANK" in os.environ:
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    if "WORLD_SIZE" not in os.environ and "OMPI_COMM_WORLD_SIZE" in os.environ:
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
    if "LOCAL_RANK" not in os.environ and "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
    os.environ["MASTER_ADDR"] = _master_addr_from_ompi()
    os.environ.setdefault("MASTER_PORT", "29500")


def setup():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the openmpi-cuda smoke test")

    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", os.environ.get("LOCAL_RANK", "0")))

    # Universal midstream OpenMPI is a process launcher only (no CUDA MCA).
    # Prefer NCCL for GPU collectives even when torch reports MPI as available.
    _export_ompi_env_for_torch()
    print(
        f"rank={os.environ.get('RANK')} NCCL rendezvous "
        f"MASTER_ADDR={os.environ['MASTER_ADDR']} MASTER_PORT={os.environ['MASTER_PORT']}",
        flush=True,
    )
    dist.init_process_group(backend="nccl", init_method="env://")
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
