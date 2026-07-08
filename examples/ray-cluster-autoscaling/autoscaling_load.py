"""
CPU load script for the Ray cluster autoscaling example.

Submit this script with the Ray Jobs API to queue more single-CPU tasks than the
cluster can run at its current size. That backlog triggers Ray in-tree autoscaling
(scale-up). Tasks sleep long enough to observe workers before the cluster scales
back down when idle.
"""

import os
import time

import ray


def main():
    ray.init(address="auto")

    concurrency = int(os.getenv("AUTOSCALING_TASKS", "3"))
    sleep_s = int(os.getenv("AUTOSCALING_TASK_SLEEP_S", "180"))

    @ray.remote(num_cpus=1)
    def burn_cpu():
        time.sleep(sleep_s)
        return True

    futures = [burn_cpu.remote() for _ in range(concurrency)]
    ray.get(futures)


if __name__ == "__main__":
    main()
