# Fine-Tune Llama Models with Ray and DeepSpeed on OpenShift AI

This example demonstrates how to fine-tune LLMs with Ray on OpenShift AI, using HF Transformers, Accelerate, PEFT (LoRA), and DeepSpeed, for Llama models.
It adapts the _Fine-tuning Llama-2 series models with DeepSpeed, Accelerate, and Ray Train TorchTrainer_[^1] example from the Ray project, so it runs using the Distributed Workloads stack, on OpenShift AI.

> [!IMPORTANT]
> This example has been tested with the configurations listed in the [validation](#validation) section.
> Its configuration space is highly dimensional, with application configuration tightly coupled to runtime / hardware configuration.
> It is your responsibility to adapt it, and validate it works as expected, with your configuration(s), on your target environment(s).

## Requirements

* An OpenShift cluster with OpenShift AI (RHOAI) 2.10+ installed:
  * The `codeflare`, `dashboard`, `ray` and `workbenches` components enabled;
* Sufficient worker nodes for your configuration(s) with NVIDIA GPUs (Ampere-based or newer recommended) or AMD GPUs (AMD Instinct MI300X);
* An AWS S3 bucket to store experimentation results.

## Setup

* Access the OpenShift AI dashboard, for example from the top navigation bar menu:
![](./docs/01.png)
* Log in, then go to _Data Science Projects_ and create a project:
![](./docs/02.png)
* Once the project is created, click on _Create a workbench_:
![](./docs/03.png)
* Then create a workbench with the following settings:
![](./docs/04a.png)
![](./docs/04b.png)
> [!NOTE]
> You can reuse an existing data connection, if you have one already configured for S3.
* And click on _Open_ when it's ready:
![](./docs/05.png)
* From the Notebook server, clone this repository, i.e., `https://github.com/opendatahub-io/distributed-workloads.git`:
![](./docs/06.png)
* Navigate to the `distributed-workloads/examples/ray-finetune-llm-deepspeed` directory and open the `ray-finetune-llm-deepspeed` notebook:
![](./docs/07.png)
* Finally, change the connection parameters for the CodeFlare SDK.
  Mind the Ray cluster is configured to use NVIDIA GPUs by default.
  If you use different accelerators, e.g. AMD GPUs, the Ray cluster configuration must be changed accordingly.

## Experimentation

Once you've reviewed your setup is correct, you can execute the notebook step-by-step.
It creates a Ray cluster, and submits the fine-tuning job to it.

Once you've run the `cluster.details()` command, it prints the Ray cluster dashboard URL, that you can click to access the UI:

![Ray Dashboard](./docs/dashboard.png)

You can also setup [TensorBoard](https://github.com/tensorflow/tensorboard) to visualise your training experiments, as you change hyper-parameters, by running the following commands from a terminal:

* Install TensorBoard in the Ray head node:
    ```console
    kubectl exec `kubectl get pod -l ray.io/node-type=head -o name` -- pip install tensorboard
    ```
* Start TensorBoard server:
    ```console
    kubectl exec `kubectl get pod -l ray.io/node-type=head -o name` -- tensorboard --logdir /tmp/ray --bind_all --port 6006
    ```
* Port-foward the TensorBoard UI endpoint:
    ```console
    kubectl port-forward `kubectl get pod -l ray.io/node-type=head -o name` 6006:6006
    ```

You can then access TensorBoard from your Web browser at http://localhost:6006.

Here is an example of a visualization, that compares experimentations with different combinations of context length and batch size:
![TensorBoard](./docs/tensorboard.png)

> [!IMPORTANT]
> TensorBoard is not part of OpenShift AI.

By default, at the end of each epoch, the checkpoints are stored in the configured S3 bucket, e.g.:

![AWS S3](./docs/s3.png)

## Validation

This example has been validated on the following configurations:

### Llama 3 8B - GSM8k - LoRA

* OpenShift cluster:
  * ROSA-hosted 4.14.20
  * 5 `g5.8xlarge` (NVIDIA A10 GPU) worker nodes
* Ray cluster:
    ```python
    ClusterConfiguration(
        num_workers=4,
        worker_cpu_requests=8,
        worker_cpu_limits=16,
        head_cpu_requests=8,
        head_cpu_limits=8,
        worker_memory_requests=32,
        worker_memory_limits=64,
        head_memory_requests=64,
        head_memory_limits=64,
        head_extended_resource_requests={'nvidia.com/gpu':1},
        worker_extended_resource_requests={'nvidia.com/gpu':1},
    )
    ```
* Ray job:
    ```python
    ray_finetune_llm_deepspeed.py "
        "--model-name=meta-llama/Meta-Llama-3.1-8B "
        "--ds-config=./deepspeed_configs/zero_3_offload_optim_param.json "
        f"--storage-path=s3://{s3_bucket}/ray_finetune_llm_deepspeed/ "
        "--lora "
        "--num-devices=5 "
        "--batch-size-per-device=6 "
        "--eval-batch-size-per-device=8 "
    ```

### Llama 3 8B - GSM8k - LoRA

* OpenShift cluster:
  * OCP 4.14.35, 4 AMD MI300X GPUs / single node
* Ray cluster:
    ```python
    ClusterConfiguration(
        num_workers=3,
        worker_cpu_requests=8,
        worker_cpu_limits=16,
        head_cpu_requests=16,
        head_cpu_limits=16,
        worker_memory_requests=96,
        worker_memory_limits=96,
        head_memory_requests=96,
        head_memory_limits=96,
        head_extended_resource_requests={'amd.com/gpu':1},
        worker_extended_resource_requests={'amd.com/gpu':1},
        image="quay.io/rhoai/ray:2.35.0-py311-rocm61-torch24-fa26",
    )
    ```
* Ray job:
    ```python
    ray_finetune_llm_deepspeed.py "
        "--model-name=meta-llama/Meta-Llama-3.1-8B "
        "--ds-config=./deepspeed_configs/zero_3_offload_optim_param.json "
        f"--storage-path=s3://{s3_bucket}/ray_finetune_llm_deepspeed/ "
        "--lora "
        "--num-devices=4 "
        "--batch-size-per-device=64 "
        "--eval-batch-size-per-device=64 "
        "--ctx-len=1024"
    ```

### Llama 2 7B - GSM8k - LoRA

* OpenShift cluster:
  * ROSA-hosted 4.14.20
  * 6 `g5.8xlarge` (NVIDIA A10 GPU) worker nodes
* Ray cluster:
    ```python
    ClusterConfiguration(
        num_workers=5,
        worker_cpu_requests=8,
        worker_cpu_limits=8,
        head_cpu_requests=16,
        head_cpu_limits=16,
        worker_memory_requests=48,
        worker_memory_limits=48,
        head_memory_requests=48,
        head_memory_limits=48,
        head_extended_resource_requests={'nvidia.com/gpu':1},
        worker_extended_resource_requests={'nvidia.com/gpu':1},
    )
    ```
* Ray job:
    ```python
    ray_finetune_llm_deepspeed.py "
        "--model-name=meta-llama/Llama-2-7b-chat-hf "
        "--ds-config=./deepspeed_configs/zero_3_llama_2_7b.json "
        f"--storage-path=s3://{s3_bucket}/ray_finetune_llm_deepspeed/ "
        "--lora "
        "--num-devices=6 "
        "--batch-size-per-device=16 "
        "--eval-batch-size-per-device=32 "
    ```

### Llama 2 13B - GSM8k - LoRA

* OpenShift cluster:
  * ROSA-hosted 4.14.20
  * 6 `g5.8xlarge` (NVIDIA A10 GPU) worker nodes
* Ray cluster:
    ```python
    ClusterConfiguration(
        num_workers=5,
        worker_cpu_requests=8,
        worker_cpu_limits=8,
        head_cpu_requests=16,
        head_cpu_limits=16,
        worker_memory_requests=48,
        worker_memory_limits=48,
        head_memory_requests=48,
        head_memory_limits=48,
        head_extended_resource_requests={'nvidia.com/gpu':1},
        worker_extended_resource_requests={'nvidia.com/gpu':1},
    )
    ```
* Ray job:
    ```python
    ray_finetune_llm_deepspeed.py "
        "--model-name=meta-llama/Llama-2-13b-chat-hf "
        "--ds-config=./deepspeed_configs/zero_3_llama_2_13b.json "
        f"--storage-path=s3://{s3_bucket}/ray_finetune_llm_deepspeed/ "
        "--lora "
        "--num-devices=6 "
        "--batch-size-per-device=16 "
        "--eval-batch-size-per-device=32 "
    ```

### Llama 2 70B - GSM8k - LoRA

* OpenShift cluster:
  * DGX A100 Server (8x NVIDIA A100 / 40GB HBM)
* Ray cluster:
    ```python
    ClusterConfiguration(
        num_workers=7,
        worker_cpu_requests=16,
        worker_cpu_limits=16,
        head_cpu_requests=16,
        head_cpu_limits=16,
        worker_memory_requests=128,
        worker_memory_limits=128,
        head_memory_requests=128,
        head_memory_limits=128,
        head_extended_resource_requests={'nvidia.com/gpu':1},
        worker_extended_resource_requests={'nvidia.com/gpu':1},
    )
    ```
* Ray job:
    ```python
    ray_finetune_llm_deepspeed.py "
        "--model-name=meta-llama/Llama-2-70b-chat-hf "
        "--ds-config=./deepspeed_configs/zero_3_llama_2_70b.json "
        f"--storage-path=s3://{s3_bucket}/ray_finetune_llm_deepspeed/ "
        "--lora "
        "--num-devices=8 "
        "--batch-size-per-device=8 "
        "--eval-batch-size-per-device=8 "
    ```


[^1]: https://github.com/ray-project/ray/tree/master/doc/source/templates/04_finetuning_llms_with_deepspeed
