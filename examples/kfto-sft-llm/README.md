# LLM Fine-Tuning with Kubeflow Training on OpenShift AI

This example demonstrates how to fine-tune LLMs with the Kubeflow Training operator on OpenShift AI.
It uses HuggingFace SFTTrainer, with PEFT for LoRA and qLoRA, and PyTorch FSDP to distribute the training on multiple GPUs / nodes.

> [!IMPORTANT]
> This example has been tested with the configurations listed in the [validation](#validation) section.
> Its configuration space is highly dimensional, and tightly coupled to runtime / hardware configuration.
> You need to adapt it, and validate it works as expected, with your configuration(s), on your target environment(s).

## Requirements

* An OpenShift cluster with OpenShift AI (RHOAI) 2.17+ installed:
  * The `dashboard`, `trainingoperator` and `workbenches` components enabled
* Sufficient worker nodes for your configuration(s) with NVIDIA GPUs (Ampere-based or newer recommended) or AMD GPUs (AMD Instinct MI300X or newer recommended)
* A dynamic storage provisioner supporting RWX PVC provisioning

## Setup

* Access the OpenShift AI dashboard, for example from the top navigation bar menu:
![](./docs/01.png)
* Log in, then go to _Data Science Projects_ and create a project:
![](./docs/02.png)
* Once the project is created, click on _Create a workbench_:
![](./docs/03.png)
* Then create a workbench with the following settings:
    * Select the `PyTorch` (or the `ROCm-PyTorch`) notebook image:
    ![](./docs/04a.png)
    * Select the `Medium` container size and add an accelerator:
    ![](./docs/04b.png)
        > [!NOTE]
        > Adding an accelerator is only needed to test the fine-tuned model from within the workbench so you can spare an accelerator if needed.
    * Create a storage that'll be shared between the workbench and the fine-tuning runs.
    Make sure it uses a storage class with RWX capability and give it enough size according to the size of the model you want to fine-tune:
        ![](./docs/04c.png)
        > [!NOTE]
        > You can attach an existing shared storage if you already have one instead.
    * Review the storage configuration and click "Create workbench":
    ![](./docs/04d.png)
* From "Workbenches" page, click on _Open_ when the workbench you've just created becomes ready:
![](./docs/05.png)
* From the workbench, clone this repository, i.e., `https://github.com/opendatahub-io/distributed-workloads.git`
![](./docs/06.png)
* Navigate to the `distributed-workloads/examples/kfto-sft-llm` directory and open the `sft` notebook

You can now proceed with the instructions from the notebook. Enjoy!

## Validation

This example has been validated with the following configurations:

### Llama 3.1 8B Instruct - GSM8k Dataset - LoRA - 8x NVIDIA A100/80G

* Infrastructure:
  * OpenShift AI 2.17
  * 8x NVIDIA-A100-SXM4-80GB
* Configuration:
    ```yaml
    # Model
    model_name_or_path: Meta-Llama/Meta-Llama-3.1-8B-Instruct
    model_revision: main
    torch_dtype: bfloat16
    attn_implementation: flash_attention_2

    # PEFT / LoRA
    use_peft: true
    lora_r: 16
    lora_alpha: 8
    lora_dropout: 0.05
    lora_target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # QLoRA (BitsAndBytes)
    load_in_4bit: false
    load_in_8bit: false

    # Dataset
    dataset_name: gsm8k
    dataset_config: main

    # SFT
    max_seq_length: 1024
    packing: false

    # Training
    per_device_train_batch_size: 64
    per_device_eval_batch_size: 64

    bf16: true
    tf32: false

    learning_rate: 1.0e-4
    warmup_steps: 10
    lr_scheduler_type: inverse_sqrt

    optim: adamw_torch_fused
    max_grad_norm: 1.0

    # FSDP
    fsdp: "full_shard auto_wrap offload"
    fsdp_config:
    activation_checkpointing: true
    ```
* Job:
    ```yaml
    num_workers: 8
    num_procs_per_worker: 1
    resources_per_worker:
        "nvidia.com/gpu": 1
        "memory": 96Gi
        "cpu": 4
    base_image: quay.io/modh/training:py311-cuda121-torch241
    env_vars:
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
        "NCCL_DEBUG": "INFO"
    ```
* Metrics:
    ![](./docs/run01.png)

### Llama 3.3 70B Instruct - GSM8k Dataset - LoRA - 16x NVIDIA A100/80G

* Infrastructure:
  * OpenShift AI 2.17
  * 16x NVIDIA-A100-SXM4-80GB
* Configuration:
    ```yaml
    # Model
    model_name_or_path: meta-llama/Llama-3.3-70B-Instruct
    model_revision: main
    torch_dtype: bfloat16
    attn_implementation: flash_attention_2

    # PEFT / LoRA
    use_peft: true
    lora_target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_modules_to_save: []
    lora_r: 16
    lora_alpha: 8
    lora_dropout: 0.05

    # QLoRA (BitsAndBytes)
    load_in_4bit: false
    load_in_8bit: false

    # Dataset
    dataset_name: gsm8k
    dataset_config: main

    # SFT
    max_seq_length: 1024
    packing: false
    use_liger: true

    # Training
    per_device_train_batch_size: 64
    per_device_eval_batch_size: 64

    bf16: true
    tf32: false

    learning_rate: 2.0e-4
    warmup_steps: 10
    lr_scheduler_type: inverse_sqrt

    optim: adamw_torch_fused
    max_grad_norm: 1.0

    # FSDP
    fsdp: "full_shard auto_wrap"
    fsdp_config:
    activation_checkpointing: true
    ```
* Job:
    ```yaml
    num_workers: 16
    num_procs_per_worker: 1
    resources_per_worker:
        "amd.com/gpu": 1
        "memory": 192Gi
        "cpu": 4
    base_image: quay.io/modh/training:py311-cuda121-torch241
    env_vars:
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
        "NCCL_DEBUG": "INFO"
    ```
* Metrics:
    ![](./docs/run02.png)

### Llama 3.1 8B Instruct - GSM8k Dataset - LoRA - 8x AMD Instinct MI300X

* Infrastructure:
  * OpenShift AI 2.17
  * 8x AMD Instinct MI300X
* Configuration:
    ```yaml
    # Model
    model_name_or_path: Meta-Llama/Meta-Llama-3.1-8B-Instruct
    model_revision: main
    torch_dtype: bfloat16
    attn_implementation: flash_attention_2

    # PEFT / LoRA
    use_peft: true
    lora_target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_modules_to_save: []
    lora_r: 16
    lora_alpha: 8
    lora_dropout: 0.05

    # QLoRA (BitsAndBytes)
    load_in_4bit: false
    load_in_8bit: false

    # Dataset
    dataset_name: gsm8k
    dataset_config: main

    # SFT
    max_seq_length: 4096
    packing: false
    use_liger: true

    # Training
    per_device_train_batch_size: 128
    per_device_eval_batch_size: 128

    bf16: true
    tf32: false

    learning_rate: 2.0e-4
    warmup_steps: 10
    lr_scheduler_type: inverse_sqrt

    optim: adamw_torch_fused
    max_grad_norm: 1.0

    # FSDP
    fsdp: "full_shard auto_wrap"
    fsdp_config:
    activation_checkpointing: true
    ```
* Job:
    ```yaml
    num_workers: 8
    num_procs_per_worker: 1
    resources_per_worker:
        "amd.com/gpu": 1
        "memory": 96Gi
        "cpu": 4
    base_image: quay.io/modh/training:py311-rocm62-torch241
    env_vars:
        "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True"
        "NCCL_DEBUG": "INFO"
    ```
* Metrics:
    ![](./docs/run03.png)
    Blue: with Liger kernels

    Orange: without Liger kernels
