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
* From the workbench, clone this repository, i.e., `https://github.com/opendatahub-io/distributed-workloads.git`:
![](./docs/06.png)
* Navigate to the `distributed-workloads/examples/kfto-sft-llm` directory and open the `sft` notebook

You can now proceed with the instructions from the notebook. Enjoy!

## Validation

This example has been validated with the following configurations:

### Llama 3.3 70B Instruct - GSM8k - LoRA

* Cluster:
  * OpenShift AI 2.17
  * 16x `gx2-80x1280x8a100` nodes on IBM Cloud (NVIDIA-A100-SXM4-80GB GPU)
* Configuration:
    ```yaml
    # Model
    model_name_or_path: meta-llama/Llama-3.3-70B-Instruct
    model_revision: main
    torch_dtype: bfloat16
    attn_implementation: flash_attention_2

    # PEFT / LoRA
    use_peft: true
    lora_target_modules: "all-linear"
    lora_modules_to_save: ["lm_head", "embed_tokens"]
    lora_r: 16
    lora_alpha: 8
    lora_dropout: 0.05

    # Quantization / BitsAndBytes
    load_in_4bit: false
    load_in_8bit: false

    # Datasets
    dataset_name: gsm8k
    dataset_config: main

    # SFT
    max_seq_length: 1024
    packing: false

    # Training
    per_device_train_batch_size: 32
    per_device_eval_batch_size: 32

    bf16: true
    tf32: false

    # FSDP
    fsdp: "full_shard auto_wrap offload"
    fsdp_config:
    activation_checkpointing: true
    ```

### Llama 3.1 8B Instruct - GSM8k - LoRA

* Cluster:
  * OpenShift AI 2.17
  * 8x `gx2-80x1280x8a100` nodes on IBM Cloud (NVIDIA-A100-SXM4-80GB GPU)
* Configuration:
    ```yaml
    # Model
    model_name_or_path: Meta-Llama/Meta-Llama-3.1-8B-Instruct
    model_revision: main
    torch_dtype: bfloat16
    attn_implementation: flash_attention_2

    # PEFT / LoRA
    use_peft: true
    lora_target_modules: "all-linear"
    lora_modules_to_save: ["lm_head", "embed_tokens"]
    lora_r: 16
    lora_alpha: 8
    lora_dropout: 0.05

    # Quantization / BitsAndBytes
    load_in_4bit: false
    load_in_8bit: false

    # Datasets
    dataset_name: gsm8k
    dataset_config: main

    # SFT
    max_seq_length: 1024
    packing: false

    # Training
    per_device_train_batch_size: 32
    per_device_eval_batch_size: 32

    bf16: true
    tf32: false

    # FSDP
    fsdp: "full_shard auto_wrap offload"
    fsdp_config:
    activation_checkpointing: true
    ```
