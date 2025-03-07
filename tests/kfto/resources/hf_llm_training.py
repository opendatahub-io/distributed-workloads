# Copyright 2023.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Reference: https://github.com/kubeflow/training-operator/blob/master/sdk/python/kubeflow/trainer/hf_llm_training.py


import argparse
import logging
from urllib.parse import urlparse
import json
import os
import time
from datetime import datetime

from datasets import load_dataset, Dataset
from datasets.distributed import split_dataset_by_node
from peft import LoraConfig, get_peft_model
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForImageClassification,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
)
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist


class CustomTensorBoardCallback(TrainerCallback):
    def __init__(self, log_dir=None, exclude_metrics=None):
        self.exclude_metrics = exclude_metrics or []
        self.writer = None
        self.log_dir = log_dir
        self.epoch_start_time = None
        self.total_forward_time = 0
        self.total_backward_time = 0
        self.total_batches = 0
    
    def on_train_begin(self, args, state, control, **kwargs):
        # Initialize TensorBoard writer at the start of training, only for the main process.
        if dist.get_rank() == 0:
            if self.log_dir is None:
                self.log_dir = args.logging_dir
            self.writer = SummaryWriter(log_dir=self.log_dir)

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Aggregate metrics across all ranks and log only from rank 0.
        if logs is None:
            return

        aggregated_logs = {}
        
        for key, value in logs.items():
            if key not in self.exclude_metrics:  # Remove unwanted metrics
                tensor_value = torch.tensor(value, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")

                # Aggregate across all ranks
                dist.all_reduce(tensor_value, op=dist.ReduceOp.SUM)  
                tensor_value /= dist.get_world_size()  

                aggregated_logs[key] = tensor_value.item()  

        if dist.get_rank() == 0:
            for key, value in aggregated_logs.items():
                self.writer.add_scalar(f"kfto-pytorch/{key}", value, state.global_step)

            self.writer.flush() 

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        print(f"Epoch {state.epoch +1} starting...")
        if dist.get_rank() == 0:
            self.writer.add_scalar("kfto-pytorch/epoch", state.epoch +1, state.global_step)
    
    def on_step_begin(self, args, state, control, **kwargs):
        self.start_forward_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        forward_time = time.time() - self.start_forward_time
        self.total_forward_time += forward_time

        start_backward_time = time.time()
        torch.cuda.synchronize() # Wait for GPU operations to finish before timing

        backward_time = time.time() - start_backward_time
        self.total_backward_time += backward_time

        self.total_batches += 1
        avg_forward_time = self.total_forward_time / self.total_batches
        avg_backward_time = self.total_backward_time / self.total_batches

        gpu_memory_peak = torch.cuda.max_memory_allocated() / (1024 ** 2) # convert to mb

        if dist.get_rank() == 0:
            self.writer.add_scalar("kfto-pytorch/forward_time", avg_forward_time, state.global_step)
            self.writer.add_scalar("kfto-pytorch/backward_time", avg_backward_time, state.global_step)
            self.writer.add_scalar("kfto-pytorch/gpu_memory_peak_mb", gpu_memory_peak, state.global_step)

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - self.epoch_start_time

        if dist.get_rank() == 0:
            self.writer.add_scalar("kfto-pytorch/epoch_duration", epoch_duration, state.global_step)

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        for key, value in metrics.items():
             if key not in self.exclude_metrics and dist.get_rank() == 0:
                self.writer.add_scalar(f"kfto-pytorch/{key}", value, state.global_step)
                self.writer.flush()

# Configure logger.
log_formatter = logging.Formatter(
    "%(asctime)s %(levelname)-8s %(message)s", "%Y-%m-%dT%H:%M:%SZ"
)
logger = logging.getLogger(__file__)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)


def setup_model_and_tokenizer(model_uri, transformer_type, model_dir):
    # Set up the model and tokenizer
    parsed_uri = urlparse(model_uri)
    model_name = parsed_uri.netloc + parsed_uri.path

    model = transformer_type.from_pretrained(
        pretrained_model_name_or_path=model_name,
        cache_dir=model_dir,
        local_files_only=True,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name,
        cache_dir=model_dir,
        local_files_only=True,
    )

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
        # If running in a distributed setting, synchronize model parameters across workers.
        if dist.is_initialized():
            dist.broadcast(param.data, src=0)

    return model, tokenizer

# This function is a modified version of the original.
def load_and_preprocess_data(dataset_file, transformer_type, tokenizer):
    # Load and preprocess the dataset
    logger.info("Load and preprocess dataset")

    file_path = os.path.realpath(dataset_file)

    dataset=load_dataset('json',data_files=file_path)

    if transformer_type != AutoModelForImageClassification:
        logger.info(f"Dataset specification: {dataset}")
        logger.info("-" * 40)

        logger.info("Tokenize dataset")
        # TODO (andreyvelich): Discuss how user should set the tokenizer function.
        dataset = dataset.map(
            lambda x: tokenizer(x["output"], padding=True, truncation=True, max_length=128),
            batched=True,
            keep_in_memory=True
        )

    # Check if dataset contains `train` key. Otherwise, load full dataset to train_data.
    if "train" in dataset:
        train_data = dataset["train"]
    else:
        train_data = dataset

    try:
        eval_data = dataset["eval"]
    except Exception:
        eval_data = None
        logger.info("Evaluation dataset is not found")

    if eval_data is None:  # If no eval split exists, create one
        logger.info("Creating eval split from train data")
        # Split the train data into 90% train and 10% eval (validation)
        train_valid_split = train_data.train_test_split(test_size=0.1)
        train_data = train_valid_split["train"]
        eval_data = train_valid_split["test"]
        logger.info(f"Created eval split with {len(eval_data)} examples")

    # Distribute dataset across PyTorchJob workers.
    RANK = int(os.environ["RANK"])
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    logger.info(
        f"Distributed dataset across PyTorchJob workers. WORLD_SIZE: {WORLD_SIZE}, RANK: {RANK}"
    )
    if isinstance(train_data, Dataset):
        train_data = split_dataset_by_node(
            train_data,
            rank=RANK,
            world_size=WORLD_SIZE,
        )
    if isinstance(eval_data, Dataset):
        eval_data = split_dataset_by_node(
            eval_data,
            rank=RANK,
            world_size=WORLD_SIZE,
        )

    return train_data, eval_data


def setup_peft_model(model, lora_config):
    # Set up the PEFT model
    lora_config = LoraConfig(**json.loads(lora_config))
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    return model


def train_model(model, transformer_type, train_data, eval_data, tokenizer, train_args):
    # Allow for each run to be saved in a new directory to allow multiple runs to show on tensorboard
    log_dir = f"/mnt/logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}" 
    # Exclude unwanted default metrics
    exclude_metrics = ['grad_norm', 'total_flos', 'train_runtime', 'train_samples_per_second', 'train_steps_per_second']
    # Setup the Trainer.
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=train_args,
        callbacks=[CustomTensorBoardCallback(log_dir=log_dir, exclude_metrics=exclude_metrics)]
    )

    # TODO (andreyvelich): Currently, data collator is supported only for casual LM Transformer.
    if transformer_type == AutoModelForCausalLM:
        logger.info("Add data collector for language modeling")
        logger.info("-" * 40)
        trainer.data_collator = DataCollatorForLanguageModeling(
            tokenizer,
            pad_to_multiple_of=8,
            mlm=False,
        )

    # Train and save the model.
    trainer.train()

    # Using trainer.evaluate() for default eval metrics 
    if eval_data is not None:
        eval_results = trainer.evaluate(eval_dataset=eval_data)

    trainer.save_model()
    logger.info("parallel_mode: '{0}'".format(trainer.args.parallel_mode))
    logger.info("is_model_parallel: '{0}'".format(trainer.is_model_parallel))
    logger.info("model_wrapped: '{0}'".format(trainer.model_wrapped))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Script for training a model with PEFT configuration."
    )

    parser.add_argument("--model_uri", help="model uri")
    parser.add_argument("--transformer_type", help="model transformer type")
    parser.add_argument("--model_dir", help="directory containing model")
    parser.add_argument("--dataset_file", help="dataset file path")
    parser.add_argument("--lora_config", help="lora_config")
    parser.add_argument(
        "--training_parameters", help="hugging face training parameters"
    )
    parser.add_argument("--log_dir", type=str, default="/mnt/logs", help="TensorBoard log directory")

    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Starting HuggingFace LLM Trainer")
    args = parse_arguments()
    train_args = TrainingArguments(**json.loads(args.training_parameters))
    transformer_type = getattr(transformers, args.transformer_type)

    logger.info("Setup model and tokenizer")
    model, tokenizer = setup_model_and_tokenizer(
        args.model_uri, transformer_type, args.model_dir
    )

    logger.info("Preprocess dataset")
    train_data, eval_data = load_and_preprocess_data(
        args.dataset_file, transformer_type, tokenizer
    )

    logger.info("Setup LoRA config for model")
    model = setup_peft_model(model, args.lora_config)

    logger.info("Start model training")
    train_model(model, transformer_type, train_data, eval_data, tokenizer, train_args)

    logger.info("Training is complete")
