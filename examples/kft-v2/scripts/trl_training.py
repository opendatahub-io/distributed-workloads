def trl_train():
    """TRL training script with distributed coordination and checkpointing."""

    import os
    import json
    import time
    import signal
    import torch
    import numpy
    from numpy.core.multiarray import _reconstruct
    import torch.serialization
    import torch.distributed as dist
    from datetime import datetime
    from pathlib import Path
    from typing import Optional
    from datasets import load_dataset, load_from_disk
    from transformers import (
        AutoTokenizer,
        TrainingArguments,
        TrainerState,
        TrainerControl,
        TrainerCallback,
        set_seed,
    )
    from transformers.trainer_utils import get_last_checkpoint
    from trl import (
        ModelConfig,
        ScriptArguments,
        SFTConfig,
        SFTTrainer,
        TrlParser,
        get_peft_config,
    )

    torch.serialization.add_safe_globals([_reconstruct, numpy.ndarray, numpy.dtype, numpy.dtypes.UInt32DType])

    class ProgressionTracker:
        """Tracks and writes training progression."""

        def __init__(
            self,
            total_epochs: int,
            steps_per_epoch: int,
            status_file_path: Optional[str] = None,
            update_interval: int = 30,
        ):
            self.total_epochs = total_epochs
            self.steps_per_epoch = steps_per_epoch
            self.total_steps = total_epochs * steps_per_epoch
            self.status_file_path = status_file_path or os.getenv(
                "TRAINJOB_PROGRESSION_FILE_PATH", "/tmp/training_progression.json"
            )
            self.update_interval = update_interval
            self.start_time = time.time()
            self.last_update_time = 0
            self.current_epoch = 0
            self.current_step = 0
            self.metrics = {}

        def update_step(self, epoch: int, step: int, loss: float = None, learning_rate: float = None, checkpoint_dir: str = None, **kwargs):
            self.current_epoch = epoch
            if 'global_step' in kwargs:
                self.current_step = int(kwargs['global_step'])
            else:
                self.current_step = (epoch - 1) * self.steps_per_epoch + step

            training_metrics = {}
            generic_metrics = {}

            if loss is not None:
                training_metrics["loss"] = str(loss)
            if learning_rate is not None:
                training_metrics["learning_rate"] = str(learning_rate)

            if checkpoint_dir and os.path.exists(checkpoint_dir):
                try:
                    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint-') or f.startswith('epoch-')]
                    if checkpoints:
                        training_metrics["checkpoints_stored"] = len(checkpoints)
                        def get_checkpoint_number(name):
                            try:
                                return int(name.split('-')[1].split('.')[0])
                            except (IndexError, ValueError):
                                return -1
                        latest_checkpoint = os.path.join(checkpoint_dir, max(checkpoints, key=get_checkpoint_number))
                        training_metrics["latest_checkpoint_path"] = latest_checkpoint
                except (OSError, ValueError):
                    pass

            for key, value in kwargs.items():
                str_value = str(value)

                if key in ['accuracy', 'train_accuracy']:
                    training_metrics["accuracy"] = str_value
                else:
                    generic_metrics[key] = str_value

            self.training_metrics = training_metrics
            self.generic_metrics = generic_metrics

            current_time = time.time()
            if current_time - self.last_update_time >= self.update_interval:
                message = f"Training step {self.current_step}/{self.total_steps}"
                self.write_status(message)
                self.last_update_time = current_time

        def update_epoch(self, epoch: int, checkpoint_dir: str = None, **metrics):
            self.current_epoch = epoch

            training_metrics = {}
            generic_metrics = {}

            for key, value in metrics.items():
                str_value = str(value)

                if key in ['loss', 'avg_loss', 'train_loss']:
                    training_metrics["loss"] = str_value
                elif key in ['accuracy', 'train_accuracy']:
                    training_metrics["accuracy"] = str_value
                else:
                    generic_metrics[key] = str_value

            if checkpoint_dir and os.path.exists(checkpoint_dir):
                try:
                    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint-') or f.startswith('epoch-')]
                    if checkpoints:
                        training_metrics["checkpoints_stored"] = len(checkpoints)
                        def get_checkpoint_number(name):
                            try:
                                return int(name.split('-')[1].split('.')[0])
                            except (IndexError, ValueError):
                                return -1
                        latest_checkpoint = os.path.join(checkpoint_dir, max(checkpoints, key=get_checkpoint_number))
                        training_metrics["latest_checkpoint_path"] = latest_checkpoint
                except (OSError, ValueError):
                    pass

            self.training_metrics = training_metrics
            self.generic_metrics = generic_metrics
            message = f"Completed epoch {epoch}/{self.total_epochs}"
            self.write_status(message)

        def write_status(self, message: str = "Training in progress"):
            """Write training status to file."""
            try:
                current_time = time.time()

                status_data = {
                    "message": message,
                    "timestamp": int(current_time),
                    "start_time": int(self.start_time),
                    "current_step": self.current_step,
                    "total_steps": self.total_steps,
                    "current_epoch": self.current_epoch,
                    "total_epochs": self.total_epochs,
                }

                if self.total_steps > 0:
                    percentage = (self.current_step / self.total_steps) * 100
                    status_data["percentage_complete"] = f"{percentage:.2f}"

                    if self.current_step > 0:
                        elapsed_time = current_time - self.start_time
                        time_per_step = elapsed_time / self.current_step
                        remaining_steps = self.total_steps - self.current_step
                        eta_seconds = int(remaining_steps * time_per_step)
                        days, hours, minutes, seconds = eta_seconds // 86400, (eta_seconds % 86400) // 3600, (eta_seconds % 3600) // 60, eta_seconds % 60
                        eta_formatted = ""
                        if days > 0: eta_formatted += f"{days}d"
                        if hours > 0: eta_formatted += f"{hours}h"
                        if minutes > 0: eta_formatted += f"{minutes}m"
                        if seconds > 0 or eta_formatted == "": eta_formatted += f"{seconds}s"
                        status_data["estimated_time_remaining"] = eta_formatted

                if hasattr(self, 'training_metrics') and self.training_metrics:
                    status_data["training_metrics"] = self.training_metrics

                if hasattr(self, 'generic_metrics') and self.generic_metrics:
                    status_data["metrics"] = self.generic_metrics

                temp_file = f"{self.status_file_path}.tmp"
                with open(temp_file, "w") as f:
                    json.dump(status_data, f, indent=2)
                os.rename(temp_file, self.status_file_path)

            except Exception as e:
                print(f"Failed to write progression status: {e}")

    original_torch_load = torch.load
    def patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs: kwargs['weights_only'] = False
        if 'map_location' not in kwargs: kwargs['map_location'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        return original_torch_load(*args, **kwargs)
    torch.load = patched_torch_load

    class DistributedCheckpointCallback(TrainerCallback):
        def __init__(self, output_dir: str, progression_tracker: Optional[ProgressionTracker] = None):
            self.output_dir = output_dir
            self.checkpoint_requested = False
            self.save_triggered = False
            self.checkpoint_stream = None
            self.sigterm_tensor = None
            self.progression_tracker = progression_tracker

            self.checkpoint_enabled = os.environ.get('CHECKPOINT_ENABLED', 'false').lower() == 'true'
            self.checkpoint_uri = os.environ.get('CHECKPOINT_URI', '/workspace/checkpoints')

            self.progress_file = os.environ.get('TRAINING_PROGRESS_FILE', '/workspace/training_progress.json')


        def _log_message(self, message: str):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

        def _write_progress(self, state: TrainerState):
            rank = int(os.environ.get('RANK', '0'))
            if rank != 0:
                return

            if self.progression_tracker:
                try:
                    latest_loss = 0.0
                    latest_lr = 0.0
                    if state.log_history:
                        latest_log = state.log_history[-1]
                        latest_loss = latest_log.get('loss', latest_log.get('train_loss', latest_log.get('training_loss', 0.0)))
                        latest_lr = latest_log.get('learning_rate', latest_log.get('lr', latest_log.get('train_lr', 0.0)))

                    epoch = max(1, int(state.epoch)) if state.epoch is not None else 1
                    step_in_epoch = (state.global_step - 1) % self.progression_tracker.steps_per_epoch if self.progression_tracker.steps_per_epoch > 0 else 0

                    self.progression_tracker.update_step(
                        epoch=epoch,
                        step=step_in_epoch,
                        loss=latest_loss,
                        learning_rate=latest_lr,
                        checkpoint_dir=self.output_dir,
                        global_step=state.global_step,
                        max_steps=state.max_steps,
                        num_train_epochs=state.num_train_epochs
                    )
                except Exception as e:
                    print(f"ProgressionTracker update failed: {e}")
                    self._write_simple_progress(state)
            else:
                self._write_simple_progress(state)

        def _write_simple_progress(self, state: TrainerState):
            try:
                latest_loss = latest_lr = 0.0
                if state.log_history:
                    latest_log = state.log_history[-1]
                    latest_loss = latest_log.get('loss', latest_log.get('train_loss', latest_log.get('training_loss', 0.0)))
                    latest_lr = latest_log.get('learning_rate', latest_log.get('lr', latest_log.get('train_lr', 0.0)))

                progress_data = {
                    "epoch": int(state.epoch) if state.epoch else 1,
                    "totalEpochs": int(state.num_train_epochs) if state.num_train_epochs else 1,
                    "step": state.global_step,
                    "totalSteps": state.max_steps,
                    "loss": f"{latest_loss:.4f}",
                    "learningRate": f"{latest_lr:.6f}",
                    "percentComplete": f"{(state.global_step / state.max_steps * 100):.1f}" if state.max_steps > 0 else "0.0",
                    "lastUpdateTime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }

                temp_file = self.progress_file + '.tmp'
                with open(temp_file, 'w') as f:
                    json.dump(progress_data, f, indent=2)
                os.rename(temp_file, self.progress_file)
                os.chmod(self.progress_file, 0o644)

            except Exception as e:
                pass

        def _init_distributed_signal_tensor(self):
            try:
                if dist.is_initialized():
                    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
                    self.sigterm_tensor = torch.zeros(1, dtype=torch.float32, device=device)
                    self._log_message(f"Initialized distributed SIGTERM tensor on device: {device}")
                else:
                    self._log_message("Distributed training not initialized - using local SIGTERM handling only")
            except Exception as e:
                self._log_message(f"Failed to initialize distributed SIGTERM tensor: {e}. Using local handling only.")

        def _check_distributed_sigterm(self):
            try:
                if dist.is_initialized() and self.sigterm_tensor is not None:
                    dist.all_reduce(self.sigterm_tensor, op=dist.ReduceOp.MAX)
                    return self.sigterm_tensor.item() > 0.5
            except Exception as e:
                self._log_message(f"Distributed SIGTERM check failed: {e}. Using local signal only.")
            return self.checkpoint_requested

        def _sigterm_handler(self, signum, frame):
            rank = os.environ.get("RANK", "-1")
            self._log_message(f"Rank {rank}: SIGTERM received, flagging for distributed checkpoint.")
            self.checkpoint_requested = True
            if self.sigterm_tensor is not None:
                self.sigterm_tensor.fill_(1.0)

        def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            rank = os.environ.get("RANK", "-1")
            os.makedirs(self.output_dir, exist_ok=True)
            self._init_distributed_signal_tensor()

            if torch.cuda.is_available():
                self.checkpoint_stream = torch.cuda.Stream()
                self._log_message(f"Rank {rank}: Created dedicated CUDA stream for checkpointing.")

            signal.signal(signal.SIGTERM, self._sigterm_handler)
            self._log_message(f"Rank {rank}: Distributed SIGTERM handler registered.")

            try:
                if dist.is_initialized():
                    dist.barrier()
                    self._log_message(f"Rank {rank}: Distributed coordination setup synchronized across all ranks")
            except Exception as e:
                self._log_message(f"Rank {rank}: Failed to synchronize distributed setup: {e}")

        def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            if state.global_step % args.logging_steps == 0:
                self._write_progress(state)

            if self.progression_tracker and state.global_step % max(1, args.logging_steps // 2) == 0:
                rank = int(os.environ.get('RANK', '0'))
                if rank == 0:
                    # Extract current metrics
                    latest_loss = 0.0
                    latest_lr = 0.0
                    if state.log_history:
                        latest_log = state.log_history[-1]
                        latest_loss = latest_log.get('loss', latest_log.get('train_loss', latest_log.get('training_loss', 0.0)))
                        latest_lr = latest_log.get('learning_rate', latest_log.get('lr', latest_log.get('train_lr', 0.0)))

                    epoch = max(1, int(state.epoch)) if state.epoch is not None else 1
                    step_in_epoch = (state.global_step - 1) % self.progression_tracker.steps_per_epoch if self.progression_tracker.steps_per_epoch > 0 else 0

                    current_time = time.time()
                    elapsed_time = current_time - self.progression_tracker.start_time

                    batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
                    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
                        batch_size *= int(os.environ.get('WORLD_SIZE', '1'))

                    total_samples_processed = state.global_step * batch_size
                    samples_per_second = total_samples_processed / elapsed_time if elapsed_time > 0 else 0

                    self.progression_tracker.update_step(
                        epoch=epoch,
                        step=step_in_epoch,
                        loss=latest_loss,
                        learning_rate=latest_lr,
                        checkpoint_dir=self.output_dir,
                        global_step=state.global_step,
                        max_steps=state.max_steps,
                        train_samples_per_second=f"{samples_per_second:.2f}",
                        train_runtime=f"{elapsed_time:.1f}",
                        world_size=os.environ.get('WORLD_SIZE', '1'),
                        local_rank=os.environ.get('LOCAL_RANK', '0')
                    )

            if self._check_distributed_sigterm() and not self.save_triggered:
                rank = os.environ.get("RANK", "-1")
                self._log_message(f"Rank {rank}: Distributed SIGTERM detected, initiating checkpoint at step {state.global_step}.")
                self.save_triggered = True
                control.should_save = True
                control.should_training_stop = True

        def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            self._write_progress(state)

            if self.progression_tracker:
                rank = int(os.environ.get('RANK', '0'))
                if rank == 0:
                    self.progression_tracker.current_step = self.progression_tracker.total_steps
                    self.progression_tracker.write_status("Training completed")

            rank = os.environ.get("RANK", "-1")
            if rank == "0" and self.checkpoint_requested:
                self._log_message(f"Rank {rank}: Training ended due to distributed SIGTERM checkpoint request.")

        def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            if self.progression_tracker:
                rank = int(os.environ.get('RANK', '0'))
                if rank == 0:
                    epoch = max(1, int(state.epoch)) if state.epoch is not None else 1
                    latest_loss = 0.0
                    if state.log_history:
                        latest_log = state.log_history[-1]
                        latest_loss = latest_log.get('loss', latest_log.get('train_loss', latest_log.get('training_loss', 0.0)))

                    self.progression_tracker.update_epoch(
                        epoch=epoch,
                        checkpoint_dir=self.output_dir,
                        avg_loss=latest_loss,
                        global_step=state.global_step,
                        max_steps=state.max_steps
                    )

        def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            rank = os.environ.get("RANK", "-1")
            if rank == "0":
                if self.progression_tracker:
                    try:
                        trainer_state_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}", 'trainer_state.json')
                        if os.path.exists(trainer_state_path):
                            with open(trainer_state_path, 'r') as f:
                                trainer_state_data = json.load(f)

                            trainer_state_data['training_start_time'] = self.progression_tracker.start_time

                            with open(trainer_state_path, 'w') as f:
                                json.dump(trainer_state_data, f, indent=2)

                            self._log_message(f"Rank {rank}: Saved training start time to checkpoint.")
                    except Exception as e:
                        self._log_message(f"Rank {rank}: Failed to save training start time: {e}")

                self._log_message(f"Rank {rank}: Checkpoint save completed.")
                if self.checkpoint_requested:
                    self._log_message(f"Rank {rank}: Distributed SIGTERM-triggered checkpoint save finished successfully.")

    def setup_distributed():
        """Initialize distributed training."""
        node_rank = int(os.getenv('PET_NODE_RANK', '0'))
        num_nodes = int(os.getenv('PET_NNODES', '1'))
        nproc_per_node = int(os.getenv('PET_NPROC_PER_NODE', '1'))
        master_addr = os.getenv('PET_MASTER_ADDR', 'localhost')
        master_port = os.getenv('PET_MASTER_PORT', '29500')

        local_rank = int(os.getenv('LOCAL_RANK', '0'))
        world_size = num_nodes * nproc_per_node
        global_rank = node_rank * nproc_per_node + local_rank

        os.environ['RANK'] = str(global_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port

        if world_size > 1:
            try:
                torch.distributed.init_process_group(
                    backend='gloo',
                    rank=global_rank,
                    world_size=world_size
                )
                torch.distributed.barrier()
            except Exception as e:
                print(f"Warning: Failed to initialize distributed training: {e}")

        return local_rank, global_rank, world_size

    def load_dataset_from_initializer():
        """Load dataset from initializer or download."""
        dataset_dir = Path("/workspace/dataset")

        if dataset_dir.exists() and any(dataset_dir.iterdir()):
            try:
                full_dataset = load_from_disk(str(dataset_dir))
                if isinstance(full_dataset, dict):
                    train_dataset = full_dataset.get('train', full_dataset.get('train[:100]'))
                    test_dataset = full_dataset.get('test', full_dataset.get('test[:20]'))
                else:
                    train_size = min(100, len(full_dataset) - 20)
                    train_dataset = full_dataset.select(range(train_size))
                    test_dataset = full_dataset.select(range(train_size, min(train_size + 20, len(full_dataset))))

                return train_dataset, test_dataset
            except Exception as e:
                print(f"Failed to load from initializer: {e}")

        dataset_name = os.getenv('DATASET_NAME', 'tatsu-lab/alpaca')
        train_split = os.getenv('DATASET_TRAIN_SPLIT', 'train[:100]')
        test_split = os.getenv('DATASET_TEST_SPLIT', 'train[100:120]')

        train_dataset = load_dataset(dataset_name, split=train_split)
        test_dataset = load_dataset(dataset_name, split=test_split)

        return train_dataset, test_dataset

    def load_model_from_initializer():
        """Load model and tokenizer."""
        model_dir = Path("/workspace/model")

        if model_dir.exists() and any(model_dir.iterdir()):
            model_path = str(model_dir)
        else:
            model_path = os.getenv('MODEL_NAME', 'gpt2')

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            if tokenizer.chat_template is None:
                tokenizer.chat_template = (
                    "{% for message in messages %}"
                    "{% if message['role'] == 'user' %}"
                    "### Instruction:\n{{ message['content'] }}\n"
                    "{% elif message['role'] == 'assistant' %}"
                    "### Response:\n{{ message['content'] }}{{ eos_token }}\n"
                    "{% endif %}"
                    "{% endfor %}"
                )

            return model_path, tokenizer

        except Exception as e:
            print(f"Error loading model: {e}")
            model_path = 'gpt2'
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return model_path, tokenizer

    def prepare_datasets(train_dataset, test_dataset, tokenizer):
        """Prepare datasets for training."""
        def template_dataset(sample):
            if 'instruction' in sample and 'output' in sample:
                messages = [
                    {"role": "user", "content": sample['instruction']},
                    {"role": "assistant", "content": sample['output']},
                ]
            elif 'question' in sample and 'answer' in sample:
                messages = [
                    {"role": "user", "content": sample['question']},
                    {"role": "assistant", "content": sample['answer']},
                ]
            else:
                content = str(sample.get('text', sample.get('content', 'Sample text')))
                messages = [
                    {"role": "user", "content": "Complete this text:"},
                    {"role": "assistant", "content": content},
                ]

            return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

        train_columns = list(train_dataset.features.keys())
        train_columns.remove('text') if 'text' in train_columns else None

        train_dataset = train_dataset.map(template_dataset, remove_columns=train_columns)

        if test_dataset is not None:
            test_columns = list(test_dataset.features.keys())
            test_columns.remove('text') if 'text' in test_columns else None
            test_dataset = test_dataset.map(template_dataset, remove_columns=test_columns)

        return train_dataset, test_dataset

    def get_training_parameters():
        """Get training parameters."""
        checkpoint_dir = Path(os.getenv('CHECKPOINT_URI', '/workspace/checkpoints'))
        checkpoint_enabled = os.getenv('CHECKPOINT_ENABLED', 'false').lower() == 'true'
        checkpoint_interval = os.getenv('CHECKPOINT_INTERVAL', '30s')
        max_checkpoints = int(os.getenv('CHECKPOINT_MAX_RETAIN', '5'))

        parameters = {
            'model_name_or_path': os.getenv('MODEL_NAME', 'gpt2'),
            'model_revision': 'main',
            'torch_dtype': 'bfloat16',
            'use_peft': True,
            'lora_r': int(os.getenv('LORA_R', '16')),
            'lora_alpha': int(os.getenv('LORA_ALPHA', '32')),
            'lora_dropout': float(os.getenv('LORA_DROPOUT', '0.1')),
            'lora_target_modules': ['c_attn', 'c_proj'],  # GPT-2 specific
            'dataset_name': os.getenv('DATASET_NAME', 'tatsu-lab/alpaca'),
            'dataset_config': 'main',
            'dataset_train_split': os.getenv('DATASET_TRAIN_SPLIT', 'train[:100]'),
            'dataset_test_split': os.getenv('DATASET_TEST_SPLIT', 'train[100:120]'),
            'num_train_epochs': int(os.getenv('MAX_EPOCHS', '3')),
            'per_device_train_batch_size': int(os.getenv('BATCH_SIZE', '2')),
            'per_device_eval_batch_size': int(os.getenv('BATCH_SIZE', '2')),
            'eval_strategy': 'steps',
            'eval_steps': int(os.getenv('EVAL_STEPS', '25')),
            'bf16': torch.cuda.is_available(),  # Only use bf16 if CUDA is available
            'fp16': not torch.cuda.is_available(),  # Use fp16 for CPU training
            'learning_rate': float(os.getenv('LEARNING_RATE', '5e-5')),
            'warmup_steps': int(os.getenv('WARMUP_STEPS', '10')),
            'lr_scheduler_type': 'cosine',
            'optim': 'adamw_torch',
            'max_grad_norm': 1.0,
            'seed': 42,
            'gradient_accumulation_steps': int(os.getenv('GRADIENT_ACCUMULATION_STEPS', '4')),
            'save_strategy': 'steps',
            'save_steps': int(os.getenv('SAVE_STEPS', '20')),
            'save_total_limit': max_checkpoints if checkpoint_enabled else None,
            'logging_strategy': 'steps',
            'logging_steps': int(os.getenv('LOGGING_STEPS', '5')),
            'report_to': [],
            'output_dir': str(checkpoint_dir),
            # Fix DDP parameter marking issue with LoRA
            'gradient_checkpointing': False,  # Disable gradient checkpointing to avoid DDP conflicts
            'ddp_find_unused_parameters': False,  # Optimize DDP performance
            'ddp_backend': 'gloo',  # Use gloo backend for better LoRA compatibility
            'dataloader_pin_memory': False,  # Disable pin memory for CPU training
        }


        return parameters


    """Training function."""

    import os

    local_rank, global_rank, world_size = setup_distributed()

    if world_size > 1:
        try:
            if dist.is_initialized():
                dist.barrier()
        except Exception as e:
            print(f"Warning: Failed to synchronize distributed setup: {e}")

    os.makedirs("/workspace/cache/transformers", exist_ok=True)
    os.makedirs("/workspace/cache", exist_ok=True)
    os.makedirs("/workspace/cache/datasets", exist_ok=True)

    parameters = get_training_parameters()
    checkpoint_dir = Path(parameters['output_dir'])
    os.makedirs(checkpoint_dir, exist_ok=True)


    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_dict(parameters)

    set_seed(training_args.seed)

    model_path, tokenizer = load_model_from_initializer()
    train_dataset, test_dataset = load_dataset_from_initializer()
    train_dataset, test_dataset = prepare_datasets(train_dataset, test_dataset, tokenizer)

    progression_tracker = None

    callbacks = [
        DistributedCheckpointCallback(str(checkpoint_dir), progression_tracker)
    ]

    # Fix DDP parameter marking issue with LoRA
    if world_size > 1:
        # Set static graph for DDP to avoid parameter marking issues
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
        
    trainer = SFTTrainer(
        model=model_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=get_peft_config(model_args),
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    
    # Apply DDP fix after trainer initialization
    if world_size > 1:
        try:
            # Check if model is wrapped in DDP
            if hasattr(trainer.model, 'module'):
                # Set static graph to prevent parameter marking issues
                trainer.model._set_static_graph()
                print(f"Applied DDP static graph fix for distributed training")
            elif hasattr(trainer.accelerator.unwrap_model(trainer.model), 'base_model'):
                # For PEFT models, ensure proper DDP handling
                unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
                if hasattr(unwrapped_model, '_ddp_params_and_buffers_to_ignore'):
                    print(f"PEFT model detected, DDP parameters properly configured")
        except Exception as e:
            print(f"Warning: Could not apply DDP static graph fix: {e}")

    if trainer.accelerator.is_main_process and hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    checkpoint = get_last_checkpoint(training_args.output_dir)
    resume_from_epoch = 0
    resume_from_step = 0

    if checkpoint is not None:
        try:
            checkpoint_files = os.listdir(checkpoint)
            if 'trainer_state.json' not in checkpoint_files:
                checkpoint = None
            else:
                trainer_state_path = os.path.join(checkpoint, 'trainer_state.json')
                if os.path.exists(trainer_state_path):
                    with open(trainer_state_path, 'r') as f:
                        trainer_state = json.load(f)
                        resume_from_epoch = int(trainer_state.get('epoch', 0))
                        resume_from_step = int(trainer_state.get('global_step', 0))
                        print(f"Resuming from checkpoint: epoch {resume_from_epoch}, step {resume_from_step}")
        except Exception as e:
            print(f"Checkpoint validation failed: {e}")
            checkpoint = None

    if world_size == 1 or global_rank == 0:
        train_dataset_size = len(train_dataset) if train_dataset else 1000  # fallback
        batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        if world_size > 1:
            batch_size *= world_size
        steps_per_epoch = max(1, train_dataset_size // batch_size)

        num_epochs = int(training_args.num_train_epochs)
        # Fix: Total epochs should always be the configured number, not additive
        total_epochs_planned = num_epochs

        progression_tracker = ProgressionTracker(
            total_epochs=total_epochs_planned,
            steps_per_epoch=steps_per_epoch,
            update_interval=int(os.getenv('PROGRESSION_UPDATE_INTERVAL', '10'))
        )

        if checkpoint is not None and (resume_from_epoch > 0 or resume_from_step > 0):
            # Set current progress based on what was actually completed
            progression_tracker.current_epoch = resume_from_epoch
            # Use global_step if available, otherwise calculate from epochs
            if resume_from_step > 0:
                progression_tracker.current_step = resume_from_step
            else:
                # Calculate steps from completed epochs
                progression_tracker.current_step = resume_from_epoch * steps_per_epoch

            try:
                trainer_state_path = os.path.join(checkpoint, 'trainer_state.json')
                if os.path.exists(trainer_state_path):
                    with open(trainer_state_path, 'r') as f:
                        trainer_state = json.load(f)
                        if 'training_start_time' in trainer_state:
                            progression_tracker.start_time = trainer_state['training_start_time']
                            print(f"Restored original training start time from checkpoint")
                        else:
                            current_time = time.time()
                            if progression_tracker.current_step > 0 and progression_tracker.total_steps > 0:
                                # Better estimation: assume average 2 seconds per step (more realistic)
                                estimated_elapsed = progression_tracker.current_step * 2.0
                                progression_tracker.start_time = current_time - estimated_elapsed
                                print(f"Estimated original training start time based on completed steps: {progression_tracker.current_step} steps")
            except Exception as e:
                print(f"Could not restore training start time: {e}")

            progression_tracker.write_status(f"Training resumed from epoch {resume_from_epoch}")
        else:
            progression_tracker.current_epoch = 0
            progression_tracker.current_step = 0
            progression_tracker.write_status("Training started")

    if callbacks and len(callbacks) > 0:
        callbacks[0].progression_tracker = progression_tracker

    if world_size > 1:
        try:
            if dist.is_initialized():
                dist.barrier()
        except Exception as e:
            print(f"Warning: Failed to synchronize distributed processes: {e}")

    try:
        trainer.train(resume_from_checkpoint=checkpoint)
    except Exception as e:
        print(f"Training failed: {e}")
        if checkpoint is not None:
            try:
                if progression_tracker:
                    progression_tracker.current_epoch = 0
                    progression_tracker.current_step = 0
                    progression_tracker.write_status("Training restarted from scratch after checkpoint failure")
                trainer.train(resume_from_checkpoint=None)
            except Exception as retry_e:
                print(f"Training failed even from scratch: {retry_e}")
                raise retry_e
        else:
            raise

    trainer.save_model(training_args.output_dir)

    if progression_tracker and (world_size == 1 or global_rank == 0):
        progression_tracker.current_step = progression_tracker.total_steps
        progression_tracker.write_status("Training completed successfully")

        print("Waiting for progression status to be captured...")
        time.sleep(30)

    if world_size > 1:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception as e:
            print(f"Warning: Failed to cleanup distributed process group: {e}")

        