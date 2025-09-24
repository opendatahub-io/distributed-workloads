def training_func(parameters=None):
    """Fine-tune IBM Granite model on synthetic data using TRL SFTTrainer"""
    import random
    import json
    import os
    from datasets import Dataset
    from transformers import AutoTokenizer, set_seed
    from trl import ModelConfig, ScriptArguments, SFTConfig, SFTTrainer, TrlParser, get_peft_config, get_quantization_config, get_kbit_device_map

    print("Starting Granite fine-tuning on synthetic data...")
    
    # Ensure cache directories exist
    print(f"HuggingFace cache directory: {os.environ['HF_HOME']}")
    os.makedirs(os.environ['HF_HOME'], exist_ok=True)
    os.makedirs(os.environ['HF_DATASETS_CACHE'], exist_ok=True)
    os.makedirs('/shared/models', exist_ok=True)
    
    if parameters is None:
        # Parse arguments using TRL parser
        parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
        script_args, training_args, model_args = parser.parse_args_and_config()
    else:        
        # Create ModelConfig from parameters
        model_args = ModelConfig(
            model_name_or_path=parameters['model_name_or_path'],
            model_revision=parameters.get('model_revision', 'main'),
            torch_dtype=parameters.get('torch_dtype', 'bfloat16'),
            attn_implementation=parameters.get('attn_implementation', 'flash_attention_2'),
            use_peft=parameters.get('use_peft', True),
            lora_r=parameters.get('lora_r', 16),
            lora_alpha=parameters.get('lora_alpha', 8),
            lora_dropout=parameters.get('lora_dropout', 0.05),
            lora_target_modules=parameters.get('lora_target_modules', []),
            lora_modules_to_save=parameters.get('lora_modules_to_save', []),
            load_in_4bit=parameters.get('load_in_4bit', False),
            load_in_8bit=parameters.get('load_in_8bit', False),
            trust_remote_code=True,
        )
        
        # Store parameters that don't belong to standard TRL configs
        dataset_batch_size = parameters.get('dataset_batch_size', 1000)
        
        # Create SFTConfig from parameters
        training_args = SFTConfig(
            output_dir=parameters.get('output_dir', '/shared/models/granite-3.1-2b-instruct-synthetic'),
            max_seq_length=parameters.get('max_seq_length', 1024),
            packing=parameters.get('packing', False),
            num_train_epochs=parameters.get('num_train_epochs', 3),
            per_device_train_batch_size=parameters.get('per_device_train_batch_size', 8),
            per_device_eval_batch_size=parameters.get('per_device_eval_batch_size', 8),
            auto_find_batch_size=parameters.get('auto_find_batch_size', False),
            eval_strategy=parameters.get('eval_strategy', 'epoch'),
            bf16=parameters.get('bf16', True),
            tf32=parameters.get('tf32', False),
            learning_rate=parameters.get('learning_rate', 2.0e-4),
            warmup_steps=parameters.get('warmup_steps', 10),
            lr_scheduler_type=parameters.get('lr_scheduler_type', 'inverse_sqrt'),
            optim=parameters.get('optim', 'adamw_torch_fused'),
            max_grad_norm=parameters.get('max_grad_norm', 1.0),
            seed=parameters.get('seed', 42),
            gradient_accumulation_steps=parameters.get('gradient_accumulation_steps', 1),
            gradient_checkpointing=parameters.get('gradient_checkpointing', False),
            gradient_checkpointing_kwargs=parameters.get('gradient_checkpointing_kwargs', {'use_reentrant': False}),
            fsdp=parameters.get('fsdp', 'full_shard auto_wrap'),
            fsdp_config=parameters.get('fsdp_config', {
                'activation_checkpointing': True,
                'cpu_ram_efficient_loading': False,
                'sync_module_states': True,
                'use_orig_params': True,
                'limit_all_gathers': False
            }),
            save_strategy=parameters.get('save_strategy', 'epoch'),
            save_total_limit=parameters.get('save_total_limit', 1),
            resume_from_checkpoint=parameters.get('resume_from_checkpoint', False),
            log_level=parameters.get('log_level', 'warning'),
            logging_strategy=parameters.get('logging_strategy', 'steps'),
            logging_steps=parameters.get('logging_steps', 1),
            report_to=parameters.get('report_to', ['tensorboard']),
        )
        
        # Store dataset-related parameters separately (not part of standard ScriptArguments)
        dataset_name = parameters.get('dataset_name', 'synthetic_gsm8k')
        dataset_config = parameters.get('dataset_config', 'main')
        dataset_train_split = parameters.get('dataset_train_split', 'train')
        dataset_test_split = parameters.get('dataset_test_split', 'test')
        dataset_text_field = parameters.get('dataset_text_field', 'text')
        dataset_kwargs = parameters.get('dataset_kwargs', {
            'add_special_tokens': False,
            'append_concat_token': False
        })
        
        # Create ScriptArguments with required dataset_name parameter
        script_args = ScriptArguments(dataset_name=dataset_name)
    
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    # Ensure output directory exists
    os.makedirs(training_args.output_dir, exist_ok=True)
    print(f"Output directory: {training_args.output_dir}")
    
    # Model and tokenizer configuration with CPU/GPU compatibility
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Adjust configuration based on available hardware
    quantization_config = get_quantization_config(model_args) if device == "cuda" else None
    torch_dtype = getattr(torch, model_args.torch_dtype) if device == "cuda" and hasattr(model_args, 'torch_dtype') else torch.float32
    
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation if device == "cuda" else "eager",
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing or training_args.fsdp_config.get("activation_checkpointing", False) else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    
    # Check if model exists in shared PVC first, otherwise download from HuggingFace Hub
    shared_model_path = f"/shared/models/{model_args.model_name_or_path.replace('/', '_')}"
    
    if os.path.exists(shared_model_path):
        model_path = shared_model_path
        print(f"Loading model from shared PVC: {model_path}")
    else:
        model_path = model_args.model_name_or_path
        print(f"Model not found in shared PVC, will download from HuggingFace Hub: {model_path}")
        print(f"Downloaded model will be cached in: {os.environ['HF_HOME']}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=model_args.trust_remote_code, 
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load synthetic dataset generated by Ray workers
    print("Loading synthetic dataset...")
    dataset_paths = [
        "/shared/synthetic_data/synthetic_dataset.json",
        "/shared/synthetic_data/final_synthetic_dataset.json"
    ]
    
    synthetic_data = None
    for path in dataset_paths:
        try:
            with open(path, "r") as f:
                synthetic_data = json.load(f)
            print(f"Loaded synthetic dataset from: {path}")
            break
        except FileNotFoundError:
            continue
    
    if synthetic_data is None:
        print("Synthetic dataset not found in any expected location:")
        for path in dataset_paths:
            print(f"   - {path}")
        print("Please run Ray preprocessing first.")
        raise FileNotFoundError("No synthetic dataset found")
    
    # Handle both list and dict formats
    if isinstance(synthetic_data, list):
        # Convert flat list to train/test split
        total_samples = len(synthetic_data)
        split_idx = int(total_samples * 0.8)  # 80/20 split
        train_data = synthetic_data[:split_idx]
        test_data = synthetic_data[split_idx:]
        print(f"Converted list format: {len(train_data)} train, {len(test_data)} test samples")
    else:
        # Dict format with train/test keys
        train_data = synthetic_data.get("train", [])
        test_data = synthetic_data.get("test", [])
        print(f"Dict format: {len(train_data)} train, {len(test_data)} test samples")
    
    # Convert to HuggingFace datasets format with chat template
    train_samples = []
    for item in train_data:
        # Format as conversation for instruction tuning
        messages = [
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["answer"]},
        ]
        train_samples.append({"messages": messages})
    
    test_samples = []
    for item in test_data[:100]:  # Limit eval set size
        messages = [
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["answer"]},
        ]
        test_samples.append({"messages": messages})
    
    # Create datasets
    train_dataset = Dataset.from_list(train_samples)
    test_dataset = Dataset.from_list(test_samples) if training_args.eval_strategy != "no" else None
    
    # Apply chat template
    def template_dataset(sample):
        return {"text": tokenizer.apply_chat_template(sample["messages"], tokenize=False, add_generation_prompt=False)}
    
    train_dataset = train_dataset.map(template_dataset, remove_columns=["messages"])
    if test_dataset is not None:
        test_dataset = test_dataset.map(template_dataset, remove_columns=["messages"])
    
    print(f"Dataset prepared: {len(train_dataset)} train samples")
    
    # Log few random samples from training set
    with training_args.main_process_first(desc="Log few samples from the training set"):
        for index in random.sample(range(len(train_dataset)), 2):
            print(f"Sample {index}:")
            print(train_dataset[index]["text"])
            print("-" * 50)
    
    # Initialize SFT trainer
    trainer = SFTTrainer(
        model=model_path,  # Use the resolved model path (shared PVC or HuggingFace Hub)
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=get_peft_config(model_args),
        processing_class=tokenizer,
    )
    
    # Print trainable parameters info
    if trainer.accelerator.is_main_process and hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()
    
    # Start training
    print("Starting training...")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(training_args.output_dir)
    
    with training_args.main_process_first(desc="Training completed"):
        print(f"Training completed successfully!")
        print(f"Model checkpoint saved to: {training_args.output_dir}")
        print(f"Training info:")
        print(f"   - Model: {model_args.model_name_or_path}")
        print(f"   - Epochs: {training_args.num_train_epochs}")
        print(f"   - LoRA rank: {model_args.lora_r if model_args.use_peft else 'N/A'}")
        print(f"   - Synthetic samples: {len(train_dataset)}")


if __name__ == "__main__":
    training_func()
