import random
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
# from trl import setup_chat_format
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer, TrlParser

# Comment in if you want to use the Llama 3 instruct template but make sure to add modules_to_save
# LLAMA_3_CHAT_TEMPLATE="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

# Anthropic/Vicuna like template without the need for special tokens
LLAMA_3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}"
        "{% elif message['role'] == 'user' %}"
            "{{ '\n\nHuman: ' + message['content'] +  eos_token }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '\n\nAssistant: ' }}"
    "{% endif %}"
)


@dataclass
class ScriptArguments:
    # Model
    model_id_or_path: str = field(
        default=None, metadata={"help": "Model ID or path to use for SFT training"}
    )
    torch_dtype: str = field(
        default="bfloat16", metadata={"help": "PyTorch model type"}
    )
    attn_implementation: str = field(
        default="sdpa", metadata={"help": "The attention implementation to use in the model."}
    )
    # Dataset
    dataset_id_or_path: str = field(
        default=None, metadata={"help": "Path to the dataset"},
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset configuration."}
    )
    # BnB
    use_bnb: bool = field(
        default=False, metadata={"help": "Whether to use quantization with BitsAndBytes or not."}
    )
    load_in_4bit: bool = field(
        default=True, metadata={"help": "Enable 4-bit quantization by replacing the linear layers with FP4/NF4 layers."}
    )
    use_peft: bool = field(
        default=False, metadata={"help": "Whether to use PEFT or not."}
    )
    # PEFT / LoRA
    lora_r: int = field(
        default=16, metadata={"help": "Lora attention dimension / rank"}
    )
    lora_alpha: int = field(
        default=16, metadata={"help": "The alpha parameter for Lora scaling"}
    )
    lora_dropout: float = field(
        default=0.5, metadata={"help": "The dropout probability for LoRA layers"}
    )
    lora_target_modules: Optional[Union[list[str], str]] = field(
        default="all-linear", metadata={"help": "The names of the modules to apply the LoRA adapter to"}
    )
    lora_modules_to_save: List[str] = field(
        default_factory=list, metadata={"help": "List of extra modules apart from adapter layers to be trained"}
    )


def training_function(script_args, training_args):
    # Datasets
    train_dataset = load_dataset(script_args.dataset_id_or_path, name=script_args.dataset_config_name, split="train")
    test_dataset = load_dataset(script_args.dataset_id_or_path, name=script_args.dataset_config_name, split="test")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
    
    # Templatize datasets
    def template_dataset(sample):
        # return{"text": tokenizer.apply_chat_template(examples["messages"], tokenize=False)}
        messages = [
            {"role": "user", "content": sample['question']},
            {"role": "assistant", "content": sample['answer']},
        ]
        return{"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    train_dataset = train_dataset.map(template_dataset, remove_columns=["question", "answer"])
    # test_dataset = test_dataset.map(template_dataset, remove_columns=["messages"])
    test_dataset = test_dataset.map(template_dataset, remove_columns=["question", "answer"])

    # Check random samples
    with training_args.main_process_first(
        desc="Log a few random samples from the processed training set"
    ):
        for index in random.sample(range(len(train_dataset)), 2):
            print(train_dataset[index]["text"])

    # Model    
    # torch_dtype = torch.bfloat16
    torch_dtype = script_args.torch_dtype
    quant_storage_dtype = torch.bfloat16

    # BitsAndBytes
    quantization_config = None
    if script_args.use_bnb:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=script_args.load_in_4bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id_or_path,
        quantization_config=quantization_config,
        attn_implementation=script_args.attn_implementation,
        torch_dtype=quant_storage_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # LoRA
    peft_config = None
    if script_args.use_peft:
        peft_config = LoraConfig(
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            r=script_args.lora_r,
            bias="none",
            target_modules=script_args.lora_target_modules,
            task_type="CAUSAL_LM",
            modules_to_save=script_args.lora_modules_to_save,
        )

    # SFT
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        dataset_text_field="text",
        eval_dataset=test_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        },
    )

    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()
    
if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig))
    script_args, training_args = parser.parse_args_and_config()

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    set_seed(training_args.seed)

    training_function(script_args, training_args)
