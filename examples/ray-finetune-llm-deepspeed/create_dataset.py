from datasets import load_dataset
import json
import os

cache_dir="../../datasets"
if not os.path.exists(cache_dir):
    cache_dir=""
dataset = load_dataset("gsm8k", "main", cache_dir=cache_dir)

def gsm8k_qa_tokens_template():
    dataset = load_dataset("gsm8k", "main")
    dataset_splits = {"train": dataset["train"], "test": dataset["test"]}

    if not os.path.exists("data"):
        os.mkdir("data")

    with open("data/config.json", "w") as f:
        config = {
            "chat_template": "{{ messages }}",
            "special_tokens": ["<START_Q>", "<END_Q>", "<START_A>", "<END_A>"],
        }
        f.write(json.dumps(config))

    for key, ds in dataset_splits.items():
        with open(f"data/{key}.jsonl", "w") as f:
            for item in ds:
                i = {"messages": (
                    f"<START_Q>{item['question']}<END_Q>"
                    f"<START_A>{item['answer']}<END_A>"
                )}
                f.write(json.dumps(i) + "\n")


def gsm8k_qa_no_tokens_template():
    dataset = load_dataset("gsm8k", "main")
    dataset_splits = {"train": dataset["train"], "test": dataset["test"]}

    if not os.path.exists("data"):
        os.mkdir("data")

    with open("data/config.json", "w") as f:
        config = {
            "chat_template": (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                "{{ message['content'] }}"
                "{% elif message['role'] == 'user' %}"
                "{{ '\n\nQuestion: ' + message['content'] +  eos_token }}"
                "{% elif message['role'] == 'assistant' %}"
                "{{ '\n\nAnswer: '  + message['content'] +  eos_token  }}"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '\n\nAnswer: ' }}"
                "{% endif %}"
            )
        }
        f.write(json.dumps(config))

    for key, ds in dataset_splits.items():
        with open(f"data/{key}.jsonl", "w") as f:
            for item in ds:
                i = {"messages": [
                    {"role": "user", "content": item['question']},
                    {"role": "assistant", "content": item['answer']},
                ]}
                f.write(json.dumps(i) + "\n")


def gsm8k_hf_chat_template():
    dataset = load_dataset("gsm8k", "main")
    dataset_splits = {"train": dataset["train"], "test": dataset["test"]}

    if not os.path.exists("data"):
        os.mkdir("data")

    for key, ds in dataset_splits.items():
        with open(f"data/{key}.jsonl", "w") as f:
            for item in ds:
                i = {"messages": [
                    {"role": "user", "content": item['question']},
                    {"role": "assistant", "content": item['answer']},
                ]}
                f.write(json.dumps(i) + "\n")


if __name__ == "__main__":
    gsm8k_hf_chat_template()
