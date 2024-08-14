from datasets import load_dataset
import json
import os


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


if __name__ == "__main__":
    gsm8k_qa_tokens_template()
