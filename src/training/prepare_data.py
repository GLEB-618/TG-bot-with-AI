from datasets import load_dataset
from transformers import AutoTokenizer
import yaml
from shared import DATA_DIR, BASE_MODEL, DATASET_FILE

with open("training/config.yaml", "r") as f:
    config = yaml.safe_load(f)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
eos = tokenizer.eos_token

def format_pair(example):
    prompt = f"<|user|>\n{example['instruction']}\n<|assistant|>\n{example['response']}{eos}"
    tokens = tokenizer(prompt, truncation=True, padding="max_length", max_length=config["max_length"])
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

def get_dataset():
    raw = load_dataset("json", data_files=str(DATA_DIR / DATASET_FILE), split="train")
    return raw.map(format_pair)
