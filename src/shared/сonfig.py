from transformers.utils.quantization_config import BitsAndBytesConfig
from dotenv import load_dotenv
import os, torch

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "0")
HF_TOKEN = os.getenv("HF_TOKEN", "0")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
BASE_MODEL = os.getenv("BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
USE_BASE_MODEL = os.getenv("USE_BASE_MODEL", "true").lower() == "true"
DATASET_FILE = os.getenv("DATASET_FILE", "file.jsonl")

def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )