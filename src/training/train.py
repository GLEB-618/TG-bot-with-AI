from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.training_args import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch
import yaml

from training.callbacks import callbacks
from shared import BASE_MODEL, CHECKPOINTS_DIR, train_logger, DATA_DIR, DATASET_FILE, get_next_versioned_dir

with open("training/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Модель и токенизатор
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config, device_map="auto")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Подгрузка датасета
dataset = load_dataset("json", data_files=str(DATA_DIR / DATASET_FILE), split="train")

def format_chat(example):
    messages = example["messages"]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)[0]
    return {"input_ids": input_ids}

tokenized = dataset.map(format_chat)

training_args = TrainingArguments(
    output_dir=str(CHECKPOINTS_DIR),
    per_device_train_batch_size=config["per_device_train_batch_size"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    num_train_epochs=config["num_train_epochs"],
    learning_rate=config["learning_rate"],
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer, # type: ignore
    train_dataset=tokenized,
    args=training_args,
    dataset_text_field="input_ids", # type: ignore
    callbacks=callbacks,
)

model.config.use_cache = False # type: ignore
train_logger.info("🚀 Начало обучения!")
trainer.train()

CHECKPOINTS_DIR = get_next_versioned_dir(CHECKPOINTS_DIR)

# Сохранение
model.save_pretrained(str(CHECKPOINTS_DIR))
tokenizer.save_pretrained(str(CHECKPOINTS_DIR))
