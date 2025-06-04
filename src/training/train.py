from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch, yaml
from pathlib import Path

from training.prepare_data import get_dataset
from training.collate_fn import collate_fn
from training.callbacks import callbacks

from shared import CHECKPOINTS_DIR, train_logger


with open("training/config.yaml", "r") as f:
    config = yaml.safe_load(f)

bnb_config = dict(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    config["model_name"],
    device_map="auto",
    trust_remote_code=True,
    **bnb_config
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir=str(CHECKPOINTS_DIR),
    per_device_train_batch_size=config["per_device_train_batch_size"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    num_train_epochs=config["num_train_epochs"],
    learning_rate=config["learning_rate"],
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=get_dataset(), # type: ignore
    data_collator=collate_fn,
    callbacks=callbacks
)

model.config.use_cache = False # type: ignore
train_logger.info("Начало обучения!")
trainer.train()

def get_next_versioned_dir(base_dir: Path, prefix: str = "final_model_v") -> Path:
    i = 1
    while (base_dir / f"{prefix}{i}").exists():
        i += 1
    return base_dir / f"{prefix}{i}"

FINAL_DIR = get_next_versioned_dir(CHECKPOINTS_DIR)

model.save_pretrained(str(FINAL_DIR))
tokenizer.save_pretrained(str(FINAL_DIR))
