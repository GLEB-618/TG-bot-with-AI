from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling
import yaml

with open("training/config.yaml", "r") as f:
    config = yaml.safe_load(f)

tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)