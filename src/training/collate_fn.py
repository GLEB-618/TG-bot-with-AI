from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from shared import BASE_MODEL


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)