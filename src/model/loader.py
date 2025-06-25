from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, LocalTokenNotFoundError
from shared import CHECKPOINTS_DIR, get_versioned_dir, USE_BASE_MODEL, BASE_MODEL, model_logger, get_bnb_config, HF_TOKEN
import torch

FINAL_DIR = get_versioned_dir(CHECKPOINTS_DIR)

model_logger.info(f"Model: {BASE_MODEL}")
if not USE_BASE_MODEL:
    model_logger.info(f"Path: {FINAL_DIR}")

def load_model():
    try:
        login(token=HF_TOKEN)
        bnb_config = get_bnb_config()

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            # torch_dtype=torch.float16,
            device_map="auto",
        )

        model = base_model if USE_BASE_MODEL else PeftModel.from_pretrained(base_model, FINAL_DIR)
        model.eval()

        return model, tokenizer
    except GatedRepoError:

        model_logger.error(f"Access to the '{BASE_MODEL}' model is restricted. Check that the request is approved on HF.")
        exit(1)

    except RepositoryNotFoundError:

        model_logger.error(f"Repository '{BASE_MODEL}' not found or name is incorrect.")
        exit(1)

    except LocalTokenNotFoundError:
        model_logger.error("Invalid Hugging Face token entered.")
        exit(1)

    except Exception as e:

        model_logger.error(f"An unexpected error occurred: {e}")
        exit(1)

