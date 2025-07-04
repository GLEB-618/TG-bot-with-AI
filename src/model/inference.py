from shared import Timer, model_logger
from ollama import AsyncClient


oll = AsyncClient(host="http://localhost:11434")
model_logger.info("Model loaded")


async def generate_response(messages) -> str:
    model_logger.debug(f"Start of response generation: {messages}")
    with Timer(logger=model_logger, label="Response generation"):
        
        response = await oll.chat(
            model="huihui_ai/qwen3-abliterated:8b-v2", # huihui_ai/qwen3-abliterated:8b-v2
            messages=messages,
            stream=False,
            options={
                "num_predict": 4096
            }
        )

        return response["message"]["content"]