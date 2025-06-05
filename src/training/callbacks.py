from transformers.trainer_callback import TrainerCallback
import torch, math, time
from shared import train_logger

class EvalLoggerCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            eval_loss = metrics["eval_loss"]
            perplexity = math.exp(eval_loss) if eval_loss < 100 else float("inf")
            train_logger.info(f"\nEval loss: {eval_loss:.4f} | Perplexity: {perplexity:.2f}\n")

class TextGenerationCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        tokenizer = kwargs["tokenizer"]

        prompt = [{"role": "user", "content": "Готов к сегодняшней ночи?"}]
        input_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt=True).to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=50,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        train_logger.info(f"Генерация после эпохи {state.epoch:.0f}: {decoded}")

class TrainLoggerCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.epoch_start = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        train_logger.info("🚀 Обучение началось")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()
        if state.epoch is not None:
            train_logger.info(f"📚 Начало эпохи {int(state.epoch) + 1}")

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start:
            epoch_duration = time.time() - self.epoch_start
            if state.epoch is not None:
                train_logger.info(f"✅ Эпоха {int(state.epoch)} завершена за {epoch_duration:.2f} сек")

    def on_save(self, args, state, control, **kwargs):
        ckpt_path = f"{args.output_dir}/checkpoint-{state.global_step}"
        train_logger.info(f"💾 Чекпоинт сохранён: {ckpt_path}")

    def on_train_end(self, args, state, control, **kwargs):
        if self.start_time is not None:
            total_time = time.time() - self.start_time
            train_logger.info(f"🏁 Обучение завершено за {total_time:.2f} сек")

# Подключаем их
callbacks = [EvalLoggerCallback(), TextGenerationCallback()]