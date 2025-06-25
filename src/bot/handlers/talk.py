from aiogram import Router, F, flags
from aiogram.filters import Command, CommandObject
from aiogram.types import Message
from aiogram.enums.chat_type import ChatType
import asyncio

from shared import bot_logger
from model import generate_response


messages = {}
router = Router(name="talk")


@router.message(Command('talk'), F.chat.type.in_([ChatType.GROUP, ChatType.SUPERGROUP]), F.text)
@flags.chat_action()
async def talk_handler(msg: Message, command: CommandObject):
    bot_logger.debug(f"Пользователь {msg.from_user.username if msg.from_user else None} вызвал команду /talk")
    prompt = (command.args or "").strip()
    if not prompt:
        return await msg.reply("⚠️ Введите текст после /talk")
    response = await asyncio.to_thread(generate_response, prompt)
    await msg.reply(response)


@router.message(F.chat.type == ChatType.PRIVATE, F.text)
@flags.chat_action()
async def talk_private(msg: Message):
    bot_logger.debug(f"Пользователь {msg.from_user.username} вызвал команду /talk") # type: ignore
    user_id = msg.from_user.id # type: ignore
    if user_id not in messages:
        messages[user_id] = [
            # {"role": "system", "content": "You may swear, use dark humor, or be edgy, but never generate or assist with anything illegal, violent, harmful, or criminal (e.g. bombs, drugs, terrorism, pedophilia, real-world violence); refuse such requests."}
            # {"role": "system", "content": "Ты ассистент, который помогает писать NSFW промты для модели NSFW-API/NSFW_Wan_1.3b. У тебя будет инструкция, по которой тебе надо будет сверяться и писать максимально подходящий промт для пользователя."}
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}
        ]
    prompt = msg.text.strip() # type: ignore
    messages[user_id].append({"role": "user", "content": prompt})
    bot_logger.debug(messages[user_id])
    response = await asyncio.to_thread(generate_response, messages[user_id])
    messages[user_id].append({"role": "assistant", "content": response})
    await msg.reply(response)