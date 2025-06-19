from aiogram import Router, F, Bot, flags
from aiogram.filters import Command, CommandObject
from aiogram.types import Message
from aiogram.enums.chat_type import ChatType
import asyncio

from shared import bot_logger
from model import generate_response


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
async def talk_private(msg: Message, bot: Bot):
    prompt = msg.text.strip() # type: ignore
    response = await asyncio.to_thread(generate_response, prompt)
    await msg.reply(response)