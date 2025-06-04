from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from shared import bot_logger


router = Router()


def setup(r: Router):
    r.message.register(talk, Command('talk'))


async def talk(msg: Message):
    bot_logger.debug(f"Пользователь {msg.from_user.username if msg.from_user else None} вызвал команду /upload")
    prompt = msg.text.removeprefix("/send@abAIv_bot").strip() # type: ignore
    print(prompt)
    if not prompt:
        await msg.reply("Ты не ввёл текст после команды /send")
        return
    think_msg = await msg.reply("🤔 Думаю...")
    # response = await generate_reply(prompt)
    # await think_msg.delete()
    # await msg.reply(response, parse_mode=None)