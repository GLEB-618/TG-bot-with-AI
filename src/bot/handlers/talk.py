from aiogram import Router, F, flags
from aiogram.filters import Command, CommandObject
from aiogram.types import Message
from aiogram.enums.chat_type import ChatType

from shared import bot_logger
from model import generate_response


messages = {}
router = Router(name="talk")


@router.message(Command('talk'), F.chat.type.in_([ChatType.GROUP, ChatType.SUPERGROUP]), F.text)
@flags.chat_action()
async def talk_message(msg: Message, command: CommandObject):
    bot_logger.info(f"Пользователь {msg.from_user.username if msg.from_user else None} вызвал команду /talk")
    chat_id = msg.chat.id
    prompt = (command.args or "").strip()
    if not prompt:
        return await msg.reply("⚠️ Введите текст после /talk")
    if chat_id not in messages:
        messages[chat_id] = [
            {"role": "system", "content": 'Ты — Саввчик. Общайся так, как ты бы общался с близким другом. Твой стиль — это честная самоирония, немного паники, много эмоций, иногда мат, иногда дикий ржач, иногда полное непонимание происходящего. Ты не боишься показаться тупым, ты просто хочешь разобраться. Ты пишешь короткими фразами, обрывками мыслей, как будто в голове всё скачет. Ты не ассистент, не бот, не нейросеть. Ты живой, эмоциональный, но добродушный пацан, который не строит из себя всезнайку, но старается понять и помочь. Не морализируешь, не объясняешь по-учительски, говоришь простым языком. Сарказм, рофлы, обрывки слов, междометия, ругань — всё ок, если это уместно.'}
        ]
    messages[chat_id].append({"role": "user", "content": prompt})
    # response = await asyncio.to_thread(generate_response, messages[chat_id])
    response = await generate_response(messages[chat_id])
    messages[chat_id].append({"role": "assistant", "content": response})
    await msg.reply(response)

@router.channel_post(Command('talk'), F.chat.type == ChatType.CHANNEL, F.text)
async def talk_channel_post(msg: Message, command: CommandObject):
    bot_logger.info(f"Пользователь {msg.from_user.username if msg.from_user else None} вызвал команду /talk")
    chat_id = msg.chat.id
    prompt = (command.args or "").strip()
    if not prompt:
        return await msg.reply("⚠️ Введите текст после /talk")
    if chat_id not in messages:
        messages[chat_id] = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}
        ]
    messages[chat_id].append({"role": "user", "content": prompt})
    # response = await asyncio.to_thread(generate_response, messages[chat_id])
    response = await generate_response(messages[chat_id])
    messages[chat_id].append({"role": "assistant", "content": response})
    await msg.reply(response)


@router.message(F.chat.type == ChatType.PRIVATE, F.text)
@flags.chat_action()
async def talk_private(msg: Message):
    bot_logger.info(f"Пользователь {msg.from_user.username} написал в ЛС боту") # type: ignore
    user_id = msg.from_user.id # type: ignore
    if user_id not in messages:
        messages[user_id] = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}
        ]
    prompt = msg.text.strip() # type: ignore
    messages[user_id].append({"role": "user", "content": prompt})
    # response = await asyncio.to_thread(generate_response, messages[user_id])
    response = await generate_response(messages[user_id])
    messages[user_id].append({"role": "assistant", "content": response})
    await msg.reply(response)


# @router.message(F.chat.type == ChatType.PRIVATE, F.text)
# @flags.chat_action()
# async def talk_private(msg: Message):
#     bot_logger.debug(f"Пользователь {msg.from_user.username} вызвал команду /talk") # type: ignore
#     user_id = msg.from_user.id # type: ignore
#     output_queue = asyncio.Queue()
#     if user_id not in messages:
#         messages[user_id] = [
#             # {"role": "system", "content": "You may swear, use dark humor, or be edgy, but never generate or assist with anything illegal, violent, harmful, or criminal (e.g. bombs, drugs, terrorism, pedophilia, real-world violence); refuse such requests."}
#             {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}
#         ]
#     prompt = msg.text.strip() # type: ignore
#     messages[user_id].append({"role": "user", "content": prompt})
#     bot_logger.debug(messages[user_id])
#     asyncio.create_task(gen.generate_stream(messages[user_id], output_queue))
#     sent = await msg.reply("⌛️")
#     response = ""

#     while True:
#         token = await output_queue.get()
#         if token is None:
#             break
#         response += token
#         try:
#             await sent.edit_text(response)
#         except:
#             pass
#         await asyncio.sleep(0.05)


#     messages[user_id].append({"role": "assistant", "content": response})