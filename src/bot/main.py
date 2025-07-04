import asyncio
from aiogram import Bot, Dispatcher, Router, types
from aiogram.utils.chat_action import ChatActionMiddleware
from shared import BOT_TOKEN, bot_logger
from handlers import router_talk


# Всплывающий список команд бота
async def set_commands(bot: Bot):
    commands = [
        types.BotCommand(command="talk", description="/talk <текст> | Нейро АбAIв ответит")
    ]
    await bot.set_my_commands(commands)
    bot_logger.debug("Commands added")


# Запуск бота
async def run_bot():
    try:
        router = Router()
        bot = Bot(token=BOT_TOKEN)
        dp = Dispatcher() 

        dp.include_routers(router, router_talk)
        bot_logger.debug("Routers are connected")

        dp.message.middleware(ChatActionMiddleware())

        await set_commands(bot)
        bot_logger.info("The bot has started working")
        await dp.start_polling(bot)
    except Exception as e:
        bot_logger.error(f"Error in operation: {e}")

if __name__ == "__main__":
    asyncio.run(run_bot())