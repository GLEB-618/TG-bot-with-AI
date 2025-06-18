import asyncio
from aiogram import Bot, Dispatcher, Router, types
from shared import BOT_TOKEN, bot_logger
from handlers import setup_all_routers


router = Router()
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher() 

setup_all_routers(router)
dp.include_router(router)
bot_logger.debug("Routers are connected")


# Всплывающий список команд бота
async def set_commands(bot: Bot):
    commands = [
        types.BotCommand(command="talk", description="/talk <текст> | Нейро АбAIв ответит")
    ]
    await bot.set_my_commands(commands)
    bot_logger.debug("Commands added")


# Запуск бота
async def run_bot() -> None:
    try:
        bot_logger.info("The bot has started working")
        await set_commands(bot)
        await dp.start_polling(bot)
    except Exception as e:
        bot_logger.error(f"Error in operation: {e}")

if __name__ == "__main__":
    asyncio.run(run_bot())