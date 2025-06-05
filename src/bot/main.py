import asyncio
from aiogram import Bot, Dispatcher, Router, types
from shared import TOKEN, bot_logger
from handlers import setup_all_routers


router = Router()
bot = Bot(token=TOKEN)
dp = Dispatcher() 

setup_all_routers(router)
dp.include_router(router)


# Всплывающий список команд бота
async def set_commands(bot: Bot):
    commands = [
        types.BotCommand(command="send", description="/send <текст> | Нейро АбAIв ответит")
    ]
    await bot.set_my_commands(commands)


# Запуск бота
async def run_bot() -> None:
    try:
        bot_logger.info("Бот начал работу!")
        await set_commands(bot)
        await dp.start_polling(bot)
    except Exception as e:
        bot_logger.error(f"Ошибка в работе бота: {e}")

if __name__ == "__main__":
    asyncio.run(run_bot())