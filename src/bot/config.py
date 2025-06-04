from dotenv import load_dotenv
import os
from shared import bot_logger

load_dotenv()

if os.getenv("TOKEN"):
    TOKEN = os.getenv("TOKEN")
else:
    TOKEN = "0"
    bot_logger.warning("TOKEN не был указан в .env")