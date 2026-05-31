import os
from dotenv import load_dotenv

load_dotenv()

TG_TOKEN = os.getenv("TG_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not TG_TOKEN or not DEEPSEEK_API_KEY:
    raise RuntimeError("TG_TOKEN и DEEPSEEK_API_KEY должны быть заданы в .env файле")
