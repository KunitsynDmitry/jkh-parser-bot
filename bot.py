import threading
import time
import traceback
from datetime import datetime

import telebot

from config import TG_TOKEN
from graph_agent import process_message

bot = telebot.TeleBot(TG_TOKEN)

# ==========================================
# 1. RATE LIMITER (in-memory)
# ==========================================
_user_last_request: dict[str, float] = {}
RATE_LIMIT_SEC = 3


def check_rate_limit(user_id: str) -> bool:
    now = time.time()
    last = _user_last_request.get(user_id, 0)
    if now - last < RATE_LIMIT_SEC:
        return False
    _user_last_request[user_id] = now
    return True


# ==========================================
# 2. БЕЗОПАСНЫЙ ВЫЗОВ ГРАФА С ТАЙМАУТОМ
# ==========================================
AGENT_TIMEOUT_SEC = 30


def _run_graph(text: str, user_id: int, result_container: list):
    try:
        result_container.append(process_message(text, user_id))
    except Exception as e:
        result_container.append(e)


def call_graph(text: str, user_id: int) -> str:
    result_container: list = []
    thread = threading.Thread(target=_run_graph, args=(text, user_id, result_container))
    thread.start()
    thread.join(timeout=AGENT_TIMEOUT_SEC)

    if thread.is_alive():
        raise TimeoutError(f"Граф не ответил за {AGENT_TIMEOUT_SEC} сек")

    if not result_container:
        raise RuntimeError("Граф не вернул результат")

    result = result_container[0]
    if isinstance(result, Exception):
        raise result
    return result


# ==========================================
# 3. ЛОГИКА ТЕЛЕГРАМ-БОТА
# ==========================================
@bot.message_handler(func=lambda message: True)
def process_complaint(message):
    if message.chat.type == 'private':
        pass
    elif message.chat.type in ['group', 'supergroup']:
        if not message.reply_to_message:
            return
        original_post = message.reply_to_message
        post_text = original_post.text or original_post.caption or ""
        if "#жкх_интерактив_v2" not in post_text and "#жкх_интерактив" not in post_text:
            return

    user_id = str(message.from_user.id)

    if not check_rate_limit(user_id):
        bot.reply_to(message, "⏳ Слишком много запросов. Подождите немного.")
        return

    bot.send_chat_action(message.chat.id, 'typing')

    try:
        reply = call_graph(message.text, message.from_user.id)
        bot.reply_to(message, reply, parse_mode="Markdown")

    except TimeoutError:
        bot.reply_to(message, "⏳ Сервис временно перегружен. Попробуйте через минуту.")
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] Ошибка: {e}")
        traceback.print_exc()
        bot.reply_to(message, "❌ Ошибка парсинга. Система не смогла классифицировать запрос.")


# ==========================================
# 4. ЗАПУСК
# ==========================================
if __name__ == "__main__":
    print("Диспетчер ЖКХ v3.0 (LangGraph) запущен...")
    bot.polling(none_stop=True)
