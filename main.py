"""Транспорт: Telegram polling. Никакой логики, только приём/отправка сообщений."""
import telebot

from config import TG_TOKEN
from db import init_db
from dispatcher import process_message

init_db()

bot = telebot.TeleBot(TG_TOKEN)


@bot.message_handler(func=lambda m: True)
def handle(message):
    # Личные сообщения — принимаем всё
    if message.chat.type == 'private':
        pass
    # Группа — только реплай с хэштегом
    elif message.chat.type in ['group', 'supergroup']:
        if not message.reply_to_message:
            return
        original = message.reply_to_message
        post_text = original.text or original.caption or ""
        if "#жкх_интерактив_v2" not in post_text and "#жкх_интерактив" not in post_text:
            return

    bot.send_chat_action(message.chat.id, 'typing')
    reply = process_message(message.text, message.from_user.id)
    bot.reply_to(message, reply, parse_mode="Markdown")


if __name__ == "__main__":
    print("Диспетчер ЖКХ v3.1 запущен...")
    bot.polling(none_stop=True)
