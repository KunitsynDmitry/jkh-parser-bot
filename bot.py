import hashlib
import os
import threading
import time
import traceback
from datetime import datetime
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider
import telebot

# ==========================================
# 1. НАСТРОЙКИ КЛЮЧЕЙ
# ==========================================
load_dotenv()
TG_TOKEN = os.getenv("TG_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not TG_TOKEN or not DEEPSEEK_API_KEY:
    raise RuntimeError("TG_TOKEN и DEEPSEEK_API_KEY должны быть заданы в .env файле")

bot = telebot.TeleBot(TG_TOKEN)

# ==========================================
# 2. RATE LIMITER (in-memory)
# ==========================================
_user_last_request: dict[str, float] = {}
RATE_LIMIT_SEC = 3  # минимум 3 секунды между запросами от одного пользователя


def check_rate_limit(user_id: str) -> bool:
    now = time.time()
    last = _user_last_request.get(user_id, 0)
    if now - last < RATE_LIMIT_SEC:
        return False
    _user_last_request[user_id] = now
    return True


# ==========================================
# 3. СХЕМА ДАННЫХ
# ==========================================
class SingleIssue(BaseModel):
    category: str = Field(description="Категория (строго одна из): Водоснабжение, Отопление, Электрика, Лифт, Общедомовое имущество, Двор")
    urgency_level: str = Field(description="Уровень срочности: Низкий, Средний, Высокий, Критический")
    location_details: str = Field(description="ГДЕ ИМЕННО проблема (этаж, подъезд, квартира). НЕ адрес заявителя, а место поломки.")
    dry_summary: str = Field(description="Сухая выжимка проблемы в 1-2 предложениях")

    @field_validator('category')
    @classmethod
    def check_category(cls, v: str) -> str:
        valid = {"Водоснабжение", "Отопление", "Электрика", "Лифт", "Общедомовое имущество", "Двор"}
        if v not in valid:
            raise ModelRetry(f"Категория '{v}' не относится к ЖКХ. Выбери из: {', '.join(sorted(valid))}. Если проблема не в этом списке — не включай её в issues.")
        return v

    @field_validator('location_details')
    @classmethod
    def check_location(cls, v: str) -> str:
        val_lower = v.lower()

        if "уточнен" in val_lower or "неизвестно" in val_lower:
            return "Требуется уточнение"

        # Заявитель описывает СВОЙ адрес, а не адрес проблемы
        applicant_centric = ["я живу", "мой адрес", "моя квартира", "проживаю на", "я нахожусь"]
        if any(term in val_lower for term in applicant_centric):
            raise ModelRetry("Ты указал адрес ЗАЯВИТЕЛЯ, а не адрес ПРОБЛЕМЫ. Вычисли, где именно произошла поломка (сосед снизу = этажом ниже, сверху = этажом выше). Если координаты неизвестны, напиши 'Требуется уточнение'.")

        # Относительные координаты без конкретики
        relative_terms = ["надо мной", "над нами", "тут", "здесь", "где-то", "непонятно где"]
        if any(term in val_lower for term in relative_terms):
            raise ModelRetry("Локация указана слишком размыто. Если нет точных координат, напиши 'Требуется уточнение'.")

        if not any(char.isdigit() for char in v):
            raise ModelRetry("В локации нет цифр (этажа, квартиры, подъезда). Если координаты неизвестны, напиши 'Требуется уточнение'.")

        return v


class ComplaintReport(BaseModel):
    applicant_id: str = Field(description="Хеш ID заявителя")
    issues: List[SingleIssue] = Field(description="Список всех найденных проблем")
    emotional_intensity: int = Field(description="Оценка эмоционального накала текста от 1 до 10")
    threatens_lawsuit: bool = Field(description="Угрожает ли автор судом или проверками? (True/False)")
    non_jkh_issues: List[str] = Field(default_factory=list, description="Проблемы из сообщения, НЕ относящиеся к ЖКХ (шум, поведение жильцов, музыка, драки и т.п.)")


# ==========================================
# 4. НАСТРОЙКА АГЕНТА
# ==========================================
deepseek_model = OpenAIChatModel(
    model_name='deepseek-chat',
    provider=DeepSeekProvider(api_key=DEEPSEEK_API_KEY)
)

agent = Agent(
    model=deepseek_model,
    output_type=ComplaintReport,
    deps_type=str,
    retries=2  # ограничиваем ретраи, чтобы не жечь бюджет
)


@agent.system_prompt
def dynamic_prompt(ctx: RunContext[str]) -> str:
    now = datetime.now()
    day_type = "ВЫХОДНОЙ" if now.weekday() >= 5 else "РАБОЧИЙ"
    return (
        f"Ты - бездушный диспетчер ЖКХ. Сейчас {now.strftime('%H:%M')}, день {day_type}. "
        f"Твоя задача перевести эмоциональную жалобу жильца в сухой структурированный отчет. "
        f"ID заявителя для отчета: {ctx.deps}. "
        "Если в тексте описано несколько разных проблем — создай для каждой отдельный элемент в списке issues. "
        "ВНИМАНИЕ НА ЛОКАЦИЮ: в location_details указывай МЕСТО ПРОБЛЕМЫ, а не адрес заявителя. "
        "Если автор пишет «я живу на 5 этаже, а сосед снизу сверлит потолок» — проблема на 4 этаже (этажом ниже), "
        "а НЕ на 5-м. «Сосед сверху заливает» — проблема этажом ВЫШЕ автора. "
        "«Сосед по стояку», «за стеной», «в соседнем подъезде» — вычисляй точный адрес проблемы относительно автора. "
        "Категории проблем строго: Водоснабжение, Отопление, Электрика, Лифт, Общедомовое имущество, Двор. "
        "Общедомовое имущество — это повреждения стен, перекрытий, потолков, пола, сверление, штробление, резка несущих конструкций, "
        "трещины, протечки кровли, подтопление подвала, разрушение фасада. "
        "КРИТИЧНО: сверление потолка/стен, резка болгаркой, штробы в перекрытиях — это повреждение общедомового имущества, "
        "относится к ЖКХ, а НЕ к полиции. ВСЕГДА выделяй такие проблемы в отдельный issue. "
        "ВАЖНО: если сообщение содержит ТОЛЬКО криминал/поведение жильцов (драки, наркотики, алкоголь, избиения, шум, музыка) "
        "без проблем ЖКХ — верни ПУСТОЙ список issues. "
        "Если в сообщении есть И жкх-проблемы, И не-жкх — жкх помести в issues, а не-жкх перечисли в non_jkh_issues краткими фразами. "
        "Не выдумывай проблемы, если их нет в компетенции ЖКХ."
    )


# ==========================================
# 5. БЕЗОПАСНЫЙ ВЫЗОВ АГЕНТА С ТАЙМАУТОМ
# ==========================================
AGENT_TIMEOUT_SEC = 30


def _run_agent_sync(text: str, deps: str, result_container: list):
    try:
        result_container.append(agent.run_sync(text, deps=deps))
    except Exception as e:
        result_container.append(e)


def call_agent(text: str, deps: str) -> ComplaintReport:
    result_container: list = []
    thread = threading.Thread(target=_run_agent_sync, args=(text, deps, result_container))
    thread.start()
    thread.join(timeout=AGENT_TIMEOUT_SEC)

    if thread.is_alive():
        raise TimeoutError(f"DeepSeek не ответил за {AGENT_TIMEOUT_SEC} сек")

    if not result_container:
        raise RuntimeError("Агент не вернул результат")

    result = result_container[0]
    if isinstance(result, Exception):
        raise result
    return result.output


# ==========================================
# 6. ЛОГИКА ТЕЛЕГРАМ-БОТА
# ==========================================
def hash_user_id(telegram_id: int) -> str:
    return hashlib.sha256(str(telegram_id).encode()).hexdigest()[:12]


@bot.message_handler(func=lambda message: True)
def process_complaint(message):
    # Правило 1: в личных сообщениях — принимаем всё
    if message.chat.type == 'private':
        pass
    # Правило 2: в группе — только реплай с хэштегом
    elif message.chat.type in ['group', 'supergroup']:
        if not message.reply_to_message:
            return
        original_post = message.reply_to_message
        post_text = original_post.text or original_post.caption or ""
        if "#жкх_интерактив_v2" not in post_text and "#жкх_интерактив" not in post_text:
            return

    user_id = str(message.from_user.id)

    # Rate limit
    if not check_rate_limit(user_id):
        bot.reply_to(message, "⏳ Слишком много запросов. Подождите немного.")
        return

    bot.send_chat_action(message.chat.id, 'typing')

    try:
        applicant_hash = hash_user_id(message.from_user.id)
        report = call_agent(message.text, deps=applicant_hash)

        if not report.issues:
            bot.reply_to(message, "🤷 Это не относится к компетенции ЖКХ. Я принимаю проблемы с водоснабжением, отоплением, электрикой, лифтами, общедомовым имуществом и состоянием двора. По вопросам шума, драк, наркотиков и поведения жильцов — обращайтесь в полицию.")
            return

        json_str = report.model_dump_json(indent=2, exclude={"non_jkh_issues"})
        reply = f"📑 **Тикет сформирован:**\n\n```json\n{json_str}\n```"

        if report.non_jkh_issues:
            note = "\n".join(f"• {item}" for item in report.non_jkh_issues)
            reply += f"\n\n⚠️ *Описанные ниже проблемы не относятся к ЖКХ. Для их решения обратитесь в другие службы (полиция, администрация):*\n{note}"

        bot.reply_to(message, reply, parse_mode="Markdown")

    except TimeoutError:
        bot.reply_to(message, "⏳ Сервис временно перегружен. Попробуйте через минуту.")
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] Ошибка парсинга: {e}")
        traceback.print_exc()
        bot.reply_to(message, "❌ Ошибка парсинга. Система не смогла классифицировать запрос.")


# ==========================================
# 7. ЗАПУСК
# ==========================================
if __name__ == "__main__":
    print("Диспетчер ЖКХ v2.1 запущен...")
    bot.polling(none_stop=True)
