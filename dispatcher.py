"""
Граф обработки жалобы ЖКХ.

Узлы (Nodes):
  1. RateLimiter   — защита от спама
  2. Extractor     — вызов Pydantic-агента (DeepSeek)
  3. QualityGate   — проверка результата, роутинг ответа

Поток:
  user_input → RateLimiter → Extractor → QualityGate → END
"""
import hashlib
import re
import time
from datetime import datetime
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

from config import DEEPSEEK_API_KEY


# ═══════════════════════════════════════════
# 1. СХЕМА ДАННЫХ
# ═══════════════════════════════════════════
class SingleIssue(BaseModel):
    category: str = Field(description="Категория (строго одна из): Водоснабжение, Отопление, Электрика, Лифт, Общедомовое имущество, Двор")
    urgency_level: str = Field(description="Уровень срочности: Низкий, Средний, Высокий, Критический")
    problem_scope: str = Field(description="Масштаб: Квартира (проблема внутри жилого помещения — залив, проводка, батарея), Подъезд/Этаж (подъезд, лестница, общий коридор, холл), Дом (весь дом, крыша, подвал, фасад, стояк), Двор (улица, придомовая территория)")
    location_details: str = Field(description="ГДЕ ИМЕННО проблема (этаж, подъезд, квартира). НЕ адрес заявителя, а место поломки.")
    dry_summary: str = Field(description="Сухая выжимка проблемы в 1-2 предложениях")

    @field_validator('category')
    @classmethod
    def check_category(cls, v: str) -> str:
        valid = {"Водоснабжение", "Отопление", "Электрика", "Лифт", "Общедомовое имущество", "Двор"}
        if v not in valid:
            raise ModelRetry(f"Категория '{v}' не относится к ЖКХ. Выбери из: {', '.join(sorted(valid))}. Если проблема не в этом списке — не включай её в issues.")
        return v

    @field_validator('problem_scope')
    @classmethod
    def check_scope(cls, v: str) -> str:
        valid = {"Квартира", "Подъезд/Этаж", "Дом", "Двор"}
        if v not in valid:
            raise ModelRetry(f"problem_scope '{v}' некорректен. Выбери из: {', '.join(sorted(valid))}.")
        return v

    @field_validator('location_details')
    @classmethod
    def check_location(cls, v: str) -> str:
        val_lower = v.lower()

        if val_lower.strip().startswith("требуется уточнение") or val_lower.strip().startswith("неизвестно"):
            return "Требуется уточнение"

        applicant_centric = ["я живу", "мой адрес", "моя квартира", "проживаю на", "я нахожусь"]
        if any(term in val_lower for term in applicant_centric):
            raise ModelRetry("Ты указал адрес ЗАЯВИТЕЛЯ, а не адрес ПРОБЛЕМЫ. Вычисли, где именно произошла поломка (сосед снизу = этажом ниже, сверху = этажом выше). Если координаты неизвестны, напиши 'Требуется уточнение'.")

        relative_terms = ["надо мной", "над нами", "тут", "здесь", "там", "у нас", "где-то", "непонятно где"]
        if any(term in val_lower for term in relative_terms):
            raise ModelRetry("Локация указана слишком размыто. Если нет точных координат, напиши 'Требуется уточнение'.")

        # Цифры больше не требуем здесь — enforce_location_by_scope проверяет
        # детализацию по problem_scope и молча корректирует при необходимости.
        # ModelRetry за «нет цифр» давил на модель, провоцируя галлюцинации номеров.

        return v

    @model_validator(mode='after')
    def enforce_location_by_scope(self):
        """Квартирные проблемы требуют номера квартиры. Проблемы в подъезде — подъезда/этажа.
        Не поднимаем ModelRetry — если данных нет в тексте, модель не сможет их выдумать.
        Вместо этого молча корректируем location_details на 'Требуется уточнение'."""
        if "Требуется уточнение" in self.location_details:
            return self

        if self.problem_scope == "Квартира":
            has_apt = bool(re.search(r'кв(?:\.|артир[а-яё])?\s*\d+', self.location_details.lower()))
            if not has_apt:
                self.location_details = 'Требуется уточнение'

        if self.problem_scope == "Подъезд/Этаж":
            has_entrance = bool(re.search(r'подъезд\s*\d+', self.location_details.lower()))
            has_floor = bool(re.search(r'этаж\s*\d+', self.location_details.lower()))
            if not (has_entrance or has_floor):
                self.location_details = 'Требуется уточнение'

        return self


class ComplaintReport(BaseModel):
    applicant_id: str = Field(description="Хеш ID заявителя")
    issues: list[SingleIssue] = Field(description="Список всех найденных проблем")
    emotional_intensity: int = Field(description="Оценка эмоционального накала текста от 1 до 10")
    threatens_lawsuit: bool = Field(description="Угрожает ли автор судом или проверками? (True/False)")
    non_jkh_issues: list[str] = Field(default_factory=list, description="Проблемы из сообщения, НЕ относящиеся к ЖКХ")


# ═══════════════════════════════════════════
# 2. PYDANTIC-АГЕНТ
# ═══════════════════════════════════════════
deepseek_model = OpenAIChatModel(
    model_name='deepseek-chat',
    provider=DeepSeekProvider(api_key=DEEPSEEK_API_KEY)
)

agent = Agent(
    model=deepseek_model,
    output_type=ComplaintReport,
    deps_type=str,
    retries=2
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
        "Если автор пишет «я живу на 5 этаже, а сосед снизу сверлит потолок» — проблема на 4 этаже (этажом ниже). "
        "«Сосед сверху заливает» — проблема этажом ВЫШЕ автора. "
        "Категории проблем строго: Водоснабжение, Отопление, Электрика, Лифт, Общедомовое имущество, Двор. "
        "Общедомовое имущество — повреждения стен, перекрытий, потолков, пола, сверление, штробление, резка несущих конструкций. "
        "КРИТИЧНО: сверление потолка/стен, резка болгаркой — это повреждение общедомового имущества, относится к ЖКХ. "
        "problem_scope определяй так: Квартира — проблема внутри жилого помещения (залив из квартиры, проводка, батарея, сантехника); "
        "Подъезд/Этаж — проблема в подъезде, на лестнице, в общем коридоре, в лифтовом холле; "
        "Дом — крыша, подвал, фасад, стояк, общедомовые коммуникации; Двор — улица, придомовая территория, парковка. "
        "ЗАПРЕЩЕНО ВЫДУМЫВАТЬ номера квартир, этажей, подъездов. Ты можешь использовать в location_details ТОЛЬКО цифры, "
        "которые ЯВНО названы в тексте жалобы. Гадать, додумывать, предполагать номера — НЕЛЬЗЯ. "
        "Если проблема в Квартире, а номер квартиры не указан в тексте — пиши 'Требуется уточнение'. "
        "Если проблема в Подъезде/На Этаже, а подъезд/этаж не указан — пиши 'Требуется уточнение'. "
        "Пример: «сосед сверху заливает» без номера квартиры → location_details='Требуется уточнение', НЕ придумывай номер. "
        "Пример: «в подъезде пахнет газом» без номера подъезда → 'Требуется уточнение'. "
        "ВАЖНО: если сообщение содержит ТОЛЬКО криминал (драки, наркотики, алкоголь, избиения, шум, музыка) "
        "без проблем ЖКХ — верни ПУСТОЙ список issues. "
        "Если есть И жкх-проблемы, И не-жкх — жкх в issues, не-жкх в non_jkh_issues краткими фразами."
    )


# ═══════════════════════════════════════════
# 3. СОСТОЯНИЕ ГРАФА
# ═══════════════════════════════════════════
class AgentState(TypedDict, total=False):
    user_input: str
    accumulated_context: str  # полный контекст диалога (исходная жалоба + уточнения)
    applicant_hash: str
    complaint: ComplaintReport | None
    needs_clarification: bool
    reply_message: str
    blocked: bool  # rate limiter flag


# ═══════════════════════════════════════════
# 4. УЗЛЫ ГРАФА
# ═══════════════════════════════════════════
RATE_LIMIT_SEC = 3
_user_last_request: dict[str, float] = {}


def node_rate_limiter(state: AgentState) -> AgentState:
    """Защита от спама: не чаще RATE_LIMIT_SEC на пользователя."""
    now = time.time()
    user_key = state.get("applicant_hash", "anon")
    last = _user_last_request.get(user_key, 0)

    if now - last < RATE_LIMIT_SEC:
        return {
            **state,
            "blocked": True,
            "reply_message": "⏳ Слишком много запросов. Подождите немного.",
        }

    _user_last_request[user_key] = now
    return {**state, "blocked": False}


def node_extractor(state: AgentState) -> AgentState:
    """Вызывает Pydantic-агента (DeepSeek) и кладёт результат в состояние."""
    if state.get("blocked"):
        return state

    text = state.get("accumulated_context") or state["user_input"]
    result = agent.run_sync(text, deps=state["applicant_hash"])
    return {**state, "complaint": result.output}


def node_quality_gate(state: AgentState) -> AgentState:
    """Проверяет качество: отдать JSON / запросить уточнение / отклонить."""
    if state.get("blocked"):
        return state

    complaint = state.get("complaint")

    if not complaint or not complaint.issues:
        return {
            **state,
            "needs_clarification": False,
            "reply_message": (
                "🤷 Это не относится к компетенции ЖКХ. "
                "Я принимаю проблемы с водоснабжением, отоплением, электрикой, лифтами, "
                "общедомовым имуществом и состоянием двора. "
                "По вопросам шума, драк, наркотиков и поведения жильцов — обращайтесь в полицию."
            ),
        }

    missing_issues = [
        issue for issue in complaint.issues
        if "Требуется уточнение" in issue.location_details
    ]
    if missing_issues:
        lines = [f"• {issue.dry_summary}" for issue in missing_issues]
        scopes = {issue.problem_scope for issue in missing_issues}

        if "Квартира" in scopes and "Подъезд/Этаж" in scopes:
            ask = "Для проблем в квартире укажите номер квартиры. Для проблем в подъезде — подъезд и этаж"
        elif "Квартира" in scopes:
            ask = "Укажите номер квартиры"
        elif "Подъезд/Этаж" in scopes:
            ask = "Укажите подъезд и этаж"
        elif "Дом" in scopes:
            ask = "Укажите, о какой части дома идёт речь (крыша, подвал, фасад)"
        else:
            ask = "Укажите расположение проблемы"

        return {
            **state,
            "needs_clarification": True,
            "reply_message": (
                "📍 Я понял суть проблемы, но не хватает точного адреса. Уточните, пожалуйста:\n"
                + "\n".join(lines)
                + f"\n\n{ask}."
            ),
        }

    json_str = complaint.model_dump_json(indent=2, exclude={"non_jkh_issues"})
    reply = f"📑 **Тикет сформирован:**\n\n```json\n{json_str}\n```"

    if complaint.non_jkh_issues:
        note = "\n".join(f"• {item}" for item in complaint.non_jkh_issues)
        reply += f"\n\n⚠️ *Описанные ниже проблемы не относятся к ЖКХ. Для их решения обратитесь в другие службы (полиция, администрация):*\n{note}"

    return {**state, "needs_clarification": False, "reply_message": reply}


# ═══════════════════════════════════════════
# 5. СБОРКА ГРАФА
# ═══════════════════════════════════════════
def _build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("rate_limiter", node_rate_limiter)
    builder.add_node("extractor", node_extractor)
    builder.add_node("quality_gate", node_quality_gate)

    builder.set_entry_point("rate_limiter")
    builder.add_edge("rate_limiter", "extractor")
    builder.add_edge("extractor", "quality_gate")
    builder.add_edge("quality_gate", END)

    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[("dispatcher", "ComplaintReport")]
    )
    memory = MemorySaver(serde=serde)
    return builder.compile(checkpointer=memory)


_graph = _build_graph()


# ═══════════════════════════════════════════
# 6. ТОЧКА ВХОДА ДЛЯ ТРАНСПОРТА
# ═══════════════════════════════════════════
def process_message(text: str, telegram_user_id: int) -> str:
    """
    Единственная публичная функция.
    Вызывается транспортом (main.py) на каждое сообщение.
    Возвращает строку-ответ для отправки пользователю.
    """
    applicant_hash = hashlib.sha256(str(telegram_user_id).encode()).hexdigest()[:12]
    thread_cfg = {"configurable": {"thread_id": str(telegram_user_id)}}

    # Склеиваем контекст, если предыдущий шаг запросил уточнение
    accumulated = text
    try:
        prev_state = _graph.get_state(thread_cfg)
        if prev_state and prev_state.values:
            if prev_state.values.get("needs_clarification"):
                old_context = prev_state.values.get("accumulated_context") or prev_state.values.get("user_input") or ""
                accumulated = f"{old_context}\nУточнение от жильца: {text}"
    except Exception:
        pass  # первый запуск для этого thread_id — состояния ещё нет

    result = _graph.invoke(
        {"user_input": text, "accumulated_context": accumulated, "applicant_hash": applicant_hash},
        config=thread_cfg,
    )
    return result["reply_message"]
