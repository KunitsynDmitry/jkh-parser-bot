"""
Мини-LangGraph для ЖКХ-диспетчера.
Граф: Extractor -> QualityGate -> END
MemorySaver хранит состояние в RAM для будущих многошаговых диалогов.
"""
import hashlib
from datetime import datetime
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

from config import DEEPSEEK_API_KEY


# ==========================================
# 1. СХЕМА ДАННЫХ (без изменений)
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

        applicant_centric = ["я живу", "мой адрес", "моя квартира", "проживаю на", "я нахожусь"]
        if any(term in val_lower for term in applicant_centric):
            raise ModelRetry("Ты указал адрес ЗАЯВИТЕЛЯ, а не адрес ПРОБЛЕМЫ. Вычисли, где именно произошла поломка (сосед снизу = этажом ниже, сверху = этажом выше). Если координаты неизвестны, напиши 'Требуется уточнение'.")

        relative_terms = ["надо мной", "над нами", "тут", "здесь", "где-то", "непонятно где"]
        if any(term in val_lower for term in relative_terms):
            raise ModelRetry("Локация указана слишком размыто. Если нет точных координат, напиши 'Требуется уточнение'.")

        if not any(char.isdigit() for char in v):
            raise ModelRetry("В локации нет цифр (этажа, квартиры, подъезда). Если координаты неизвестны, напиши 'Требуется уточнение'.")

        return v


class ComplaintReport(BaseModel):
    applicant_id: str = Field(description="Хеш ID заявителя")
    issues: list[SingleIssue] = Field(description="Список всех найденных проблем")
    emotional_intensity: int = Field(description="Оценка эмоционального накала текста от 1 до 10")
    threatens_lawsuit: bool = Field(description="Угрожает ли автор судом или проверками? (True/False)")
    non_jkh_issues: list[str] = Field(default_factory=list, description="Проблемы из сообщения, НЕ относящиеся к ЖКХ")


# ==========================================
# 2. PYDANTIC-АГЕНТ
# ==========================================
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
# 3. СОСТОЯНИЕ ГРАФА
# ==========================================
class AgentState(TypedDict, total=False):
    user_input: str
    applicant_hash: str
    complaint: ComplaintReport | None
    needs_clarification: bool
    reply_message: str


# ==========================================
# 4. УЗЛЫ ГРАФА
# ==========================================
def node_extractor(state: AgentState) -> AgentState:
    """Вызывает Pydantic-агента и кладёт результат в состояние."""
    result = agent.run_sync(state["user_input"], deps=state["applicant_hash"])
    return {
        **state,
        "complaint": result.output,
    }


def node_quality_gate(state: AgentState) -> AgentState:
    """Проверяет качество результата и решает: отдать JSON или запросить уточнение."""
    complaint = state["complaint"]

    # Случай 1: пустой тикет — не ЖКХ
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

    # Случай 2: не хватает локации
    missing_locations = [
        f"• {issue.dry_summary}"
        for issue in complaint.issues
        if "Требуется уточнение" in issue.location_details
    ]
    if missing_locations:
        return {
            **state,
            "needs_clarification": True,
            "reply_message": (
                "📍 Я понял суть проблемы, но не хватает точного адреса. Уточните, пожалуйста:\n"
                + "\n".join(missing_locations)
                + "\n\nУкажите этаж, номер квартиры или подъезда."
            ),
        }

    # Случай 3: всё хорошо — формируем JSON
    json_str = complaint.model_dump_json(indent=2, exclude={"non_jkh_issues"})
    reply = f"📑 **Тикет сформирован:**\n\n```json\n{json_str}\n```"

    if complaint.non_jkh_issues:
        note = "\n".join(f"• {item}" for item in complaint.non_jkh_issues)
        reply += f"\n\n⚠️ *Описанные ниже проблемы не относятся к ЖКХ. Для их решения обратитесь в другие службы (полиция, администрация):*\n{note}"

    return {
        **state,
        "needs_clarification": False,
        "reply_message": reply,
    }


# ==========================================
# 5. СБОРКА ГРАФА
# ==========================================
def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("extractor", node_extractor)
    builder.add_node("quality_gate", node_quality_gate)

    builder.set_entry_point("extractor")
    builder.add_edge("extractor", "quality_gate")
    builder.add_edge("quality_gate", END)

    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[("graph_agent", "ComplaintReport")]
    )
    memory = MemorySaver(serde=serde)
    return builder.compile(checkpointer=memory)


# Экземпляр графа (синглтон)
graph = build_graph()


# ==========================================
# 6. ТОЧКА ВХОДА ДЛЯ БОТА
# ==========================================
def process_message(text: str, telegram_user_id: int) -> str:
    """
    Вызывается из bot.py на каждое сообщение.
    Возвращает строку-ответ для отправки пользователю.
    """
    applicant_hash = hashlib.sha256(str(telegram_user_id).encode()).hexdigest()[:12]
    thread_id = str(telegram_user_id)

    result = graph.invoke(
        {
            "user_input": text,
            "applicant_hash": applicant_hash,
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    return result["reply_message"]
