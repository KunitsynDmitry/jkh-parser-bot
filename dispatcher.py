"""
Граф обработки сообщений ЖКХ-бота с Fan-out / Fan-in.

Узлы:
  1. RateLimiter      — защита от спама
  2. IntentRouter     — классификация намерений (create_ticket / check_status)
  3. Extractor        — вызов Pydantic-агента (DeepSeek), извлечение жалобы + master_notes
  4. QualityGate      — проверка качества, запрос уточнений
  5. CreateTicket     — MCP-инструмент: INSERT в SQLite
  6. CheckStatus      — MCP-инструмент: SELECT из SQLite
  7. Aggregator       — сбор результатов из параллельных веток → ответ жильцу

Поток (Fan-out / Fan-in):
  user_input → RateLimiter → IntentRouter
    ├─ create_ticket: Extractor → QualityGate → CreateTicket ──┐
    └─ check_status:  CheckStatus ─────────────────────────────┤
                                                                ▼
                                                          Aggregator → END
"""
import asyncio
import hashlib
import json
import re
import sys
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
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from config import DEEPSEEK_API_KEY
from db import find_recent_ticket


# ═══════════════════════════════════════════
# 1. СХЕМА ДАННЫХ (Pydantic)
# ═══════════════════════════════════════════
class SingleIssue(BaseModel):
    category: str = Field(
        description="Категория (строго одна из): Отопление, Сантехника, Электрика, "
                    "Лифты, Уборка, Двор и фасад, Общедомовое имущество, Прочее"
    )
    urgency_level: str = Field(
        description="Уровень срочности: Критично (авария), Высокая, Плановая"
    )
    problem_scope: str = Field(
        description="Масштаб: Квартира, Подъезд/Этаж, Дом, Двор"
    )
    location_details: str = Field(
        description="ГДЕ ИМЕННО проблема (этаж, подъезд, квартира). НЕ адрес заявителя."
    )
    dry_summary: str = Field(
        description="Сухая выжимка проблемы в 1-2 предложениях"
    )
    master_notes: str = Field(
        default="",
        description="Правдоподобный комментарий от лица мастера УК. "
                    "Всегда в неконкретном будущем времени. Примеры: "
                    "'Осмотрели проблему, материалы приедут на следующей неделе, "
                    "сразу приступим.' или 'Включили в план ремонта, ориентировочно "
                    "в ближайшие дни направим специалиста.'"
    )

    @field_validator('category')
    @classmethod
    def check_category(cls, v: str) -> str:
        valid = {
            "Отопление", "Сантехника", "Электрика",
            "Лифты", "Уборка", "Двор и фасад", "Общедомовое имущество", "Прочее"
        }
        if v not in valid:
            raise ModelRetry(
                f"Категория '{v}' недопустима. Выбери из: {', '.join(sorted(valid))}."
            )
        return v

    @field_validator('problem_scope')
    @classmethod
    def check_scope(cls, v: str) -> str:
        valid = {"Квартира", "Подъезд/Этаж", "Дом", "Двор"}
        if v not in valid:
            raise ModelRetry(
                f"problem_scope '{v}' некорректен. Выбери из: {', '.join(sorted(valid))}."
            )
        return v

    @field_validator('location_details')
    @classmethod
    def check_location(cls, v: str) -> str:
        val_lower = v.lower()
        if val_lower.strip().startswith("требуется уточнение") or val_lower.strip().startswith("неизвестно"):
            return "Требуется уточнение"

        applicant_centric = [
            "я живу", "мой адрес", "моя квартира", "проживаю на", "я нахожусь"
        ]
        if any(term in val_lower for term in applicant_centric):
            raise ModelRetry(
                "Ты указал адрес ЗАЯВИТЕЛЯ, а не адрес ПРОБЛЕМЫ. "
                "Вычисли место поломки. Если координаты неизвестны — 'Требуется уточнение'."
            )

        relative_terms = [
            "надо мной", "над нами", "тут", "здесь", "там",
            "у нас", "где-то", "непонятно где"
        ]
        if any(term in val_lower for term in relative_terms):
            raise ModelRetry(
                "Локация слишком размыта. Если нет точных координат — 'Требуется уточнение'."
            )
        return v

    @model_validator(mode='after')
    def enforce_location_by_scope(self):
        if "Требуется уточнение" in self.location_details:
            return self

        if self.problem_scope == "Квартира":
            has_apt = bool(re.search(
                r'кв(?:\.|артир[а-яё])?\s*\d+', self.location_details.lower()
            ))
            if not has_apt:
                self.location_details = 'Требуется уточнение'

        if self.problem_scope == "Подъезд/Этаж":
            _ord = (
                r'(?:перв|втор|трет|четв[её]рт|пят|шест|седьм|восьм|девят|десят'
                r'|одиннадцат|двенадцат|тринадцат|четырнадцат|пятнадцат|шестнадцат'
                r'|семнадцат|восемнадцат|девятнадцат|двадцат)'
            )
            _ent = rf'(?:подъезд\s*(\d+|{_ord}))|(?:(\d+|{_ord})\S{{0,5}}\s*подъезд)'
            _flr = rf'(?:этаж\s*(\d+|{_ord}))|(?:(\d+|{_ord})\S{{0,5}}\s*этаж)'
            has_entrance = bool(re.search(_ent, self.location_details.lower()))
            has_floor = bool(re.search(_flr, self.location_details.lower()))
            if not (has_entrance or has_floor):
                self.location_details = 'Требуется уточнение'
        return self


class ComplaintReport(BaseModel):
    applicant_id: str = Field(description="Хеш ID заявителя")
    issues: list[SingleIssue] = Field(description="Список всех найденных проблем")
    emotional_intensity: int = Field(description="Оценка эмоционального накала от 1 до 10")
    threatens_lawsuit: bool = Field(description="Угрожает ли автор судом?")
    non_jkh_issues: list[str] = Field(
        default_factory=list,
        description="Проблемы НЕ ЖКХ"
    )


class IntentResult(BaseModel):
    intents: list[str] = Field(
        description="Намерения: create_ticket (новая жалоба/проблема), "
                    "check_status (вопрос о статусе старых заявок). Может быть оба."
    )
    search_category: str | None = Field(
        default=None,
        description="Если check_status — какую категорию ищет пользователь "
                    "(null если категория не указана)"
    )
    search_ticket_id: int | None = Field(
        default=None,
        description="Если пользователь спрашивает про конкретную заявку по номеру "
                    "(«заявка 9», «статус тикета №5») — указать номер. Иначе null."
    )


# ═══════════════════════════════════════════
# 2. PYDANTIC-АГЕНТЫ (DeepSeek)
# ═══════════════════════════════════════════
deepseek_model = OpenAIChatModel(
    model_name='deepseek-chat',
    provider=DeepSeekProvider(api_key=DEEPSEEK_API_KEY)
)

extractor_agent = Agent(
    model=deepseek_model,
    output_type=ComplaintReport,
    deps_type=str,
    retries=2,
)


@extractor_agent.system_prompt
def extractor_prompt(ctx: RunContext[str]) -> str:
    now = datetime.now()
    day_type = "ВЫХОДНОЙ" if now.weekday() >= 5 else "РАБОЧИЙ"
    return (
        f"Ты — диспетчер ЖКХ. Сейчас {now.strftime('%H:%M')}, день {day_type}. "
        f"Твоя задача: перевести эмоциональную жалобу жильца в сухой структурированный отчет. "
        f"ID заявителя: {ctx.deps}. "
        "Несколько проблем → отдельный элемент в issues для каждой. "
        "В location_details указывай МЕСТО ПРОБЛЕМЫ, не адрес заявителя. "
        "«Я живу на 5 этаже, сосед снизу сверлит» → проблема на 4 этаже. "
        "Категории (строго): Отопление, Сантехника, Электрика, Лифты, Уборка, "
        "Двор и фасад, Общедомовое имущество, Прочее. "
        "problem_scope: Квартира (проблема внутри квартиры), Подъезд/Этаж (подъезд/лестница/коридор), "
        "Дом (крыша/подвал/фасад/стояк), Двор (улица/территория). "
        "ЗАПРЕЩЕНО выдумывать номера квартир, этажей, подъездов. "
        "Только цифры из текста. Нет в тексте → 'Требуется уточнение'. "
        "В location_details всегда ЦИФРЫ: '2-й этаж', не 'второй этаж'. "
        "Для КАЖДОГО обращения сгенерируй master_notes — правдоподобный комментарий "
        "от лица мастера УК в НЕКОНКРЕТНОМ БУДУЩЕМ времени. Не пиши конкретных дат. Примеры: "
        "'Осмотрели проблему, материалы приедут на следующей неделе, сразу приступим к работам.' "
        "'Зафиксировали обращение, включили в план ремонта на ближайшие дни.' "
        "'Согласовали замену, ожидаем поставку, ориентировочно приступим через неделю.' "
        "Если только криминал/шум/драки без проблем ЖКХ → пустой список issues. "
        "Если и ЖКХ, и не-ЖКХ → ЖКХ в issues, остальное в non_jkh_issues."
    )


intent_agent = Agent(
    model=deepseek_model,
    output_type=IntentResult,
    retries=1,
)


@intent_agent.system_prompt
def intent_prompt(ctx: RunContext[str]) -> str:
    return (
        "Ты — классификатор намерений для ЖКХ-диспетчера. Определи по сообщению, "
        "что хочет пользователь:\n"
        "- create_ticket: создать новую заявку (жалоба на проблему ЖКХ, просьба починить)\n"
        "- check_status: спросить о статусе старых заявок (вопросы вроде "
        "'что с моей заявкой', 'когда починят', 'что там с отоплением')\n"
        "Может быть ОБА сразу. Например: 'Трубу прорвало! И что там с электриком?' → "
        "['create_ticket', 'check_status'].\n"
        "Если check_status и пользователь упомянул категорию — заполни search_category "
        "СТРОГО одним из значений: Отопление, Сантехника, Электрика, Лифты, Уборка, "
        "Двор и фасад, Общедомовое имущество, Прочее.\n"
        "Если пользователь сказал 'лифт' или 'лифт' — пиши 'Лифты'. "
        "'электрика' → 'Электрика', 'отопление'/'батарея' → 'Отопление'.\n"
        "Если пользователь спрашивает про КОНКРЕТНУЮ заявку по номеру "
        "('заявка 9', 'статус по заявке 5', 'тикет №3', 'что с заявкой 12') — "
        "поставь search_ticket_id равным этому номеру.\n"
        "Если сообщение вообще не про ЖКХ и не про заявки → пустой список intents."
    )


# ═══════════════════════════════════════════
# 3. MCP-КЛИЕНТ (подключение к mcp_server.py)
# ═══════════════════════════════════════════
class McpManager:
    """MCP-клиент для вызова инструментов mcp_server.py.
    Каждый вызов создаёт новый подпроцесс (stdio-транспорт).
    При миграции на PostgreSQL достаточно заменить command/args —
    логика агента не меняется."""

    @classmethod
    async def call_tool(cls, name: str, arguments: dict) -> dict:
        transport = StdioTransport(
            command=sys.executable,
            args=["mcp_server.py"],
        )
        async with Client(transport) as client:
            result = await client.call_tool(name, arguments)
            return json.loads(result.content[0].text)


# ═══════════════════════════════════════════
# 4. СОСТОЯНИЕ ГРАФА
# ═══════════════════════════════════════════
class AgentState(TypedDict, total=False):
    user_input: str
    accumulated_context: str
    applicant_hash: str
    telegram_id: int

    # intent routing
    intents: list[str]
    search_category: str | None
    search_ticket_id: int | None

    # extraction results
    complaint: ComplaintReport | None
    needs_clarification: bool
    clarification_message: str

    # результаты параллельных веток — накапливаются последовательно
    # (при переходе на истинный параллелизм — заменить на Annotated[list, operator.add])
    branch_results: list[dict]

    # final
    reply_message: str
    blocked: bool


# ═══════════════════════════════════════════
# 5. УЗЛЫ ГРАФА
# ═══════════════════════════════════════════
RATE_LIMIT_SEC = 3
_user_last_request: dict[str, float] = {}


async def node_rate_limiter(state: AgentState) -> AgentState:
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


async def node_intent_router(state: AgentState) -> AgentState:
    """Классифицирует намерения пользователя: создать тикет / проверить статус."""
    if state.get("blocked"):
        return state

    text = state.get("accumulated_context") or state["user_input"]
    result = await intent_agent.run(text)
    return {
        **state,
        "intents": result.output.intents,
        "search_category": result.output.search_category,
        "search_ticket_id": result.output.search_ticket_id,
        "complaint": None,
        "needs_clarification": False,
        "clarification_message": "",
        "branch_results": [],
    }


async def node_extractor(state: AgentState) -> AgentState:
    """Вызывает Pydantic-агента (DeepSeek) — извлекает жалобу и генерирует master_notes."""
    if state.get("blocked"):
        return state

    text = state.get("accumulated_context") or state["user_input"]
    result = await extractor_agent.run(text, deps=state["applicant_hash"])
    return {**state, "complaint": result.output}


async def node_quality_gate(state: AgentState) -> AgentState:
    """Проверяет качество извлечённой жалобы: всё есть / нужны уточнения / не ЖКХ."""
    if state.get("blocked"):
        return state

    complaint = state.get("complaint")

    if not complaint or not complaint.issues:
        return {
            **state,
            "needs_clarification": False,
            "branch_results": state.get("branch_results", []) + [{"type": "no_jkh_issues"}],
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
            "clarification_message": (
                "📍 Я понял суть проблемы, но не хватает точного адреса. Уточните, пожалуйста:\n"
                + "\n".join(lines)
                + f"\n\n{ask}."
            ),
        }

    # Всё хорошо — можно создавать тикет
    return {**state, "needs_clarification": False}


async def node_create_ticket(state: AgentState) -> AgentState:
    """Создаёт тикеты в SQLite через MCP-инструмент."""
    complaint = state.get("complaint")
    if not complaint or not complaint.issues:
        return state

    tg_id = state["telegram_id"]
    results = []

    for issue in complaint.issues:
        try:
            dup = find_recent_ticket(tg_id, issue.category, issue.dry_summary)
            if dup:
                results.append({
                    "type": "ticket_duplicate",
                    "ticket_id": dup["id"],
                    "category": issue.category,
                })
                continue

            data = await McpManager.call_tool("create_ticket", {
                "telegram_id": tg_id,
                "category": issue.category,
                "urgency": issue.urgency_level,
                "description": issue.dry_summary,
                "status": "В процессе",
                "master_notes": issue.master_notes or "",
            })
            results.append({
                "type": "ticket_created",
                "ticket_id": data["ticket_id"],
                "category": issue.category,
                "urgency": issue.urgency_level,
                "description": issue.dry_summary,
                "master_notes": issue.master_notes,
            })
        except Exception as e:
            results.append({
                "type": "ticket_error",
                "category": issue.category,
                "error": str(e),
            })

    existing = state.get("branch_results", [])
    return {**state, "branch_results": existing + results}


async def node_check_status(state: AgentState) -> AgentState:
    """Проверяет статус заявок пользователя через MCP-инструмент."""
    tg_id = state["telegram_id"]
    category = state.get("search_category")
    ticket_id = state.get("search_ticket_id")

    existing = state.get("branch_results", [])

    try:
        data = await McpManager.call_tool("search_tickets", {
            "telegram_id": tg_id,
            "category": category,
            "ticket_id": ticket_id,
        })
        return {
            **state,
            "branch_results": existing + [{
                "type": "status_check",
                "tickets": data.get("tickets", []),
                "count": data.get("count", 0),
                "search_category": category,
                "search_ticket_id": ticket_id,
            }],
        }
    except Exception as e:
        return {
            **state,
            "branch_results": existing + [{
                "type": "status_error",
                "error": str(e),
            }],
        }


async def node_aggregator(state: AgentState) -> AgentState:
    """Собирает результаты всех веток и синтезирует финальный ответ."""
    if state.get("blocked"):
        return state  # reply_message уже установлен в rate_limiter

    branch_results = state.get("branch_results", [])
    needs_clarification = state.get("needs_clarification", False)
    clarification_message = state.get("clarification_message", "")
    complaint = state.get("complaint")

    # Собираем части ответа
    parts: list[str] = []

    # --- Часть 1: Результаты создания тикетов ---
    created = [r for r in branch_results if r.get("type") == "ticket_created"]
    create_errors = [r for r in branch_results if r.get("type") == "ticket_error"]
    duplicates = [r for r in branch_results if r.get("type") == "ticket_duplicate"]

    if duplicates:
        ids = ", ".join(f"№{d['ticket_id']}" for d in duplicates)
        parts.append(f"Заявка уже зарегистрирована ({ids}).")

    if created:
        lines = ["📑 **Новые заявки:**"]
        for t in created:
            lines.append(
                f"• №{t['ticket_id']} — {t['category']} ({t['urgency']}): {t['description']}"
            )
        if complaint and complaint.issues:
            json_str = complaint.model_dump_json(indent=2, exclude={"applicant_id", "non_jkh_issues"})
            lines.append(f"\n```json\n{json_str}\n```")
        parts.append("\n".join(lines))

    if create_errors:
        lines = ["⚠️ **Ошибки при создании заявок:**"]
        for e in create_errors:
            lines.append(f"• {e.get('category', '?')}: {e['error']}")
        parts.append("\n".join(lines))

    # --- Часть 2: Результаты проверки статуса ---
    status_results = [r for r in branch_results if r.get("type") == "status_check"]
    status_errors = [r for r in branch_results if r.get("type") == "status_error"]

    for result in status_results:
        tickets = result.get("tickets", [])
        search_cat = result.get("search_category")
        search_tid = result.get("search_ticket_id")

        if not tickets:
            if search_tid:
                parts.append(f"🔍 Заявка №{search_tid} не найдена.")
            elif search_cat:
                parts.append(
                    f"🔍 По категории *«{search_cat}»* у вас пока нет заявок."
                )
            else:
                parts.append("🔍 У вас пока нет активных заявок.")
        else:
            header = (
                f"📋 **Ваши заявки{f' по категории «{search_cat}»' if search_cat else ''}:**"
            )
            lines = [header]
            for t in tickets:
                lines.append(
                    f"• №{t['id']} — {t['category']} ({t['urgency']}): {t['description']}\n"
                    f"  Статус: *{t['status']}*\n"
                    f"  _{t.get('master_notes', '')}_"
                )
            parts.append("\n".join(lines))

    for e in status_errors:
        parts.append(f"⚠️ Не удалось проверить статус заявок: {e['error']}")

    # --- Часть 3: Запрос уточнения (если quality_gate не прошёл) ---
    if needs_clarification and clarification_message:
        parts.append(clarification_message)

    # --- Часть 4: Не-ЖКХ (если есть) ---
    if complaint and complaint.non_jkh_issues:
        note = "\n".join(f"• {item}" for item in complaint.non_jkh_issues)
        parts.append(
            f"\n⚠️ *Не относится к ЖКХ:*\n{note}"
        )

    # --- Часть 5: Вообще ничего не понятно ---
    no_jkh = any(r.get("type") == "no_jkh_issues" for r in branch_results)
    if no_jkh and not created and not status_results:
        parts.append(
            "🤷 Это не относится к компетенции ЖКХ. "
            "Я принимаю проблемы с отоплением, сантехникой, электрикой, лифтами, "
            "уборкой, состоянием двора и фасада.\n"
            "По вопросам шума, драк и поведения жильцов — обращайтесь в полицию."
        )

    if not parts:
        parts.append("🤔 Не удалось обработать ваш запрос. Попробуйте переформулировать.")

    reply = "\n\n".join(parts)
    return {**state, "reply_message": reply}


# ═══════════════════════════════════════════
# 6. МАРШРУТИЗАЦИЯ
# ═══════════════════════════════════════════
def route_after_intent(state: AgentState) -> str:
    """После классификации намерений: куда направляем сообщение."""
    if state.get("blocked"):
        return "aggregator"

    intents = state.get("intents", [])
    has_create = "create_ticket" in intents
    has_check = "check_status" in intents

    if has_create:
        return "extractor"
    elif has_check:
        return "check_status"
    else:
        return "aggregator"


def route_after_quality(state: AgentState) -> str:
    """После проверки качества: создаём тикет или пропускаем."""
    intents = state.get("intents", [])
    has_check = "check_status" in intents

    if state.get("needs_clarification"):
        # Уточнение не останавливает проверку статуса
        return "check_status" if has_check else "aggregator"
    else:
        return "create_ticket"


def route_after_create(state: AgentState) -> str:
    """После создания тикета: проверяем статус или сразу агрегируем."""
    intents = state.get("intents", [])
    return "check_status" if "check_status" in intents else "aggregator"


# ═══════════════════════════════════════════
# 7. СБОРКА ГРАФА
# ═══════════════════════════════════════════
def _build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("rate_limiter", node_rate_limiter)
    builder.add_node("intent_router", node_intent_router)
    builder.add_node("extractor", node_extractor)
    builder.add_node("quality_gate", node_quality_gate)
    builder.add_node("create_ticket", node_create_ticket)
    builder.add_node("check_status", node_check_status)
    builder.add_node("aggregator", node_aggregator)

    builder.set_entry_point("rate_limiter")

    builder.add_edge("rate_limiter", "intent_router")

    builder.add_conditional_edges(
        "intent_router", route_after_intent, {
            "extractor": "extractor",
            "check_status": "check_status",
            "aggregator": "aggregator",
        }
    )

    builder.add_edge("extractor", "quality_gate")

    builder.add_conditional_edges(
        "quality_gate", route_after_quality, {
            "create_ticket": "create_ticket",
            "check_status": "check_status",
            "aggregator": "aggregator",
        }
    )

    builder.add_conditional_edges(
        "create_ticket", route_after_create, {
            "check_status": "check_status",
            "aggregator": "aggregator",
        }
    )

    builder.add_edge("check_status", "aggregator")
    builder.add_edge("aggregator", END)

    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[("dispatcher", "ComplaintReport")]
    )
    memory = MemorySaver(serde=serde)
    return builder.compile(checkpointer=memory)


_graph = _build_graph()


# ═══════════════════════════════════════════
# 8. ТОЧКА ВХОДА ДЛЯ ТРАНСПОРТА
# ═══════════════════════════════════════════
def process_message(text: str, telegram_user_id: int) -> str:
    """
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
                old_context = (
                    prev_state.values.get("accumulated_context")
                    or prev_state.values.get("user_input")
                    or ""
                )
                accumulated = f"{old_context}\nУточнение от жильца: {text}"
            # Дедупликация: тот же текст → не плодим новый тикет
            elif prev_state.values.get("user_input") == text:
                prev_results = prev_state.values.get("branch_results", [])
                created = [r for r in prev_results if r.get("type") == "ticket_created"]
                if created:
                    ids = ", ".join(f"№{t['ticket_id']}" for t in created)
                    return f"Заявка уже зарегистрирована ({ids})."
    except Exception:
        pass

    async def _invoke():
        result = await _graph.ainvoke(
            {
                "user_input": text,
                "accumulated_context": accumulated,
                "applicant_hash": applicant_hash,
                "telegram_id": telegram_user_id,
            },
            config=thread_cfg,
        )
        return result["reply_message"]

    return asyncio.run(_invoke())
