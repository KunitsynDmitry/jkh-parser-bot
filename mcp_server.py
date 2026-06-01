"""MCP-сервер для SQLite базы ЖКХ-бота.

Запуск: python mcp_server.py
Протокол: stdio (стандартный для MCP)
Назначение: дать ИИ-агенту инструменты для работы с заявками через протокол MCP.
              При миграции на PostgreSQL достаточно заменить этот файл —
              логика агента в dispatcher.py не изменится.
"""
from fastmcp import FastMCP

from db import init_db, insert_ticket, search_tickets as db_search

mcp = FastMCP("ЖКХ SQLite Server")


@mcp.tool()
def create_ticket(
    telegram_id: int,
    category: str,
    urgency: str,
    description: str,
    status: str,
    master_notes: str,
) -> dict:
    """Создать новую заявку в базе данных.

    Args:
        telegram_id: ID пользователя в Telegram (для фильтрации)
        category: Категория проблемы (Отопление, Сантехника, Электрика, ...)
        urgency: Срочность (Критично (авария), Высокая, Плановая)
        description: Сухая выжимка проблемы
        status: Статус заявки (всегда 'В процессе' для демо)
        master_notes: Комментарий мастера УК (в неконкретном будущем времени)
    """
    init_db()
    ticket_id = insert_ticket(
        telegram_id, category, urgency, description, status, master_notes
    )
    return {"ticket_id": ticket_id, "status": "created"}


@mcp.tool()
def search_tickets(
    telegram_id: int,
    category: str | None = None,
    ticket_id: int | None = None,
) -> dict:
    """Найти заявки пользователя.

    Args:
        telegram_id: ID пользователя в Telegram
        category: Опциональный фильтр по категории проблемы
        ticket_id: Опциональный фильтр по номеру заявки
    """
    init_db()
    tickets = db_search(telegram_id, category, ticket_id)
    return {"tickets": tickets, "count": len(tickets)}


if __name__ == "__main__":
    init_db()
    mcp.run(transport="stdio")
