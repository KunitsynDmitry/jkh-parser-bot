"""SQLite-прослойка для ЖКХ-бота. Одна плоская таблица tickets."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "housing.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_id BIGINT NOT NULL,
            category TEXT NOT NULL,
            urgency TEXT NOT NULL,
            description TEXT NOT NULL,
            status TEXT DEFAULT 'В процессе',
            master_notes TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def insert_ticket(
    telegram_id: int,
    category: str,
    urgency: str,
    description: str,
    status: str,
    master_notes: str,
) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO tickets (telegram_id, category, urgency, description, status, master_notes) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (telegram_id, category, urgency, description, status, master_notes),
    )
    conn.commit()
    ticket_id = cur.lastrowid
    conn.close()
    return ticket_id


def _extract_location_nums(text: str) -> set[str]:
    """Вытаскивает номера подъездов и этажей из текста (цифры и слова)."""
    import re
    word_to_digit = {
        'перв': '1', 'втор': '2', 'трет': '3', 'четв': '4', 'пят': '5',
        'шест': '6', 'седьм': '7', 'восьм': '8', 'девят': '9', 'десят': '10',
    }
    nums: set[str] = set()
    # Цифры перед "подъезд"/"этаж"/"под"/"эт"
    nums.update(re.findall(r'(\d+)\s*(?:подъезд|этаж|под\b|эт\b)', text))
    # Цифры после "подъезд"/"этаж"
    nums.update(re.findall(r'(?:подъезд|этаж)\s*(\d+)', text))
    # Словесные: "второго подъезда", "третий этаж"
    for word, digit in word_to_digit.items():
        if re.search(rf'\b{word}\w*\s+(?:подъезд|этаж|под\b)', text):
            nums.add(digit)
    return nums


def find_recent_ticket(
    telegram_id: int, category: str, description: str = "", minutes: int = 5,
) -> dict | None:
    """Найти тикет по категории за последние N минут. Если описание про другое место — не дубликат."""
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM tickets WHERE telegram_id = ? AND category = ? "
        "AND created_at > datetime('now', ? || ' minutes') "
        "ORDER BY created_at DESC LIMIT 1",
        (telegram_id, category, f'-{minutes}'),
    ).fetchone()
    conn.close()
    if not row:
        return None

    ticket = dict(row)
    if description:
        new_nums = _extract_location_nums(description.lower())
        old_nums = _extract_location_nums(ticket["description"].lower())
        # Если оба содержат номера и они не пересекаются → разные места
        if new_nums and old_nums and not (new_nums & old_nums):
            return None
    return ticket


def search_tickets(
    telegram_id: int,
    category: str | None = None,
    ticket_id: int | None = None,
) -> list[dict]:
    conn = get_conn()
    query = "SELECT * FROM tickets WHERE telegram_id = ?"
    params: list = [telegram_id]

    if ticket_id is not None:
        query += " AND id = ?"
        params.append(ticket_id)
    elif category:
        query += " AND category = ?"
        params.append(category)

    query += " ORDER BY created_at DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]
