from __future__ import annotations

"""
SQLite database for BFT auth: users, invite codes, per-user watchlists.

CLI usage for invite codes:
    python -m webapp.backend.database                    # random code, 1 use
    python -m webapp.backend.database BETA2026 10        # specific code, 10 uses
"""

import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(os.environ.get("BFT_DATA_DIR", Path(__file__).parent.parent / "data")) / "bft_auth.db"


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS invite_codes (
            code TEXT PRIMARY KEY,
            max_uses INTEGER NOT NULL DEFAULT 1,
            use_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS watchlists (
            user_id INTEGER NOT NULL UNIQUE,
            name TEXT NOT NULL DEFAULT 'Custom',
            symbols TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")


# ── Users ─────────────────────────────────────────────────────────

def get_user_by_email(email: str) -> dict | None:
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower(),)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict | None:
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def create_user(email: str, password_hash: str) -> int:
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO users (email, password_hash) VALUES (?, ?)",
        (email.lower(), password_hash),
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return user_id


# ── Invite Codes ──────────────────────────────────────────────────

def validate_invite_code(code: str) -> bool:
    conn = get_db()
    row = conn.execute("SELECT * FROM invite_codes WHERE code = ?", (code,)).fetchone()
    conn.close()
    if not row:
        return False
    return row["use_count"] < row["max_uses"]


def consume_invite_code(code: str, email: str) -> bool:
    """Atomically consume an invite code. Returns True if successful."""
    conn = get_db()
    cur = conn.execute(
        "UPDATE invite_codes SET use_count = use_count + 1 "
        "WHERE code = ? AND use_count < max_uses",
        (code,),
    )
    conn.commit()
    consumed = cur.rowcount > 0
    conn.close()
    return consumed


def create_invite_code(code: str, max_uses: int = 1):
    conn = get_db()
    conn.execute(
        "INSERT OR IGNORE INTO invite_codes (code, max_uses) VALUES (?, ?)",
        (code, max_uses),
    )
    conn.commit()
    conn.close()


# ── Watchlists ────────────────────────────────────────────────────

def get_watchlist(user_id: int) -> dict | None:
    conn = get_db()
    row = conn.execute("SELECT * FROM watchlists WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return {"name": row["name"], "symbols": row["symbols"].split(",") if row["symbols"] else []}


def set_watchlist(user_id: int, symbols: list[str], name: str = "Custom"):
    conn = get_db()
    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        """INSERT INTO watchlists (user_id, name, symbols, updated_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(user_id) DO UPDATE SET
               name = excluded.name,
               symbols = excluded.symbols,
               updated_at = excluded.updated_at""",
        (user_id, name, ",".join(symbols), now),
    )
    conn.commit()
    conn.close()


# ── CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import secrets
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    init_db()

    code = sys.argv[1] if len(sys.argv) > 1 else secrets.token_urlsafe(8)
    max_uses = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    create_invite_code(code, max_uses)
    print(f"Invite code created: {code}  (max_uses={max_uses})")
