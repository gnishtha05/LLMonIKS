"""
db.py — Data-access layer for gita.db
──────────────────────────────────────
Thin query functions that replace the old JSON-loading logic.
All functions open/close connections internally.
"""

import os
import sqlite3

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DATA_DIR, "gita.db")


def _get_connection() -> sqlite3.Connection:
    """Return a connection to gita.db with row_factory = sqlite3.Row."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def get_all_theme_names() -> list[str]:
    """Return all theme names ordered by serial_no."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT theme_name FROM themes ORDER BY serial_no"
        ).fetchall()
        return [row["theme_name"] for row in rows]
    finally:
        conn.close()


def get_verse_ids_for_theme(theme_name: str) -> list[str]:
    """Return verse IDs associated with a given theme."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            """
            SELECT tv.verse_id
            FROM theme_verses tv
            JOIN themes t ON t.theme_id = tv.theme_id
            WHERE t.theme_name = ?
            """,
            (theme_name,),
        ).fetchall()
        return [row["verse_id"] for row in rows]
    finally:
        conn.close()


def get_verses_by_ids(verse_ids: list[str]) -> list[dict]:
    """
    Return verse data dicts for a list of verse IDs.
    Each dict has keys: verse_id, shloka, bhashya_hindi.
    Order follows the input list.
    """
    if not verse_ids:
        return []

    conn = _get_connection()
    try:
        placeholders = ",".join("?" for _ in verse_ids)
        rows = conn.execute(
            f"SELECT verse_id, shloka, bhashya_hindi FROM verses WHERE verse_id IN ({placeholders})",
            verse_ids,
        ).fetchall()

        # Build a lookup to preserve input order
        lookup = {row["verse_id"]: dict(row) for row in rows}
        return [lookup[vid] for vid in verse_ids if vid in lookup]
    finally:
        conn.close()


def get_verse(verse_id: str) -> dict | None:
    """Return a single verse's data, or None if not found."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT verse_id, shloka, bhashya_skt, bhashya_hindi FROM verses WHERE verse_id = ?",
            (verse_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()
