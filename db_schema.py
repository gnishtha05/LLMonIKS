"""
db_schema.py — SQLite Migration Script
───────────────────────────────────────
Creates gita.db and imports data from:
  • themes_verses.json  → themes + theme_verses tables
  • final_data.json     → verses table

Run:  python db_schema.py
"""

import json
import os
import sqlite3

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DATA_DIR, "gita.db")

SCHEMA_SQL = """
-- Core verse data (from final_data.json)
CREATE TABLE IF NOT EXISTS verses (
    verse_id      TEXT PRIMARY KEY,   -- e.g. "2.47"
    chapter       INTEGER NOT NULL,
    verse_num     INTEGER NOT NULL,
    shloka        TEXT NOT NULL,
    bhashya_skt   TEXT NOT NULL DEFAULT '',
    bhashya_hindi TEXT NOT NULL DEFAULT ''
);

-- Themes (from themes_verses.json)
CREATE TABLE IF NOT EXISTS themes (
    theme_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    serial_no   INTEGER NOT NULL,
    theme_name  TEXT NOT NULL UNIQUE
);

-- Many-to-many: themes ↔ verses
CREATE TABLE IF NOT EXISTS theme_verses (
    theme_id    INTEGER NOT NULL REFERENCES themes(theme_id),
    verse_id    TEXT NOT NULL REFERENCES verses(verse_id),
    PRIMARY KEY (theme_id, verse_id)
);

-- Indexes for fast lookup
CREATE INDEX IF NOT EXISTS idx_verses_chapter ON verses(chapter);
CREATE INDEX IF NOT EXISTS idx_theme_verses_verse ON theme_verses(verse_id);
CREATE INDEX IF NOT EXISTS idx_theme_verses_theme ON theme_verses(theme_id);
"""


def normalise_verse_id(verse_id: str) -> str:
    """
    Strip leading zeros so that '3.09' → '3.9', '4.01' → '4.1'.
    This ensures theme_verses references match the verses table keys.
    """
    parts = verse_id.split(".")
    if len(parts) == 2:
        chapter = str(int(parts[0]))
        verse = str(int(parts[1]))
        return f"{chapter}.{verse}"
    return verse_id


def migrate():
    # ── Connect & create schema ──────────────────────────────────────────────
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"[migrate] Removed existing {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_SQL)
    print("[migrate] Schema created.")

    # ── Import verses from final_data.json ───────────────────────────────────
    final_data_path = os.path.join(DATA_DIR, "final_data.json")
    with open(final_data_path, "r", encoding="utf-8") as f:
        final_data: dict = json.load(f)

    verse_rows = []
    for verse_id, data in final_data.items():
        parts = verse_id.split(".")
        chapter = int(parts[0]) if len(parts) >= 1 else 0
        verse_num = int(parts[1]) if len(parts) >= 2 else 0
        verse_rows.append((
            verse_id,
            chapter,
            verse_num,
            data.get("shloka", ""),
            data.get("bhashya_skt", ""),
            data.get("bhashya_hindi", ""),
        ))

    conn.executemany(
        "INSERT INTO verses (verse_id, chapter, verse_num, shloka, bhashya_skt, bhashya_hindi) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        verse_rows,
    )
    print(f"[migrate] Inserted {len(verse_rows)} verses.")

    # ── Import themes from themes_verses.json ────────────────────────────────
    themes_path = os.path.join(DATA_DIR, "themes_verses.json")
    with open(themes_path, "r", encoding="utf-8") as f:
        themes_data: list[dict] = json.load(f)

    # Collect all valid verse IDs for foreign-key safety
    valid_verse_ids = set(final_data.keys())

    themes_inserted = 0
    mappings_inserted = 0
    skipped_mappings = 0

    for entry in themes_data:
        theme_name = entry["theme"].strip().replace("\xa0", "")
        serial_no = entry.get("serial_no", 0)

        conn.execute(
            "INSERT INTO themes (serial_no, theme_name) VALUES (?, ?)",
            (serial_no, theme_name),
        )
        theme_id = conn.execute(
            "SELECT theme_id FROM themes WHERE theme_name = ?", (theme_name,)
        ).fetchone()[0]
        themes_inserted += 1

        for raw_vid in entry["verses"]:
            norm_vid = normalise_verse_id(raw_vid)
            if norm_vid in valid_verse_ids:
                conn.execute(
                    "INSERT OR IGNORE INTO theme_verses (theme_id, verse_id) VALUES (?, ?)",
                    (theme_id, norm_vid),
                )
                mappings_inserted += 1
            else:
                print(f"  [Warning] Verse {raw_vid} (normalised: {norm_vid}) not in verses table — skipped.")
                skipped_mappings += 1

    conn.commit()
    conn.close()

    print(f"[migrate] Inserted {themes_inserted} themes.")
    print(f"[migrate] Inserted {mappings_inserted} theme-verse mappings ({skipped_mappings} skipped).")
    print(f"[migrate] Database saved to {DB_PATH}")


if __name__ == "__main__":
    migrate()
