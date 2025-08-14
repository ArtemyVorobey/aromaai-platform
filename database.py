import sqlite3
from datetime import datetime
import uuid

DB_FILE = "history.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id TEXT PRIMARY KEY,
        molecule TEXT,
        predicted_path TEXT,
        confidence REAL,
        timestamp TEXT
    )
    """)
    conn.commit()
    conn.close()

def add_history(molecule: str, predicted_path: str, confidence: float):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "INSERT INTO history VALUES (?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), molecule, predicted_path, confidence, datetime.utcnow().isoformat() + "Z")
    )
    conn.commit()
    conn.close()

def get_history(limit: int = None):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    query = "SELECT id, molecule, predicted_path, confidence, timestamp FROM history ORDER BY rowid DESC"
    if limit:
        query += f" LIMIT {limit}"
    c.execute(query)
    rows = c.fetchall()
    conn.close()
    return [dict(zip(["id","molecule","predicted_path","confidence","timestamp"], row)) for row in rows]

def clear_history():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM history")
    conn.commit()
    conn.close()
