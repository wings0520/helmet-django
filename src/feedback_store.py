"""Persistent local feedback in SQLite; one connection per operation."""
from contextlib import closing
from pathlib import Path
from datetime import datetime, timezone
import json
import sqlite3


def init_db(db_path):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=15)) as conn, conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            submission_id TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            user_name TEXT NOT NULL,
            rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
            comment TEXT NOT NULL,
            image_name TEXT,
            image_sha256 TEXT,
            result_json TEXT
        )''')


def save_feedback(db_path, submission_id, user_name, rating, comment,
                  image_name=None, image_sha256=None, result=None):
    comment = comment.strip()
    user_name = user_name.strip()
    if not comment:
        raise ValueError('Vui lòng nhập nội dung feedback.')
    if len(comment) > 5000 or len(user_name) > 100:
        raise ValueError('Tên tối đa 100 ký tự; feedback tối đa 5000 ký tự.')
    if not isinstance(rating, int) or not 1 <= rating <= 5:
        raise ValueError('Đánh giá phải từ 1 đến 5.')
    if not submission_id:
        raise ValueError('Thiếu mã gửi feedback.')
    init_db(db_path)
    with closing(sqlite3.connect(db_path, timeout=15)) as conn, conn:
        cursor = conn.execute('''INSERT INTO feedback
            (submission_id, created_at, user_name, rating, comment,
             image_name, image_sha256, result_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(submission_id) DO NOTHING''',
            (submission_id, datetime.now(timezone.utc).isoformat(), user_name,
             rating, comment, image_name, image_sha256,
             json.dumps(result, ensure_ascii=False) if result is not None else None))
        return cursor.rowcount == 1
