from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from .config import DB_PATH, LIBRARY_DIR, ensure_dirs


@dataclass
class Document:
    doc_id: str
    title: str
    source_type: str
    source: str
    content_path: str
    added_at: str
    tags: list[str]


@dataclass
class Interest:
    interest_id: str
    topic: str
    notes: str
    created_at: str


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source TEXT NOT NULL,
    content_path TEXT NOT NULL,
    added_at TEXT NOT NULL,
    tags_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS interests (
    interest_id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    notes TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_document(
    title: str,
    source_type: str,
    source: str,
    content_text: str,
    tags: Optional[Iterable[str]] = None,
) -> Document:
    init_db()
    doc_id = str(uuid.uuid4())
    tags_list = list(tags or [])
    content_path = str(LIBRARY_DIR / f"{doc_id}.txt")
    added_at = _now_iso()

    Path(content_path).write_text(content_text, encoding="utf-8")

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO documents
                (doc_id, title, source_type, source, content_path, added_at, tags_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (doc_id, title, source_type, source, content_path, added_at, json.dumps(tags_list)),
        )

    return Document(
        doc_id=doc_id,
        title=title,
        source_type=source_type,
        source=source,
        content_path=content_path,
        added_at=added_at,
        tags=tags_list,
    )


def list_documents(limit: int = 200) -> list[Document]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT doc_id, title, source_type, source, content_path, added_at, tags_json
            FROM documents
            ORDER BY added_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    docs = []
    for row in rows:
        docs.append(
            Document(
                doc_id=row["doc_id"],
                title=row["title"],
                source_type=row["source_type"],
                source=row["source"],
                content_path=row["content_path"],
                added_at=row["added_at"],
                tags=json.loads(row["tags_json"]),
            )
        )
    return docs


def load_document_text(doc: Document) -> str:
    path = Path(doc.content_path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def add_interest(topic: str, notes: str = "") -> Interest:
    init_db()
    interest_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO interests (interest_id, topic, notes, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (interest_id, topic, notes, _now_iso()),
        )
    return Interest(interest_id=interest_id, topic=topic, notes=notes, created_at=_now_iso())


def list_interests(limit: int = 50) -> list[Interest]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT interest_id, topic, notes, created_at
            FROM interests
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    interests = []
    for row in rows:
        interests.append(
            Interest(
                interest_id=row["interest_id"],
                topic=row["topic"],
                notes=row["notes"],
                created_at=row["created_at"],
            )
        )
    return interests


def delete_interest(interest_id: str) -> int:
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM interests WHERE interest_id = ?",
            (interest_id,),
        )
    return int(cur.rowcount or 0)


def clear_interests() -> int:
    init_db()
    with _connect() as conn:
        cur = conn.execute("DELETE FROM interests")
    return int(cur.rowcount or 0)
