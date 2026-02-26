from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .config import DB_PATH, LIBRARY_DIR, ensure_dirs


_HASH_BACKFILL_DONE = False


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


@dataclass
class Method:
    method_id: str
    method: str
    notes: str
    created_at: str


@dataclass
class DocumentStats:
    doc_id: str
    relevance_score: float
    last_used_at: str | None
    last_seen_at: str | None
    archived: bool


@dataclass
class DecayResult:
    updated: int
    archived: int


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source TEXT NOT NULL,
    content_path TEXT NOT NULL,
    content_hash TEXT,
    added_at TEXT NOT NULL,
    tags_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS document_sources (
    doc_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source TEXT NOT NULL,
    source_canonical TEXT NOT NULL,
    added_at TEXT NOT NULL,
    PRIMARY KEY (doc_id, source)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_document_sources_source
    ON document_sources(source);

CREATE INDEX IF NOT EXISTS idx_document_sources_doc_id
    ON document_sources(doc_id);

CREATE INDEX IF NOT EXISTS idx_document_sources_canonical
    ON document_sources(source_canonical);

CREATE TABLE IF NOT EXISTS interests (
    interest_id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    notes TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS methods (
    method_id TEXT PRIMARY KEY,
    method TEXT NOT NULL,
    notes TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS document_stats (
    doc_id TEXT PRIMARY KEY,
    relevance_score REAL NOT NULL,
    last_used_at TEXT,
    last_seen_at TEXT,
    last_used_run INTEGER,
    last_seen_run INTEGER,
    archived INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS graph_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kg_entities (
    entity_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    type TEXT NOT NULL,
    aliases_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    first_seen_at TEXT,
    last_seen_at TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_kg_entities_user_canonical
    ON kg_entities(user_id, canonical_name);

CREATE INDEX IF NOT EXISTS idx_kg_entities_type
    ON kg_entities(type);

CREATE TABLE IF NOT EXISTS kg_users (
    user_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kg_relations (
    relation_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_id TEXT NOT NULL,
    claim_id TEXT,
    confidence REAL NOT NULL,
    start_date TEXT,
    end_date TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_kg_relations_subject
    ON kg_relations(user_id, subject_id);

CREATE INDEX IF NOT EXISTS idx_kg_relations_object
    ON kg_relations(user_id, object_id);

CREATE INDEX IF NOT EXISTS idx_kg_relations_predicate
    ON kg_relations(user_id, predicate);

CREATE INDEX IF NOT EXISTS idx_kg_relations_time
    ON kg_relations(start_date, end_date);

CREATE INDEX IF NOT EXISTS idx_kg_relations_claim
    ON kg_relations(claim_id);

CREATE TABLE IF NOT EXISTS kg_claims (
    claim_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    topic TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    first_seen_at TEXT,
    last_seen_at TEXT,
    confidence REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_kg_claims_user_norm
    ON kg_claims(user_id, normalized_text);

CREATE TABLE IF NOT EXISTS kg_mentions (
    mention_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    claim_id TEXT,
    span_start INTEGER,
    span_end INTEGER,
    sentence TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_kg_mentions_doc
    ON kg_mentions(doc_id);

CREATE INDEX IF NOT EXISTS idx_kg_mentions_entity
    ON kg_mentions(entity_id);

CREATE INDEX IF NOT EXISTS idx_kg_mentions_claim
    ON kg_mentions(claim_id);

CREATE TABLE IF NOT EXISTS kg_evidence (
    evidence_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    relation_id TEXT NOT NULL,
    claim_id TEXT,
    quote TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_kg_evidence_doc
    ON kg_evidence(doc_id);

CREATE INDEX IF NOT EXISTS idx_kg_evidence_relation
    ON kg_evidence(relation_id);

CREATE INDEX IF NOT EXISTS idx_kg_evidence_claim
    ON kg_evidence(claim_id);

CREATE TABLE IF NOT EXISTS kg_user_profile (
    profile_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    salience REAL NOT NULL,
    start_date TEXT,
    end_date TEXT,
    source_doc_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_kg_profile_entity
    ON kg_user_profile(entity_id);

CREATE TABLE IF NOT EXISTS kg_contradictions (
    contradiction_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    claim_id_a TEXT NOT NULL,
    claim_id_b TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_kg_contradictions_claims
    ON kg_contradictions(claim_id_a, claim_id_b);

CREATE TABLE IF NOT EXISTS kg_doc_state (
    doc_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    extracted_at TEXT NOT NULL,
    status TEXT NOT NULL
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
    _ensure_documents_schema()
    _ensure_document_stats_schema()
    _ensure_kg_schema()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_TRACKING_QUERY_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "gclid",
    "fbclid",
    "yclid",
    "mc_cid",
    "mc_eid",
}


def _canonicalize_url(url: str) -> str:
    text = (url or "").strip()
    if not text:
        return ""
    try:
        parsed = urlparse(text)
    except ValueError:
        return text
    if not parsed.scheme or not parsed.netloc:
        return text
    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower() if parsed.hostname else parsed.netloc.lower()
    port = parsed.port
    if port:
        default = (scheme == "https" and port == 443) or (scheme == "http" and port == 80)
        if not default:
            host = f"{host}:{port}"
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=False)
        if key.lower() not in _TRACKING_QUERY_KEYS
    ]
    query = urlencode(sorted(query_pairs))
    return urlunparse((scheme, host, path, "", query, ""))


def _canonicalize_source(source_type: str, source: str) -> str:
    text = (source or "").strip()
    if not text:
        return ""
    if source_type in {"web", "url"}:
        canonical = _canonicalize_url(text)
        return canonical or text
    return text


def _normalize_text_for_hash(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in cleaned.split("\n")]
    lines = [line for line in lines if line]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def compute_content_hash(text: str) -> str:
    normalized = _normalize_text_for_hash(text)
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def add_document(
    title: str,
    source_type: str,
    source: str,
    content_text: str,
    tags: Optional[Iterable[str]] = None,
    content_hash: str | None = None,
) -> Document:
    init_db()
    _ensure_documents_schema()
    doc_id = str(uuid.uuid4())
    tags_list = list(tags or [])
    content_path = str(LIBRARY_DIR / f"{doc_id}.txt")
    added_at = _now_iso()

    Path(content_path).write_text(content_text, encoding="utf-8")
    if content_hash is None:
        content_hash = compute_content_hash(content_text)
    source_canonical = _canonicalize_source(source_type, source)
    run_id = get_current_run_id()

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO documents
                (doc_id, title, source_type, source, content_path, content_hash, added_at, tags_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                title,
                source_type,
                source,
                content_path,
                content_hash,
                added_at,
                json.dumps(tags_list),
            ),
        )
        initial_score = 0.6 if source_type == "web" else 1.0
        conn.execute(
            """
            INSERT INTO document_stats (doc_id, relevance_score, last_seen_at, last_seen_run, archived)
            VALUES (?, ?, ?, ?, 0)
            """,
            (doc_id, initial_score, added_at, run_id),
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO document_sources
                (doc_id, source_type, source, source_canonical, added_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (doc_id, source_type, source, source_canonical, added_at),
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


def _fetch_document_by_id(conn: sqlite3.Connection, doc_id: str) -> Document | None:
    row = conn.execute(
        """
        SELECT doc_id, title, source_type, source, content_path, added_at, tags_json
        FROM documents
        WHERE doc_id = ?
        """,
        (doc_id,),
    ).fetchone()
    if not row:
        return None
    return Document(
        doc_id=row["doc_id"],
        title=row["title"],
        source_type=row["source_type"],
        source=row["source"],
        content_path=row["content_path"],
        added_at=row["added_at"],
        tags=json.loads(row["tags_json"]),
    )


def _find_doc_id_by_source(conn: sqlite3.Connection, source: str, source_canonical: str) -> str | None:
    row = conn.execute(
        "SELECT doc_id FROM documents WHERE source = ? LIMIT 1",
        (source,),
    ).fetchone()
    if row:
        return row["doc_id"]
    row = conn.execute(
        "SELECT doc_id FROM document_sources WHERE source = ? LIMIT 1",
        (source,),
    ).fetchone()
    if row:
        return row["doc_id"]
    if source_canonical:
        row = conn.execute(
            "SELECT doc_id FROM document_sources WHERE source_canonical = ? LIMIT 1",
            (source_canonical,),
        ).fetchone()
        if row:
            return row["doc_id"]
    return None


def _find_doc_id_by_hash(conn: sqlite3.Connection, content_hash: str) -> str | None:
    if not content_hash:
        return None
    row = conn.execute(
        "SELECT doc_id FROM documents WHERE content_hash = ? LIMIT 1",
        (content_hash,),
    ).fetchone()
    if row:
        return row["doc_id"]
    return None


def _add_document_source(
    conn: sqlite3.Connection,
    doc_id: str,
    source_type: str,
    source: str,
    source_canonical: str,
    added_at: str | None = None,
) -> None:
    if not source or not doc_id:
        return
    timestamp = added_at or _now_iso()
    conn.execute(
        """
        INSERT OR IGNORE INTO document_sources
            (doc_id, source_type, source, source_canonical, added_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (doc_id, source_type, source, source_canonical, timestamp),
    )


def add_document_dedup(
    title: str,
    source_type: str,
    source: str,
    content_text: str,
    tags: Optional[Iterable[str]] = None,
) -> tuple[Document, bool, str | None]:
    init_db()
    _ensure_documents_schema()
    _ensure_document_hashes()
    source_canonical = _canonicalize_source(source_type, source)
    with _connect() as conn:
        doc_id = _find_doc_id_by_source(conn, source, source_canonical)
        if doc_id:
            _add_document_source(conn, doc_id, source_type, source, source_canonical)
            existing = _fetch_document_by_id(conn, doc_id)
            if existing:
                return existing, False, "source"
    content_hash = compute_content_hash(content_text)
    if content_hash:
        with _connect() as conn:
            doc_id = _find_doc_id_by_hash(conn, content_hash)
            if doc_id:
                _add_document_source(conn, doc_id, source_type, source, source_canonical)
                existing = _fetch_document_by_id(conn, doc_id)
                if existing:
                    return existing, False, "content_hash"
    doc = add_document(
        title=title,
        source_type=source_type,
        source=source,
        content_text=content_text,
        tags=tags,
        content_hash=content_hash,
    )
    return doc, True, None


def list_documents(limit: int = 200) -> list[Document]:
    init_db()
    _ensure_document_stats()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT d.doc_id, d.title, d.source_type, d.source, d.content_path, d.added_at, d.tags_json
            FROM documents d
            JOIN document_stats s ON d.doc_id = s.doc_id
            WHERE s.archived = 0
            ORDER BY d.added_at DESC
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


def list_documents_by_source(
    source_type: str,
    limit: int | None = None,
    include_archived: bool = False,
) -> list[Document]:
    init_db()
    _ensure_document_stats()
    archived_clause = "" if include_archived else "AND s.archived = 0"
    limit_clause = "" if limit is None else "LIMIT ?"
    params: list[object] = [source_type]
    if limit is not None:
        params.append(limit)
    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT d.doc_id, d.title, d.source_type, d.source, d.content_path, d.added_at, d.tags_json
            FROM documents d
            JOIN document_stats s ON d.doc_id = s.doc_id
            WHERE d.source_type = ?
              {archived_clause}
            ORDER BY d.added_at DESC
            {limit_clause}
            """,
            tuple(params),
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


def _ensure_document_stats() -> None:
    _ensure_document_stats_schema()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO document_stats (doc_id, relevance_score, last_seen_at, archived)
            SELECT d.doc_id,
                   CASE WHEN d.source_type = 'web' THEN 0.6 ELSE 1.0 END,
                   d.added_at,
                   0
            FROM documents d
            WHERE d.doc_id NOT IN (SELECT doc_id FROM document_stats)
            """
        )
        conn.execute(
            """
            UPDATE document_stats
            SET last_seen_run = COALESCE(last_seen_run, 0),
                last_used_run = COALESCE(last_used_run, 0)
            """
        )


def _ensure_document_stats_schema() -> None:
    with _connect() as conn:
        rows = conn.execute("PRAGMA table_info(document_stats)").fetchall()
        cols = {row["name"] for row in rows}
        if "last_used_run" not in cols:
            conn.execute("ALTER TABLE document_stats ADD COLUMN last_used_run INTEGER")
        if "last_seen_run" not in cols:
            conn.execute("ALTER TABLE document_stats ADD COLUMN last_seen_run INTEGER")


def _ensure_documents_schema() -> None:
    with _connect() as conn:
        rows = conn.execute("PRAGMA table_info(documents)").fetchall()
        cols = {row["name"] for row in rows}
        if "content_hash" not in cols:
            conn.execute("ALTER TABLE documents ADD COLUMN content_hash TEXT")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_sources (
                doc_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source TEXT NOT NULL,
                source_canonical TEXT NOT NULL,
                added_at TEXT NOT NULL,
                PRIMARY KEY (doc_id, source)
            )
            """
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_document_sources_source ON document_sources(source)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_document_sources_doc_id ON document_sources(doc_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_document_sources_canonical ON document_sources(source_canonical)"
        )
        rows = conn.execute(
            "SELECT doc_id, source_type, source, added_at FROM documents"
        ).fetchall()
        for row in rows:
            canonical = _canonicalize_source(row["source_type"], row["source"])
            conn.execute(
                """
                INSERT OR IGNORE INTO document_sources
                    (doc_id, source_type, source, source_canonical, added_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (row["doc_id"], row["source_type"], row["source"], canonical, row["added_at"]),
            )


def _ensure_document_hashes() -> None:
    global _HASH_BACKFILL_DONE
    if _HASH_BACKFILL_DONE:
        return
    with _connect() as conn:
        rows = conn.execute(
            "SELECT doc_id, content_path FROM documents WHERE content_hash IS NULL OR content_hash = ''"
        ).fetchall()
    if not rows:
        _HASH_BACKFILL_DONE = True
        return
    for row in rows:
        try:
            text = Path(row["content_path"]).read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        digest = compute_content_hash(text)
        if not digest:
            continue
        with _connect() as conn:
            conn.execute(
                "UPDATE documents SET content_hash = ? WHERE doc_id = ?",
                (digest, row["doc_id"]),
            )
    _HASH_BACKFILL_DONE = True


def _ensure_kg_schema() -> None:
    with _connect() as conn:
        rows = conn.execute("PRAGMA table_info(kg_users)").fetchall()
        if not rows:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS kg_users (user_id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL)"
            )
        rows = conn.execute("PRAGMA table_info(kg_relations)").fetchall()
        cols = {row["name"] for row in rows}
        if rows and "claim_id" not in cols:
            conn.execute("ALTER TABLE kg_relations ADD COLUMN claim_id TEXT")
        rows = conn.execute("PRAGMA table_info(kg_mentions)").fetchall()
        cols = {row["name"] for row in rows}
        if rows and "claim_id" not in cols:
            conn.execute("ALTER TABLE kg_mentions ADD COLUMN claim_id TEXT")
        rows = conn.execute("PRAGMA table_info(kg_evidence)").fetchall()
        cols = {row["name"] for row in rows}
        if rows and "claim_id" not in cols:
            conn.execute("ALTER TABLE kg_evidence ADD COLUMN claim_id TEXT")
        rows = conn.execute("PRAGMA table_info(kg_user_profile)").fetchall()
        if rows:
            cols = {row["name"] for row in rows}
            needs_rebuild = "profile_type" in cols
            if not needs_rebuild:
                dup = conn.execute(
                    """
                    SELECT 1
                    FROM kg_user_profile
                    GROUP BY user_id, entity_id
                    HAVING COUNT(*) > 1
                    LIMIT 1
                    """
                ).fetchone()
                needs_rebuild = bool(dup)

            if needs_rebuild:
                existing = conn.execute(
                    """
                    SELECT profile_id, user_id, entity_id, salience, start_date, end_date,
                           source_doc_id, created_at, updated_at
                    FROM kg_user_profile
                    """
                ).fetchall()
                merged: dict[tuple[str, str], sqlite3.Row] = {}
                for row in existing:
                    key = (row["user_id"], row["entity_id"])
                    current = merged.get(key)
                    if not current:
                        merged[key] = row
                        continue
                    current_salience = float(current["salience"] or 0.0)
                    candidate_salience = float(row["salience"] or 0.0)
                    if candidate_salience > current_salience:
                        merged[key] = row
                        continue
                    if candidate_salience == current_salience:
                        if (row["updated_at"] or "") > (current["updated_at"] or ""):
                            merged[key] = row

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kg_user_profile_new (
                        profile_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        entity_id TEXT NOT NULL,
                        salience REAL NOT NULL,
                        start_date TEXT,
                        end_date TEXT,
                        source_doc_id TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    """
                )
                for row in merged.values():
                    conn.execute(
                        """
                        INSERT INTO kg_user_profile_new
                            (profile_id, user_id, entity_id, salience, start_date, end_date,
                             source_doc_id, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            row["profile_id"],
                            row["user_id"],
                            row["entity_id"],
                            row["salience"],
                            row["start_date"],
                            row["end_date"],
                            row["source_doc_id"],
                            row["created_at"],
                            row["updated_at"],
                        ),
                    )
                conn.execute("DROP TABLE kg_user_profile")
                conn.execute("ALTER TABLE kg_user_profile_new RENAME TO kg_user_profile")

            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_kg_profile_user_entity
                    ON kg_user_profile(user_id, entity_id);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_kg_profile_entity
                    ON kg_user_profile(entity_id);
                """
            )


def mark_documents_seen(doc_ids: Iterable[str]) -> None:
    init_db()
    _ensure_document_stats()
    now = _now_iso()
    run_id = get_current_run_id()
    with _connect() as conn:
        for doc_id in doc_ids:
            conn.execute(
                """
                UPDATE document_stats
                SET last_seen_at = ?, last_seen_run = ?
                WHERE doc_id = ?
                """,
                (now, run_id, doc_id),
            )


def mark_documents_used(doc_ids: Iterable[str], boost: float = 0.2) -> None:
    init_db()
    _ensure_document_stats()
    now = _now_iso()
    run_id = get_current_run_id()
    with _connect() as conn:
        for doc_id in doc_ids:
            row = conn.execute(
                "SELECT relevance_score FROM document_stats WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            score = float(row["relevance_score"]) if row else 0.0
            new_score = min(1.0, score + boost)
            conn.execute(
                """
                UPDATE document_stats
                SET relevance_score = ?, last_used_at = ?, last_used_run = ?
                WHERE doc_id = ?
                """,
                (new_score, now, run_id, doc_id),
            )


def update_document_stats(
    doc_id: str,
    relevance_score: float | None = None,
    last_seen_at: str | None = None,
    archived: int | None = None,
    last_seen_run: int | None = None,
    last_used_run: int | None = None,
) -> None:
    init_db()
    _ensure_document_stats()
    updates = []
    params: list[object] = []
    if relevance_score is not None:
        updates.append("relevance_score = ?")
        params.append(relevance_score)
    if last_seen_at is not None:
        updates.append("last_seen_at = ?")
        params.append(last_seen_at)
    if archived is not None:
        updates.append("archived = ?")
        params.append(archived)
    if last_seen_run is not None:
        updates.append("last_seen_run = ?")
        params.append(last_seen_run)
    if last_used_run is not None:
        updates.append("last_used_run = ?")
        params.append(last_used_run)
    if not updates:
        return
    params.append(doc_id)
    with _connect() as conn:
        conn.execute(
            f"UPDATE document_stats SET {', '.join(updates)} WHERE doc_id = ?",
            tuple(params),
        )


def decay_web_documents(
    decay_per_run: float,
    archive_threshold: float,
    archive_runs_unused: int,
) -> DecayResult:
    init_db()
    _ensure_document_stats()
    current_run = get_current_run_id()
    updated = 0
    archived = 0
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT d.doc_id, d.added_at, d.source_type,
                   s.relevance_score, s.last_used_run, s.last_seen_run, s.archived
            FROM documents d
            JOIN document_stats s ON d.doc_id = s.doc_id
            WHERE d.source_type = 'web' AND s.archived = 0
            """
        ).fetchall()

        for row in rows:
            score = float(row["relevance_score"] or 0.0)
            last_used_run = int(row["last_used_run"] or 0)
            last_seen_run = int(row["last_seen_run"] or 0)
            anchor_run = last_used_run or last_seen_run or current_run
            runs_since = max(0, current_run - anchor_run)
            decayed = score * (decay_per_run ** runs_since)
            archive = 0
            if decayed < archive_threshold:
                if runs_since >= archive_runs_unused:
                    archive = 1
            if archive:
                archived += 1
            conn.execute(
                """
                UPDATE document_stats
                SET relevance_score = ?, archived = ?
                WHERE doc_id = ?
                """,
                (decayed, archive, row["doc_id"]),
            )
            updated += 1
    return DecayResult(updated=updated, archived=archived)


def document_exists(source: str) -> bool:
    init_db()
    _ensure_documents_schema()
    source_canonical = _canonicalize_source("url", source)
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT 1 FROM documents WHERE source = ? LIMIT 1
            """,
            (source,),
        ).fetchone()
        if row is not None:
            return True
        row = conn.execute(
            """
            SELECT 1 FROM document_sources WHERE source = ? LIMIT 1
            """,
            (source,),
        ).fetchone()
        if row is not None:
            return True
        if source_canonical:
            row = conn.execute(
                """
                SELECT 1 FROM document_sources WHERE source_canonical = ? LIMIT 1
                """,
                (source_canonical,),
            ).fetchone()
    return row is not None


def get_documents_by_ids(doc_ids: Iterable[str]) -> list[Document]:
    init_db()
    _ensure_document_stats()
    doc_ids = [doc_id for doc_id in doc_ids if doc_id]
    if not doc_ids:
        return []
    placeholders = ",".join("?" for _ in doc_ids)
    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT d.doc_id, d.title, d.source_type, d.source, d.content_path, d.added_at, d.tags_json
            FROM documents d
            JOIN document_stats s ON d.doc_id = s.doc_id
            WHERE s.archived = 0 AND d.doc_id IN ({placeholders})
            """,
            tuple(doc_ids),
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
    # Preserve input ordering when possible
    order = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    docs.sort(key=lambda doc: order.get(doc.doc_id, len(order)))
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


def add_method(method: str, notes: str = "") -> Method:
    init_db()
    method_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO methods (method_id, method, notes, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (method_id, method, notes, _now_iso()),
        )
    return Method(method_id=method_id, method=method, notes=notes, created_at=_now_iso())


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


def list_methods(limit: int = 50) -> list[Method]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT method_id, method, notes, created_at
            FROM methods
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    methods = []
    for row in rows:
        methods.append(
            Method(
                method_id=row["method_id"],
                method=row["method"],
                notes=row["notes"],
                created_at=row["created_at"],
            )
        )
    return methods


def delete_interest(interest_id: str) -> int:
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM interests WHERE interest_id = ?",
            (interest_id,),
        )
    return int(cur.rowcount or 0)


def delete_method(method_id: str) -> int:
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM methods WHERE method_id = ?",
            (method_id,),
        )
    return int(cur.rowcount or 0)


def clear_interests() -> int:
    init_db()
    with _connect() as conn:
        cur = conn.execute("DELETE FROM interests")
    return int(cur.rowcount or 0)


def clear_methods() -> int:
    init_db()
    with _connect() as conn:
        cur = conn.execute("DELETE FROM methods")
    return int(cur.rowcount or 0)


def clear_library_and_graph(clear_files: bool = True) -> dict[str, int]:
    init_db()
    removed_files = 0
    if clear_files:
        for path in LIBRARY_DIR.glob("*"):
            if path.is_file():
                path.unlink()
                removed_files += 1
    with _connect() as conn:
        graph_meta = conn.execute("DELETE FROM graph_meta").rowcount
        kg_entities = conn.execute("DELETE FROM kg_entities").rowcount
        kg_relations = conn.execute("DELETE FROM kg_relations").rowcount
        kg_claims = conn.execute("DELETE FROM kg_claims").rowcount
        kg_mentions = conn.execute("DELETE FROM kg_mentions").rowcount
        kg_evidence = conn.execute("DELETE FROM kg_evidence").rowcount
        kg_profiles = conn.execute("DELETE FROM kg_user_profile").rowcount
        kg_contra = conn.execute("DELETE FROM kg_contradictions").rowcount
        kg_doc_state = conn.execute("DELETE FROM kg_doc_state").rowcount
        stats = conn.execute("DELETE FROM document_stats").rowcount
        docs = conn.execute("DELETE FROM documents").rowcount
    return {
        "documents": int(docs or 0),
        "document_stats": int(stats or 0),
        "graph_meta": int(graph_meta or 0),
        "kg_entities": int(kg_entities or 0),
        "kg_relations": int(kg_relations or 0),
        "kg_claims": int(kg_claims or 0),
        "kg_mentions": int(kg_mentions or 0),
        "kg_evidence": int(kg_evidence or 0),
        "kg_profiles": int(kg_profiles or 0),
        "kg_contradictions": int(kg_contra or 0),
        "kg_doc_state": int(kg_doc_state or 0),
        "files_removed": removed_files,
    }


def get_meta(key: str) -> str | None:
    init_db()
    with _connect() as conn:
        row = conn.execute("SELECT value FROM graph_meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(key: str, value: str) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO graph_meta (key, value) VALUES (?, ?)",
            (key, value),
        )


def get_current_run_id() -> int:
    value = get_meta("research_run_count")
    try:
        return int(value) if value is not None else 0
    except Exception:
        return 0


def increment_research_run() -> int:
    current = get_current_run_id()
    current += 1
    set_meta("research_run_count", str(current))
    return current
