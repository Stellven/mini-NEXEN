from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Callable
from datetime import datetime, timezone
from typing import Iterable

from . import db
from .llm import LLMClient, log_task_event
from .text_utils import tokenize


DEFAULT_USER_ID = "default"

PREDICATE_CANONICAL = {
    "is a": "is_a",
    "is an": "is_a",
    "is": "is_a",
    "are": "is_a",
    "type of": "is_a",
    "part of": "part_of",
    "subset of": "part_of",
    "uses": "uses",
    "use": "uses",
    "utilizes": "uses",
    "employs": "uses",
    "affects": "affects",
    "impact": "affects",
    "impacts": "affects",
    "influences": "affects",
    "causes": "causes",
    "leads to": "causes",
    "enables": "enables",
    "supports": "enables",
    "requires": "requires",
    "depends on": "requires",
    "improves": "improves",
    "reduces": "reduces",
    "limits": "reduces",
    "compares to": "compares_to",
    "similar to": "related_to",
    "related to": "related_to",
}

PREDICATE_CHOICES = sorted(set(PREDICATE_CANONICAL.values()))
PROFILE_TYPE = "profile signal"
ORG_TOKENS = {"inc", "corp", "co", "company", "ltd", "llc", "gmbh", "plc", "ag"}
LAB_TOKENS = {"lab", "labs", "laboratory", "laboratories"}
UNIVERSITY_TOKENS = {"university", "college", "institute", "school", "polytechnic"}


def _infer_entity_type(name: str) -> str:
    text = (name or "").strip()
    lowered = text.casefold()
    if not lowered:
        return "Thing"
    if any(token in lowered for token in UNIVERSITY_TOKENS):
        return "University"
    if any(token in lowered for token in LAB_TOKENS):
        return "Lab"
    if any(token in lowered.split() for token in ORG_TOKENS):
        return "Company"
    if re.match(r"^[A-Z][a-z]+\\s+[A-Z][a-z]+", text):
        return "Person"
    return "Thing"


@dataclass
class KGEntity:
    entity_id: str
    name: str
    canonical_name: str
    type: str


@dataclass
class KGRelation:
    relation_id: str
    subject_id: str
    predicate: str
    object_id: str
    claim_id: str | None
    confidence: float


@dataclass
class KGEvidence:
    evidence_id: str
    relation_id: str
    doc_id: str
    claim_id: str | None
    quote: str
    confidence: float


@dataclass
class KGSubgraph:
    entities: list[KGEntity]
    relations: list[KGRelation]
    evidence: list[KGEvidence]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonicalize_name(name: str) -> str:
    text = (name or "").strip().casefold()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s\-_/]", "", text)
    return text.strip()


def _normalize_claim_text(text: str) -> str:
    cleaned = " ".join((text or "").split())
    cleaned = cleaned.strip().casefold()
    cleaned = re.sub(r"[^a-z0-9\s\-_,.;:]", "", cleaned)
    return cleaned.strip()


def _normalize_predicate(text: str) -> str:
    cleaned = (text or "").strip().casefold()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return "related_to"
    for key, canonical in PREDICATE_CANONICAL.items():
        if cleaned == key:
            return canonical
    if cleaned in PREDICATE_CHOICES:
        return cleaned
    return "related_to"


def _extract_json_array(text: str) -> list[dict]:
    if not text:
        return []
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    candidate = text[start : end + 1]
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def _extract_json_object(text: str) -> dict:
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    candidate = text[start : end + 1]
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _triple_prompt(text: str) -> str:
    return json.dumps(
        {
            "instructions": {
                "output_format": "JSON array only",
                "schema": [
                    {
                        "subject": "string",
                        "predicate": "one of: " + ", ".join(PREDICATE_CHOICES),
                        "object": "string",
                        "claim": "natural language statement",
                        "evidence": "short quote or sentence",
                        "confidence": "0-1 float",
                    }
                ],
                "rules": [
                    "Use short, canonical entity names.",
                    "Prefer predicates from the list; if unsure use related_to.",
                    "Provide evidence for each triple. If you cannot quote evidence, omit the triple.",
                    "Return 3-12 triples maximum.",
                ],
            },
            "text": text[:4000],
        },
        indent=2,
    )


def extract_triples(llm: LLMClient | None, text: str) -> list[dict]:
    if not llm or not text.strip():
        return []
    prompt = _triple_prompt(text)
    response = llm.generate(
        system_prompt="You extract knowledge graph triples.",
        user_prompt=prompt,
        task="kg triples",
        agent="KGExtractor",
    )
    triples = _extract_json_array(response or "")
    normalized: list[dict] = []
    for item in triples:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject") or "").strip()
        predicate = str(item.get("predicate") or "").strip()
        obj = str(item.get("object") or "").strip()
        claim_text = str(item.get("claim") or "").strip()
        evidence = str(item.get("evidence") or "").strip()
        if not subject or not obj:
            continue
        try:
            confidence = float(item.get("confidence") or 0.6)
        except (TypeError, ValueError):
            confidence = 0.6
        normalized.append(
            {
                "subject": subject,
                "predicate": _normalize_predicate(predicate),
                "object": obj,
                "claim": claim_text,
                "evidence": evidence,
                "confidence": max(0.0, min(1.0, confidence)),
            }
        )
    return normalized


def _profile_prompt(text: str) -> str:
    return json.dumps(
        {
            "instructions": {
                "output_format": "JSON array only",
                "schema": [
                    {
                        "entity": "string",
                        "salience": "0-1 float",
                        "evidence": "short quote or sentence",
                        "start_date": "YYYY-MM-DD or null",
                        "end_date": "YYYY-MM-DD or null",
                    }
                ],
                "rules": [
                    "Use short, canonical entity names.",
                    "Prefer concrete topics or goals over vague terms.",
                    "Assume the user is a domain expert; prioritize advanced or technical interests over basic concepts.",
                    "Return 3-10 items maximum.",
                ],
            },
            "text": text[:4000],
        },
        indent=2,
    )


def extract_profile_items(llm: LLMClient | None, text: str) -> list[dict]:
    if not llm or not text.strip():
        return []
    prompt = _profile_prompt(text)
    response = llm.generate(
        system_prompt="You extract user profile signals from documents.",
        user_prompt=prompt,
        task="profile extraction",
        agent="Profiler",
    )
    items = _extract_json_array(response or "")
    normalized: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        entity = str(item.get("entity") or "").strip()
        if not entity:
            continue
        evidence = str(item.get("evidence") or "").strip()
        start_date = str(item.get("start_date") or "").strip() or None
        end_date = str(item.get("end_date") or "").strip() or None
        try:
            salience = float(item.get("salience") or 0.6)
        except (TypeError, ValueError):
            salience = 0.6
        normalized.append(
            {
                "entity": entity,
                "salience": max(0.0, min(1.0, salience)),
                "evidence": evidence,
                "start_date": start_date,
                "end_date": end_date,
            }
        )
    return normalized


def detect_contradictions(
    store: KGStore,
    llm: LLMClient | None = None,
    max_pairs: int = 40,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    pairs = store.contradiction_candidates(limit=max_pairs)
    if not pairs:
        return 0
    updates = 0
    total = len(pairs)
    if progress:
        progress(0, total)
    for idx, (claim_a, claim_b) in enumerate(pairs, start=1):
        text_a = store.get_claim_text(claim_a) or ""
        text_b = store.get_claim_text(claim_b) or ""
        if not text_a or not text_b:
            if progress:
                progress(idx, total)
            continue
        if llm:
            prompt = json.dumps(
                {
                    "claim_a": text_a,
                    "claim_b": text_b,
                    "instructions": {
                        "output_format": "{\"contradiction\": true|false, \"confidence\": 0-1}",
                        "definition": "Contradiction means the two claims cannot both be true in the same context.",
                    },
                },
                indent=2,
            )
            response = llm.generate(
                system_prompt="You judge whether two claims contradict.",
                user_prompt=prompt,
                task="contradiction check",
                agent="KGContradiction",
            )
            payload = _extract_json_object(response or "")
            is_contra = bool(payload.get("contradiction"))
            try:
                conf = float(payload.get("confidence") or 0.4)
            except (TypeError, ValueError):
                conf = 0.4
            if is_contra:
                store.attach_contradiction(claim_a, claim_b, confidence=conf)
                updates += 1
        else:
            store.attach_contradiction(claim_a, claim_b, confidence=0.2)
            updates += 1
        if progress:
            progress(idx, total)
    if progress:
        progress(total, total)
    return updates


class KGStore:
    def __init__(self, user_id: str = DEFAULT_USER_ID):
        self.user_id = user_id
        db.init_db()
        self._ensure_user()

    def _ensure_user(self) -> None:
        now = _now_iso()
        with db._connect() as conn:
            row = conn.execute(
                "SELECT user_id FROM kg_users WHERE user_id = ?",
                (self.user_id,),
            ).fetchone()
            if row:
                return
            conn.execute(
                "INSERT INTO kg_users (user_id, name, created_at) VALUES (?, ?, ?)",
                (self.user_id, self.user_id, now),
            )

    def upsert_entity(self, name: str, type_name: str = "Thing", aliases: Iterable[str] | None = None) -> str:
        name = (name or "").strip()
        if not name:
            raise ValueError("Entity name is required.")
        canonical = _canonicalize_name(name)
        if not canonical:
            canonical = name.casefold()
        inferred_type = type_name or "Thing"
        if inferred_type == "Thing":
            inferred_type = _infer_entity_type(name)
        aliases_list = sorted({alias.strip() for alias in (aliases or []) if alias and alias.strip()})
        now = _now_iso()
        with db._connect() as conn:
            row = conn.execute(
                """
                SELECT entity_id FROM kg_entities
                WHERE user_id = ? AND canonical_name = ?
                """,
                (self.user_id, canonical),
            ).fetchone()
            if row:
                conn.execute(
                    """
                    UPDATE kg_entities
                    SET updated_at = ?, last_seen_at = ?
                    WHERE entity_id = ?
                    """,
                    (now, now, row["entity_id"]),
                )
                return str(row["entity_id"])
            entity_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO kg_entities
                    (entity_id, user_id, name, canonical_name, type, aliases_json,
                     created_at, updated_at, first_seen_at, last_seen_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entity_id,
                    self.user_id,
                    name,
                    canonical,
                    inferred_type,
                    json.dumps(aliases_list),
                    now,
                    now,
                    now,
                    now,
                ),
            )
        return entity_id

    def upsert_claim(
        self,
        text: str,
        topic: str | None = None,
        confidence: float = 0.6,
    ) -> str:
        claim_text = " ".join((text or "").split()).strip()
        if not claim_text:
            raise ValueError("Claim text is required.")
        normalized = _normalize_claim_text(claim_text)
        if not normalized:
            normalized = claim_text.casefold()
        now = _now_iso()
        with db._connect() as conn:
            row = conn.execute(
                """
                SELECT claim_id, confidence FROM kg_claims
                WHERE user_id = ? AND normalized_text = ?
                """,
                (self.user_id, normalized),
            ).fetchone()
            if row:
                current_conf = float(row["confidence"] or 0.0)
                new_conf = max(current_conf, max(0.0, min(1.0, confidence)))
                conn.execute(
                    """
                    UPDATE kg_claims
                    SET updated_at = ?, last_seen_at = ?, confidence = ?
                    WHERE claim_id = ?
                    """,
                    (now, now, new_conf, row["claim_id"]),
                )
                return str(row["claim_id"])
            claim_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO kg_claims
                    (claim_id, user_id, text, normalized_text, topic,
                     created_at, updated_at, first_seen_at, last_seen_at, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    claim_id,
                    self.user_id,
                    claim_text,
                    normalized,
                    topic,
                    now,
                    now,
                    now,
                    now,
                    max(0.0, min(1.0, confidence)),
                ),
            )
        return claim_id

    def add_relation(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        claim_id: str | None = None,
        confidence: float = 0.6,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> str:
        relation_id = str(uuid.uuid4())
        now = _now_iso()
        with db._connect() as conn:
            conn.execute(
                """
                INSERT INTO kg_relations
                    (relation_id, user_id, subject_id, predicate, object_id, claim_id,
                     confidence, start_date, end_date, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    relation_id,
                    self.user_id,
                    subject_id,
                    _normalize_predicate(predicate),
                    object_id,
                    claim_id,
                    max(0.0, min(1.0, confidence)),
                    start_date,
                    end_date,
                    now,
                    now,
                ),
            )
        return relation_id

    def add_evidence(
        self,
        doc_id: str,
        relation_id: str,
        quote: str,
        claim_id: str | None = None,
        confidence: float = 0.6,
    ) -> str:
        evidence_id = str(uuid.uuid4())
        now = _now_iso()
        with db._connect() as conn:
            conn.execute(
                """
                INSERT INTO kg_evidence
                    (evidence_id, user_id, doc_id, relation_id, claim_id, quote, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence_id,
                    self.user_id,
                    doc_id,
                    relation_id,
                    claim_id,
                    quote.strip(),
                    max(0.0, min(1.0, confidence)),
                    now,
                ),
            )
        return evidence_id

    def add_mention(
        self,
        doc_id: str,
        entity_id: str,
        claim_id: str | None = None,
        sentence: str | None = None,
        span_start: int | None = None,
        span_end: int | None = None,
    ) -> str:
        mention_id = str(uuid.uuid4())
        now = _now_iso()
        with db._connect() as conn:
            conn.execute(
                """
                INSERT INTO kg_mentions
                    (mention_id, user_id, doc_id, entity_id, claim_id, span_start, span_end, sentence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mention_id,
                    self.user_id,
                    doc_id,
                    entity_id,
                    claim_id,
                    span_start,
                    span_end,
                    sentence,
                    now,
                ),
            )
        return mention_id

    def set_profile_edge(
        self,
        entity_id: str,
        salience: float,
        start_date: str | None = None,
        end_date: str | None = None,
        source_doc_id: str | None = None,
    ) -> str:
        now = _now_iso()
        salience = max(0.0, min(1.0, salience))
        with db._connect() as conn:
            row = conn.execute(
                """
                SELECT profile_id, salience FROM kg_user_profile
                WHERE user_id = ? AND entity_id = ?
                """,
                (self.user_id, entity_id),
            ).fetchone()
            if row:
                current_salience = float(row["salience"] or 0.0)
                if salience < current_salience:
                    salience = current_salience
                conn.execute(
                    """
                    UPDATE kg_user_profile
                    SET salience = ?, start_date = ?, end_date = ?, source_doc_id = ?, updated_at = ?
                    WHERE profile_id = ?
                    """,
                    (
                        salience,
                        start_date,
                        end_date,
                        source_doc_id,
                        now,
                        row["profile_id"],
                    ),
                )
                return str(row["profile_id"])
            profile_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO kg_user_profile
                    (profile_id, user_id, entity_id, salience,
                     start_date, end_date, source_doc_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    profile_id,
                    self.user_id,
                    entity_id,
                    salience,
                    start_date,
                    end_date,
                    source_doc_id,
                    now,
                    now,
                ),
            )
        return profile_id

    def search_entities(self, term: str, limit: int = 10) -> list[KGEntity]:
        term = (term or "").strip()
        if not term:
            return []
        canonical = _canonicalize_name(term)
        canonical_like = f"%{canonical}%" if canonical else ""
        raw_like = f"%{term.casefold()}%"
        with db._connect() as conn:
            rows = conn.execute(
                """
                SELECT entity_id, name, canonical_name, type
                FROM kg_entities
                WHERE user_id = ? AND (canonical_name LIKE ? OR name LIKE ?)
                LIMIT ?
                """,
                (self.user_id, canonical_like or raw_like, raw_like, limit),
            ).fetchall()
        return [
            KGEntity(
                entity_id=row["entity_id"],
                name=row["name"],
                canonical_name=row["canonical_name"],
                type=row["type"],
            )
            for row in rows
        ]

    def seed_entities_from_terms(self, terms: Iterable[str], limit: int = 12) -> list[KGEntity]:
        seeds: list[KGEntity] = []
        seen = set()
        for term in terms:
            if not term:
                continue
            for entity in self.search_entities(term, limit=limit):
                if entity.entity_id in seen:
                    continue
                seeds.append(entity)
                seen.add(entity.entity_id)
                if len(seeds) >= limit:
                    return seeds
        return seeds

    def get_profile_terms(self, limit: int = 5) -> list[str]:
        with db._connect() as conn:
            rows = conn.execute(
                """
                SELECT e.name
                FROM kg_user_profile p
                JOIN kg_entities e ON p.entity_id = e.entity_id
                WHERE p.user_id = ?
                ORDER BY p.salience DESC
                LIMIT ?
                """,
                (self.user_id, limit),
            ).fetchall()
        return [row["name"] for row in rows if row["name"]]

    def clear_profile(self) -> int:
        with db._connect() as conn:
            cur = conn.execute(
                """
                DELETE FROM kg_user_profile
                WHERE user_id = ?
                """,
                (self.user_id,),
            )
        return int(cur.rowcount or 0)

    def get_entities_by_type(self, type_name: str, limit: int = 50) -> list[KGEntity]:
        with db._connect() as conn:
            rows = conn.execute(
                """
                SELECT entity_id, name, canonical_name, type
                FROM kg_entities
                WHERE user_id = ? AND type = ?
                LIMIT ?
                """,
                (self.user_id, type_name, limit),
            ).fetchall()
        return [
            KGEntity(
                entity_id=row["entity_id"],
                name=row["name"],
                canonical_name=row["canonical_name"],
                type=row["type"],
            )
            for row in rows
        ]

    def get_relations_for_entity(
        self,
        entity_id: str,
        hops: int = 1,
        min_confidence: float = 0.0,
        predicates: Iterable[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[KGRelation]:
        if hops <= 1:
            predicate_filter = ""
            params: list[object] = [self.user_id, entity_id, entity_id, min_confidence]
            if predicates:
                predicate_filter = "AND predicate IN ({})".format(
                    ",".join("?" for _ in predicates)
                )
                params.extend(list(predicates))
            date_filter = ""
            if start_date and end_date:
                date_filter = "AND (start_date IS NULL OR start_date >= ?) AND (end_date IS NULL OR end_date <= ?)"
                params.extend([start_date, end_date])
            elif start_date:
                date_filter = "AND (start_date IS NULL OR start_date >= ?)"
                params.append(start_date)
            elif end_date:
                date_filter = "AND (end_date IS NULL OR end_date <= ?)"
                params.append(end_date)
            with db._connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT relation_id, subject_id, predicate, object_id, claim_id, confidence
                    FROM kg_relations
                    WHERE user_id = ?
                      AND (subject_id = ? OR object_id = ?)
                      AND confidence >= ?
                      {predicate_filter}
                      {date_filter}
                    """,
                    tuple(params),
                ).fetchall()
            return [
                KGRelation(
                    relation_id=row["relation_id"],
                    subject_id=row["subject_id"],
                    predicate=row["predicate"],
                    object_id=row["object_id"],
                    claim_id=row["claim_id"],
                    confidence=float(row["confidence"] or 0.0),
                )
                for row in rows
            ]
        sub = self.subgraph([entity_id], hops=hops, min_confidence=min_confidence)
        return sub.relations

    def get_claims_for_entity(
        self,
        entity_id: str,
        min_confidence: float = 0.0,
    ) -> list[dict]:
        with db._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.claim_id, c.text, c.confidence
                FROM kg_claims c
                JOIN kg_relations r ON r.claim_id = c.claim_id
                WHERE c.user_id = ?
                  AND r.user_id = ?
                  AND (r.subject_id = ? OR r.object_id = ?)
                  AND c.confidence >= ?
                """,
                (self.user_id, self.user_id, entity_id, entity_id, min_confidence),
            ).fetchall()
        return [
            {"claim_id": row["claim_id"], "text": row["text"], "confidence": float(row["confidence"] or 0.0)}
            for row in rows
        ]

    def get_profile(
        self,
        limit: int = 50,
    ) -> list[dict]:
        with db._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT p.profile_id, p.salience, p.start_date, p.end_date,
                       p.source_doc_id, e.name, e.entity_id
                FROM kg_user_profile p
                JOIN kg_entities e ON p.entity_id = e.entity_id
                WHERE p.user_id = ?
                ORDER BY p.salience DESC
                LIMIT ?
                """,
                (self.user_id, limit),
            ).fetchall()
        return [
            {
                "profile_id": row["profile_id"],
                "salience": float(row["salience"] or 0.0),
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "source_doc_id": row["source_doc_id"],
                "entity": row["name"],
                "entity_id": row["entity_id"],
            }
            for row in rows
        ]

    def count_contradictions(
        self,
        max_confidence: float | None = None,
        older_than: str | None = None,
    ) -> int:
        params: list[object] = [self.user_id]
        where = ""
        if max_confidence is not None and older_than is not None:
            where = "AND (confidence <= ? OR created_at < ?)"
            params.extend([float(max_confidence), older_than])
        elif max_confidence is not None:
            where = "AND confidence <= ?"
            params.append(float(max_confidence))
        elif older_than is not None:
            where = "AND created_at < ?"
            params.append(older_than)
        with db._connect() as conn:
            row = conn.execute(
                f"""
                SELECT COUNT(*) AS count
                FROM kg_contradictions
                WHERE user_id = ?
                {where}
                """,
                tuple(params),
            ).fetchone()
        return int(row["count"] or 0)

    def get_contradictions_for_claims(
        self,
        claim_ids: Iterable[str],
        limit: int = 200,
    ) -> list[dict]:
        ids = [cid for cid in dict.fromkeys(claim_ids) if cid]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with db._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT claim_id_a, claim_id_b, confidence
                FROM kg_contradictions
                WHERE user_id = ?
                  AND (claim_id_a IN ({placeholders}) OR claim_id_b IN ({placeholders}))
                LIMIT ?
                """,
                (self.user_id, *ids, *ids, limit),
            ).fetchall()
        return [
            {
                "claim_id_a": row["claim_id_a"],
                "claim_id_b": row["claim_id_b"],
                "confidence": float(row["confidence"] or 0.0),
            }
            for row in rows
        ]

    def resolve_alias(self, name: str) -> str | None:
        canonical = _canonicalize_name(name)
        if not canonical:
            return None
        with db._connect() as conn:
            row = conn.execute(
                """
                SELECT canonical_name FROM kg_entities
                WHERE user_id = ? AND canonical_name = ?
                """,
                (self.user_id, canonical),
            ).fetchone()
        return row["canonical_name"] if row else None

    def merge_entities(self, entity_id_a: str, entity_id_b: str) -> None:
        if not entity_id_a or not entity_id_b or entity_id_a == entity_id_b:
            return
        now = _now_iso()
        with db._connect() as conn:
            row_a = conn.execute(
                """
                SELECT profile_id, salience, start_date, end_date, source_doc_id
                FROM kg_user_profile
                WHERE user_id = ? AND entity_id = ?
                """,
                (self.user_id, entity_id_a),
            ).fetchone()
            row_b = conn.execute(
                """
                SELECT profile_id, salience, start_date, end_date, source_doc_id
                FROM kg_user_profile
                WHERE user_id = ? AND entity_id = ?
                """,
                (self.user_id, entity_id_b),
            ).fetchone()
            if row_b:
                if row_a:
                    salience = max(float(row_a["salience"] or 0.0), float(row_b["salience"] or 0.0))
                    start_date = row_a["start_date"] or row_b["start_date"]
                    end_date = row_a["end_date"] or row_b["end_date"]
                    source_doc_id = row_a["source_doc_id"] or row_b["source_doc_id"]
                    conn.execute(
                        """
                        UPDATE kg_user_profile
                        SET salience = ?, start_date = ?, end_date = ?, source_doc_id = ?, updated_at = ?
                        WHERE profile_id = ?
                        """,
                        (salience, start_date, end_date, source_doc_id, now, row_a["profile_id"]),
                    )
                    conn.execute(
                        "DELETE FROM kg_user_profile WHERE profile_id = ?",
                        (row_b["profile_id"],),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE kg_user_profile
                        SET entity_id = ?, updated_at = ?
                        WHERE profile_id = ?
                        """,
                        (entity_id_a, now, row_b["profile_id"]),
                    )
            conn.execute(
                "UPDATE kg_relations SET subject_id = ? WHERE subject_id = ?",
                (entity_id_a, entity_id_b),
            )
            conn.execute(
                "UPDATE kg_relations SET object_id = ? WHERE object_id = ?",
                (entity_id_a, entity_id_b),
            )
            conn.execute(
                "UPDATE kg_mentions SET entity_id = ? WHERE entity_id = ?",
                (entity_id_a, entity_id_b),
            )
            conn.execute("DELETE FROM kg_entities WHERE entity_id = ?", (entity_id_b,))

    def get_entity_evidence(self, entity_id: str, limit: int = 5) -> list[dict]:
        if not entity_id:
            return []
        with db._connect() as conn:
            rows = conn.execute(
                """
                SELECT ev.quote, ev.confidence, ev.doc_id, d.title
                FROM kg_relations r
                JOIN kg_evidence ev ON ev.relation_id = r.relation_id
                LEFT JOIN documents d ON d.doc_id = ev.doc_id
                WHERE r.user_id = ? AND (r.subject_id = ? OR r.object_id = ?)
                ORDER BY ev.confidence DESC
                LIMIT ?
                """,
                (self.user_id, entity_id, entity_id, limit),
            ).fetchall()
        return [
            {
                "quote": row["quote"],
                "confidence": float(row["confidence"] or 0.0),
                "doc_id": row["doc_id"],
                "title": row["title"] or "",
            }
            for row in rows
        ]

    def get_entity_mentions(self, entity_id: str, limit: int = 5) -> list[dict]:
        if not entity_id:
            return []
        with db._connect() as conn:
            rows = conn.execute(
                """
                SELECT m.sentence, m.doc_id, d.title
                FROM kg_mentions m
                LEFT JOIN documents d ON d.doc_id = m.doc_id
                WHERE m.user_id = ? AND m.entity_id = ?
                ORDER BY m.created_at DESC
                LIMIT ?
                """,
                (self.user_id, entity_id, limit),
            ).fetchall()
        return [
            {
                "sentence": row["sentence"],
                "doc_id": row["doc_id"],
                "title": row["title"] or "",
            }
            for row in rows
        ]

    def contradiction_candidates(self, limit: int = 50) -> list[tuple[str, str]]:
        with db._connect() as conn:
            rows = conn.execute(
                """
                SELECT subject_id, predicate, COUNT(DISTINCT object_id) AS obj_count
                FROM kg_relations
                WHERE user_id = ? AND claim_id IS NOT NULL
                GROUP BY subject_id, predicate
                HAVING obj_count > 1
                LIMIT ?
                """,
                (self.user_id, limit),
            ).fetchall()
        pairs: list[tuple[str, str]] = []
        for row in rows:
            subject_id = row["subject_id"]
            predicate = row["predicate"]
            with db._connect() as conn:
                rels = conn.execute(
                    """
                    SELECT DISTINCT claim_id
                    FROM kg_relations
                    WHERE user_id = ? AND subject_id = ? AND predicate = ? AND claim_id IS NOT NULL
                    LIMIT 6
                    """,
                    (self.user_id, subject_id, predicate),
                ).fetchall()
            claim_ids = [r["claim_id"] for r in rels if r["claim_id"]]
            for i in range(len(claim_ids)):
                for j in range(i + 1, len(claim_ids)):
                    pairs.append((claim_ids[i], claim_ids[j]))
        return pairs[:limit]

    def get_claim_text(self, claim_id: str) -> str | None:
        if not claim_id:
            return None
        with db._connect() as conn:
            row = conn.execute(
                "SELECT text FROM kg_claims WHERE user_id = ? AND claim_id = ?",
                (self.user_id, claim_id),
            ).fetchone()
        return row["text"] if row else None

    def subgraph(
        self,
        seed_entity_ids: Iterable[str],
        hops: int = 1,
        min_confidence: float = 0.0,
        limit_edges: int = 200,
    ) -> KGSubgraph:
        seeds = [sid for sid in seed_entity_ids if sid]
        if not seeds:
            return KGSubgraph(entities=[], relations=[], evidence=[])
        frontier = set(seeds)
        visited = set(seeds)
        relations: list[KGRelation] = []
        seen_relations: set[str] = set()
        for _ in range(max(1, hops)):
            if not frontier:
                break
            with db._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT relation_id, subject_id, predicate, object_id, claim_id, confidence
                    FROM kg_relations
                    WHERE user_id = ?
                      AND confidence >= ?
                      AND (subject_id IN ({}) OR object_id IN ({}))
                    LIMIT ?
                    """.format(
                        ",".join("?" for _ in frontier), ",".join("?" for _ in frontier)
                    ),
                    (self.user_id, min_confidence, *frontier, *frontier, limit_edges),
                ).fetchall()
            new_frontier = set()
            for row in rows:
                relation_id = row["relation_id"]
                if relation_id in seen_relations:
                    continue
                relation = KGRelation(
                    relation_id=relation_id,
                    subject_id=row["subject_id"],
                    predicate=row["predicate"],
                    object_id=row["object_id"],
                    claim_id=row["claim_id"] if "claim_id" in row.keys() else None,
                    confidence=float(row["confidence"] or 0.0),
                )
                relations.append(relation)
                seen_relations.add(relation_id)
                if relation.subject_id not in visited:
                    new_frontier.add(relation.subject_id)
                    visited.add(relation.subject_id)
                if relation.object_id not in visited:
                    new_frontier.add(relation.object_id)
                    visited.add(relation.object_id)
            frontier = new_frontier
            if len(relations) >= limit_edges:
                break

        entity_rows: list[KGEntity] = []
        if visited:
            with db._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT entity_id, name, canonical_name, type
                    FROM kg_entities
                    WHERE user_id = ? AND entity_id IN ({})
                    """.format(",".join("?" for _ in visited)),
                    (self.user_id, *visited),
                ).fetchall()
            entity_rows = [
                KGEntity(
                    entity_id=row["entity_id"],
                    name=row["name"],
                    canonical_name=row["canonical_name"],
                    type=row["type"],
                )
                for row in rows
            ]

        evidence_rows: list[KGEvidence] = []
        relation_ids = [rel.relation_id for rel in relations]
        if relation_ids:
            with db._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT evidence_id, relation_id, doc_id, claim_id, quote, confidence
                    FROM kg_evidence
                    WHERE user_id = ? AND relation_id IN ({})
                    """.format(",".join("?" for _ in relation_ids)),
                    (self.user_id, *relation_ids),
                ).fetchall()
            evidence_rows = [
                KGEvidence(
                    evidence_id=row["evidence_id"],
                    relation_id=row["relation_id"],
                    doc_id=row["doc_id"],
                    claim_id=row["claim_id"] if "claim_id" in row.keys() else None,
                    quote=row["quote"],
                    confidence=float(row["confidence"] or 0.0),
                )
                for row in rows
            ]

        return KGSubgraph(entities=entity_rows, relations=relations, evidence=evidence_rows)

    def evidence_doc_ids(self, relation_ids: Iterable[str]) -> list[str]:
        rel_ids = [rid for rid in relation_ids if rid]
        if not rel_ids:
            return []
        with db._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT doc_id FROM kg_evidence
                WHERE user_id = ? AND relation_id IN ({})
                """.format(",".join("?" for _ in rel_ids)),
                (self.user_id, *rel_ids),
            ).fetchall()
        return [row["doc_id"] for row in rows if row["doc_id"]]

    def attach_contradiction(self, claim_id_a: str, claim_id_b: str, confidence: float = 0.6) -> str:
        contradiction_id = str(uuid.uuid4())
        now = _now_iso()
        with db._connect() as conn:
            row = conn.execute(
                """
                SELECT contradiction_id FROM kg_contradictions
                WHERE user_id = ?
                  AND (
                        (claim_id_a = ? AND claim_id_b = ?)
                     OR (claim_id_a = ? AND claim_id_b = ?)
                  )
                """,
                (self.user_id, claim_id_a, claim_id_b, claim_id_b, claim_id_a),
            ).fetchone()
            if row:
                return str(row["contradiction_id"])
            conn.execute(
                """
                INSERT INTO kg_contradictions
                    (contradiction_id, user_id, claim_id_a, claim_id_b, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    contradiction_id,
                    self.user_id,
                    claim_id_a,
                    claim_id_b,
                    max(0.0, min(1.0, confidence)),
                    now,
                ),
            )
        return contradiction_id

    def is_doc_extracted(self, doc_id: str) -> bool:
        if not doc_id:
            return False
        with db._connect() as conn:
            row = conn.execute(
                """
                SELECT status FROM kg_doc_state WHERE doc_id = ? AND user_id = ?
                """,
                (doc_id, self.user_id),
            ).fetchone()
        return bool(row and row["status"] in {"ok", "empty"})

    def mark_doc_extracted(self, doc_id: str, status: str = "ok") -> None:
        if not doc_id:
            return
        now = _now_iso()
        with db._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO kg_doc_state (doc_id, user_id, extracted_at, status)
                VALUES (?, ?, ?, ?)
                """,
                (doc_id, self.user_id, now, status),
            )


def build_seed_terms(topic: str, interests: Iterable[str], hints: Iterable[str]) -> list[str]:
    terms = [topic] + [t for t in interests if t] + [h for h in hints if h]
    cleaned = []
    seen = set()
    for term in terms:
        text = (term or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        cleaned.append(text)
        seen.add(key)
    return cleaned


def extract_and_store(
    store: KGStore,
    llm: LLMClient | None,
    doc_id: str,
    text: str,
    topic: str | None = None,
    max_triples: int = 12,
) -> int:
    if not doc_id or not text.strip():
        return 0
    if not llm:
        return 0
    if store.is_doc_extracted(doc_id):
        return 0
    triples = extract_triples(llm, text)
    if not triples:
        store.mark_doc_extracted(doc_id, status="empty")
        return 0
    added = 0
    for triple in triples[:max_triples]:
        evidence = (triple.get("evidence") or "").strip()
        if not evidence:
            continue
        subject_id = store.upsert_entity(triple["subject"])
        object_id = store.upsert_entity(triple["object"])
        claim_text = (triple.get("claim") or "").strip()
        if not claim_text and evidence:
            claim_text = evidence
        if not claim_text:
            claim_text = f"{triple['subject']} {triple['predicate']} {triple['object']}"
        claim_id = store.upsert_claim(
            claim_text,
            topic=topic,
            confidence=triple.get("confidence", 0.6),
        )
        relation_id = store.add_relation(
            subject_id,
            triple["predicate"],
            object_id,
            claim_id=claim_id,
            confidence=triple.get("confidence", 0.6),
        )
        store.add_evidence(
            doc_id,
            relation_id,
            evidence,
            claim_id=claim_id,
            confidence=triple.get("confidence", 0.6),
        )
        store.add_mention(doc_id, subject_id, claim_id=claim_id, sentence=evidence)
        store.add_mention(doc_id, object_id, claim_id=claim_id, sentence=evidence)
        added += 1
    if added == 0:
        store.mark_doc_extracted(doc_id, status="empty")
        return 0
    store.mark_doc_extracted(doc_id, status="ok")
    return added


def update_profile_from_mentions(
    store: KGStore,
    doc_ids: Iterable[str],
    limit: int = 10,
) -> int:
    doc_list = [doc_id for doc_id in doc_ids if doc_id]
    if not doc_list:
        return 0
    with db._connect() as conn:
        rows = conn.execute(
            """
            SELECT entity_id, COUNT(*) AS count
            FROM kg_mentions
            WHERE user_id = ? AND doc_id IN ({})
            GROUP BY entity_id
            ORDER BY count DESC
            LIMIT ?
            """.format(",".join("?" for _ in doc_list)),
            (store.user_id, *doc_list, limit),
        ).fetchall()
    if not rows:
        return 0
    max_count = max(int(row["count"] or 1) for row in rows)
    updates = 0
    for row in rows:
        count = int(row["count"] or 0)
        salience = 0.2 + 0.8 * (count / max_count)
        store.set_profile_edge(row["entity_id"], salience)
        updates += 1
    return updates


def apply_profile_items(
    store: KGStore,
    doc_id: str,
    items: Iterable[dict],
) -> int:
    updates = 0
    for item in items:
        entity = str(item.get("entity") or "").strip()
        if not entity:
            continue
        salience = float(item.get("salience") or 0.6)
        start_date = item.get("start_date")
        end_date = item.get("end_date")
        evidence = str(item.get("evidence") or "").strip()
        entity_id = store.upsert_entity(entity)
        store.set_profile_edge(
            entity_id,
            salience=salience,
            start_date=start_date,
            end_date=end_date,
            source_doc_id=doc_id,
        )
        if evidence:
            store.add_mention(doc_id, entity_id, sentence=evidence)
        updates += 1
    return updates


def seed_terms_from_query(query: str, extra: Iterable[str]) -> list[str]:
    query_terms = tokenize(query)
    seed_terms = list(dict.fromkeys([query] + list(extra) + query_terms))
    return [term for term in seed_terms if term]


def log_subgraph_summary(subgraph: KGSubgraph) -> None:
    log_task_event(
        "KG subgraph: "
        f"entities={len(subgraph.entities)} "
        f"relations={len(subgraph.relations)} "
        f"evidence={len(subgraph.evidence)}"
    )


def build_subgraph_for_terms(
    store: KGStore,
    terms: Iterable[str],
    hops: int = 1,
    min_confidence: float = 0.3,
    limit_edges: int = 200,
) -> KGSubgraph:
    seeds = store.seed_entities_from_terms(terms, limit=12)
    return store.subgraph(
        [entity.entity_id for entity in seeds],
        hops=hops,
        min_confidence=min_confidence,
        limit_edges=limit_edges,
    )


def build_full_subgraph(
    store: KGStore,
    min_confidence: float = 0.0,
    limit_edges: int = 200,
) -> KGSubgraph:
    with db._connect() as conn:
        rows = conn.execute(
            """
            SELECT relation_id, subject_id, predicate, object_id, claim_id, confidence
            FROM kg_relations
            WHERE user_id = ? AND confidence >= ?
            LIMIT ?
            """,
            (store.user_id, min_confidence, limit_edges),
        ).fetchall()
    relations = [
        KGRelation(
            relation_id=row["relation_id"],
            subject_id=row["subject_id"],
            predicate=row["predicate"],
            object_id=row["object_id"],
            claim_id=row["claim_id"] if "claim_id" in row.keys() else None,
            confidence=float(row["confidence"] or 0.0),
        )
        for row in rows
    ]
    if not relations:
        return KGSubgraph(entities=[], relations=[], evidence=[])

    entity_ids = {rel.subject_id for rel in relations} | {rel.object_id for rel in relations}
    with db._connect() as conn:
        entity_rows = conn.execute(
            """
            SELECT entity_id, name, canonical_name, type
            FROM kg_entities
            WHERE user_id = ? AND entity_id IN ({})
            """.format(",".join("?" for _ in entity_ids)),
            (store.user_id, *entity_ids),
        ).fetchall()
    entities = [
        KGEntity(
            entity_id=row["entity_id"],
            name=row["name"],
            canonical_name=row["canonical_name"],
            type=row["type"],
        )
        for row in entity_rows
    ]

    relation_ids = [rel.relation_id for rel in relations]
    with db._connect() as conn:
        evidence_rows = conn.execute(
            """
            SELECT evidence_id, relation_id, doc_id, claim_id, quote, confidence
            FROM kg_evidence
            WHERE user_id = ? AND relation_id IN ({})
            """.format(",".join("?" for _ in relation_ids)),
            (store.user_id, *relation_ids),
        ).fetchall()
    evidence = [
        KGEvidence(
            evidence_id=row["evidence_id"],
            relation_id=row["relation_id"],
            doc_id=row["doc_id"],
            claim_id=row["claim_id"] if "claim_id" in row.keys() else None,
            quote=row["quote"],
            confidence=float(row["confidence"] or 0.0),
        )
        for row in evidence_rows
    ]

    return KGSubgraph(entities=entities, relations=relations, evidence=evidence)


def _escape_dot(text: str) -> str:
    cleaned = (text or "").replace("\\", "\\\\").replace("\"", "\\\"")
    cleaned = cleaned.replace("\n", " ").strip()
    return cleaned


def render_dot(
    subgraph: KGSubgraph,
    user_id: str | None = None,
    profile_edges: list[dict] | None = None,
) -> str:
    node_lines = []
    edge_lines = []
    evidence_map: dict[str, list[str]] = {}
    for ev in subgraph.evidence:
        evidence_map.setdefault(ev.relation_id, []).append(ev.quote)
    source_counts: dict[str, dict[str, int]] = {}
    if subgraph.evidence:
        doc_ids = {ev.doc_id for ev in subgraph.evidence if ev.doc_id}
        docs = db.get_documents_by_ids(doc_ids)
        doc_types = {doc.doc_id: doc.source_type for doc in docs}
        rel_sources: dict[str, dict[str, set[str]]] = {}
        for ev in subgraph.evidence:
            doc_id = ev.doc_id
            if not doc_id:
                continue
            source_type = (doc_types.get(doc_id) or "").strip().lower() or "unknown"
            if source_type in {"file", "note", "url"}:
                bucket = "local"
            elif source_type == "web":
                bucket = "web"
            else:
                bucket = source_type
            rel_sources.setdefault(ev.relation_id, {}).setdefault(bucket, set()).add(doc_id)
        for rel_id, buckets in rel_sources.items():
            source_counts[rel_id] = {bucket: len(ids) for bucket, ids in buckets.items()}

    entity_labels: dict[str, str] = {}
    for entity in subgraph.entities:
        label = _escape_dot(entity.name or entity.canonical_name or entity.entity_id)
        entity_labels[entity.entity_id] = label
        node_lines.append(f"  \"{entity.entity_id}\" [label=\"{label}\"];")

    if user_id and profile_edges:
        user_node_id = f"user:{user_id}"
        node_lines.append(
            f"  \"{user_node_id}\" [label=\"User:{_escape_dot(user_id)}\", shape=ellipse, style=filled, fillcolor=\"#E8E8E8\"];"
        )
        for edge in profile_edges:
            entity_id = edge.get("entity_id") or ""
            if not entity_id:
                continue
            if entity_id not in entity_labels:
                label = _escape_dot(edge.get("entity") or entity_id)
                entity_labels[entity_id] = label
                node_lines.append(f"  \"{entity_id}\" [label=\"{label}\"];")
            ptype = _escape_dot(PROFILE_TYPE)
            salience = edge.get("salience")
            if isinstance(salience, (int, float)):
                edge_label = f"{ptype} ({salience:.2f})"
            else:
                edge_label = ptype
            edge_lines.append(
                f"  \"{user_node_id}\" -> \"{entity_id}\" [label=\"{edge_label}\"];"
            )

    for rel in subgraph.relations:
        label = _escape_dot(rel.predicate)
        ev_quotes = evidence_map.get(rel.relation_id, [])
        parts: list[str] = []
        parts.append(f"c={rel.confidence:.2f}")
        if ev_quotes:
            parts.append(f"n={len(ev_quotes)}")
        if parts:
            label = f"{label} ({', '.join(parts)})"
        counts = source_counts.get(rel.relation_id)
        if counts:
            local_count = counts.get("local", 0)
            web_count = counts.get("web", 0)
            label = f"{label} [L{local_count}/W{web_count}]"
        edge_lines.append(
            f"  \"{rel.subject_id}\" -> \"{rel.object_id}\" [label=\"{label}\"];"
        )

    lines = ["digraph KG {", "  rankdir=LR;", "  node [shape=box];"]
    lines.extend(node_lines)
    lines.extend(edge_lines)
    lines.append("}")
    return "\n".join(lines)


def render_html(
    subgraph: KGSubgraph,
    title: str = "KG Visualization",
    user_id: str | None = None,
    profile_edges: list[dict] | None = None,
) -> str:
    nodes = []
    edges = []
    evidence_map: dict[str, list[str]] = {}
    evidence_items: dict[str, list[dict[str, object]]] = {}
    source_counts: dict[str, dict[str, int]] = {}
    doc_lookup: dict[str, db.Document] = {}
    if subgraph.evidence:
        doc_ids = {ev.doc_id for ev in subgraph.evidence if ev.doc_id}
        docs = db.get_documents_by_ids(doc_ids)
        doc_lookup = {doc.doc_id: doc for doc in docs}
        doc_types = {doc.doc_id: doc.source_type for doc in docs}
        rel_sources: dict[str, dict[str, set[str]]] = {}
        for ev in subgraph.evidence:
            doc_id = ev.doc_id
            if doc_id:
                source_type = (doc_types.get(doc_id) or "").strip().lower() or "unknown"
            else:
                source_type = "unknown"
            if source_type in {"file", "note", "url"}:
                bucket = "local"
            elif source_type == "web":
                bucket = "web"
            else:
                bucket = source_type
            rel_sources.setdefault(ev.relation_id, {}).setdefault(bucket, set()).add(
                doc_id or f"unknown:{ev.evidence_id}"
            )
            evidence_map.setdefault(ev.relation_id, []).append(ev.quote)
            doc = doc_lookup.get(doc_id) if doc_id else None
            evidence_items.setdefault(ev.relation_id, []).append(
                {
                    "quote": ev.quote,
                    "confidence": float(ev.confidence or 0.0),
                    "doc_id": doc_id,
                    "doc_title": doc.title if doc else "",
                    "source_type": doc.source_type if doc else source_type,
                    "source": doc.source if doc else "",
                    "added_at": doc.added_at if doc else "",
                }
            )
        for rel_id, buckets in rel_sources.items():
            source_counts[rel_id] = {bucket: len(ids) for bucket, ids in buckets.items()}

    def _latest_evidence_at(items: list[dict[str, object]]) -> str:
        latest: datetime | None = None
        for item in items:
            raw = str(item.get("added_at") or "")
            if not raw:
                continue
            try:
                parsed = datetime.fromisoformat(raw)
            except ValueError:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            if latest is None or parsed > latest:
                latest = parsed
        return latest.isoformat() if latest else ""

    node_ids: set[str] = set()
    for entity in subgraph.entities:
        nodes.append(
            {
                "id": entity.entity_id,
                "label": entity.name or entity.canonical_name or entity.entity_id[:8],
                "title": entity.type or "Entity",
            }
        )
        node_ids.add(entity.entity_id)

    if user_id and profile_edges:
        user_node_id = f"user:{user_id}"
        nodes.append(
            {
                "id": user_node_id,
                "label": f"User:{user_id}",
                "title": "User",
                "color": "#E8E8E8",
                "shape": "ellipse",
                "is_user": True,
            }
        )
        node_ids.add(user_node_id)
        for edge in profile_edges:
            entity_id = edge.get("entity_id") or ""
            if not entity_id:
                continue
            if entity_id not in node_ids:
                nodes.append(
                    {
                        "id": entity_id,
                        "label": edge.get("entity") or entity_id[:8],
                        "title": "Profile",
                    }
                )
                node_ids.add(entity_id)
            ptype = PROFILE_TYPE
            salience = edge.get("salience")
            label = f"{ptype} ({salience:.2f})" if isinstance(salience, (int, float)) else ptype
            edges.append(
                {
                    "from": user_node_id,
                    "to": entity_id,
                    "label": label,
                    "title": label,
                    "arrows": "to",
                }
            )

    for rel in subgraph.relations:
        ev_quotes = evidence_map.get(rel.relation_id, [])
        title_lines = []
        if rel.predicate:
            title_lines.append(rel.predicate)
        for quote in ev_quotes[:3]:
            title_lines.append(quote)
        counts = source_counts.get(rel.relation_id)
        if counts:
            local_count = counts.get("local", 0)
            web_count = counts.get("web", 0)
            unknown_count = sum(
                value for key, value in counts.items() if key not in {"local", "web"}
            )
            source_line = f"Sources: local={local_count} web={web_count}"
            if unknown_count:
                source_line += f" other={unknown_count}"
            title_lines.append(source_line)
        edges.append(
            {
                "id": rel.relation_id,
                "from": rel.subject_id,
                "to": rel.object_id,
                "label": rel.predicate,
                "confidence": float(rel.confidence or 0.0),
                "title": "\n".join(title_lines),
                "arrows": "to",
            }
        )

    for edge in edges:
        rel_id = edge.get("id")
        if not rel_id:
            continue
        ev_items = evidence_items.get(rel_id, [])
        counts = source_counts.get(rel_id, {})
        local_count = counts.get("local", 0)
        web_count = counts.get("web", 0)
        other_count = sum(
            value for key, value in counts.items() if key not in {"local", "web"}
        )
        label = edge.get("label") or ""
        parts: list[str] = []
        confidence = edge.get("confidence")
        if isinstance(confidence, (int, float)):
            parts.append(f"c={confidence:.2f}")
        if ev_items:
            parts.append(f"n={len(ev_items)}")
        if parts:
            label = f"{label} ({', '.join(parts)})"
        if counts:
            label = f"{label} [L{local_count}/W{web_count}"
            if other_count:
                label = f"{label}/O{other_count}"
            label = f"{label}]"
        edge["label"] = label
        latest_at = _latest_evidence_at(ev_items)
        if latest_at:
            edge["title"] = "\n".join(
                [edge.get("title") or "", f"Latest evidence: {latest_at}"]
            ).strip()
        source_summary = f"local={local_count} web={web_count}"
        if other_count:
            source_summary += f" other={other_count}"
        edge["source_summary"] = source_summary
        edge["latest_evidence_at"] = latest_at
        edge["evidence_items"] = ev_items

    payload = {
        "title": title,
        "nodes": nodes,
        "edges": edges,
        "user_node_id": f"user:{user_id}" if user_id else "",
    }
    node_count = len(nodes)
    edge_count = len(edges)
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{_escape_dot(title)}</title>
  <style>
    html, body {{
      height: 100%;
      margin: 0;
      padding: 0;
      background: #f5f5f2;
      font-family: "IBM Plex Mono", "Space Mono", monospace;
    }}
    #layout {{
      height: 100%;
      display: flex;
    }}
    #network {{
      flex: 1;
      min-width: 0;
    }}
    #details {{
      width: 360px;
      border-left: 1px solid #ddd;
      background: #fafafa;
      padding: 16px;
      box-sizing: border-box;
      overflow: auto;
    }}
    #summary {{
      font-size: 12px;
      color: #555;
      margin-bottom: 12px;
    }}
    #fitBtn {{
      display: inline-block;
      margin-top: 6px;
      padding: 4px 8px;
      border: 1px solid #bbb;
      background: #fff;
      font-size: 12px;
      cursor: pointer;
    }}
    #details h2 {{
      margin: 0 0 8px 0;
      font-size: 16px;
    }}
    .control {{
      margin-bottom: 12px;
      font-size: 13px;
    }}
    .meta {{
      font-size: 12px;
      color: #555;
      margin-bottom: 8px;
    }}
    .evidence {{
      border-top: 1px solid #e0e0e0;
      padding-top: 8px;
      margin-top: 8px;
      font-size: 12px;
    }}
    .evidence-item {{
      margin-bottom: 10px;
      padding-bottom: 8px;
      border-bottom: 1px dashed #e0e0e0;
    }}
    .evidence-item:last-child {{
      border-bottom: none;
    }}
  </style>
  <script src="https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.js"></script>
</head>
<body>
  <div id="layout">
    <div id="network"></div>
    <aside id="details">
      <div class="control">
        <label>
          <input type="checkbox" id="toggleUser" checked>
          Show user node
        </label>
      </div>
      <div id="summary">
        Nodes: <span id="node-count">{node_count}</span> |
        Edges: <span id="edge-count">{edge_count}</span>
        <div><button id="fitBtn" type="button">Fit View</button></div>
      </div>
      <h2>Edge Details</h2>
      <div id="edge-details" class="meta">Select an edge to see evidence.</div>
    </aside>
  </div>
  <script>
    const payload = {json.dumps(payload)};
    const baseNodes = payload.nodes || [];
    const baseEdges = payload.edges || [];
    if (typeof vis === "undefined") {{
      const panel = document.getElementById("edge-details");
      panel.textContent = "vis-network failed to load. Check your network connection or firewall, or use a local copy of the library.";
      throw new Error("vis-network not available");
    }}
    const nodes = new vis.DataSet([]);
    const edges = new vis.DataSet([]);
    const container = document.getElementById("network");
    const data = {{ nodes: nodes, edges: edges }};
    window.addEventListener("error", (event) => {{
      const panel = document.getElementById("edge-details");
      const message = event?.message || "Unknown error";
      panel.textContent = "Viewer error: " + message;
    }});
    window.addEventListener("unhandledrejection", (event) => {{
      const panel = document.getElementById("edge-details");
      const reason = event?.reason || "Unhandled rejection";
      panel.textContent = "Viewer error: " + reason;
    }});
    const options = {{
      nodes: {{
        shape: "box",
        font: {{ color: "#111", size: 14 }},
        color: {{ background: "#fef6d1", border: "#333" }},
      }},
      edges: {{
        font: {{ align: "middle" }},
        color: {{ color: "#333" }},
        arrows: {{ to: {{ enabled: true }} }},
        width: 1.4,
        smooth: {{ type: "continuous" }},
      }},
      physics: {{
        stabilization: true,
        barnesHut: {{ gravitationalConstant: -22000, springLength: 140 }},
      }}
    }};
    const network = new vis.Network(container, data, options);

    function escapeHtml(value) {{
      return String(value || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }}

    function applyUserFilter(showUser) {{
      const filteredNodes = showUser ? baseNodes : baseNodes.filter(n => !n.is_user);
      const nodeIds = new Set(filteredNodes.map(n => n.id));
      const filteredEdges = showUser
        ? baseEdges
        : baseEdges.filter(e => nodeIds.has(e.from) && nodeIds.has(e.to));
      nodes.clear();
      edges.clear();
      nodes.add(filteredNodes);
      edges.add(filteredEdges);
      document.getElementById("node-count").textContent = String(filteredNodes.length);
      document.getElementById("edge-count").textContent = String(filteredEdges.length);
    }}

    function renderEdgeDetails(edge) {{
      const panel = document.getElementById("edge-details");
      if (!edge) {{
        panel.textContent = "Select an edge to see evidence.";
        return;
      }}
      const title = escapeHtml(edge.label || edge.title || "Edge");
      const latest = edge.latest_evidence_at ? escapeHtml(edge.latest_evidence_at) : "Unknown";
      const sources = escapeHtml(edge.source_summary || "Unknown");
      let html = `<div class="meta"><strong>${{title}}</strong></div>`;
      html += `<div class="meta">Latest evidence: ${{latest}}</div>`;
      html += `<div class="meta">Sources: ${{sources}}</div>`;
      const evidence = edge.evidence_items || [];
      if (!evidence.length) {{
        html += `<div class="evidence">No evidence attached to this edge.</div>`;
        panel.innerHTML = html;
        return;
      }}
      html += `<div class="evidence"><strong>Evidence</strong></div>`;
      for (const item of evidence) {{
        const quote = escapeHtml(item.quote || "");
        const addedAt = escapeHtml(item.added_at || "Unknown");
        const sourceType = escapeHtml(item.source_type || "Unknown");
        const titleText = escapeHtml(item.doc_title || item.doc_id || "");
        html += `<div class="evidence-item">`;
        html += `<div class="meta">${{sourceType}} | ${{addedAt}}</div>`;
        if (titleText) {{
          html += `<div class="meta">${{titleText}}</div>`;
        }}
        if (quote) {{
          html += `<div>${{quote}}</div>`;
        }}
        html += `</div>`;
      }}
      panel.innerHTML = html;
    }}

    applyUserFilter(true);
    network.once("stabilizationIterationsDone", () => {{
      network.fit({{ animation: true }});
    }});
    document.getElementById("toggleUser").addEventListener("change", (event) => {{
      applyUserFilter(event.target.checked);
      network.fit({{ animation: true }});
    }});
    document.getElementById("fitBtn").addEventListener("click", () => {{
      network.fit({{ animation: true }});
    }});

    network.on("selectEdge", (params) => {{
      if (!params.edges.length) {{
        renderEdgeDetails(null);
        return;
      }}
      const edge = edges.get(params.edges[0]);
      renderEdgeDetails(edge);
    }});
    network.on("deselectEdge", () => renderEdgeDetails(null));
  </script>
</body>
</html>
"""
