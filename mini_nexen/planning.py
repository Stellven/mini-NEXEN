from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from . import db
from .config import GRAPH_TOP_CLUSTERS, PLANS_DIR, ensure_dirs
from .db import Document, Interest, Method, load_document_text
from .embeddings import EmbeddingClient, EmbeddingConfig, cosine_similarity, normalize
from .graph import GraphManager
from .llm import LLMClient, LLMClientError, log_task_event
from .llm_prompts import SYSTEM_OUTLINE_PROMPT, SYSTEM_PLAN_PROMPT, outline_prompt, plan_prompt, refine_prompt
from .text_utils import top_sentences, tokenize

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_CJK_WORD_RE = re.compile(r"[\u4e00-\u9fff]+")

DOC_CHUNK_CAP = 3
PER_CLUSTER_CHUNK_LIMIT = 30
MERGED_CHUNK_MAX_CHARS = 1200

_JSON_KEY_MAP = {
    "scope": "scope",
    "key_questions": "key_questions",
    "key questions": "key_questions",
    "key-questions": "key_questions",
    "questions": "key_questions",
    "keywords": "keywords",
    "gaps": "gaps",
    "gap": "gaps",
    "notes": "notes",
    "readiness": "readiness",
    "retrieval_queries": "retrieval_queries",
    "retrieval queries": "retrieval_queries",
    "retrieval query": "retrieval_queries",
    "范围": "scope",
    "关键问题": "key_questions",
    "关键词": "keywords",
    "缺口": "gaps",
    "差距": "gaps",
    "缺口与检索需求": "gaps",
    "差距与检索需求": "gaps",
    "备注": "notes",
    "就绪度": "readiness",
    "准备度": "readiness",
    "准备状态": "readiness",
    "检索查询": "retrieval_queries",
    "检索查询语句": "retrieval_queries",
    "检索关键词": "retrieval_queries",
}

OUTLINE_MIN_WORDS = 1000
OUTLINE_MAX_WORDS = 2000
OUTLINE_MIN_CJK_RATIO = 0.4


@dataclass
class SourceBrief:
    doc: Document
    highlights: list[str]


@dataclass
class PlanDraft:
    topic: str
    created_at: str
    round_number: int
    scope: list[str]
    key_questions: list[str]
    keywords: list[str]
    source_types: list[str]
    source_briefs: list[SourceBrief]
    gaps: list[str]
    readiness: str
    notes: list[str]
    retrieval_queries: list[str]
    graph_top_clusters: int
    top_k_docs: int


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def summarize_interests(interests: Iterable[Interest]) -> list[str]:
    summary = []
    seen = set()
    for interest in interests:
        item = interest.topic
        if item not in seen:
            seen.add(item)
            summary.append(item)
    return summary


def summarize_methods(methods: Iterable[Method]) -> list[str]:
    summary = []
    seen = set()
    for method in methods:
        item = method.method
        if item not in seen:
            seen.add(item)
            summary.append(item)
    return summary


def build_keywords(topic: str, interests: Iterable[Interest], extra: Iterable[str] | None = None) -> list[str]:
    tokens = tokenize(topic)
    for interest in interests:
        tokens.extend(tokenize(interest.topic))
    if extra:
        for item in extra:
            tokens.extend(tokenize(item))
    unique = []
    seen = set()
    for token in tokens:
        if token and token not in seen:
            seen.add(token)
            unique.append(token)
    return unique[:40]


def create_source_briefs(
    documents: Iterable[Document],
    keywords: Iterable[str],
    highlights_per_doc: int = 10,
    doc_text_overrides: dict[str, str] | None = None,
) -> list[SourceBrief]:
    from .llm import log_task_event

    briefs = []
    total_highlights = 0
    counts: dict[str, int] = {}
    for doc in documents:
        text = doc_text_overrides.get(doc.doc_id) if doc_text_overrides else None
        if not text:
            text = load_document_text(doc)
        highlights = top_sentences(text, keywords, limit=highlights_per_doc)
        total_highlights += len(highlights)
        briefs.append(SourceBrief(doc=doc, highlights=highlights))
        counts[doc.source_type] = counts.get(doc.source_type, 0) + 1
    log_task_event(
        f"Source briefs: docs={len(briefs)} highlights={total_highlights} per_doc={highlights_per_doc}"
    )
    if counts:
        breakdown = " ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        log_task_event(f"Source briefs by source_type: {breakdown}")
    return briefs


def _compact_snippet(text: str, limit: int = 220) -> str:
    compacted = " ".join((text or "").split())
    if len(compacted) <= limit:
        return compacted
    trimmed = compacted[:limit].rsplit(" ", 1)[0]
    return (trimmed or compacted[:limit]).rstrip() + "..."


def select_docs_by_cluster_round_robin(
    query: str,
    embed_config: EmbeddingConfig | None,
    top_clusters: int,
    top_k_docs: int,
    source_type: str | None = None,
    per_cluster_chunk_limit: int = PER_CLUSTER_CHUNK_LIMIT,
    min_similarity: float = 0.2,
) -> tuple[list[str], list[tuple[float, str, str]], dict[str, str]]:
    if not query or not embed_config:
        return [], [], {}
    graph = GraphManager(embed_config=embed_config, llm=None, semantic_labels=False)
    scored = graph.score_clusters(query)
    if not scored:
        return [], [], {}
    selected_clusters = [(sim, cid, label) for sim, cid, label in scored if sim >= min_similarity]
    if not selected_clusters:
        return [], [], {}
    selected_clusters = selected_clusters[: max(1, int(top_clusters))]

    cluster_chunks: dict[str, list[str]] = {}
    with db._connect() as conn:
        for _, cluster_id, _ in selected_clusters:
            params = [cluster_id]
            source_clause = ""
            if source_type:
                source_clause = "AND d.source_type = ?"
                params.append(source_type)
            params.append(int(per_cluster_chunk_limit))
            rows = conn.execute(
                f"""
                SELECT c.doc_id
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id
                JOIN document_stats s ON s.doc_id = d.doc_id
                WHERE c.cluster_id = ?
                  AND s.archived = 0
                  {source_clause}
                ORDER BY c.similarity DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
            cluster_chunks[cluster_id] = [row["doc_id"] for row in rows]

    selected_clusters = [item for item in selected_clusters if cluster_chunks.get(item[1])]
    if not selected_clusters:
        return [], [], {}

    selected_docs: list[str] = []
    doc_cluster_map: dict[str, str] = {}
    seen_docs: set[str] = set()
    progress = True
    while len(selected_docs) < int(top_k_docs) and progress:
        progress = False
        for _, cluster_id, _ in selected_clusters:
            queue = cluster_chunks.get(cluster_id, [])
            while queue and queue[0] in seen_docs:
                queue.pop(0)
            if not queue:
                continue
            doc_id = queue.pop(0)
            selected_docs.append(doc_id)
            seen_docs.add(doc_id)
            doc_cluster_map[doc_id] = cluster_id
            progress = True
            if len(selected_docs) >= int(top_k_docs):
                break
    return selected_docs, selected_clusters, doc_cluster_map


def merge_doc_chunks(
    doc_ids: Iterable[str],
    per_doc_cap: int = DOC_CHUNK_CAP,
    max_chars: int = MERGED_CHUNK_MAX_CHARS,
) -> tuple[dict[str, str], dict[str, int]]:
    overrides: dict[str, str] = {}
    counts: dict[str, int] = {}
    ids = [doc_id for doc_id in doc_ids if doc_id]
    if not ids:
        return overrides, counts
    per_doc_cap = max(1, int(per_doc_cap))
    per_chunk_limit = max(200, int(max_chars / per_doc_cap))
    with db._connect() as conn:
        for doc_id in ids:
            rows = conn.execute(
                """
                SELECT c.text
                FROM chunks c
                JOIN document_stats s ON s.doc_id = c.doc_id
                WHERE c.doc_id = ? AND s.archived = 0 AND c.cluster_id IS NOT NULL
                ORDER BY c.similarity DESC
                LIMIT ?
                """,
                (doc_id, per_doc_cap),
            ).fetchall()
            parts = []
            for row in rows:
                snippet = _compact_snippet(row["text"], limit=per_chunk_limit)
                if snippet:
                    parts.append(snippet)
            counts[doc_id] = len(parts)
            if not parts:
                continue
            merged = "\n\n".join(parts)
            merged = _compact_snippet(merged, limit=max_chars)
            overrides[doc_id] = merged
    return overrides, counts


def _prepare_theme_evidence(
    snippets: list[tuple[float, str, str]],
    max_docs: int,
    max_snippets: int,
    max_chars: int = 420,
) -> list[dict]:
    grouped: dict[str, list[tuple[float, str]]] = {}
    for sim, text, title in snippets:
        grouped.setdefault(title, []).append((sim, text))
    docs = []
    for title, items in grouped.items():
        items.sort(key=lambda item: item[0], reverse=True)
        evidence = []
        for _, text in items[: max_snippets]:
            snippet = _compact_snippet(text, limit=max_chars)
            if snippet:
                evidence.append(snippet)
        if evidence:
            docs.append({"title": title, "evidence": evidence})
    docs.sort(key=lambda item: len(item["evidence"]), reverse=True)
    return docs[:max_docs]


def _build_theme_embedding_config(llm: LLMClient) -> EmbeddingConfig | None:
    provider = (llm.config.provider or "").lower()
    if not provider:
        return None
    if provider == "gemini":
        return EmbeddingConfig(provider="gemini", model="gemini-embedding-001", api_key=llm.config.api_key)
    if provider == "lmstudio":
        return EmbeddingConfig(provider="lmstudio", base_url=llm.config.base_url, api_key=llm.config.api_key)
    return None


def _merge_theme_label(llm: LLMClient, labels: list[str]) -> str:
    prompt = {
        "instruction": "Create a concise merged label (2-6 words) that captures the shared idea.",
        "labels": labels,
    }
    response = llm.generate(
        system_prompt="You merge topic labels.",
        user_prompt=json.dumps(prompt, indent=2),
        task="merge theme label",
        agent="Theme",
    )
    merged = ""
    parsed = _extract_json(response or "")
    if isinstance(parsed, dict):
        candidate = parsed.get("merged_label") or parsed.get("label") or parsed.get("theme")
        if isinstance(candidate, str):
            merged = candidate.strip()
    if not merged:
        merged = (response or "").strip().strip("\"'`")
    merged = merged.replace("\n", " ").strip()
    if not merged:
        raise LLMClientError("LLM returned empty merged theme label.")
    if len(merged) > 80:
        merged = merged[:80].strip()
    return merged


def _merge_themes_by_label_similarity(
    llm: LLMClient,
    themes: list[dict],
    threshold: float,
) -> list[dict]:
    if len(themes) < 2:
        return themes
    embed_config = _build_theme_embedding_config(llm)
    if not embed_config:
        raise LLMClientError("Embedding config required for theme merging.")
    client = EmbeddingClient(embed_config)
    labels = [theme["label"] for theme in themes]
    vectors = client.embed_texts(labels)
    if len(vectors) != len(labels):
        raise LLMClientError("Failed to embed theme labels for merging.")
    vectors = [normalize(vec) for vec in vectors]

    visited = set()
    merged: list[dict] = []
    for i, label in enumerate(labels):
        if i in visited:
            continue
        group = [i]
        visited.add(i)
        for j in range(i + 1, len(labels)):
            if j in visited:
                continue
            sim = cosine_similarity(vectors[i], vectors[j])
            if sim >= threshold:
                group.append(j)
                visited.add(j)
        if len(group) == 1:
            merged.append(themes[i])
            continue

        group_labels = [labels[idx] for idx in group]
        merged_label = _merge_theme_label(llm, group_labels)

        combined_docs: dict[str, list[str]] = {}
        chunk_count = 0
        similarities: list[float] = []
        for idx in group:
            theme = themes[idx]
            chunk_count += int(theme.get("chunk_count") or 0)
            similarity = theme.get("similarity")
            if isinstance(similarity, (int, float)):
                similarities.append(float(similarity))
            for doc in theme.get("documents", []):
                title = doc.get("title") or "Unknown document"
                evidence = doc.get("evidence") or []
                combined_docs.setdefault(title, [])
                for item in evidence:
                    if item not in combined_docs[title]:
                        combined_docs[title].append(item)

        merged_similarity = max(similarities) if similarities else None
        documents = [{"title": title, "evidence": evidence} for title, evidence in combined_docs.items()]
        merged.append(
            {
                "label": merged_label,
                "similarity": merged_similarity,
                "bullets": [],
                "documents": documents,
                "chunk_count": chunk_count,
                "doc_count": len(documents),
            }
        )
    return merged


def _summarize_theme_bullets(
    llm: LLMClient,
    themes: list[dict],
    max_bullets: int,
) -> dict[str, list[str]]:
    payload = []
    for theme in themes:
        payload.append(
            {
                "theme": theme["label"],
                "chunk_count": theme.get("chunk_count", 0),
                "doc_count": theme.get("doc_count", 0),
                "documents": theme["documents"],
            }
        )
    prompt = {
        "instruction": (
            "You summarize recurring themes based on the evidence provided. "
            "Use ONLY the evidence provided. Do NOT quote text; paraphrase. "
            "Avoid words like 'user', 'interest', or 'focus'. "
            "Write all natural-language content in Chinese. "
            "Keep paper titles, dataset names, benchmarks, model names, APIs, and acronyms in English. "
            "Return JSON only: "
            "{\"themes\":[{\"theme\":\"...\",\"bullets\":[{\"bullet\":\"...\",\"sources\":[\"title1\",\"title2\"]}]}]}."
        ),
        "requirements": [
            f"Provide 2-{max_bullets} bullets per theme.",
            "Bullets must be full-sentence reasoning, not keyword lists.",
            "Highlight recurring patterns, commonalities, and contradictions when present.",
            "Explicitly connect the evidence to the theme label.",
            "Each bullet must be an object with fields: bullet (string) and sources (array of document titles).",
            "Sources may be empty, but include 1-2 titles when they improve traceability.",
        ],
        "themes": payload,
    }
    response = llm.generate(
        system_prompt="You summarize evidence into concise, traceable bullets.",
        user_prompt=json.dumps(prompt, indent=2),
        task="interest themes",
        agent="Theme",
    )
    parsed = _extract_json(response)
    if not parsed:
        raise LLMClientError("LLM returned invalid JSON for interest themes.")
    items = parsed.get("themes")
    if not isinstance(items, list):
        raise LLMClientError("LLM returned invalid themes payload.")
    result: dict[str, list[str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        label = str(item.get("theme", "")).strip()
        bullets = item.get("bullets")
        if not label or not isinstance(bullets, list):
            continue
        cleaned = []
        for bullet in bullets:
            if isinstance(bullet, dict):
                text = str(
                    bullet.get("bullet")
                    or bullet.get("text")
                    or bullet.get("claim")
                    or ""
                ).strip()
                sources = bullet.get("sources") or bullet.get("source") or []
                source_items: list[str] = []
                if isinstance(sources, list):
                    source_items = [str(src).strip() for src in sources if str(src).strip()]
                elif isinstance(sources, str):
                    if sources.strip():
                        source_items = [sources.strip()]
                if text:
                    if source_items:
                        text = f"{text} Sources: {', '.join(source_items)}"
                    cleaned.append(text)
                continue
            text = str(bullet).strip()
            if text:
                cleaned.append(text)
        if cleaned:
            result[label] = cleaned[:max_bullets]
    if not result:
        raise LLMClientError("LLM returned empty interest themes.")
    return result


def build_interest_themes(
    topic: str,
    interests: list[Interest],
    query_hints: list[str] | None = None,
    llm: LLMClient | None = None,
    top_k_docs: int | None = None,
    max_themes: int = 6,
    max_docs: int = 3,
    max_bullets: int = 4,
    max_snippets: int = 3,
) -> list[dict]:
    if not llm:
        raise LLMClientError("LLM is required for interest theme summaries.")

    query_parts = [topic]
    for interest in interests:
        if interest.topic:
            query_parts.append(interest.topic)
    if query_hints:
        query_parts.extend(query_hints)
    query = " ".join(part for part in query_parts if part).strip()

    top_k = int(top_k_docs or max_docs * max_themes)
    embed_config = _build_theme_embedding_config(llm)
    selected_doc_ids, selected_clusters, doc_cluster_map = select_docs_by_cluster_round_robin(
        query,
        embed_config,
        max_themes,
        top_k,
        source_type="file",
    )

    themes: list[dict] = []
    if selected_doc_ids and selected_clusters:
        db.init_db()
        docs = db.get_documents_by_ids(selected_doc_ids)
        doc_titles = {doc.doc_id: doc.title for doc in docs}
        merged_texts, chunk_counts = merge_doc_chunks(selected_doc_ids, per_doc_cap=DOC_CHUNK_CAP)
        selected_order = {doc_id: idx for idx, doc_id in enumerate(selected_doc_ids)}
        for sim, cluster_id, label in selected_clusters:
            doc_ids = [
                doc_id for doc_id, cid in doc_cluster_map.items() if cid == cluster_id
            ]
            if not doc_ids:
                continue
            doc_ids.sort(key=lambda doc_id: selected_order.get(doc_id, 0))
            documents = []
            for doc_id in doc_ids[:max_docs]:
                text = merged_texts.get(doc_id) or ""
                snippet = _compact_snippet(text, limit=420)
                if not snippet:
                    continue
                title = doc_titles.get(doc_id) or "Unknown document"
                documents.append({"title": title, "evidence": [snippet]})
            if not documents:
                continue
            chunk_count = sum(chunk_counts.get(doc_id, 0) for doc_id in doc_ids)
            themes.append(
                {
                    "label": label or "Unlabeled Theme",
                    "similarity": sim,
                    "bullets": [],
                    "documents": documents,
                    "chunk_count": chunk_count,
                    "doc_count": len(doc_ids),
                }
            )

    if not themes:
        db.init_db()
        with db._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.cluster_id, c.text, c.similarity, d.doc_id, d.title, cl.label
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id
                JOIN document_stats s ON s.doc_id = d.doc_id
                LEFT JOIN clusters cl ON cl.cluster_id = c.cluster_id
                WHERE d.source_type = 'file'
                  AND s.archived = 0
                  AND c.cluster_id IS NOT NULL
                """
            ).fetchall()
        if not rows:
            return []
        clusters: dict[str, dict] = {}
        for row in rows:
            cluster_id = row["cluster_id"]
            entry = clusters.setdefault(
                cluster_id,
                {
                    "label": row["label"] or "Unlabeled Theme",
                    "chunk_count": 0,
                    "doc_counts": Counter(),
                    "doc_titles": {},
                    "texts": [],
                    "snippets": [],
                },
            )
            entry["chunk_count"] += 1
            doc_id = row["doc_id"]
            entry["doc_counts"][doc_id] += 1
            entry["doc_titles"][doc_id] = row["title"]
            if row["text"]:
                entry["texts"].append(row["text"])
                entry["snippets"].append(
                    (
                        float(row["similarity"] or 0.0),
                        row["text"],
                        row["title"] or "Unknown document",
                    )
                )
        ordered = sorted(clusters.values(), key=lambda item: item["chunk_count"], reverse=True)
        for item in ordered[:max_themes]:
            documents = _prepare_theme_evidence(item["snippets"], max_docs=max_docs, max_snippets=max_snippets)
            themes.append(
                {
                    "label": item["label"],
                    "similarity": None,
                    "bullets": [],
                    "documents": documents,
                    "chunk_count": item["chunk_count"],
                    "doc_count": len(item["doc_counts"]),
                }
            )
    if themes:
        summaries = _summarize_theme_bullets(llm, themes, max_bullets=max_bullets)
        for theme in themes:
            label = theme["label"]
            bullets = summaries.get(label)
            if not bullets:
                log_task_event(f"Theme summary missing for '{label}'; leaving bullets empty.")
                theme["bullets"] = []
                continue
            theme["bullets"] = bullets
    return themes


def is_ready(plan: PlanDraft, min_sources: int = 3) -> tuple[bool, list[str]]:
    gaps = list(plan.gaps)
    if len(plan.source_briefs) < min_sources:
        gaps.append("Insufficient sources to finalize the plan.")
    ready = len(gaps) == 0
    return ready, gaps


def _extract_json(text: str) -> dict:
    if not text:
        return {}
    for replace_quotes in (False, True):
        normalized = _normalize_json_text(text, replace_quotes=replace_quotes)
        for repair in (False, True):
            candidate = _repair_json_text(normalized) if repair else normalized
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start == -1 or end == -1 or end <= start:
                continue
            try:
                payload = json.loads(candidate[start : end + 1])
                return _normalize_payload_keys(payload)
            except json.JSONDecodeError:
                continue
    return {}


def _extract_json_list(text: str) -> list[object]:
    if not text:
        return []
    for replace_quotes in (False, True):
        normalized = _normalize_json_text(text, replace_quotes=replace_quotes)
        for repair in (False, True):
            candidate = _repair_json_text(normalized) if repair else normalized
            start = candidate.find("[")
            end = candidate.rfind("]")
            if start == -1 or end == -1 or end <= start:
                continue
            try:
                payload = json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
            return payload if isinstance(payload, list) else []
    return []


def _outline_from_text(text: str, limit: int = 50) -> list[str]:
    lines = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            line = line.lstrip("#").strip()
        if line[:1] in {"-", "*", "•"}:
            line = line[1:].strip()
        else:
            match = re.match(r"^\\d+[\\).]\\s+(.*)$", line)
            if match:
                line = match.group(1).strip()
        if line:
            lines.append(line)
        if len(lines) >= limit:
            break
    return lines


def _parse_plan_from_text(text: str) -> dict[str, list[str]]:
    sections = {
        "scope": [],
        "key_questions": [],
        "keywords": [],
        "gaps": [],
        "notes": [],
        "retrieval_queries": [],
    }
    current: str | None = None
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        lower = line.casefold().strip(":：")
        if lower.startswith("#"):
            lower = lower.lstrip("#").strip().strip(":：")
        if lower in {"scope"}:
            current = "scope"
            continue
        if lower in {"范围"}:
            current = "scope"
            continue
        if lower in {"key questions", "questions", "key_question", "key_questions"}:
            current = "key_questions"
            continue
        if lower in {"关键问题"}:
            current = "key_questions"
            continue
        if lower.startswith("keywords"):
            current = "keywords"
            # allow inline keywords after colon
            if ":" in line or "：" in line:
                inline = line.split(":", 1)[1].strip() if ":" in line else line.split("：", 1)[1].strip()
                if inline:
                    parts = [p.strip() for p in re.split(r"[;,]", inline) if p.strip()]
                    sections["keywords"].extend(parts)
            continue
        if lower.startswith("关键词"):
            current = "keywords"
            if ":" in line or "：" in line:
                inline = line.split(":", 1)[1].strip() if ":" in line else line.split("：", 1)[1].strip()
                if inline:
                    parts = [p.strip() for p in re.split(r"[;,，]", inline) if p.strip()]
                    sections["keywords"].extend(parts)
            continue
        if lower.startswith("gaps"):
            current = "gaps"
            continue
        if lower.startswith("缺口") or lower.startswith("差距"):
            current = "gaps"
            continue
        if lower.startswith("notes"):
            current = "notes"
            continue
        if lower.startswith("备注"):
            current = "notes"
            continue
        if lower.startswith("retrieval queries") or lower.startswith("retrieval_queries"):
            current = "retrieval_queries"
            continue
        if lower.startswith("检索查询") or lower.startswith("检索关键词"):
            current = "retrieval_queries"
            continue

        item = line
        if item.startswith(("-", "*", "•")):
            item = item[1:].strip()
        else:
            match = re.match(r"^\\d+[\\).]\\s+(.*)$", item)
            if match:
                item = match.group(1).strip()
        if not current:
            continue
        if current == "keywords":
            parts = [p.strip() for p in re.split(r"[;,]", item) if p.strip()]
            sections["keywords"].extend(parts)
        else:
            sections[current].append(item)
    return sections


def _truncate_text(text: str, limit: int = 2000) -> str:
    if not text:
        return "<empty>"
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated {len(text) - limit} chars]"


def _log_llm_failure(agent: str, stage: str, response: str) -> None:
    snippet = _truncate_text(response).replace("\r", "\\r").replace("\n", "\\n")
    log_task_event(f"{agent} | LLM {stage} raw response (truncated): {snippet}")


def _normalize_json_text(text: str, replace_quotes: bool = False) -> str:
    if not text:
        return ""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    replacements = {
        "：": ":",
        "，": ",",
        "、": ",",
        "；": ";",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "｛": "{",
        "｝": "}",
        "\u3000": " ",
    }
    if replace_quotes:
        replacements.update(
            {
                "“": "\"",
                "”": "\"",
                "‘": "\"",
                "’": "\"",
            }
        )
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _repair_json_text(text: str) -> str:
    if not text:
        return ""
    # Fix invalid unicode escapes like "\4e9b" -> "\u4e9b"
    text = re.sub(r"\\([0-9a-fA-F]{4})", r"\\u\1", text)
    # Escape any remaining invalid backslash escapes.
    text = re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", text)
    return text


def _normalize_payload_keys(payload: object) -> dict:
    if not isinstance(payload, dict):
        return {}
    normalized: dict[str, object] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            continue
        key_clean = key.strip()
        key_fold = key_clean.casefold().replace("-", " ").replace("_", " ").strip()
        mapped = _JSON_KEY_MAP.get(key_clean)
        if not mapped:
            mapped = _JSON_KEY_MAP.get(key_fold)
        if not mapped:
            mapped = _JSON_KEY_MAP.get(key_fold.replace(" ", " "))
        if not mapped:
            mapped = key_clean
        normalized[mapped] = value
    return normalized


def _clean_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_outline(outline: list[object]) -> list[str]:
    normalized: list[str] = []
    for item in outline:
        if isinstance(item, str):
            text = item.strip()
            if text:
                normalized.append(text)
            continue
        if isinstance(item, dict):
            title = str(item.get("title") or item.get("step") or item.get("name") or "").strip()
            if not title:
                continue
            lines = [title]
            substeps = item.get("substeps") or item.get("steps") or item.get("items") or []
            if isinstance(substeps, list):
                for sub in substeps:
                    if isinstance(sub, str):
                        sub_text = sub.strip()
                        if sub_text:
                            lines.append(f"- {sub_text}")
                        continue
                    if isinstance(sub, dict):
                        sub_text = str(sub.get("text") or sub.get("title") or "").strip()
                        if sub_text:
                            lines.append(f"- {sub_text}")
                        sub_sub = sub.get("substeps") or sub.get("items") or []
                        if isinstance(sub_sub, list):
                            for sub_item in sub_sub:
                                if isinstance(sub_item, str):
                                    sub_item_text = sub_item.strip()
                                    if sub_item_text:
                                        lines.append(f"  - {sub_item_text}")
            normalized.append("\n".join(lines))
    return normalized


def _count_words(text: str) -> int:
    tokens = tokenize(text)
    english_tokens = len([token for token in tokens if token])
    # Approximate CJK word count by contiguous CJK sequences.
    cjk_words = len(_CJK_WORD_RE.findall(text or ""))
    return english_tokens + cjk_words


def outline_word_count(outline: list[str]) -> int:
    return _count_words(" ".join(outline))


def outline_cjk_ratio(outline: list[str]) -> float:
    text = " ".join(outline)
    cjk_tokens = len(_CJK_RE.findall(text))
    english_tokens = len([token for token in tokenize(text) if token])
    total = cjk_tokens + english_tokens
    return cjk_tokens / total if total else 0.0


def _apply_llm_fields(plan: PlanDraft, payload: dict) -> PlanDraft:
    plan.scope = _clean_list(payload.get("scope")) or plan.scope
    plan.key_questions = _clean_list(payload.get("key_questions")) or plan.key_questions
    plan.keywords = _clean_list(payload.get("keywords")) or plan.keywords
    plan.gaps = _clean_list(payload.get("gaps")) or plan.gaps
    plan.notes = _clean_list(payload.get("notes")) or plan.notes
    plan.retrieval_queries = _clean_list(payload.get("retrieval_queries")) or plan.retrieval_queries
    readiness = payload.get("readiness")
    if isinstance(readiness, str) and readiness.strip():
        plan.readiness = readiness.strip().lower()
    return plan


def _base_plan(
    topic: str,
    interests: list[Interest],
    documents: list[Document],
    round_number: int,
    keywords_seed: list[str],
    graph_top_clusters: int | None = None,
    top_k_docs: int | None = None,
    doc_text_overrides: dict[str, str] | None = None,
) -> PlanDraft:
    cluster_limit = int(graph_top_clusters or GRAPH_TOP_CLUSTERS)
    doc_limit = int(top_k_docs or 0)
    source_types = []
    seen = set()
    for doc in documents:
        if doc.source_type not in seen:
            seen.add(doc.source_type)
            source_types.append(doc.source_type)

    source_briefs = create_source_briefs(
        documents,
        keywords_seed,
        doc_text_overrides=doc_text_overrides,
    )

    return PlanDraft(
        topic=topic,
        created_at=_now_iso(),
        round_number=round_number,
        scope=[],
        key_questions=[],
        keywords=[],
        source_types=source_types,
        source_briefs=source_briefs,
        gaps=[],
        readiness="draft",
        notes=[],
        retrieval_queries=[],
        graph_top_clusters=cluster_limit,
        top_k_docs=doc_limit,
    )


def _validate_llm_plan(plan: PlanDraft) -> None:
    missing = []
    if not plan.scope:
        missing.append("scope")
    if not plan.key_questions:
        missing.append("key_questions")
    if not plan.keywords:
        missing.append("keywords")
    if missing:
        raise LLMClientError(f"LLM output missing fields: {', '.join(missing)}")


def llm_draft_plan(
    llm: LLMClient,
    topic: str,
    interests: list[Interest],
    methods: list[Method],
    extracted_interests: list[str] | None,
    documents: list[Document],
    round_number: int,
    graph_top_clusters: int | None = None,
    top_k_docs: int | None = None,
    doc_text_overrides: dict[str, str] | None = None,
    skill_guidance: list[str] | None = None,
) -> PlanDraft:
    keywords_seed = build_keywords(topic, interests)
    base_plan = _base_plan(
        topic,
        interests,
        documents,
        round_number,
        keywords_seed,
        graph_top_clusters,
        top_k_docs,
        doc_text_overrides,
    )
    prompt = plan_prompt(
        topic,
        interests,
        methods,
        documents,
        keywords_seed,
        extracted_interests=extracted_interests,
        doc_text_overrides=doc_text_overrides,
        skill_guidance=skill_guidance,
    )
    response = llm.generate(SYSTEM_PLAN_PROMPT, prompt, task="plan draft", agent="Planner")
    payload = _extract_json(response)
    if not payload:
        parsed = _parse_plan_from_text(response)
        if any(parsed.values()):
            payload = parsed
            log_task_event("Planner: recovered plan draft from text fallback.")
        else:
            _log_llm_failure("Planner", "plan draft", response)
            raise LLMClientError("LLM returned invalid JSON for plan draft.")
    plan = _apply_llm_fields(base_plan, payload)
    if not plan.keywords:
        plan.keywords = keywords_seed or [topic]
    if not plan.scope:
        plan.scope = [f"Research and summarize {topic}."]
    if not plan.key_questions:
        plan.key_questions = [f"What are the core ideas and open questions in {topic}?"]
    try:
        _validate_llm_plan(plan)
    except LLMClientError as exc:
        _log_llm_failure("Planner", "plan draft", response)
        raise LLMClientError(str(exc)) from exc
    return plan


def llm_refine_plan(
    llm: LLMClient,
    plan: PlanDraft,
    documents: list[Document],
    interests: list[Interest],
    methods: list[Method],
    extracted_interests: list[str] | None,
    round_number: int,
    graph_top_clusters: int | None = None,
    top_k_docs: int | None = None,
    doc_text_overrides: dict[str, str] | None = None,
    skill_guidance: list[str] | None = None,
) -> PlanDraft:
    keywords_seed = plan.keywords or build_keywords(plan.topic, interests)
    base_plan = _base_plan(
        plan.topic,
        interests,
        documents,
        round_number,
        keywords_seed,
        graph_top_clusters,
        top_k_docs,
        doc_text_overrides,
    )
    base_plan.readiness = "refined"
    prior_plan = {
        "scope": plan.scope,
        "key_questions": plan.key_questions,
        "keywords": plan.keywords,
        "gaps": plan.gaps,
        "notes": plan.notes,
        "readiness": plan.readiness,
    }
    prompt = refine_prompt(
        plan.topic,
        prior_plan,
        interests,
        methods,
        documents,
        keywords_seed,
        extracted_interests=extracted_interests,
        doc_text_overrides=doc_text_overrides,
        skill_guidance=skill_guidance,
    )
    response = llm.generate(SYSTEM_PLAN_PROMPT, prompt, task="plan refinement", agent="Planner")
    payload = _extract_json(response)
    if not payload:
        parsed = _parse_plan_from_text(response)
        if any(parsed.values()):
            payload = parsed
            log_task_event("Planner: recovered plan refinement from text fallback.")
        else:
            _log_llm_failure("Planner", "plan refinement", response)
            raise LLMClientError("LLM returned invalid JSON for plan refinement.")
    refined = _apply_llm_fields(base_plan, payload)
    if not refined.keywords:
        refined.keywords = keywords_seed or [plan.topic]
    if not refined.scope:
        refined.scope = plan.scope or [f"Research and summarize {plan.topic}."]
    if not refined.key_questions:
        refined.key_questions = plan.key_questions or [f"What are the core ideas and open questions in {plan.topic}?"]
    try:
        _validate_llm_plan(refined)
    except LLMClientError as exc:
        _log_llm_failure("Planner", "plan refinement", response)
        raise LLMClientError(str(exc)) from exc
    return refined


def _save_outline_snapshot(
    outline: list[str] | None,
    raw: str | None,
    prefix: str,
    label: str,
) -> None:
    ensure_dirs()
    safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "_", label).strip("_")
    if outline:
        path = PLANS_DIR / f"{prefix}_{safe_label}.json"
        payload = {"label": label, "outline": outline}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if raw:
        path = PLANS_DIR / f"{prefix}_{safe_label}_raw.txt"
        path.write_text(raw, encoding="utf-8")


def llm_build_outline(
    llm: LLMClient,
    topic: str,
    documents: list[Document],
    interests: list[Interest],
    methods: list[Method],
    keywords: list[str],
    doc_text_overrides: dict[str, str] | None = None,
    skill_guidance: list[str] | None = None,
    active_skills: list[str] | None = None,
    run_id: int | None = None,
    save_prefix: str | None = None,
) -> list[str]:
    def _parse_outline_response(response: str, attempt: int, task: str) -> list[str] | None:
        payload = _extract_json(response)
        outline = payload.get("outline")
        cleaned: list[str] = []
        if isinstance(outline, list):
            cleaned = _normalize_outline(outline)
        if not cleaned:
            list_payload = _extract_json_list(response)
            if list_payload:
                cleaned = _normalize_outline(list_payload)
                if cleaned:
                    log_task_event(f"Outliner: recovered outline from JSON list fallback ({task}).")
        if not cleaned:
            fallback = _outline_from_text(response)
            if fallback:
                log_task_event(f"Outliner: recovered outline from text fallback ({task}).")
                cleaned = fallback
        if not cleaned:
            _log_llm_failure("Outliner", f"{task} attempt {attempt}", response)
            return None
        return cleaned

    def _translate_outline_to_chinese(items: list[str]) -> list[str]:
        prompt = {
            "outline": items,
            "instructions": [
                "Translate all items to Chinese.",
                "Keep paper titles, dataset names, benchmarks, model names, APIs, and acronyms in English.",
                "Return JSON only as {\"outline\": [...]} with the same structure depth.",
            ],
        }
        response = llm.generate(
            system_prompt="You translate research plan outlines to Chinese. Return JSON only.",
            user_prompt=json.dumps(prompt, indent=2),
            task="outline translate",
            agent="Outliner",
        )
        payload = _extract_json(response)
        outline = payload.get("outline")
        if isinstance(outline, list):
            cleaned = _normalize_outline(outline)
            if cleaned:
                return cleaned
        list_payload = _extract_json_list(response)
        if list_payload:
            cleaned = _normalize_outline(list_payload)
            if cleaned:
                return cleaned
        return []
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    prefix = save_prefix or f"{timestamp}_outline"
    if run_id is not None and save_prefix is None:
        prefix = f"{prefix}_run_{run_id}"

    def _attempt(prompt_text: str, attempt: int) -> tuple[list[str], int, float] | None:
        response = llm.generate(SYSTEM_OUTLINE_PROMPT, prompt_text, task="outline", agent="Outliner")
        cleaned = _parse_outline_response(response, attempt, "outline")
        _save_outline_snapshot(cleaned, response, prefix, f"outline_attempt{attempt}")
        if not cleaned:
            return None
        count = outline_word_count(cleaned)
        ratio = outline_cjk_ratio(cleaned)
        if OUTLINE_MIN_WORDS <= count <= OUTLINE_MAX_WORDS and ratio >= OUTLINE_MIN_CJK_RATIO:
            return cleaned, count, ratio
        log_task_event(
            "Outliner: outline length out of range "
            f"(attempt={attempt} words={count} target={OUTLINE_MIN_WORDS}-{OUTLINE_MAX_WORDS})"
        )
        if ratio < OUTLINE_MIN_CJK_RATIO:
            log_task_event(
                "Outliner: outline language ratio too low "
                f"(attempt={attempt} cjk_ratio={ratio:.2f} min={OUTLINE_MIN_CJK_RATIO:.2f})"
            )
        return cleaned, count, ratio
    def _revision_prompt(
        previous_outline: list[str],
        length_hint: str,
        language_hint: str | None,
    ) -> str:
        payload = {
            "previous_outline": previous_outline,
            "instructions": {
                "output_json_schema": {
                    "outline": ["string"]
                },
                "requirements": [
                    f"Length must be {OUTLINE_MIN_WORDS}-{OUTLINE_MAX_WORDS} words in the output language.",
                    "Keep 8-12 major steps.",
                    "Each major step must include 3-5 substeps.",
                    "Each substep should be 2-3 sentences.",
                    "Preserve the original topic coverage and structure; rewrite to fit length.",
                ],
            "language_guidance": [
                "Write natural-language content in Chinese.",
                "Keep paper titles, dataset names, benchmarks, model names, APIs, and acronyms in English.",
            ],
        },
        "length_hint": length_hint,
    }
        if structure_guidance:
            payload["instructions"]["structure_guidance"] = structure_guidance
        if language_hint:
            payload["language_hint"] = language_hint
        return json.dumps(payload, indent=2)

    def _expand_prompt(
        previous_outline: list[str],
        length_hint: str,
        language_hint: str | None,
    ) -> str:
        payload = {
            "previous_outline": previous_outline,
            "instructions": {
                "output_json_schema": {
                    "outline": ["string"]
                },
                "requirements": [
                    f"Length must be at least {OUTLINE_MIN_WORDS} words in the output language.",
                    "Keep 8-12 major steps.",
                    "Each major step must include 3-5 substeps.",
                    "Each substep should be 2-3 sentences.",
                    "Do not remove content; only expand with additional detail and substeps.",
                ],
            "language_guidance": [
                "Write natural-language content in Chinese.",
                "Keep paper titles, dataset names, benchmarks, model names, APIs, and acronyms in English.",
            ],
        },
        "length_hint": length_hint,
    }
        if structure_guidance:
            payload["instructions"]["structure_guidance"] = structure_guidance
        if language_hint:
            payload["language_hint"] = language_hint
        return json.dumps(payload, indent=2)

    def _attempt_revision(
        previous_outline: list[str],
        previous_count: int,
        previous_ratio: float,
        attempt: int,
    ) -> tuple[list[str], int, float] | None:
        length_hint = (
            f"Your previous outline length was {previous_count} words. "
            f"Adjust to {OUTLINE_MIN_WORDS}-{OUTLINE_MAX_WORDS} words. "
            "You MUST be within range."
        )
        language_hint = None
        if previous_ratio < OUTLINE_MIN_CJK_RATIO:
            language_hint = (
                "Your previous outline was not sufficiently Chinese. "
                "Translate all step titles and substeps to Chinese; keep English only for paper titles, "
                "datasets, benchmarks, model names, APIs, and acronyms."
            )
        prompt_text = _revision_prompt(previous_outline, length_hint, language_hint)
        response = llm.generate(
            system_prompt="You revise research plan outlines to meet strict length constraints. Return JSON only.",
            user_prompt=prompt_text,
            task="outline revision",
            agent="Outliner",
        )
        cleaned = _parse_outline_response(response, attempt, "outline revision")
        _save_outline_snapshot(cleaned, response, prefix, f"outline_revision{attempt}")
        if not cleaned:
            return None
        count = outline_word_count(cleaned)
        ratio = outline_cjk_ratio(cleaned)
        if OUTLINE_MIN_WORDS <= count <= OUTLINE_MAX_WORDS and ratio >= OUTLINE_MIN_CJK_RATIO:
            return cleaned, count, ratio
        log_task_event(
            "Outliner: outline length still out of range after revision "
            f"(attempt={attempt} words={count} target={OUTLINE_MIN_WORDS}-{OUTLINE_MAX_WORDS})"
        )
        if ratio < OUTLINE_MIN_CJK_RATIO:
            log_task_event(
                "Outliner: outline language ratio too low after revision "
                f"(attempt={attempt} cjk_ratio={ratio:.2f} min={OUTLINE_MIN_CJK_RATIO:.2f})"
            )
        return cleaned, count, ratio

    def _attempt_expand(
        previous_outline: list[str],
        previous_count: int,
        previous_ratio: float,
        attempt: int,
    ) -> tuple[list[str], int, float] | None:
        length_hint = (
            f"Your previous outline length was {previous_count} words. "
            f"Expand to at least {OUTLINE_MIN_WORDS} words without changing the major steps. "
            "Add more substeps and elaboration (2-3 sentences per substep)."
        )
        language_hint = None
        if previous_ratio < OUTLINE_MIN_CJK_RATIO:
            language_hint = (
                "Your previous outline was not sufficiently Chinese. "
                "Translate all step titles and substeps to Chinese; keep English only for paper titles, "
                "datasets, benchmarks, model names, APIs, and acronyms."
            )
        prompt_text = _expand_prompt(previous_outline, length_hint, language_hint)
        response = llm.generate(
            system_prompt="You expand research plan outlines to meet strict minimum length. Return JSON only.",
            user_prompt=prompt_text,
            task="outline expansion",
            agent="Outliner",
        )
        cleaned = _parse_outline_response(response, attempt, "outline expansion")
        _save_outline_snapshot(cleaned, response, prefix, f"outline_expand{attempt}")
        if not cleaned:
            return None
        count = outline_word_count(cleaned)
        ratio = outline_cjk_ratio(cleaned)
        if OUTLINE_MIN_WORDS <= count <= OUTLINE_MAX_WORDS and ratio >= OUTLINE_MIN_CJK_RATIO:
            return cleaned, count, ratio
        log_task_event(
            "Outliner: outline length still out of range after expansion "
            f"(attempt={attempt} words={count} target={OUTLINE_MIN_WORDS}-{OUTLINE_MAX_WORDS})"
        )
        if ratio < OUTLINE_MIN_CJK_RATIO:
            log_task_event(
                "Outliner: outline language ratio too low after expansion "
                f"(attempt={attempt} cjk_ratio={ratio:.2f} min={OUTLINE_MIN_CJK_RATIO:.2f})"
            )
        return cleaned, count, ratio

    structure_guidance = None
    if active_skills:
        labels = [skill.replace("-", " ").title() for skill in active_skills]
        structure_guidance = [
            "Structure major steps into contiguous sections grouped by the triggered skills in this order: "
            + ", ".join(labels)
            + ".",
            "Prefix each major step title with the matching skill label (e.g., '[Systems Engineering] ...').",
            "Ensure each skill has at least one major step; do not introduce new section labels.",
        ]

    prompt = outline_prompt(
        topic,
        interests,
        methods,
        documents,
        keywords,
        doc_text_overrides=doc_text_overrides,
        structure_guidance=structure_guidance,
        skill_guidance=skill_guidance,
    )
    result = _attempt(prompt, 1)
    if result and OUTLINE_MIN_WORDS <= result[1] <= OUTLINE_MAX_WORDS and result[2] >= OUTLINE_MIN_CJK_RATIO:
        return result[0]

    previous_count = result[1] if result else 0
    previous_ratio = result[2] if result else 0.0
    length_hint = (
        f"Your previous outline length was {previous_count} words. "
        f"Expand or compress to {OUTLINE_MIN_WORDS}-{OUTLINE_MAX_WORDS} words. "
        "Add more detailed substeps (2-3 sentences each) to reach the target."
    )
    language_hint = (
        "Your previous outline was not sufficiently Chinese. "
        "Translate all step titles and substeps to Chinese; keep English only for paper titles, datasets, "
        "benchmarks, model names, APIs, and acronyms."
    )
    retry_prompt = outline_prompt(
        topic,
        interests,
        methods,
        documents,
        keywords,
        doc_text_overrides=doc_text_overrides,
        length_hint=length_hint,
        language_hint=language_hint if previous_ratio < OUTLINE_MIN_CJK_RATIO else None,
        structure_guidance=structure_guidance,
        skill_guidance=skill_guidance,
    )
    result = _attempt(retry_prompt, 2)
    if result and OUTLINE_MIN_WORDS <= result[1] <= OUTLINE_MAX_WORDS and result[2] >= OUTLINE_MIN_CJK_RATIO:
        return result[0]

    previous_count = result[1] if result else previous_count
    previous_ratio = result[2] if result else previous_ratio
    length_hint = (
        f"Length still out of range ({previous_count} words). "
        f"Target {OUTLINE_MIN_WORDS}-{OUTLINE_MAX_WORDS} words. "
        "Increase detail by expanding each major step with additional substeps."
    )
    final_prompt = outline_prompt(
        topic,
        interests,
        methods,
        documents,
        keywords,
        doc_text_overrides=doc_text_overrides,
        length_hint=length_hint,
        language_hint=language_hint if previous_ratio < OUTLINE_MIN_CJK_RATIO else None,
        structure_guidance=structure_guidance,
        skill_guidance=skill_guidance,
    )
    result = _attempt(final_prompt, 3)
    if result and result[2] >= OUTLINE_MIN_CJK_RATIO:
        if OUTLINE_MIN_WORDS <= result[1] <= OUTLINE_MAX_WORDS:
            return result[0]

    if result and result[1] < OUTLINE_MIN_WORDS:
        expanded = _attempt_expand(result[0], result[1], result[2], 4)
        if expanded and OUTLINE_MIN_WORDS <= expanded[1] <= OUTLINE_MAX_WORDS and expanded[2] >= OUTLINE_MIN_CJK_RATIO:
            return expanded[0]
        if expanded:
            result = expanded
            if result[1] < OUTLINE_MIN_WORDS:
                expanded = _attempt_expand(result[0], result[1], result[2], 5)
                if expanded and OUTLINE_MIN_WORDS <= expanded[1] <= OUTLINE_MAX_WORDS and expanded[2] >= OUTLINE_MIN_CJK_RATIO:
                    return expanded[0]
                if expanded:
                    result = expanded

    if result:
        enforced = _attempt_revision(result[0], result[1], result[2], 6)
        if enforced and OUTLINE_MIN_WORDS <= enforced[1] <= OUTLINE_MAX_WORDS and enforced[2] >= OUTLINE_MIN_CJK_RATIO:
            return enforced[0]

        translated = _translate_outline_to_chinese(result[0])
        if translated:
            _save_outline_snapshot(translated, None, prefix, "outline_translate")
            translated_count = outline_word_count(translated)
            translated_ratio = outline_cjk_ratio(translated)
            if OUTLINE_MIN_WORDS <= translated_count <= OUTLINE_MAX_WORDS and translated_ratio >= OUTLINE_MIN_CJK_RATIO:
                return translated

        log_task_event(
            "Outliner: returning best-effort outline despite length constraint failure. "
            f"Last count={result[1]} target={OUTLINE_MIN_WORDS}-{OUTLINE_MAX_WORDS}."
        )
        return result[0]

    raise LLMClientError("LLM returned invalid JSON for outline.")


def render_plan_md(
    plan: PlanDraft,
    outline: list[str],
    interests: list[Interest],
    methods: list[Method],
    llm: LLMClient | None = None,
) -> str:
    interest_summary = summarize_interests(interests)
    method_summary = summarize_methods(methods)

    lines = [
        f"# Research Plan - {plan.topic}",
        "",
        f"- Created at: {plan.created_at}",
        f"- Round: {plan.round_number}",
        f"- Readiness: {plan.readiness}",
        "",
        "## Scope",
    ]
    for item in plan.scope:
        lines.append(f"- {item}")

    lines.append("")
    lines.append("## Key Questions")
    for item in plan.key_questions:
        lines.append(f"- {item}")

    lines.append("")
    lines.append("## Keywords")
    lines.append("- " + ", ".join(plan.keywords))

    lines.append("")
    lines.append("## Source Types in Selected Docs")
    if plan.source_types:
        for item in plan.source_types:
            lines.append(f"- {item}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("## Source Types in Library")
    db.init_db()
    db._ensure_document_stats()
    with db._connect() as conn:
        rows = conn.execute(
            """
            SELECT d.source_type, COUNT(*) AS count
            FROM documents d
            JOIN document_stats s ON d.doc_id = s.doc_id
            WHERE s.archived = 0
            GROUP BY d.source_type
            ORDER BY d.source_type
            """
        ).fetchall()
    if rows:
        for row in rows:
            lines.append(f"- {row['source_type']}: {row['count']}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("## Gaps and Retrieval Needs")
    if plan.gaps:
        for item in plan.gaps:
            lines.append(f"- {item}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("## Notes")
    for item in plan.notes:
        lines.append(f"- {item}")

    lines.append("")
    lines.append("## Analysis Methods")
    if method_summary:
        for item in method_summary:
            lines.append(f"- {item}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("## Manually Added Interests")
    if interest_summary:
        for item in interest_summary:
            lines.append(f"- {item}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("## Locally Extracted Interests")
    themes = build_interest_themes(
        plan.topic,
        interests,
        query_hints=plan.retrieval_queries,
        llm=llm,
        max_themes=plan.graph_top_clusters,
        top_k_docs=plan.top_k_docs,
    )
    if not themes:
        lines.append(
            "No themes available yet. Ingest local files and run research with embeddings enabled to build the graph."
        )
    else:
        for theme in themes:
            similarity = theme.get("similarity")
            if isinstance(similarity, (int, float)):
                lines.append(f"Theme: {theme['label']} (sim={similarity:.2f})")
            else:
                lines.append(f"Theme: {theme['label']}")
            for bullet in theme["bullets"]:
                lines.append(f"- {bullet}")
            lines.append("")

    lines.append("")
    lines.append("## Research Plan")
    for idx, item in enumerate(outline, start=1):
        parts = [part for part in str(item).splitlines() if part.strip()]
        if not parts:
            continue
        lines.append(f"{idx}. {parts[0]}")
        for extra in parts[1:]:
            lines.append(f"   {extra}")

    lines.append("")
    lines.append("## Next Actions")
    lines.append("- Ingest missing sources for the identified gaps.")
    lines.append("- Run another planning round after updating the library.")

    return "\n".join(lines)
