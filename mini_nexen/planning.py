from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from . import db
from .db import Document, Interest, Method, load_document_text
from .embeddings import EmbeddingClient, EmbeddingConfig, cosine_similarity, normalize
from .llm import LLMClient, LLMClientError, log_task_event
from .llm_prompts import SYSTEM_OUTLINE_PROMPT, SYSTEM_PLAN_PROMPT, outline_prompt, plan_prompt, refine_prompt
from .text_utils import top_sentences, tokenize


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
) -> list[SourceBrief]:
    from .llm import log_task_event

    briefs = []
    total_highlights = 0
    counts: dict[str, int] = {}
    for doc in documents:
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
        for idx in group:
            theme = themes[idx]
            chunk_count += int(theme.get("chunk_count") or 0)
            for doc in theme.get("documents", []):
                title = doc.get("title") or "Unknown document"
                evidence = doc.get("evidence") or []
                combined_docs.setdefault(title, [])
                for item in evidence:
                    if item not in combined_docs[title]:
                        combined_docs[title].append(item)

        documents = [{"title": title, "evidence": evidence} for title, evidence in combined_docs.items()]
        merged.append(
            {
                "label": merged_label,
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
            "Return JSON only: {\"themes\":[{\"theme\":\"...\",\"bullets\":[...]}]}."
        ),
        "requirements": [
            f"Provide 2-{max_bullets} bullets per theme.",
            "Bullets must be full-sentence reasoning, not keyword lists.",
            "Bullets must be plain strings; do not return objects.",
            "Highlight recurring patterns, commonalities, and contradictions when present.",
            "Explicitly connect the evidence to the theme label.",
            "Sources are optional; include zero, one, or multiple titles only if it improves clarity.",
            "If listing sources, format as: Sources: title1; title2",
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
    llm: LLMClient | None = None,
    max_themes: int = 6,
    max_docs: int = 3,
    max_bullets: int = 4,
    max_snippets: int = 3,
    merge_threshold: float = 0.85,
) -> list[dict]:
    if not llm:
        raise LLMClientError("LLM is required for interest theme summaries.")
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
    themes = []
    for item in ordered[:max_themes]:
        doc_counts = item["doc_counts"].most_common(max_docs)
        doc_titles = [item["doc_titles"][doc_id] for doc_id, _ in doc_counts]
        documents = _prepare_theme_evidence(item["snippets"], max_docs=max_docs, max_snippets=max_snippets)
        themes.append(
            {
                "label": item["label"],
                "bullets": [],
                "documents": documents,
                "chunk_count": item["chunk_count"],
                "doc_count": len(item["doc_counts"]),
            }
        )
    if themes:
        themes = _merge_themes_by_label_similarity(llm, themes, threshold=merge_threshold)
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
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def _extract_json_list(text: str) -> list[object]:
    if not text:
        return []
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


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
        lower = line.casefold().strip(":")
        if lower.startswith("#"):
            lower = lower.lstrip("#").strip().strip(":")
        if lower in {"scope"}:
            current = "scope"
            continue
        if lower in {"key questions", "questions", "key_question", "key_questions"}:
            current = "key_questions"
            continue
        if lower.startswith("keywords"):
            current = "keywords"
            # allow inline keywords after colon
            if ":" in line:
                inline = line.split(":", 1)[1].strip()
                if inline:
                    parts = [p.strip() for p in re.split(r"[;,]", inline) if p.strip()]
                    sections["keywords"].extend(parts)
            continue
        if lower.startswith("gaps"):
            current = "gaps"
            continue
        if lower.startswith("notes"):
            current = "notes"
            continue
        if lower.startswith("retrieval queries") or lower.startswith("retrieval_queries"):
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
    return len([token for token in tokens if token])


def outline_word_count(outline: list[str]) -> int:
    return _count_words(" ".join(outline))


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
) -> PlanDraft:
    source_types = []
    seen = set()
    for doc in documents:
        if doc.source_type not in seen:
            seen.add(doc.source_type)
            source_types.append(doc.source_type)

    source_briefs = create_source_briefs(documents, keywords_seed)

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
    skill_guidance: list[str] | None = None,
) -> PlanDraft:
    keywords_seed = build_keywords(topic, interests)
    base_plan = _base_plan(topic, interests, documents, round_number, keywords_seed)
    prompt = plan_prompt(
        topic,
        interests,
        methods,
        documents,
        keywords_seed,
        extracted_interests=extracted_interests,
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
    skill_guidance: list[str] | None = None,
) -> PlanDraft:
    keywords_seed = plan.keywords or build_keywords(plan.topic, interests)
    base_plan = _base_plan(plan.topic, interests, documents, round_number, keywords_seed)
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


def llm_build_outline(
    llm: LLMClient,
    topic: str,
    documents: list[Document],
    interests: list[Interest],
    methods: list[Method],
    keywords: list[str],
    skill_guidance: list[str] | None = None,
) -> list[str]:
    prompt = outline_prompt(topic, interests, methods, documents, keywords, skill_guidance=skill_guidance)
    response = llm.generate(SYSTEM_OUTLINE_PROMPT, prompt, task="outline", agent="Outliner")
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
            log_task_event("Outliner: recovered outline from JSON list fallback.")
            return cleaned
    fallback = _outline_from_text(response)
    if fallback:
        log_task_event("Outliner: recovered outline from text fallback.")
        return fallback
    _log_llm_failure("Outliner", "outline", response)
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
    lines.append("## Source Types in Library")
    if plan.source_types:
        for item in plan.source_types:
            lines.append(f"- {item}")
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
    themes = build_interest_themes(llm=llm)
    if not themes:
        lines.append(
            "No themes available yet. Ingest local files and run research with embeddings enabled to build the graph."
        )
    else:
        for theme in themes:
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
