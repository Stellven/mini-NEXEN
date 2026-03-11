from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from . import db
from .config import (
    ARTIFACTS_DIR,
    DEFAULT_OUTLINE_PROFILE_REVIEW_ROUNDS,
    DEFAULT_PROFILE_TOP_K,
    ensure_dirs,
)
from .db import Document, Interest, Method
from .kg import KGStore
from .llm import LLMClient, LLMClientError, emit_progress, log_task_event
from .llm_prompts import (
    SYSTEM_OUTLINE_PROMPT,
    SYSTEM_PLAN_PROMPT,
    SYSTEM_REVIEW_PROMPT,
    outline_profile_review_prompt,
    outline_prompt,
    plan_prompt,
    plan_quality_review_prompt,
    plan_readiness_review_prompt,
    refine_prompt,
    outline_quality_review_prompt,
)
from .text_utils import tokenize

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


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
    "source_requirements": "source_requirements",
    "source requirements": "source_requirements",
    "source requirement": "source_requirements",
    "section_requirements": "section_requirements",
    "section requirements": "section_requirements",
    "sections": "section_requirements",
    "section plan": "section_requirements",
    "研究段落": "section_requirements",
    "章节需求": "section_requirements",
    "来源要求": "source_requirements",
    "来源需求": "source_requirements",
    "证据要求": "source_requirements",
    "证据需求": "source_requirements",
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
    source_requirements: dict[str, object]
    section_requirements: list[dict[str, object]]
    source_types: list[str]
    source_briefs: list[SourceBrief]
    gaps: list[str]
    readiness: str
    notes: list[str]
    retrieval_queries: list[str]
    profile_top_k: int
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
) -> list[SourceBrief]:
    from .llm import log_task_event

    briefs = []
    counts: dict[str, int] = {}
    for doc in documents:
        briefs.append(SourceBrief(doc=doc, highlights=[]))
        counts[doc.source_type] = counts.get(doc.source_type, 0) + 1
    log_task_event(f"Source briefs: docs={len(briefs)}")
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



def _prepare_profile_evidence(
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


def _summarize_profile_bullets(
    llm: LLMClient,
    signals: list[dict],
    max_bullets: int,
) -> dict[str, list[str]]:
    payload = []
    for signal in signals:
        payload.append(
            {
                "signal": signal["label"],
                "chunk_count": signal.get("chunk_count", 0),
                "doc_count": signal.get("doc_count", 0),
                "documents": signal["documents"],
            }
        )
    prompt = {
        "instruction": (
            "You summarize profile signals based on the evidence provided. "
            "Use ONLY the evidence provided. Do NOT quote text; paraphrase. "
            "Assume the reader is a domain expert and prioritize technical nuance over basic definitions. "
            "Avoid words like 'user', 'interest', or 'focus'. "
            "Write all natural-language content in Chinese. "
            "Keep paper titles, dataset names, benchmarks, model names, APIs, and acronyms in English. "
            "Return JSON only: "
            "{\"signals\":[{\"signal\":\"...\",\"bullets\":[{\"bullet\":\"...\",\"sources\":[\"title1\",\"title2\"]}]}]}."
        ),
        "requirements": [
            f"Provide 2-3 bullets per signal by default; expand up to {max_bullets} when evidence is abundant.",
            "Bullets must be full-sentence reasoning, not keyword lists.",
            "Highlight recurring patterns, commonalities, and contradictions when present.",
            "Prioritize insightful, controversial, paradigm-shifting, noteworthy, or important findings grounded in the evidence.",
            "Call out shifts in perspective or direction when supported; distinguish stable consensus from active debates.",
            "Explicitly connect the evidence to the signal label.",
            "Each bullet must be an object with fields: bullet (string) and sources (array of document titles).",
            "Sources may be empty, but include 1-2 titles when they improve traceability.",
        ],
        "signals": payload,
    }
    model_label = llm.config.model
    emit_progress("ProfileSignals", model_label, "profile signals", 0, 1, done=False)
    try:
        response = llm.generate(
            system_prompt="You summarize evidence into concise, traceable bullets.",
            user_prompt=json.dumps(prompt, indent=2),
            task="profile signals",
            agent="ProfileSignals",
        )
    finally:
        emit_progress("ProfileSignals", model_label, "profile signals", 1, 1, done=True)
    parsed = _extract_json(response)
    if not parsed:
        raise LLMClientError("LLM returned invalid JSON for profile signals.")
    items = parsed.get("signals") or parsed.get("themes")
    if not isinstance(items, list):
        raise LLMClientError("LLM returned invalid profile signal payload.")
    result: dict[str, list[str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        label = str(item.get("signal", "")).strip() or str(item.get("theme", "")).strip()
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
        raise LLMClientError("LLM returned empty profile signals.")
    return result


def build_profile_signals(
    topic: str,
    interests: list[Interest],
    query_hints: list[str] | None = None,
    llm: LLMClient | None = None,
    top_k_docs: int | None = None,
    max_signals: int = 6,
    max_docs: int = 3,
    max_bullets: int = 7,
    max_snippets: int = 3,
    *,
    use_cache: bool = True,
    cache_result: bool = False,
) -> list[dict]:
    store = KGStore()
    if use_cache:
        cached = store.get_profile_summary(max_signals=max_signals, scope="local")
        if cached is not None:
            return cached
    profile = store.get_profile(limit=max_signals)
    if not profile:
        return []

    signals: list[dict] = []
    local_sources = {"file", "note", "url"}
    for item in profile:
        entity_id = item.get("entity_id")
        label = item.get("entity") or "Unknown Topic"
        if not entity_id:
            continue
        evidence = store.get_entity_evidence(
            entity_id,
            limit=max_docs * max_snippets * 4,
            source_types=local_sources,
        )
        if not evidence:
            mentions = store.get_entity_mentions(
                entity_id,
                limit=max_docs * max_snippets * 4,
                source_types=local_sources,
            )
            evidence = [
                {
                    "quote": m.get("sentence") or "",
                    "doc_id": m.get("doc_id"),
                    "title": m.get("title") or "",
                    "confidence": 0.3,
                }
                for m in mentions
                if m.get("sentence")
            ]
        documents: dict[str, list[str]] = {}
        for item_ev in evidence:
            title = item_ev.get("title") or "Unknown document"
            quote = item_ev.get("quote") or ""
            if not quote:
                continue
            documents.setdefault(title, [])
            if quote not in documents[title]:
                documents[title].append(quote)
        doc_entries = [
            {"title": title, "evidence": quotes[:max_snippets]}
            for title, quotes in documents.items()
        ]
        if not doc_entries:
            continue
        signals.append(
            {
                "label": label,
                "similarity": item.get("salience"),
                "bullets": [],
                "documents": doc_entries[:max_docs],
                "chunk_count": 0,
                "doc_count": len(doc_entries),
            }
        )

    if not signals:
        return []

    if llm:
        try:
            summaries = _summarize_profile_bullets(llm, signals, max_bullets=max_bullets)
        except LLMClientError as exc:
            log_task_event(f"Profile signal summarization failed: {exc}")
            summaries = {}
        for signal in signals:
            label = signal["label"]
            bullets = summaries.get(label)
            if not bullets:
                log_task_event(f"Profile summary missing for '{label}'; leaving bullets empty.")
                signal["bullets"] = []
                continue
            signal["bullets"] = bullets
        if cache_result:
            store.set_profile_summary(signals, scope="local")
        return signals

    for signal in signals:
        bullets = []
        for doc in signal["documents"]:
            for quote in doc.get("evidence", [])[:max_bullets]:
                bullets.append(_compact_snippet(quote, limit=240))
                if len(bullets) >= max_bullets:
                    break
            if len(bullets) >= max_bullets:
                break
        signal["bullets"] = bullets
    if cache_result:
        store.set_profile_summary(signals, scope="local")
    return signals


def is_ready(plan: PlanDraft, min_sources: int = 3) -> tuple[bool, list[str]]:
    gaps = list(plan.gaps)
    if len(plan.source_briefs) < min_sources:
        gaps.append("Insufficient sources to finalize the plan.")
    ready = len(gaps) == 0
    return ready, gaps


def validate_plan(plan: PlanDraft) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []

    if len(plan.scope) < 3:
        errors.append("scope must include at least 3 items.")
    if len(plan.key_questions) < 3:
        errors.append("key_questions must include at least 3 items.")
    if len(plan.keywords) < 3:
        errors.append("keywords must include at least 3 items.")
    if plan.readiness not in {"draft", "refined", "ready"}:
        warnings.append("readiness should be draft/refined/ready.")

    if not isinstance(plan.source_requirements, dict) or not plan.source_requirements:
        errors.append("source_requirements must be a non-empty object.")
    else:
        required_keys = {"depth", "breadth", "rigor", "recency", "source_types", "min_sources"}
        missing = [key for key in required_keys if key not in plan.source_requirements]
        if missing:
            errors.append(f"source_requirements missing keys: {', '.join(missing)}.")

    if not isinstance(plan.section_requirements, list) or not plan.section_requirements:
        errors.append("section_requirements must be a non-empty array.")
    else:
        for idx, item in enumerate(plan.section_requirements, start=1):
            if not isinstance(item, dict):
                errors.append(f"section_requirements[{idx}] must be an object.")
                continue
            section = str(item.get("section") or "").strip()
            evidence = item.get("evidence_requirements") or {}
            if not section:
                errors.append(f"section_requirements[{idx}] missing section name.")
            if not isinstance(evidence, dict) or not evidence:
                errors.append(f"section_requirements[{idx}] missing evidence_requirements.")

    if not plan.retrieval_queries:
        warnings.append("retrieval_queries are empty.")

    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}


def _outline_evidence_alignment_ok(line: str, kg_fact_cards: list[dict]) -> bool:
    if not line or not kg_fact_cards:
        return False
    lowered = line.casefold()
    needles: list[str] = []
    for card in kg_fact_cards:
        for key in ("claim", "statement", "subject", "object"):
            text = str(card.get(key) or "").strip()
            if text and text.casefold() not in needles:
                needles.append(text.casefold())
    for needle in needles:
        if len(needle) < 3:
            continue
        if needle in lowered:
            return True
    return False


def validate_outline(
    outline: list[str],
    output_language: str,
    kg_fact_cards: list[dict] | None = None,
    profile_summary: list[dict] | None = None,
) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    if not outline:
        errors.append("outline must not be empty.")
        return {"ok": False, "errors": errors, "warnings": warnings}

    count = outline_word_count(outline)
    if count < OUTLINE_MIN_WORDS:
        errors.append(f"outline length below minimum ({count} < {OUTLINE_MIN_WORDS}).")
    if _is_chinese_language(output_language):
        ratio = outline_cjk_ratio(outline)
        if ratio < OUTLINE_MIN_CJK_RATIO:
            errors.append(
                f"outline language ratio below minimum ({ratio:.2f} < {OUTLINE_MIN_CJK_RATIO:.2f})."
            )
    evidence_cards = kg_fact_cards or []
    for line in outline:
        if "[關切證據]" in line or "[关切证据]" in line or "[profile evidence]" in line.casefold():
            if not evidence_cards:
                warnings.append("outline uses [关切证据] but no evidence cards are available; consider [关切].")
                continue
            if not _outline_evidence_alignment_ok(line, evidence_cards):
                warnings.append("outline uses [关切证据] without aligned evidence; consider [关切].")
    if profile_summary and not _outline_has_profile_tags(outline):
        warnings.append(
            "missing_profile_tags: outline has profile signals available but no [关切]/[关切证据] tags."
        )
    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}


def review_plan_readiness(
    llm: LLMClient,
    plan: PlanDraft,
    output_language: str,
) -> tuple[bool, list[str]]:
    prompt = plan_readiness_review_prompt(
        topic=plan.topic,
        plan={
            "scope": plan.scope,
            "key_questions": plan.key_questions,
            "keywords": plan.keywords,
            "gaps": plan.gaps,
            "notes": plan.notes,
            "readiness": plan.readiness,
        },
        source_briefs_count=len(plan.source_briefs),
        source_types=plan.source_types,
        output_language=output_language,
    )
    try:
        response = llm.generate(
            SYSTEM_REVIEW_PROMPT,
            prompt,
            task="plan readiness review",
            agent="Reviewer",
        )
    except Exception as exc:
        log_task_event(f"Reviewer failed: plan readiness review error={exc}")
        return is_ready(plan)
    payload = _extract_json(response)
    if not payload:
        log_task_event("Reviewer: invalid JSON for plan readiness review.")
        return is_ready(plan)
    ready_flag = payload.get("ready")
    readiness = payload.get("readiness")
    if isinstance(ready_flag, str):
        ready_flag = ready_flag.strip().lower() == "true"
    gaps = _clean_list(payload.get("gaps"))
    if isinstance(readiness, str) and readiness.strip().lower() == "ready":
        return True, gaps
    if isinstance(ready_flag, bool):
        return ready_flag, gaps
    return is_ready(plan)


def review_plan_quality(
    llm: LLMClient,
    plan: PlanDraft,
    validation: dict[str, object],
    output_language: str,
) -> dict | None:
    prompt = plan_quality_review_prompt(
        topic=plan.topic,
        plan={
            "scope": plan.scope,
            "key_questions": plan.key_questions,
            "keywords": plan.keywords,
            "source_requirements": plan.source_requirements,
            "section_requirements": plan.section_requirements,
            "gaps": plan.gaps,
            "notes": plan.notes,
            "readiness": plan.readiness,
            "retrieval_queries": plan.retrieval_queries,
        },
        validation=validation,
        output_language=output_language,
    )
    try:
        response = llm.generate(
            SYSTEM_REVIEW_PROMPT,
            prompt,
            task="plan quality review",
            agent="Reviewer",
        )
    except Exception as exc:
        log_task_event(f"Reviewer failed: plan quality review error={exc}")
        return None
    payload = _extract_json(response)
    if not payload:
        log_task_event("Reviewer: invalid JSON for plan quality review.")
        return None
    return payload


def review_outline_quality(
    llm: LLMClient,
    topic: str,
    outline: list[str],
    plan: PlanDraft,
    validation: dict[str, object],
    output_language: str,
) -> dict | None:
    plan_requirements = {
        "source_requirements": plan.source_requirements,
        "section_requirements": plan.section_requirements,
        "key_questions": plan.key_questions,
        "scope": plan.scope,
    }
    prompt = outline_quality_review_prompt(
        topic=topic,
        outline=outline,
        plan_requirements=plan_requirements,
        validation=validation,
        output_language=output_language,
    )
    try:
        response = llm.generate(
            SYSTEM_REVIEW_PROMPT,
            prompt,
            task="outline quality review",
            agent="Reviewer",
        )
    except Exception as exc:
        log_task_event(f"Reviewer failed: outline quality review error={exc}")
        return None
    payload = _extract_json(response)
    if not payload:
        log_task_event("Reviewer: invalid JSON for outline quality review.")
        return None
    return payload


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
    # Drop malformed unicode escapes like "\u法" to keep the following character.
    text = re.sub(r"\\u(?![0-9a-fA-F]{4})", "", text)
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


def normalize_bracket_tag(text: str) -> str:
    if not text:
        return ""
    cleaned = text.casefold()
    cleaned = re.sub(r"[\s\-_]+", "", cleaned)
    return cleaned


def _strip_bracket_tags(text: str, allowed_tags: set[str] | None = None) -> str:
    if not text:
        return ""
    allowed_tags = allowed_tags or set()

    def _repl(match: re.Match[str]) -> str:
        content = match.group(1).strip()
        if allowed_tags and normalize_bracket_tag(content) in allowed_tags:
            return match.group(0)
        return ""

    cleaned = re.sub(r"[\[\【]([^\]\】]+)[\]\】]", _repl, text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def _strip_profile_tags(text: str) -> str:
    if not text:
        return ""
    pattern = r"\s*[\[\【](關切證據|关切证据|關切|关切|profile evidence|profile)[\]\】]\s*"
    cleaned = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def _normalize_profile_tags(text: str) -> str:
    if not text:
        return ""
    pattern = r"[\[\【](關切證據|关切证据|關切|关切|profile evidence|profile)[\]\】]"

    def _repl(match: re.Match[str]) -> str:
        token = match.group(1)
        lowered = token.casefold()
        if "证据" in token or "evidence" in lowered:
            return "[关切证据]"
        return "[关切]"

    return re.sub(pattern, _repl, text, flags=re.IGNORECASE)


def _normalize_outline(
    outline: list[object],
    allowed_bracket_tags: set[str] | None = None,
) -> list[str]:
    normalized: list[str] = []
    for item in outline:
        if isinstance(item, str):
            text = _normalize_profile_tags(_strip_bracket_tags(item.strip(), allowed_bracket_tags))
            if text:
                normalized.append(text)
            continue
        if isinstance(item, dict):
            title = _strip_bracket_tags(
                str(item.get("title") or item.get("step") or item.get("name") or "").strip(),
                allowed_bracket_tags,
            )
            title = _normalize_profile_tags(title)
            if not title:
                continue
            lines = [title]
            substeps = item.get("substeps") or item.get("steps") or item.get("items") or []
            if isinstance(substeps, list):
                for sub in substeps:
                    if isinstance(sub, str):
                        sub_text = _normalize_profile_tags(
                            _strip_bracket_tags(sub.strip(), allowed_bracket_tags)
                        )
                        if sub_text:
                            lines.append(f"- {sub_text}")
                        continue
                    if isinstance(sub, dict):
                        sub_text = _strip_bracket_tags(
                            str(sub.get("text") or sub.get("title") or "").strip(),
                            allowed_bracket_tags,
                        )
                        sub_text = _normalize_profile_tags(sub_text)
                        if sub_text:
                            lines.append(f"- {sub_text}")
                        sub_sub = sub.get("substeps") or sub.get("items") or []
                        if isinstance(sub_sub, list):
                            for sub_item in sub_sub:
                                if isinstance(sub_item, str):
                                    sub_item_text = _normalize_profile_tags(
                                        _strip_bracket_tags(sub_item.strip(), allowed_bracket_tags)
                                    )
                                    if sub_item_text:
                                        lines.append(f"  - {sub_item_text}")
            normalized.append("\n".join(lines))
    return normalized


def _count_words(text: str) -> int:
    tokens = tokenize(text)
    english_words = len([token for token in tokens if token])
    cjk_chars = len(_CJK_RE.findall(text or ""))
    return english_words + cjk_chars


def outline_word_count(outline: list[str]) -> int:
    return _count_words(" ".join(outline))


def outline_cjk_ratio(outline: list[str]) -> float:
    text = " ".join(outline)
    cjk_chars = len(_CJK_RE.findall(text))
    english_words = len([token for token in tokenize(text) if token])
    total = cjk_chars + english_words
    return cjk_chars / total if total else 0.0


def outline_length_ok(count: int) -> bool:
    return count >= OUTLINE_MIN_WORDS


def _apply_llm_fields(plan: PlanDraft, payload: dict) -> PlanDraft:
    plan.scope = _clean_list(payload.get("scope")) or plan.scope
    plan.key_questions = _clean_list(payload.get("key_questions")) or plan.key_questions
    plan.keywords = _clean_list(payload.get("keywords")) or plan.keywords
    source_requirements = payload.get("source_requirements")
    if isinstance(source_requirements, dict):
        plan.source_requirements = source_requirements
    section_requirements = payload.get("section_requirements")
    if isinstance(section_requirements, list):
        plan.section_requirements = [item for item in section_requirements if isinstance(item, dict)]
    plan.gaps = _clean_list(payload.get("gaps")) or plan.gaps
    plan.notes = _clean_list(payload.get("notes")) or plan.notes
    plan.retrieval_queries = _clean_list(payload.get("retrieval_queries")) or plan.retrieval_queries
    readiness = payload.get("readiness")
    if isinstance(readiness, str) and readiness.strip():
        plan.readiness = readiness.strip().lower()
    return plan


def _is_chinese_language(value: str) -> bool:
    lowered = (value or "").casefold()
    return ("chinese" in lowered) or ("中文" in value) or ("zh" == lowered) or ("zh-" in lowered)


def _outline_has_profile_tags(outline: list[str]) -> bool:
    tag_markers = ("[關切", "[关切", "[profile", "[profile evidence")
    text = " ".join(outline or [])
    lowered = text.casefold()
    return any(marker.casefold() in lowered for marker in tag_markers)


def _review_outline_profile_tags(
    llm: LLMClient,
    outline: list[str],
    topic: str,
    keywords: list[str],
    profile_summary: list[dict],
    kg_fact_cards: list[dict] | None,
    output_language: str,
) -> dict | None:
    prompt = outline_profile_review_prompt(
        topic=topic,
        outline=outline,
        keywords=keywords,
        profile_summary=profile_summary,
        kg_fact_cards=kg_fact_cards,
        output_language=output_language,
    )
    try:
        response = llm.generate(
            SYSTEM_REVIEW_PROMPT,
            prompt,
            task="outline profile review",
            agent="Reviewer",
        )
    except Exception as exc:
        log_task_event(f"Reviewer failed: outline profile review error={exc}")
        return None
    payload = _extract_json(response)
    if not payload:
        log_task_event("Reviewer: invalid JSON for outline profile review.")
        return None
    relevant_labels = _clean_list(payload.get("relevant_labels"))
    missing_labels = _clean_list(payload.get("missing_labels"))
    untagged_mentions: list[dict] = []
    raw_untagged = payload.get("untagged_mentions")
    if isinstance(raw_untagged, list):
        for item in raw_untagged:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip()
            line = str(item.get("line") or "").strip()
            if label and line:
                untagged_mentions.append({"label": label, "line": line})
    suggested_additions: list[dict] = []
    raw_suggestions = payload.get("suggested_additions")
    if isinstance(raw_suggestions, list):
        for item in raw_suggestions:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip()
            why = str(item.get("why") or "").strip()
            placement = str(item.get("placement_hint") or "").strip()
            if label and (why or placement):
                suggested_additions.append(
                    {"label": label, "why": why, "placement_hint": placement}
                )
    needs_revision = payload.get("needs_revision")
    if isinstance(needs_revision, str):
        needs_revision = needs_revision.strip().lower() == "true"
    if not isinstance(needs_revision, bool):
        needs_revision = bool(missing_labels or untagged_mentions or suggested_additions)
    return {
        "relevant_labels": relevant_labels,
        "missing_labels": missing_labels,
        "untagged_mentions": untagged_mentions,
        "suggested_additions": suggested_additions,
        "needs_revision": needs_revision,
    }


def _base_plan(
    topic: str,
    interests: list[Interest],
    documents: list[Document],
    round_number: int,
    keywords_seed: list[str],
    profile_top_k: int | None = None,
    top_k_docs: int | None = None,
) -> PlanDraft:
    profile_limit = int(profile_top_k or DEFAULT_PROFILE_TOP_K)
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
    )

    return PlanDraft(
        topic=topic,
        created_at=_now_iso(),
        round_number=round_number,
        scope=[],
        key_questions=[],
        keywords=[],
        source_requirements={},
        section_requirements=[],
        source_types=source_types,
        source_briefs=source_briefs,
        gaps=[],
        readiness="draft",
        notes=[],
        retrieval_queries=[],
        profile_top_k=profile_limit,
        top_k_docs=doc_limit,
    )


def _validate_llm_plan(plan: PlanDraft) -> list[str]:
    missing = []
    if not plan.scope:
        missing.append("scope")
    if not plan.key_questions:
        missing.append("key_questions")
    if not plan.keywords:
        missing.append("keywords")
    return missing


def llm_draft_plan(
    llm: LLMClient,
    topic: str,
    interests: list[Interest],
    methods: list[Method],
    extracted_interests: list[str] | None,
    documents: list[Document],
    round_number: int,
    profile_top_k: int | None = None,
    top_k_docs: int | None = None,
    kg_fact_cards: list[dict] | None = None,
    output_language: str = "Chinese",
    profile_summary: list[dict] | None = None,
    skill_guidance: list[str] | None = None,
    revision_feedback: list[str] | None = None,
    run_id: int | None = None,
    save_prefix: str | None = None,
) -> PlanDraft:
    keywords_seed = build_keywords(topic, interests)
    base_plan = _base_plan(
        topic,
        interests,
        documents,
        round_number,
        keywords_seed,
        profile_top_k,
        top_k_docs,
    )
    prompt = plan_prompt(
        topic,
        interests,
        methods,
        documents,
        keywords_seed,
        extracted_interests=extracted_interests,
        kg_fact_cards=kg_fact_cards,
        output_language=output_language,
        profile_summary=profile_summary,
        skill_guidance=skill_guidance,
        revision_feedback=revision_feedback,
    )
    model_label = llm.config.model
    emit_progress("Planner", model_label, "plan draft", 0, 1, done=False)
    try:
        response = llm.generate(SYSTEM_PLAN_PROMPT, prompt, task="plan draft", agent="Planner")
    finally:
        emit_progress("Planner", model_label, "plan draft", 1, 1, done=True)
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
    missing = _validate_llm_plan(plan)
    if missing:
        log_task_event(f"Planner: draft missing fields: {', '.join(missing)}")
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    prefix = save_prefix or f"{timestamp}_plan"
    if run_id is not None and save_prefix is None:
        prefix = f"{timestamp}_plan_run_{run_id}"
    _save_plan_snapshot(plan, prefix, f"plan_draft_round{round_number}")
    return plan


def llm_refine_plan(
    llm: LLMClient,
    plan: PlanDraft,
    documents: list[Document],
    interests: list[Interest],
    methods: list[Method],
    extracted_interests: list[str] | None,
    round_number: int,
    profile_top_k: int | None = None,
    top_k_docs: int | None = None,
    kg_fact_cards: list[dict] | None = None,
    output_language: str = "Chinese",
    profile_summary: list[dict] | None = None,
    skill_guidance: list[str] | None = None,
    revision_feedback: list[str] | None = None,
    run_id: int | None = None,
    save_prefix: str | None = None,
) -> PlanDraft:
    keywords_seed = plan.keywords or build_keywords(plan.topic, interests)
    base_plan = _base_plan(
        plan.topic,
        interests,
        documents,
        round_number,
        keywords_seed,
        profile_top_k,
        top_k_docs,
    )
    base_plan.readiness = "refined"
    prior_plan = {
        "scope": plan.scope,
        "key_questions": plan.key_questions,
        "keywords": plan.keywords,
        "source_requirements": plan.source_requirements,
        "section_requirements": plan.section_requirements,
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
        kg_fact_cards=kg_fact_cards,
        output_language=output_language,
        profile_summary=profile_summary,
        skill_guidance=skill_guidance,
        revision_feedback=revision_feedback,
    )
    model_label = llm.config.model
    emit_progress("Planner", model_label, "plan refinement", 0, 1, done=False)
    try:
        response = llm.generate(SYSTEM_PLAN_PROMPT, prompt, task="plan refinement", agent="Planner")
    finally:
        emit_progress("Planner", model_label, "plan refinement", 1, 1, done=True)
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
    missing = _validate_llm_plan(refined)
    if missing:
        log_task_event(f"Planner: refinement missing fields: {', '.join(missing)}")
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    prefix = save_prefix or f"{timestamp}_plan"
    if run_id is not None and save_prefix is None:
        prefix = f"{timestamp}_plan_run_{run_id}"
    _save_plan_snapshot(refined, prefix, f"plan_refine_round{round_number}")
    return refined


def _save_outline_snapshot(
    outline: list[str] | None,
    prefix: str,
    label: str,
) -> None:
    ensure_dirs()
    safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "_", label).strip("_")
    if outline:
        path = ARTIFACTS_DIR / f"{prefix}_{safe_label}.json"
        payload = {"label": label, "outline": outline}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_plan_snapshot(plan: PlanDraft, prefix: str, label: str) -> None:
    ensure_dirs()
    safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "_", label).strip("_")
    path = ARTIFACTS_DIR / f"{prefix}_{safe_label}.json"
    payload = {
        "label": label,
        "topic": plan.topic,
        "created_at": plan.created_at,
        "round_number": plan.round_number,
        "readiness": plan.readiness,
        "scope": plan.scope,
        "key_questions": plan.key_questions,
        "keywords": plan.keywords,
        "source_requirements": plan.source_requirements,
        "section_requirements": plan.section_requirements,
        "gaps": plan.gaps,
        "notes": plan.notes,
        "retrieval_queries": plan.retrieval_queries,
        "source_types": plan.source_types,
        "source_briefs_count": len(plan.source_briefs),
        "profile_top_k": plan.profile_top_k,
        "top_k_docs": plan.top_k_docs,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def llm_build_outline(
    llm: LLMClient,
    topic: str,
    documents: list[Document],
    interests: list[Interest],
    methods: list[Method],
    keywords: list[str],
    plan_requirements: dict[str, object] | None = None,
    kg_fact_cards: list[dict] | None = None,
    output_language: str = "Chinese",
    profile_summary: list[dict] | None = None,
    skill_guidance: list[str] | None = None,
    active_skills: list[str] | None = None,
    skill_method_steps: dict[str, dict[str, object]] | None = None,
    run_id: int | None = None,
    save_prefix: str | None = None,
    allowed_bracket_tags: set[str] | None = None,
    profile_review_rounds: int = DEFAULT_OUTLINE_PROFILE_REVIEW_ROUNDS,
    revision_feedback: list[str] | None = None,
    internal_retries: bool = False,
) -> list[str]:
    def _parse_outline_response(response: str, attempt: int, task: str) -> list[str] | None:
        payload = _extract_json(response)
        outline = payload.get("outline")
        cleaned: list[str] = []
        if isinstance(outline, list):
            cleaned = _normalize_outline(outline, allowed_bracket_tags)
        if not cleaned:
            list_payload = _extract_json_list(response)
            if list_payload:
                cleaned = _normalize_outline(list_payload, allowed_bracket_tags)
                if cleaned:
                    log_task_event(f"Outliner: recovered outline from JSON list fallback ({task}).")
        if not cleaned:
            fallback = _outline_from_text(response)
            if fallback:
                log_task_event(f"Outliner: recovered outline from text fallback ({task}).")
                cleaned = _normalize_outline(fallback, allowed_bracket_tags)
        if not cleaned:
            _log_llm_failure("Outliner", f"{task} attempt {attempt}", response)
            return None
        return cleaned

    def _is_chinese_language(value: str) -> bool:
        lowered = (value or "").casefold()
        return ("chinese" in lowered) or ("中文" in value) or ("zh" == lowered) or ("zh-" in lowered)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    prefix = save_prefix or f"{timestamp}_outline"
    if run_id is not None and save_prefix is None:
        prefix = f"{prefix}_run_{run_id}"

    def _attempt(prompt_text: str, attempt: int) -> tuple[list[str], int, float] | None:
        model_label = llm.config.model
        emit_progress("Outliner", model_label, "outline", 0, 1, done=False)
        try:
            response = llm.generate(SYSTEM_OUTLINE_PROMPT, prompt_text, task="outline", agent="Outliner")
        finally:
            emit_progress("Outliner", model_label, "outline", 1, 1, done=True)
        cleaned = _parse_outline_response(response, attempt, "outline")
        _save_outline_snapshot(cleaned, prefix, f"outline_attempt{attempt}")
        if not cleaned:
            return None
        count = outline_word_count(cleaned)
        ratio = outline_cjk_ratio(cleaned)
        if outline_length_ok(count) and (not _is_chinese_language(output_language) or ratio >= OUTLINE_MIN_CJK_RATIO):
            return cleaned, count, ratio
        log_task_event(
            "Outliner: outline length below minimum "
            f"(attempt={attempt} words={count} min={OUTLINE_MIN_WORDS})"
        )
        if _is_chinese_language(output_language) and ratio < OUTLINE_MIN_CJK_RATIO:
            log_task_event(
                "Outliner: outline language ratio too low "
                f"(attempt={attempt} cjk_ratio={ratio:.2f} min={OUTLINE_MIN_CJK_RATIO:.2f})"
            )
        return cleaned, count, ratio
    def _revision_prompt(
        previous_outline: list[str],
        length_hint: str,
        language_hint: str | None,
        tag_requirements: list[str] | None = None,
    ) -> str:
        payload = {
            "previous_outline": previous_outline,
            "output_language": output_language,
            "instructions": {
                "output_json_schema": {
                    "outline": ["string"]
                },
                "requirements": [
                    f"Length must be at least {OUTLINE_MIN_WORDS} words in the output language.",
                    "Preserve the original topic coverage and structure; rewrite to fit length.",
                    "Keep the top-layer structure intact; if structure_guidance is provided, follow it even if step count changes.",
                    "Aim for 8-12 major steps only when no structure/method guidance applies.",
                    "Use 3-5 substeps and 2-3 sentences as defaults; vary when needed for clarity, depth, or method alignment.",
                    "Preserve explicit profile ties; keep at least one relevant tie per top-layer step when profile_summary is provided.",
                ],
                "language_guidance": [
                    f"All natural-language content MUST be in {output_language}.",
                    "Keep paper titles, dataset names, benchmarks, model names, APIs, and acronyms in English.",
                ],
            },
            "length_hint": length_hint,
        }
        if tag_requirements:
            payload["instructions"]["requirements"].extend(tag_requirements)
        if structure_guidance:
            payload["instructions"]["structure_guidance"] = structure_guidance
        if profile_summary:
            payload["profile_summary"] = profile_summary[:10]
        if language_hint:
            payload["language_hint"] = language_hint
        return json.dumps(payload, indent=2)

    def _expand_prompt(
        previous_outline: list[str],
        length_hint: str,
        language_hint: str | None,
        tag_requirements: list[str] | None = None,
    ) -> str:
        payload = {
            "previous_outline": previous_outline,
            "output_language": output_language,
            "instructions": {
                "output_json_schema": {
                    "outline": ["string"]
                },
                "requirements": [
                    f"Length must be at least {OUTLINE_MIN_WORDS} words in the output language.",
                    "Do not remove content; only expand with additional detail and substeps.",
                    "Keep the top-layer structure intact; if structure_guidance is provided, follow it even if step count changes.",
                    "Aim for 8-12 major steps only when no structure/method guidance applies.",
                    "Use 3-5 substeps and 2-3 sentences as defaults; vary when needed for clarity, depth, or method alignment.",
                    "Preserve explicit profile ties; keep at least one relevant tie per top-layer step when profile_summary is provided.",
                ],
                "language_guidance": [
                    f"All natural-language content MUST be in {output_language}.",
                    "Keep paper titles, dataset names, benchmarks, model names, APIs, and acronyms in English.",
                ],
            },
            "length_hint": length_hint,
        }
        if tag_requirements:
            payload["instructions"]["requirements"].extend(tag_requirements)
        if structure_guidance:
            payload["instructions"]["structure_guidance"] = structure_guidance
        if profile_summary:
            payload["profile_summary"] = profile_summary[:10]
        if language_hint:
            payload["language_hint"] = language_hint
        return json.dumps(payload, indent=2)

    def _attempt_revision(
        previous_outline: list[str],
        previous_count: int,
        previous_ratio: float,
        attempt: int,
        tag_requirements: list[str] | None = None,
    ) -> tuple[list[str], int, float] | None:
        length_hint = (
            f"Your previous outline length was {previous_count} words. "
            f"Ensure the outline is at least {OUTLINE_MIN_WORDS} words. "
            "You MUST meet the minimum length."
        )
        language_hint = None
        if _is_chinese_language(output_language) and previous_ratio < OUTLINE_MIN_CJK_RATIO:
            language_hint = (
                "Your previous outline was not sufficiently Chinese. "
                "Translate all step titles and substeps to Chinese; keep English only for paper titles, "
                "datasets, benchmarks, model names, APIs, and acronyms."
            )
        prompt_text = _revision_prompt(previous_outline, length_hint, language_hint, tag_requirements)
        model_label = llm.config.model
        emit_progress("Outliner", model_label, "outline revision", 0, 1, done=False)
        try:
            response = llm.generate(
                system_prompt=(
                    "You revise research plan outlines to meet strict length constraints. "
                    f"All natural-language content MUST be in {output_language}. Return JSON only."
                ),
                user_prompt=prompt_text,
                task="outline revision",
                agent="Outliner",
            )
        finally:
            emit_progress("Outliner", model_label, "outline revision", 1, 1, done=True)
        cleaned = _parse_outline_response(response, attempt, "outline revision")
        _save_outline_snapshot(cleaned, prefix, f"outline_revision{attempt}")
        if not cleaned:
            return None
        count = outline_word_count(cleaned)
        ratio = outline_cjk_ratio(cleaned)
        if outline_length_ok(count) and (not _is_chinese_language(output_language) or ratio >= OUTLINE_MIN_CJK_RATIO):
            return cleaned, count, ratio
        log_task_event(
            "Outliner: outline length still below minimum after revision "
            f"(attempt={attempt} words={count} min={OUTLINE_MIN_WORDS})"
        )
        if _is_chinese_language(output_language) and ratio < OUTLINE_MIN_CJK_RATIO:
            log_task_event(
                "Outliner: outline language ratio too low after revision "
                f"(attempt={attempt} cjk_ratio={ratio:.2f} min={OUTLINE_MIN_CJK_RATIO:.2f})"
            )
        return cleaned, count, ratio

    def _maybe_revise_for_profile_tags(
        outline: list[str],
        count: int,
        ratio: float,
        attempt: int,
    ) -> tuple[list[str], int, float]:
        if not profile_summary or not llm:
            return outline, count, ratio
        review = _review_outline_profile_tags(
            llm=llm,
            outline=outline,
            topic=topic,
            keywords=keywords,
            profile_summary=profile_summary,
            kg_fact_cards=kg_fact_cards,
            output_language=output_language,
        )
        if not review:
            return outline, count, ratio
        relevant_labels = review.get("relevant_labels") or []
        missing_labels = review.get("missing_labels") or []
        untagged_mentions = review.get("untagged_mentions") or []
        suggested_additions = review.get("suggested_additions") or []
        if not relevant_labels or (not missing_labels and not untagged_mentions and not suggested_additions):
            return outline, count, ratio
        tag_requirements = [
            "Whenever a profile theme is relevant to the topic or report and you mention it, add a brief relevance explanation ending with ' [关切]'.",
            "When a profile theme mention is supported by aligned kg_fact_cards evidence, end with ' [关切证据]' instead of ' [关切]'.",
        ]
        if missing_labels:
            tag_requirements.append(
                "Add at least one mention for each missing relevant profile label and tag it appropriately: "
                + ", ".join(missing_labels)
                + "."
            )
        if untagged_mentions:
            examples = []
            for item in untagged_mentions[:6]:
                label = item.get("label") or ""
                line = item.get("line") or ""
                if label and line:
                    examples.append(f"{label}: {line}")
            if examples:
                tag_requirements.append(
                    "The following lines mention relevant profile themes but are missing tags; fix them: "
                    + " | ".join(examples)
                )
        if suggested_additions:
            additions = []
            for item in suggested_additions[:6]:
                label = item.get("label") or ""
                why = item.get("why") or ""
                placement = item.get("placement_hint") or ""
                parts = [label]
                if placement:
                    parts.append(f"place: {placement}")
                if why:
                    parts.append(f"why: {why}")
                additions.append(" / ".join(parts))
            if additions:
                tag_requirements.append(
                    "Add the following beneficial profile-related additions and tag them appropriately: "
                    + " | ".join(additions)
                )
        revised = _attempt_revision(outline, count, ratio, attempt, tag_requirements=tag_requirements)
        if revised:
            revised_count = revised[1]
            revised_ratio = revised[2]
            if revised_count < OUTLINE_MIN_WORDS:
                expanded = _attempt_expand(
                    revised[0],
                    revised_count,
                    revised_ratio,
                    attempt + 1,
                    tag_requirements=tag_requirements,
                )
                if expanded:
                    revised = expanded
                    revised_count = revised[1]
                    revised_ratio = revised[2]
                    if revised_count < OUTLINE_MIN_WORDS:
                        return outline, count, ratio
            if _is_chinese_language(output_language) and revised_ratio < OUTLINE_MIN_CJK_RATIO:
                return outline, count, ratio
            revised_review = _review_outline_profile_tags(
                llm=llm,
                outline=revised[0],
                topic=topic,
                keywords=keywords,
                profile_summary=profile_summary,
                kg_fact_cards=kg_fact_cards,
                output_language=output_language,
            )
            if revised_review:
                if (
                    not revised_review.get("missing_labels")
                    and not revised_review.get("untagged_mentions")
                    and not revised_review.get("suggested_additions")
                ):
                    return revised[0], revised_count, revised_ratio
            if _outline_has_profile_tags(revised[0]) and not _outline_has_profile_tags(outline):
                log_task_event("Outline profile tags improved after revision; keeping revised outline.")
                return revised[0], revised_count, revised_ratio
        return outline, count, ratio

    def _apply_profile_review_loop(
        outline: list[str],
        count: int,
        ratio: float,
    ) -> list[str]:
        rounds = max(0, int(profile_review_rounds or 0))
        if rounds == 0:
            return outline
        current_outline = outline
        current_count = count
        current_ratio = ratio
        for idx in range(rounds):
            revised_outline, revised_count, revised_ratio = _maybe_revise_for_profile_tags(
                current_outline,
                current_count,
                current_ratio,
                attempt=7 + idx,
            )
            if revised_outline == current_outline:
                break
            current_outline = revised_outline
            current_count = revised_count
            current_ratio = revised_ratio
        return current_outline

    def _attempt_expand(
        previous_outline: list[str],
        previous_count: int,
        previous_ratio: float,
        attempt: int,
        tag_requirements: list[str] | None = None,
    ) -> tuple[list[str], int, float] | None:
        length_hint = (
            f"Your previous outline length was {previous_count} words. "
            f"Expand to at least {OUTLINE_MIN_WORDS} words without changing the major steps. "
            "Add more substeps and elaboration (2-3 sentences per substep)."
        )
        language_hint = None
        if _is_chinese_language(output_language) and previous_ratio < OUTLINE_MIN_CJK_RATIO:
            language_hint = (
                "Your previous outline was not sufficiently Chinese. "
                "Translate all step titles and substeps to Chinese; keep English only for paper titles, "
                "datasets, benchmarks, model names, APIs, and acronyms."
            )
        prompt_text = _expand_prompt(previous_outline, length_hint, language_hint, tag_requirements)
        model_label = llm.config.model
        emit_progress("Outliner", model_label, "outline expansion", 0, 1, done=False)
        try:
            response = llm.generate(
                system_prompt=(
                    "You expand research plan outlines to meet strict minimum length. "
                    f"All natural-language content MUST be in {output_language}. Return JSON only."
                ),
                user_prompt=prompt_text,
                task="outline expansion",
                agent="Outliner",
            )
        finally:
            emit_progress("Outliner", model_label, "outline expansion", 1, 1, done=True)
        cleaned = _parse_outline_response(response, attempt, "outline expansion")
        _save_outline_snapshot(cleaned, prefix, f"outline_expand{attempt}")
        if not cleaned:
            return None
        count = outline_word_count(cleaned)
        ratio = outline_cjk_ratio(cleaned)
        if outline_length_ok(count) and (not _is_chinese_language(output_language) or ratio >= OUTLINE_MIN_CJK_RATIO):
            return cleaned, count, ratio
        log_task_event(
            "Outliner: outline length still below minimum after expansion "
            f"(attempt={attempt} words={count} min={OUTLINE_MIN_WORDS})"
        )
        if _is_chinese_language(output_language) and ratio < OUTLINE_MIN_CJK_RATIO:
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
            "Prefix each major step title with the matching skill label using this bracketed format: "
            "'[Methodology Skill: {label}] ...' (e.g., '[Methodology Skill: Systems Engineering] ...').",
            "Do not use bracketed method tags other than the required [Methodology Skill: ...] label.",
            "Ensure each skill has at least one major step; do not introduce new section labels.",
        ]
        if skill_method_steps:
            for skill_name in active_skills:
                selection = skill_method_steps.get(skill_name) or {}
                method_name = str(selection.get("method") or "").strip()
                steps = selection.get("steps") or []
                if not method_name or not isinstance(steps, list) or not steps:
                    continue
                label = skill_name.replace("-", " ").title()
                step_list = "; ".join(str(step).strip() for step in steps if str(step).strip())
                if not step_list:
                    continue
                structure_guidance.append(
                    "For the "
                    f"{label} section, use the '{method_name}' method steps as the ONLY major steps "
                    f"in that section, in this exact order: {step_list}."
                )

    prompt = outline_prompt(
        topic,
        interests,
        methods,
        documents,
        keywords,
        plan_requirements=plan_requirements,
        kg_fact_cards=kg_fact_cards,
        output_language=output_language,
        profile_summary=profile_summary,
        structure_guidance=structure_guidance,
        skill_guidance=skill_guidance,
        revision_feedback=revision_feedback,
    )
    result = _attempt(prompt, 1)
    if not internal_retries:
        if result:
            return result[0]
        raise LLMClientError("LLM returned invalid JSON for outline.")
    if result and outline_length_ok(result[1]) and (not _is_chinese_language(output_language) or result[2] >= OUTLINE_MIN_CJK_RATIO):
        return _apply_profile_review_loop(result[0], result[1], result[2])

    previous_count = result[1] if result else 0
    previous_ratio = result[2] if result else 0.0
    length_hint = (
        f"Your previous outline length was {previous_count} words. "
        f"Expand to at least {OUTLINE_MIN_WORDS} words. "
        "Add more detailed substeps (2-3 sentences each) to reach the target."
    )
    language_hint = None
    if _is_chinese_language(output_language) and previous_ratio < OUTLINE_MIN_CJK_RATIO:
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
        plan_requirements=plan_requirements,
        kg_fact_cards=kg_fact_cards,
        output_language=output_language,
        profile_summary=profile_summary,
        length_hint=length_hint,
        language_hint=language_hint,
        structure_guidance=structure_guidance,
        skill_guidance=skill_guidance,
        revision_feedback=revision_feedback,
    )
    result = _attempt(retry_prompt, 2)
    if result and outline_length_ok(result[1]) and (not _is_chinese_language(output_language) or result[2] >= OUTLINE_MIN_CJK_RATIO):
        return _apply_profile_review_loop(result[0], result[1], result[2])

    previous_count = result[1] if result else previous_count
    previous_ratio = result[2] if result else previous_ratio
    length_hint = (
        f"Length still below minimum ({previous_count} words). "
        f"Target at least {OUTLINE_MIN_WORDS} words. "
        "Increase detail by expanding each major step with additional substeps."
    )
    final_prompt = outline_prompt(
        topic,
        interests,
        methods,
        documents,
        keywords,
        plan_requirements=plan_requirements,
        kg_fact_cards=kg_fact_cards,
        output_language=output_language,
        profile_summary=profile_summary,
        length_hint=length_hint,
        language_hint=language_hint,
        structure_guidance=structure_guidance,
        skill_guidance=skill_guidance,
        revision_feedback=revision_feedback,
    )
    result = _attempt(final_prompt, 3)
    if result and outline_length_ok(result[1]):
        if (not _is_chinese_language(output_language)) or result[2] >= OUTLINE_MIN_CJK_RATIO:
            return _apply_profile_review_loop(result[0], result[1], result[2])

    if result and result[1] < OUTLINE_MIN_WORDS:
        expanded = _attempt_expand(result[0], result[1], result[2], 4)
        if expanded and outline_length_ok(expanded[1]) and (not _is_chinese_language(output_language) or expanded[2] >= OUTLINE_MIN_CJK_RATIO):
            return _apply_profile_review_loop(expanded[0], expanded[1], expanded[2])
        if expanded:
            result = expanded
            if result[1] < OUTLINE_MIN_WORDS:
                expanded = _attempt_expand(result[0], result[1], result[2], 5)
                if expanded and outline_length_ok(expanded[1]) and (not _is_chinese_language(output_language) or expanded[2] >= OUTLINE_MIN_CJK_RATIO):
                    return _apply_profile_review_loop(expanded[0], expanded[1], expanded[2])
                if expanded:
                    result = expanded

    if result:
        enforced = _attempt_revision(result[0], result[1], result[2], 6)
        if enforced and outline_length_ok(enforced[1]) and (not _is_chinese_language(output_language) or enforced[2] >= OUTLINE_MIN_CJK_RATIO):
            return _apply_profile_review_loop(enforced[0], enforced[1], enforced[2])

        log_task_event(
            "Outliner: returning best-effort outline despite length constraint failure. "
            f"Last count={result[1]} min={OUTLINE_MIN_WORDS}."
        )
        return _apply_profile_review_loop(result[0], result[1], result[2])

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
        lines.append(f"- {_strip_profile_tags(item)}")

    lines.append("")
    lines.append("## Key Questions")
    for item in plan.key_questions:
        lines.append(f"- {_strip_profile_tags(item)}")

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
            lines.append(f"- {_strip_profile_tags(item)}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("## Notes")
    for item in plan.notes:
        lines.append(f"- {_strip_profile_tags(item)}")

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
    lines.append("## Profile Signals")
    signals = build_profile_signals(
        plan.topic,
        interests,
        query_hints=plan.retrieval_queries,
        llm=llm,
        max_signals=plan.profile_top_k,
        top_k_docs=plan.top_k_docs,
    )
    if not signals:
        lines.append(
            "No profile signals available yet. Ingest local files and run research with KG extraction enabled."
        )
    else:
        for signal in signals:
            similarity = signal.get("similarity")
            if isinstance(similarity, (int, float)):
                lines.append(f"Profile: {signal['label']} (salience={similarity:.2f})")
            else:
                lines.append(f"Profile: {signal['label']}")
            for bullet in signal["bullets"]:
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
