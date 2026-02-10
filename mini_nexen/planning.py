from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from .db import Document, Interest, load_document_text
from .llm import LLMClient, LLMClientError
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def summarize_interests(interests: Iterable[Interest]) -> list[str]:
    summary = []
    seen = set()
    for interest in interests:
        if interest.notes:
            item = f"{interest.topic} ({interest.notes})"
        else:
            item = interest.topic
        if item not in seen:
            seen.add(item)
            summary.append(item)
    return summary


def build_keywords(topic: str, interests: Iterable[Interest], extra: Iterable[str] | None = None) -> list[str]:
    tokens = tokenize(topic)
    for interest in interests:
        tokens.extend(tokenize(interest.topic))
        tokens.extend(tokenize(interest.notes))
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
    highlights_per_doc: int = 3,
) -> list[SourceBrief]:
    briefs = []
    for doc in documents:
        text = load_document_text(doc)
        highlights = top_sentences(text, keywords, limit=highlights_per_doc)
        briefs.append(SourceBrief(doc=doc, highlights=highlights))
    return briefs


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


def _clean_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _apply_llm_fields(plan: PlanDraft, payload: dict) -> PlanDraft:
    plan.scope = _clean_list(payload.get("scope")) or plan.scope
    plan.key_questions = _clean_list(payload.get("key_questions")) or plan.key_questions
    plan.keywords = _clean_list(payload.get("keywords")) or plan.keywords
    plan.gaps = _clean_list(payload.get("gaps")) or plan.gaps
    plan.notes = _clean_list(payload.get("notes")) or plan.notes
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
    documents: list[Document],
    round_number: int,
) -> PlanDraft:
    keywords_seed = build_keywords(topic, interests)
    base_plan = _base_plan(topic, interests, documents, round_number, keywords_seed)
    prompt = plan_prompt(topic, interests, documents, keywords_seed)
    response = llm.generate(SYSTEM_PLAN_PROMPT, prompt, task="plan draft", agent="Planner")
    payload = _extract_json(response)
    if not payload:
        raise LLMClientError("LLM returned invalid JSON for plan draft.")
    plan = _apply_llm_fields(base_plan, payload)
    _validate_llm_plan(plan)
    return plan


def llm_refine_plan(
    llm: LLMClient,
    plan: PlanDraft,
    documents: list[Document],
    interests: list[Interest],
    round_number: int,
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
    prompt = refine_prompt(plan.topic, prior_plan, interests, documents, keywords_seed)
    response = llm.generate(SYSTEM_PLAN_PROMPT, prompt, task="plan refinement", agent="Planner")
    payload = _extract_json(response)
    if not payload:
        raise LLMClientError("LLM returned invalid JSON for plan refinement.")
    refined = _apply_llm_fields(base_plan, payload)
    _validate_llm_plan(refined)
    return refined


def llm_build_outline(
    llm: LLMClient,
    topic: str,
    documents: list[Document],
    interests: list[Interest],
    keywords: list[str],
) -> list[str]:
    prompt = outline_prompt(topic, interests, documents, keywords)
    response = llm.generate(SYSTEM_OUTLINE_PROMPT, prompt, task="outline", agent="Outliner")
    payload = _extract_json(response)
    outline = payload.get("outline")
    if isinstance(outline, list):
        cleaned = _clean_list(outline)
        if cleaned:
            return cleaned
    raise LLMClientError("LLM returned invalid JSON for outline.")


def build_outline(
    topic: str,
    documents: list[Document],
    interests: list[Interest],
    keywords: list[str],
) -> list[str]:
    outline = [
        f"Thesis framing for {topic}",
        "Background and definitions to establish scope",
        "Current landscape and key actors",
        "Core mechanisms, methods, or technologies",
        "Evidence summary and strongest claims",
        "Points of disagreement or uncertainty",
        "Implications, risks, and ethical considerations",
        "Open research questions and validation needs",
        "Recommended next retrieval targets",
    ]

    interest_summary = summarize_interests(interests)
    if interest_summary:
        outline.append("Connections to recorded interests")

    highlights = []
    for doc in documents:
        text = load_document_text(doc)
        for sentence in top_sentences(text, keywords, limit=2):
            highlights.append(f"{doc.title}: {sentence}")

    if highlights:
        outline.append("Source-backed bullets")
        outline.extend(highlights[:12])

    return outline


def render_plan_md(
    plan: PlanDraft,
    outline: list[str],
    interests: list[Interest],
) -> str:
    interest_summary = summarize_interests(interests)

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
    lines.append("## Source Briefs")
    if plan.source_briefs:
        for brief in plan.source_briefs:
            lines.append(f"- Source: {brief.doc.title} ({brief.doc.source_type})")
            if brief.highlights:
                for highlight in brief.highlights:
                    lines.append(f"- Highlight: {highlight}")
            else:
                lines.append("- Highlight: No highlights captured yet.")
    else:
        lines.append("- No matching sources in library.")

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
    lines.append("## Recorded Interests")
    if interest_summary:
        for item in interest_summary:
            lines.append(f"- {item}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("## Detailed Outline")
    for item in outline:
        lines.append(f"- {item}")

    lines.append("")
    lines.append("## Next Actions")
    lines.append("- Ingest missing sources for the identified gaps.")
    lines.append("- Run another planning round after updating the library.")

    return "\n".join(lines)
