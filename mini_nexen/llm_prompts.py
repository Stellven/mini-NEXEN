from __future__ import annotations

import json
from typing import Iterable

from .db import Document, Interest, load_document_text
from .text_utils import top_sentences

SYSTEM_PLAN_PROMPT = """You are a research planning agent. Produce a plan that is concise, actionable, and focused on gaps.
Return ONE valid JSON object only. No markdown, no commentary, no extra text.

Requirements:
- Must include keys: scope, key_questions, keywords, gaps, notes, readiness
- scope, key_questions, keywords MUST be non-empty arrays (>= 3 items each)
- gaps and notes may be empty arrays
- readiness must be one of: "draft", "refined", "ready"
- Output must be strict JSON (double quotes, no trailing commas)

Example:
{"scope":["..."],"key_questions":["..."],"keywords":["..."],"gaps":["..."],"notes":["..."],"readiness":"draft"}
"""

SYSTEM_OUTLINE_PROMPT = """You are an outlining agent. Produce a detailed outline suitable for a manuscript or report.
Return ONE valid JSON object only. No markdown, no commentary, no extra text.

Requirements:
- Must include key: outline
- outline MUST be a non-empty array (>= 6 items)
- Output must be strict JSON (double quotes, no trailing commas)

Example:
{"outline":["Section 1","Section 2","Section 3","Section 4","Section 5","Section 6"]}
"""


def _serialize_interests(interests: Iterable[Interest]) -> list[str]:
    serialized = []
    for interest in interests:
        if interest.notes:
            serialized.append(f"{interest.topic} ({interest.notes})")
        else:
            serialized.append(interest.topic)
    return serialized


def _serialize_documents(documents: Iterable[Document], keywords: Iterable[str]) -> list[dict]:
    results = []
    for doc in documents:
        text = load_document_text(doc)
        highlights = top_sentences(text, keywords, limit=3)
        results.append(
            {
                "title": doc.title,
                "source_type": doc.source_type,
                "source": doc.source,
                "highlights": highlights,
            }
        )
    return results


def plan_prompt(
    topic: str,
    interests: list[Interest],
    documents: list[Document],
    keywords: list[str],
    skill_guidance: list[str] | None = None,
) -> str:
    payload = {
        "topic": topic,
        "interests": _serialize_interests(interests),
        "documents": _serialize_documents(documents, keywords),
        "instructions": {
            "output_json_schema": {
                "scope": ["string"],
                "key_questions": ["string"],
                "keywords": ["string"],
                "gaps": ["string"],
                "notes": ["string"],
                "readiness": "draft | refined | ready",
            }
        },
    }
    if skill_guidance:
        payload["skill_guidance"] = skill_guidance
    return json.dumps(payload, indent=2)


def refine_prompt(
    topic: str,
    prior_plan: dict,
    interests: list[Interest],
    documents: list[Document],
    keywords: list[str],
    skill_guidance: list[str] | None = None,
) -> str:
    payload = {
        "topic": topic,
        "prior_plan": prior_plan,
        "interests": _serialize_interests(interests),
        "documents": _serialize_documents(documents, keywords),
        "instructions": {
            "output_json_schema": {
                "scope": ["string"],
                "key_questions": ["string"],
                "keywords": ["string"],
                "gaps": ["string"],
                "notes": ["string"],
                "readiness": "draft | refined | ready",
            }
        },
    }
    if skill_guidance:
        payload["skill_guidance"] = skill_guidance
    return json.dumps(payload, indent=2)


def outline_prompt(
    topic: str,
    interests: list[Interest],
    documents: list[Document],
    keywords: list[str],
    skill_guidance: list[str] | None = None,
) -> str:
    payload = {
        "topic": topic,
        "interests": _serialize_interests(interests),
        "documents": _serialize_documents(documents, keywords),
        "keywords": keywords,
        "instructions": {
            "output_json_schema": {
                "outline": ["string"]
            },
            "outline_expectations": [
                "Include major sections and subtopics to expand later.",
                "List source-backed bullets when available.",
                "Emphasize gaps and future research directions.",
            ],
        },
    }
    if skill_guidance:
        payload["skill_guidance"] = skill_guidance
    return json.dumps(payload, indent=2)
