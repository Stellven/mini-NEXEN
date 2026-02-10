from __future__ import annotations

import json
from typing import Iterable

from .db import Document, Interest, load_document_text
from .text_utils import top_sentences

SYSTEM_PLAN_PROMPT = """You are a research planning agent. Produce a plan that is concise, actionable, and focused on gaps.
Return JSON only, no commentary.
"""

SYSTEM_OUTLINE_PROMPT = """You are an outlining agent. Produce a detailed outline suitable for a manuscript or report.
Return JSON only, no commentary.
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
    return json.dumps(payload, indent=2)


def refine_prompt(
    topic: str,
    prior_plan: dict,
    interests: list[Interest],
    documents: list[Document],
    keywords: list[str],
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
    return json.dumps(payload, indent=2)


def outline_prompt(
    topic: str,
    interests: list[Interest],
    documents: list[Document],
    keywords: list[str],
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
    return json.dumps(payload, indent=2)
