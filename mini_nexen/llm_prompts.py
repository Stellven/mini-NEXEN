from __future__ import annotations

import json
from typing import Iterable

from .db import Document, Interest, Method, load_document_text
from .text_utils import top_sentences

SYSTEM_PLAN_PROMPT = """You are a research planning agent. All natural-language content MUST be in Chinese. Produce a plan that is concise, actionable, and focused on gaps.
Return ONE valid JSON object only. No markdown, no commentary, no extra text.

Requirements:
- Must include keys: scope, key_questions, keywords, gaps, notes, readiness
- scope, key_questions, keywords MUST be non-empty arrays (>= 3 items each)
- keywords must include core terms from the topic and interests; include extracted interests if provided
- methods are analysis approaches; use them to frame the plan, not as the topic itself
- gaps and notes may be empty arrays
- retrieval_queries is optional; if provided, it must be a short list of search phrases
- readiness must be one of: "draft", "refined", "ready"
- Output must be strict JSON (double quotes, no trailing commas)
- All natural-language content MUST be in Chinese, except retrieval_queries must be in English.
- Keep JSON keys in English.
- Keep paper titles, dataset names, benchmarks, model names, APIs, and acronyms in English.
- Use ASCII punctuation for JSON (":" and ","), not full-width punctuation.
All natural-language content MUST be in Chinese.

Example:
{"scope":["..."],"key_questions":["..."],"keywords":["..."],"gaps":["..."],"notes":["..."],"readiness":"draft","retrieval_queries":["..."]}
"""

SYSTEM_OUTLINE_PROMPT = """You are a research planning agent. All natural-language content MUST be in Chinese. Produce a research plan (not a report outline).
Return ONE valid JSON object only. No markdown, no commentary, no extra text.

Requirements:
- Must include key: outline
- outline MUST be a non-empty array (8-12 major steps).
- Each major step must include 3-5 substeps.
- Each substep should be 2-3 sentences.
- Optional sub-substeps are allowed when helpful.
- Use imperative phrasing (e.g., "Search...", "Compare...", "Investigate...").
- Do not write report section headings.
- Total outline length MUST be 1000-2000 words across titles and substeps in the output language.
- Output must be strict JSON (double quotes, no trailing commas)
- Treat methods as analysis approaches to apply to the topic, not as the topic itself.
- If methods are provided, the research plan MUST explicitly structure steps around those methods.
- All natural-language content MUST be in Chinese.
- Keep JSON keys in English.
- Keep paper titles, dataset names, benchmarks, model names, APIs, and acronyms in English.
- Use ASCII punctuation for JSON (":" and ","), not full-width punctuation.
All natural-language content MUST be in Chinese.

Example:
{"outline":[{"title":"Search for official technical reports and release blogs to extract architecture and training details.","substeps":["List official sources and release pages to target.","Extract training scale, data sources, and capability summaries.",{"text":"Capture deployment context and version history.","substeps":["Note release cadence and changelogs.","Record published caveats or limitations."]}]},{"title":"Compare context window sizes and long-context accuracy across models.","substeps":["Collect vendor claims and independent benchmarks.","Analyze long-context recall and retrieval performance.",{"text":"Identify failure cases.","substeps":["Summarize common error patterns.","Note mitigation strategies reported by practitioners."]}]}]}
"""

SYSTEM_QUERY_UNDERSTANDING_PROMPT = (
    "You are a query understanding agent. "
    "Infer the core topic and analysis methodologies. "
    "Return JSON only."
)


def _serialize_interests(interests: Iterable[Interest]) -> list[str]:
    serialized = []
    for interest in interests:
        serialized.append(interest.topic)
    return serialized


def _serialize_methods(methods: Iterable[Method]) -> list[str]:
    serialized = []
    for method in methods:
        serialized.append(method.method)
    return serialized


def _serialize_documents(
    documents: Iterable[Document],
    keywords: Iterable[str],
    doc_text_overrides: dict[str, str] | None = None,
) -> list[dict]:
    results = []
    for doc in documents:
        text = doc_text_overrides.get(doc.doc_id) if doc_text_overrides else None
        if not text:
            text = load_document_text(doc)
        highlights = top_sentences(text, keywords, limit=10)
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
    methods: list[Method],
    documents: list[Document],
    keywords: list[str],
    extracted_interests: list[str] | None = None,
    doc_text_overrides: dict[str, str] | None = None,
    skill_guidance: list[str] | None = None,
) -> str:
    payload = {
        "topic": topic,
        "interests": _serialize_interests(interests),
        "methods": _serialize_methods(methods),
        "documents": _serialize_documents(documents, keywords, doc_text_overrides=doc_text_overrides),
        "instructions": {
            "output_json_schema": {
                "scope": ["string"],
                "key_questions": ["string"],
                "keywords": ["string"],
                "gaps": ["string"],
                "notes": ["string"],
                "retrieval_queries": ["string"],
                "readiness": "draft | refined | ready",
            },
            "language_policy": [
                "All list items must be in Chinese.",
                "retrieval_queries must be in English.",
                "Keep paper titles, dataset names, benchmarks, model names, APIs, and acronyms in English.",
            ],
            "keyword_guidance": [
                "Include key terms from the topic.",
                "Include salient terms from manually added interests.",
                "If extracted_interests are provided, incorporate them.",
                "Do not replace the topic with method names.",
            ],
            "retrieval_query_guidance": [
                "If provided, use 2-6 word search phrases.",
                "Avoid negations like 'no' or 'lack of'.",
                "Focus on entities, relationships, and concrete nouns.",
                "Keep to 2-8 total queries.",
                "Queries must be in English.",
            ],
            "method_guidance": [
                "Methods are analysis approaches (lenses), not standalone topics.",
                "Apply methods to the topic and evidence.",
            ],
        },
    }
    if extracted_interests:
        payload["extracted_interests"] = extracted_interests
    if skill_guidance:
        payload["skill_guidance"] = skill_guidance
    return json.dumps(payload, indent=2)


def query_understanding_prompt(raw_query: str, methodology_taxonomy: list[str]) -> str:
    payload = {
        "query": raw_query,
        "methodology_taxonomy": methodology_taxonomy,
        "instructions": {
            "output_json_schema": {
                "topic": "string",
                "normalized_query": "string",
                "methodologies": ["string"],
                "confidence": "0-1 float",
                "rationale": "string",
                "constraints": {"timeframe": "string | null", "region": "string | null", "industry": "string | null"},
                "audience": "string | null",
            },
            "methodology_rules": [
                "Choose methodologies only from methodology_taxonomy.",
                "Methodologies are analysis lenses, not topics.",
                "Return 1-3 methodologies.",
            ],
            "topic_rules": [
                "topic should be the core subject of the query.",
                "normalized_query should be a concise version of the query.",
                "normalized_query should avoid analysis methodology terms.",
            ],
        },
    }
    return json.dumps(payload, indent=2)


def refine_prompt(
    topic: str,
    prior_plan: dict,
    interests: list[Interest],
    methods: list[Method],
    documents: list[Document],
    keywords: list[str],
    extracted_interests: list[str] | None = None,
    doc_text_overrides: dict[str, str] | None = None,
    skill_guidance: list[str] | None = None,
) -> str:
    payload = {
        "topic": topic,
        "prior_plan": prior_plan,
        "interests": _serialize_interests(interests),
        "methods": _serialize_methods(methods),
        "documents": _serialize_documents(documents, keywords, doc_text_overrides=doc_text_overrides),
        "instructions": {
            "output_json_schema": {
                "scope": ["string"],
                "key_questions": ["string"],
                "keywords": ["string"],
                "gaps": ["string"],
                "notes": ["string"],
                "retrieval_queries": ["string"],
                "readiness": "draft | refined | ready",
            },
            "language_policy": [
                "All list items must be in Chinese.",
                "retrieval_queries must be in English.",
                "Keep paper titles, dataset names, benchmarks, model names, APIs, and acronyms in English.",
            ],
            "keyword_guidance": [
                "Include key terms from the topic.",
                "Include salient terms from manually added interests.",
                "If extracted_interests are provided, incorporate them.",
                "Do not replace the topic with method names.",
            ],
            "retrieval_query_guidance": [
                "If provided, use 2-6 word search phrases.",
                "Avoid negations like 'no' or 'lack of'.",
                "Focus on entities, relationships, and concrete nouns.",
                "Keep to 2-8 total queries.",
                "Queries must be in English.",
            ],
            "method_guidance": [
                "Methods are analysis approaches (lenses), not standalone topics.",
                "Apply methods to the topic and evidence.",
            ],
        },
    }
    if extracted_interests:
        payload["extracted_interests"] = extracted_interests
    if skill_guidance:
        payload["skill_guidance"] = skill_guidance
    return json.dumps(payload, indent=2)


def outline_prompt(
    topic: str,
    interests: list[Interest],
    methods: list[Method],
    documents: list[Document],
    keywords: list[str],
    doc_text_overrides: dict[str, str] | None = None,
    length_hint: str | None = None,
    language_hint: str | None = None,
    structure_guidance: list[str] | None = None,
    skill_guidance: list[str] | None = None,
) -> str:
    payload = {
        "topic": topic,
        "interests": _serialize_interests(interests),
        "methods": _serialize_methods(methods),
        "documents": _serialize_documents(documents, keywords, doc_text_overrides=doc_text_overrides),
        "keywords": keywords,
        "instructions": {
            "output_json_schema": {
                "outline": ["string"]
            },
            "language_policy": [
                "All natural-language content MUST be in Chinese.",
                "Keep paper titles, dataset names, benchmarks, model names, APIs, and acronyms in English.",
                "All natural-language content MUST be in Chinese.",
            ],
            "outline_expectations": [
            "Include major sections and subtopics to expand later.",
            "Provide 8-12 major steps.",
            "Each major step must include 3-5 substeps.",
            "Each substep should be 2-3 sentences.",
            "List source-backed bullets when available.",
            "Emphasize gaps and future research directions.",
            "Length must be 1000-2000 words in the output language.",
        ],
            "method_guidance": [
                "Use methods as analytical lenses applied to the topic and evidence.",
                "Do not treat methods as the topic itself.",
                "If methods are provided, align major steps to the methods and weave method names into the step text.",
                "Do not add bracketed method tags in titles.",
                "If method names are not Chinese, translate them into Chinese step titles and include the original in parentheses.",
            ],
        },
    }
    if structure_guidance:
        payload["instructions"]["structure_guidance"] = structure_guidance
    if skill_guidance:
        payload["skill_guidance"] = skill_guidance
    if length_hint:
        payload["length_hint"] = length_hint
    if language_hint:
        payload["language_hint"] = language_hint
    return json.dumps(payload, indent=2)
