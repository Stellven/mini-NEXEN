from __future__ import annotations

import json
from typing import Iterable

from .db import Document, Interest, Method

SYSTEM_PLAN_PROMPT = """You are a research planning agent. Produce a plan that is concise, actionable, and focused on gaps. All natural-language content MUST be in the requested output language.
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
- All natural-language content MUST be in the requested output language, except retrieval_queries must be in English.
- Keep JSON keys in English.
- Keep paper titles, dataset names, benchmarks, model names, APIs, and acronyms in English.
- Use ASCII punctuation for JSON (":" and ","), not full-width punctuation.
- When evidence includes confidence or recency, prefer higher-confidence and more recent evidence.
All natural-language content MUST be in the requested output language.

Example (Chinese):
{"scope":["..."],"key_questions":["..."],"keywords":["..."],"gaps":["..."],"notes":["..."],"readiness":"draft","retrieval_queries":["..."]}

Example (English):
{"scope":["..."],"key_questions":["..."],"keywords":["..."],"gaps":["..."],"notes":["..."],"readiness":"draft","retrieval_queries":["..."]}
"""

SYSTEM_OUTLINE_PROMPT = """You are a research planning agent. Produce a research plan (not a report outline). All natural-language content MUST be in the requested output language.
Return ONE valid JSON object only. No markdown, no commentary, no extra text.

Requirements:
- Must include key: outline
- outline MUST be a non-empty array of major steps (8-12 by default; may vary when methods or structure guidance require).
- Each major step should include 3-5 substeps by default; vary when it improves clarity, depth, or method alignment.
- Each substep should be 2-3 sentences by default; expand or shorten when needed for important elaboration.
- Optional sub-substeps are allowed when helpful.
- Use imperative phrasing (e.g., "Search...", "Compare...", "Investigate...").
- Do not write report section headings.
- Total outline length MUST be 1000-2000 words across titles and substeps in the output language.
- Output must be strict JSON (double quotes, no trailing commas)
- Treat methods as analysis approaches to apply to the topic, not as the topic itself.
- If methods are provided, the research plan MUST explicitly structure steps around those methods, even if the step count changes.
- All natural-language content MUST be in the requested output language.
- Keep JSON keys in English.
- Keep paper titles, dataset names, benchmarks, model names, APIs, and acronyms in English.
- Use ASCII punctuation for JSON (":" and ","), not full-width punctuation.
- When evidence includes confidence or recency, prefer higher-confidence and more recent evidence.
All natural-language content MUST be in the requested output language.

Example (Chinese):
{"outline":[{"title":"查找官方技术报告与发布博客以提取架构与训练细节。","substeps":["列出要检索的官方来源与发布页面。","提取训练规模、数据来源与能力摘要。",{"text":"捕捉部署背景与版本历史。","substeps":["记录发布节奏与更新日志。","整理公开的限制与注意事项。"]}]},{"title":"比较不同模型的上下文长度与长上下文准确性。","substeps":["收集厂商声明与独立基准。","分析长上下文召回与检索性能。",{"text":"识别失效案例。","substeps":["总结常见错误模式。","记录实践中的缓解策略。"]}]}]}

Example (English):
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
) -> list[dict]:
    results = []
    for doc in documents:
        results.append(
            {
                "title": doc.title,
                "source_type": doc.source_type,
                "source": doc.source,
                "doc_id": doc.doc_id,
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
    kg_fact_cards: list[dict] | None = None,
    output_language: str = "Chinese",
    profile_summary: list[dict] | None = None,
    skill_guidance: list[str] | None = None,
) -> str:
    payload = {
        "topic": topic,
        "interests": _serialize_interests(interests),
        "methods": _serialize_methods(methods),
        "documents": _serialize_documents(documents, keywords),
        "output_language": output_language,
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
                f"All list items must be in {output_language}.",
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
            "kg_guidance": [
                "If kg_fact_cards is provided, use it to ground claims and plan steps.",
                "Prefer higher-confidence and more recent evidence when shaping the plan.",
                "Do not copy evidence snippets verbatim; paraphrase.",
                "Do not fabricate sources that are not in kg_fact_cards or documents.",
            ],
            "profile_guidance": [
                "If profile_summary is provided, use it to personalize the plan without overriding the query.",
                "Add 1-3 Notes bullets that explicitly connect the plan to relevant profile signals.",
                "Treat profile_summary as context, not evidence; do not fabricate claims.",
            ],
        },
    }
    if extracted_interests:
        payload["extracted_interests"] = extracted_interests
    if kg_fact_cards:
        payload["kg_fact_cards"] = kg_fact_cards[:30]
    if profile_summary:
        payload["profile_summary"] = profile_summary[:10]
    if skill_guidance:
        payload["skill_guidance"] = skill_guidance
    return json.dumps(payload, indent=2)


def query_understanding_prompt(
    raw_query: str,
    methodology_taxonomy: list[str],
    profile_summary: list[dict] | None = None,
) -> str:
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
            "profile_guidance": [
                "If profile_summary is provided, use it to personalize topic framing and methodology choices.",
                "Profile signals are context, not evidence; do not override explicit query intent.",
            ],
        },
    }
    if profile_summary:
        payload["profile_summary"] = profile_summary[:10]
    return json.dumps(payload, indent=2)


def refine_prompt(
    topic: str,
    prior_plan: dict,
    interests: list[Interest],
    methods: list[Method],
    documents: list[Document],
    keywords: list[str],
    extracted_interests: list[str] | None = None,
    kg_fact_cards: list[dict] | None = None,
    output_language: str = "Chinese",
    profile_summary: list[dict] | None = None,
    skill_guidance: list[str] | None = None,
) -> str:
    payload = {
        "topic": topic,
        "prior_plan": prior_plan,
        "interests": _serialize_interests(interests),
        "methods": _serialize_methods(methods),
        "documents": _serialize_documents(documents, keywords),
        "output_language": output_language,
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
                f"All list items must be in {output_language}.",
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
            "kg_guidance": [
                "If kg_fact_cards is provided, use it to ground claims and plan steps.",
                "Prefer higher-confidence and more recent evidence when refining the plan.",
                "Do not copy evidence snippets verbatim; paraphrase.",
                "Do not fabricate sources that are not in kg_fact_cards or documents.",
            ],
            "profile_guidance": [
                "If profile_summary is provided, use it to personalize the plan without overriding the query.",
                "Add 1-3 Notes bullets that explicitly connect the plan to relevant profile signals.",
                "Treat profile_summary as context, not evidence; do not fabricate claims.",
            ],
        },
    }
    if extracted_interests:
        payload["extracted_interests"] = extracted_interests
    if kg_fact_cards:
        payload["kg_fact_cards"] = kg_fact_cards[:30]
    if profile_summary:
        payload["profile_summary"] = profile_summary[:10]
    if skill_guidance:
        payload["skill_guidance"] = skill_guidance
    return json.dumps(payload, indent=2)


def outline_prompt(
    topic: str,
    interests: list[Interest],
    methods: list[Method],
    documents: list[Document],
    keywords: list[str],
    kg_fact_cards: list[dict] | None = None,
    output_language: str = "Chinese",
    length_hint: str | None = None,
    language_hint: str | None = None,
    structure_guidance: list[str] | None = None,
    profile_summary: list[dict] | None = None,
    skill_guidance: list[str] | None = None,
) -> str:
    payload = {
        "topic": topic,
        "interests": _serialize_interests(interests),
        "methods": _serialize_methods(methods),
        "documents": _serialize_documents(documents, keywords),
        "keywords": keywords,
        "output_language": output_language,
        "instructions": {
            "output_json_schema": {
                "outline": ["string"]
            },
            "language_policy": [
                f"All natural-language content MUST be in {output_language}.",
                "Keep paper titles, dataset names, benchmarks, model names, APIs, and acronyms in English.",
                f"All natural-language content MUST be in {output_language}.",
            ],
            "outline_expectations": [
            "Include major sections and subtopics to expand later.",
            "Provide 8-12 major steps by default; if methods or structure_guidance are provided, follow them even if the step count changes.",
            "Each major step should include 3-5 substeps by default; expand or vary when it improves clarity, depth, or method alignment.",
            "Each substep should be 2-3 sentences by default; add or reduce sentences when needed for important elaboration.",
            "List source-backed bullets when available.",
            "Emphasize gaps and future research directions.",
            "Length must be 1000-2000 words in the output language.",
        ],
            "method_guidance": [
                "Use methods as analytical lenses applied to the topic and evidence.",
                "Do not treat methods as the topic itself.",
                "If methods are provided, align major steps to the methods and weave method names into the step text, even if the step count differs from the default.",
                "Do not add bracketed method tags in titles.",
                f"If method names are not in {output_language}, translate them into {output_language} step titles and include the original in parentheses.",
            ],
            "profile_guidance": [
                "If profile_summary is provided, use it to personalize the outline without overriding the query.",
                "For each top-layer step/section, include at least one explicit inline mention explaining how it ties to relevant profile signals when applicable.",
                "Keep profile ties selective and relevant; do not force ties that are misleading.",
                "Use brief natural-language explanations (no special labels required). Up to 10 profile ties per top-layer step.",
                "Treat profile_summary as context, not evidence; do not fabricate claims.",
            ],
            "kg_guidance": [
                "If kg_fact_cards is provided, use it to ground claims and outline steps.",
                "Prefer higher-confidence and more recent evidence when shaping the outline.",
                "Prefer evidence-backed relationships over unsupported assertions.",
                "Do not copy evidence snippets verbatim; paraphrase.",
                "When you use kg_fact_cards, add a short citation in-line like 'Sources: title1; title2'.",
                "Do not fabricate sources that are not in kg_fact_cards or documents.",
                "Contradictions may reflect genuine disagreements, errors, evolving consensus, or shifts in paradigm/direction over time; treat them as signals to analyze, not just errors to eliminate.",
                "Use trend stats and contradictions in kg_fact_cards to highlight agreements, shifts, and disagreements when evidence supports it.",
                "Call out stable areas (broad agreement) versus fast-moving areas (trend shifts or emerging contradictions).",
                "When trends differ across sub-communities or domains, describe them as segmented trends rather than forcing a single direction.",
                "Dates in kg_fact_cards come from document published_at timestamps when available; otherwise added_at (ingestion time).",
            ],
        },
    }
    if kg_fact_cards:
        payload["kg_fact_cards"] = kg_fact_cards[:40]
    if structure_guidance:
        payload["instructions"]["structure_guidance"] = structure_guidance
    if profile_summary:
        payload["profile_summary"] = profile_summary[:10]
    if skill_guidance:
        payload["skill_guidance"] = skill_guidance
    if length_hint:
        payload["length_hint"] = length_hint
    if language_hint:
        payload["language_hint"] = language_hint
    return json.dumps(payload, indent=2)
