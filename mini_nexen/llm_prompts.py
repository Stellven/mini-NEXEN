from __future__ import annotations

import json
from typing import Iterable

from .db import Document, Interest, Method

SYSTEM_PLAN_PROMPT = """You are a research planning agent. Produce a plan that is concise, actionable, and focused on what evidence is needed. All natural-language content MUST be in the requested output language.
Return ONE valid JSON object only. No markdown, no commentary, no extra text.

Requirements:
- Must include keys: scope, key_questions, keywords, source_requirements, section_requirements, gaps, notes, readiness
- scope, key_questions, keywords MUST be non-empty arrays (>= 3 items each)
- source_requirements MUST be an object describing global evidence characteristics (depth, breadth, rigor, recency, source_types, min_sources)
- section_requirements MUST be a non-empty array; each item must specify section, objective, subsections, and evidence_requirements
- keywords must include core terms from the topic and interests; include extracted interests if provided
- methods are analysis approaches; use them to frame the plan, not as the topic itself
- gaps and notes may be empty arrays
- retrieval_queries is optional; if provided, it must be a short list of search phrases
- readiness must be one of: "draft", "refined", "ready"
- Do NOT summarize or cite specific sources; specify the kinds of sources needed instead.
- Output must be strict JSON (double quotes, no trailing commas)
- All natural-language content MUST be in the requested output language, except retrieval_queries must be in English.
- Keep JSON keys in English.
- Keep paper titles, dataset names, benchmarks, model names, APIs, and acronyms in English.
- Use ASCII punctuation for JSON (":" and ","), not full-width punctuation.
- When evidence includes confidence or recency, prefer higher-confidence and more recent evidence.
All natural-language content MUST be in the requested output language.

Example (Chinese):
{"scope":["..."],"key_questions":["..."],"keywords":["..."],"source_requirements":{"depth":"...","breadth":"...","rigor":["..."],"recency":"...","source_types":["..."],"min_sources":3},"section_requirements":[{"section":"...","objective":"...","subsections":["..."],"evidence_requirements":{"depth":"...","breadth":"...","rigor":["..."],"recency":"...","source_types":["..."],"min_sources":2}}],"gaps":["..."],"notes":["..."],"readiness":"draft","retrieval_queries":["..."]}

Example (English):
{"scope":["..."],"key_questions":["..."],"keywords":["..."],"source_requirements":{"depth":"...","breadth":"...","rigor":["..."],"recency":"...","source_types":["..."],"min_sources":3},"section_requirements":[{"section":"...","objective":"...","subsections":["..."],"evidence_requirements":{"depth":"...","breadth":"...","rigor":["..."],"recency":"...","source_types":["..."],"min_sources":2}}],"gaps":["..."],"notes":["..."],"readiness":"draft","retrieval_queries":["..."]}
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
- Total outline length MUST be at least 1000 words across titles and substeps in the output language.
- Output must be strict JSON (double quotes, no trailing commas)
- Treat methods as analysis approaches to apply to the topic, not as the topic itself.
- If methods are provided, the research plan MUST explicitly structure steps around those methods, even if the step count changes.
- If plan_requirements are provided, align steps to the required sections and evidence needs.
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

SYSTEM_REVIEW_PROMPT = (
    "You are a careful reviewer. Evaluate compliance and relevance rigorously. "
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
    revision_feedback: list[str] | None = None,
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
                "source_requirements": {
                    "depth": "string",
                    "breadth": "string",
                    "rigor": ["string"],
                    "recency": "string",
                    "source_types": ["string"],
                    "min_sources": "number",
                },
                "section_requirements": [
                    {
                        "section": "string",
                        "objective": "string",
                        "subsections": ["string"],
                        "evidence_requirements": {
                            "depth": "string",
                            "breadth": "string",
                            "rigor": ["string"],
                            "recency": "string",
                            "source_types": ["string"],
                            "min_sources": "number",
                        },
                    }
                ],
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
                "Apply methods to the topic and evidence requirements.",
            ],
            "planning_policy": [
                "Do not summarize or cite specific sources; specify source needs instead.",
                "Treat documents as empty; do not reference evidence.",
                "Use section_requirements to break down major sections and required evidence characteristics.",
            ],
            "kg_guidance": [
                "If kg_fact_cards is provided, use it to ground claims and plan steps.",
                "Prefer higher-confidence and more recent evidence when shaping the plan.",
                "Do not copy evidence snippets verbatim; paraphrase.",
                "Do not fabricate sources that are not in kg_fact_cards or documents.",
            ],
            "profile_guidance": [
                "If profile_summary is provided, use it to personalize the plan without overriding the query.",
                "You may use profile themes or summarized themes even when supporting evidence is thin; treat them as inferred context rather than verified facts.",
                "If a profile theme aligns with kg_fact_cards evidence, you may treat that evidence as support for the theme.",
                "Do not add [关切] or [关切证据] tags anywhere in the plan output; these tags are reserved for the Research Plan outline.",
                "Treat profile_summary as context, not evidence; do not fabricate sources or claims of evidence.",
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
    if revision_feedback:
        payload["revision_feedback"] = revision_feedback[:10]
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
    revision_feedback: list[str] | None = None,
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
                "source_requirements": {
                    "depth": "string",
                    "breadth": "string",
                    "rigor": ["string"],
                    "recency": "string",
                    "source_types": ["string"],
                    "min_sources": "number",
                },
                "section_requirements": [
                    {
                        "section": "string",
                        "objective": "string",
                        "subsections": ["string"],
                        "evidence_requirements": {
                            "depth": "string",
                            "breadth": "string",
                            "rigor": ["string"],
                            "recency": "string",
                            "source_types": ["string"],
                            "min_sources": "number",
                        },
                    }
                ],
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
                "Apply methods to the topic and evidence requirements.",
            ],
            "planning_policy": [
                "Do not summarize or cite specific sources; specify source needs instead.",
                "Treat documents as empty; do not reference evidence.",
                "Use section_requirements to break down major sections and required evidence characteristics.",
            ],
            "kg_guidance": [
                "If kg_fact_cards is provided, use it to ground claims and plan steps.",
                "Prefer higher-confidence and more recent evidence when refining the plan.",
                "Do not copy evidence snippets verbatim; paraphrase.",
                "Do not fabricate sources that are not in kg_fact_cards or documents.",
            ],
            "profile_guidance": [
                "If profile_summary is provided, use it to personalize the plan without overriding the query.",
                "Do not add [关切] or [关切证据] tags anywhere in the plan output; these tags are reserved for the Research Plan outline.",
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
    if revision_feedback:
        payload["revision_feedback"] = revision_feedback[:10]
    return json.dumps(payload, indent=2)


def outline_prompt(
    topic: str,
    interests: list[Interest],
    methods: list[Method],
    documents: list[Document],
    keywords: list[str],
    plan_requirements: dict[str, object] | None = None,
    kg_fact_cards: list[dict] | None = None,
    output_language: str = "Chinese",
    length_hint: str | None = None,
    language_hint: str | None = None,
    structure_guidance: list[str] | None = None,
    profile_summary: list[dict] | None = None,
    skill_guidance: list[str] | None = None,
    revision_feedback: list[str] | None = None,
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
            "Do not include citations, references, or 'Sources:' style markers in the outline output.",
            "Emphasize gaps and future research directions.",
            "Length must be at least 1000 words in the output language.",
        ],
            "plan_guidance": [
                "If plan_requirements is provided, ensure every required section is represented in the outline.",
                "Translate each section's evidence_requirements into concrete research steps and source acquisition actions.",
                "Carry forward any depth/breadth/rigor/recency/source_type requirements into the outline steps.",
            ],
            "method_guidance": [
                "Use methods as analytical lenses applied to the topic and evidence.",
                "Do not treat methods as the topic itself.",
                "If methods are provided, align major steps to the methods and weave method names into the step text, even if the step count differs from the default.",
                "When structure_guidance provides method steps for a skill, use those steps to define the major steps for that skill; treat other methods as analytical lenses only.",
                "Do not add bracketed method tags in titles other than the required [Methodology Skill: ...] label when instructed.",
                f"If method names are not in {output_language}, translate them into {output_language} step titles and include the original in parentheses.",
            ],
            "profile_guidance": [
                "If profile_summary is provided, use it to personalize the outline without overriding the query.",
                "For each top-layer step/section, include at least one explicit inline mention explaining how it ties to relevant profile signals when applicable.",
                "When profile_summary is provided, each top-layer step must include at least one [关切] or [关切证据] tag.",
                "Keep profile ties selective and relevant; do not force ties that are misleading.",
                "Use brief natural-language explanations. Up to 10 profile ties per top-layer step.",
                "Whenever you mention a profile theme or summarized theme in ANY step, add a brief explanation of relevance and append ' [关切]' at the end of that sentence.",
                "When a profile theme mention is supported by aligned kg_fact_cards evidence, append ' [关切证据]' instead of ' [关切]'.",
                "Use [关切] when the tie is contextual or inferred without direct evidence; reserve [关切证据] for clear evidence alignment.",
                "Place profile tags in the outline steps (Research Plan). Do not use them in Notes.",
                "Treat profile_summary as context, not evidence; do not fabricate claims.",
            ],
            "kg_guidance": [
                "If kg_fact_cards is provided, use it to ground claims and outline steps.",
                "Prefer higher-confidence and more recent evidence when shaping the outline.",
                "Prefer evidence-backed relationships over unsupported assertions.",
                "Do not copy evidence snippets verbatim; paraphrase.",
                "Do not include explicit source citations or reference lists in the outline text.",
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
    if plan_requirements:
        payload["plan_requirements"] = plan_requirements
    if revision_feedback:
        payload["revision_feedback"] = revision_feedback[:10]
    if length_hint:
        payload["length_hint"] = length_hint
    if language_hint:
        payload["language_hint"] = language_hint
    return json.dumps(payload, indent=2)


def outline_profile_review_prompt(
    topic: str,
    outline: list[str],
    keywords: list[str],
    profile_summary: list[dict],
    kg_fact_cards: list[dict] | None = None,
    output_language: str = "Chinese",
) -> str:
    payload = {
        "topic": topic,
        "keywords": keywords,
        "outline": outline,
        "profile_summary": profile_summary[:10],
        "output_language": output_language,
        "instructions": {
            "output_json_schema": {
                "relevant_labels": ["string"],
                "missing_labels": ["string"],
                "untagged_mentions": [{"label": "string", "line": "string"}],
                "suggested_additions": [{"label": "string", "why": "string", "placement_hint": "string"}],
                "needs_revision": "true|false",
                "rationale": "string",
            },
            "requirements": [
                "Decide which profile labels are actually relevant to the topic and outline.",
                "Only consider relevant labels; do not force irrelevant profile themes.",
                "If a relevant label should appear in the outline but does not, list it in missing_labels.",
                "If a line mentions a relevant label but lacks [关切] or [关切证据], include it in untagged_mentions with the exact line.",
                "Identify additional relevant profile themes that would materially improve the outline if added; list them in suggested_additions with a brief why and placement_hint.",
                "Treat lines that do not start with '-' as top-layer steps.",
                "If any top-layer step lacks a [关切] or [关切证据] tag when profile_summary is provided, include a suggested_addition and set needs_revision to true.",
                "Set needs_revision to true when missing_labels or untagged_mentions are non-empty.",
                "Keep rationale brief and in the output language.",
            ],
        },
    }
    if kg_fact_cards:
        payload["kg_fact_cards"] = kg_fact_cards[:20]
    return json.dumps(payload, indent=2)


def plan_readiness_review_prompt(
    topic: str,
    plan: dict,
    source_briefs_count: int,
    source_types: list[str],
    output_language: str = "Chinese",
) -> str:
    payload = {
        "topic": topic,
        "plan": plan,
        "source_briefs_count": source_briefs_count,
        "source_types": source_types,
        "output_language": output_language,
        "instructions": {
            "output_json_schema": {
                "ready": "true|false",
                "readiness": "draft | refined | ready",
                "gaps": ["string"],
                "rationale": "string",
            },
            "requirements": [
                "Judge whether the plan is ready to proceed based on completeness, specificity, and evidence coverage.",
                "If not ready, list concrete gaps that should be addressed.",
                "Keep rationale brief and in the output language.",
            ],
        },
    }
    return json.dumps(payload, indent=2)


def plan_quality_review_prompt(
    topic: str,
    plan: dict,
    validation: dict,
    output_language: str = "Chinese",
) -> str:
    payload = {
        "topic": topic,
        "plan": plan,
        "validation": validation,
        "output_language": output_language,
        "instructions": {
            "output_json_schema": {
                "action": "accept | retry",
                "feedback": ["string"],
                "gaps": ["string"],
                "rationale": "string",
            },
            "requirements": [
                "Use validation.errors to decide if retry is required.",
                "Evaluate plan completeness, section coverage, and alignment to topic/methods.",
                "Ensure section_requirements specify depth, breadth, rigor, recency, and source types.",
                "Do not require evidence or citations; this is an evidence-needs plan.",
                "Return brief, actionable feedback for improving the plan.",
                "Keep rationale brief and in the output language.",
            ],
        },
    }
    return json.dumps(payload, indent=2)


def outline_quality_review_prompt(
    topic: str,
    outline: list[str],
    plan_requirements: dict,
    validation: dict,
    output_language: str = "Chinese",
) -> str:
    payload = {
        "topic": topic,
        "plan_requirements": plan_requirements,
        "outline": outline,
        "validation": validation,
        "output_language": output_language,
        "instructions": {
            "output_json_schema": {
                "action": "accept | outline_retry | retrieve_more | plan_retry",
                "feedback": ["string"],
                "retrieval_gaps": ["string"],
                "plan_gaps": ["string"],
                "rationale": "string",
            },
            "requirements": [
                "Use validation.errors to decide if outline_retry is required.",
                "If evidence is insufficient for plan_requirements, set action to retrieve_more and list concrete retrieval_gaps.",
                "If the plan itself is missing or inconsistent with the outline, set action to plan_retry and list plan_gaps.",
                "If [关切证据] appears without aligned evidence, advise downgrading to [关切] and provide feedback.",
                "If validation.warnings include missing_profile_tags, set action to outline_retry and request [关切]/[关切证据] tags in top-layer steps.",
                "Otherwise, accept or outline_retry with actionable feedback.",
                "Do not request citations in the outline itself; focus on research steps and evidence needs.",
                "Keep rationale brief and in the output language.",
            ],
        },
    }
    return json.dumps(payload, indent=2)
