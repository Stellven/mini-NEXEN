from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .llm import LLMClient, emit_progress
from .llm_prompts import SYSTEM_QUERY_UNDERSTANDING_PROMPT, query_understanding_prompt


DEFAULT_METHOD_TAXONOMY = [
    "Comparative analysis",
    "Literature review",
    "Gap analysis",
    "Trend analysis",
    "Risk/threat analysis",
    "Cost-benefit analysis",
    "Stakeholder analysis",
    "Systems analysis",
    "Causal analysis",
    "Benchmarking",
    "Scenario analysis",
    "Policy/regulatory analysis",
    "Market analysis",
    "Case study synthesis",
    "SWOT",
    "PESTEL",
]

_METHOD_ALIASES = {
    "benchmark": "Benchmarking",
    "benchmarking": "Benchmarking",
    "comparative": "Comparative analysis",
    "comparison": "Comparative analysis",
    "compare": "Comparative analysis",
    "system analysis": "Systems analysis",
    "systems engineering": "Systems analysis",
    "risk analysis": "Risk/threat analysis",
    "threat analysis": "Risk/threat analysis",
    "risk assessment": "Risk/threat analysis",
    "cost benefit": "Cost-benefit analysis",
    "cost-benefit": "Cost-benefit analysis",
    "stakeholder": "Stakeholder analysis",
    "trend": "Trend analysis",
    "causal": "Causal analysis",
    "root cause": "Causal analysis",
    "scenario": "Scenario analysis",
    "policy": "Policy/regulatory analysis",
    "regulatory": "Policy/regulatory analysis",
    "market": "Market analysis",
    "case study": "Case study synthesis",
    "literature review": "Literature review",
    "systematic review": "Literature review",
    "meta analysis": "Literature review",
    "meta-analysis": "Literature review",
    "swot": "SWOT",
    "pestel": "PESTEL",
}

_METHOD_PHRASE_TERMS = [
    "comparative analysis",
    "literature review",
    "systematic review",
    "meta-analysis",
    "meta analysis",
    "gap analysis",
    "trend analysis",
    "risk analysis",
    "threat analysis",
    "cost-benefit analysis",
    "cost benefit analysis",
    "stakeholder analysis",
    "systems analysis",
    "causal analysis",
    "root-cause analysis",
    "benchmarking",
    "benchmark analysis",
    "scenario analysis",
    "policy analysis",
    "regulatory analysis",
    "market analysis",
    "case study",
    "swot",
    "pestel",
]


@dataclass
class QueryUnderstanding:
    topic: str
    normalized_query: str
    methodologies: list[str]
    confidence: float
    rationale: str
    constraints: dict[str, str | None]
    audience: str | None


def _extract_json_payload(text: str) -> dict[str, Any]:
    if not text:
        return {}
    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if start_obj == -1 or end_obj == -1 or end_obj <= start_obj:
        return {}
    candidate = text[start_obj : end_obj + 1]
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _normalize_methodologies(raw: Any, taxonomy: list[str]) -> list[str]:
    if not isinstance(raw, list):
        return []
    canonical = {item.casefold(): item for item in taxonomy}
    normalized: list[str] = []
    seen = set()
    for item in raw:
        text = str(item).strip()
        if not text:
            continue
        key = text.casefold()
        match = canonical.get(key)
        if not match:
            match = _METHOD_ALIASES.get(key)
        if not match:
            for alias, canonical_item in _METHOD_ALIASES.items():
                if alias in key:
                    match = canonical_item
                    break
        if not match:
            continue
        if match.casefold() in seen:
            continue
        normalized.append(match)
        seen.add(match.casefold())
    return normalized


def _heuristic_methodologies(query: str) -> list[str]:
    if not query:
        return ["Literature review"]
    lowered = query.casefold()
    selections: list[str] = []
    patterns = [
        (r"\bcompare|comparison|versus|vs\.?\b", "Comparative analysis"),
        (r"\bbenchmark", "Benchmarking"),
        (r"\btrend|trajectory|evolution", "Trend analysis"),
        (r"\brisk|threat", "Risk/threat analysis"),
        (r"\bcost benefit|cost-benefit", "Cost-benefit analysis"),
        (r"\bstakeholder", "Stakeholder analysis"),
        (r"\bsystem|architecture|systems", "Systems analysis"),
        (r"\bcausal|root cause", "Causal analysis"),
        (r"\bscenario", "Scenario analysis"),
        (r"\bpolicy|regulat", "Policy/regulatory analysis"),
        (r"\bmarket|industry|size", "Market analysis"),
        (r"\bcase study|case studies", "Case study synthesis"),
        (r"\bswot", "SWOT"),
        (r"\bpestel", "PESTEL"),
        (r"\bliterature review|systematic review|survey|meta", "Literature review"),
        (r"\bgap analysis|gap", "Gap analysis"),
    ]
    for pattern, method in patterns:
        if re.search(pattern, lowered):
            if method not in selections:
                selections.append(method)
    if not selections:
        selections.append("Literature review")
    return selections[:3]


def _strip_method_terms(text: str, methods: list[str]) -> str:
    if not text:
        return ""
    cleaned = text
    terms = set(term.casefold() for term in _METHOD_PHRASE_TERMS)
    for method in methods:
        term = method.casefold().strip()
        if term:
            terms.add(term)
    for term in terms:
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, cleaned, flags=re.IGNORECASE):
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:-")
    return cleaned


def normalize_query_understanding(
    payload: dict[str, Any] | None,
    raw_query: str,
    taxonomy: list[str],
) -> QueryUnderstanding:
    payload = payload or {}
    topic = str(payload.get("topic") or "").strip() or raw_query
    normalized_query = str(payload.get("normalized_query") or "").strip() or topic
    methodologies = _normalize_methodologies(payload.get("methodologies"), taxonomy)
    if not methodologies:
        methodologies = _heuristic_methodologies(raw_query)
    cleaned_query = _strip_method_terms(normalized_query, methodologies)
    if cleaned_query:
        normalized_query = cleaned_query
    confidence = payload.get("confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    rationale = str(payload.get("rationale") or "").strip()
    constraints_payload = payload.get("constraints") if isinstance(payload.get("constraints"), dict) else {}
    constraints = {
        "timeframe": str(constraints_payload.get("timeframe") or "").strip() or None,
        "region": str(constraints_payload.get("region") or "").strip() or None,
        "industry": str(constraints_payload.get("industry") or "").strip() or None,
    }
    audience = str(payload.get("audience") or "").strip() or None
    return QueryUnderstanding(
        topic=topic,
        normalized_query=normalized_query,
        methodologies=methodologies,
        confidence=confidence,
        rationale=rationale,
        constraints=constraints,
        audience=audience,
    )


def infer_query_understanding(
    llm: LLMClient | None,
    raw_query: str,
    taxonomy: list[str] | None = None,
    profile_summary: list[dict] | None = None,
) -> QueryUnderstanding:
    taxonomy = taxonomy or DEFAULT_METHOD_TAXONOMY
    if not llm:
        return normalize_query_understanding({}, raw_query, taxonomy)
    prompt = query_understanding_prompt(raw_query, taxonomy, profile_summary=profile_summary)
    model_label = llm.config.model
    emit_progress("Planner", model_label, "query understanding", 0, 1, done=False)
    try:
        response = llm.generate(
            system_prompt=SYSTEM_QUERY_UNDERSTANDING_PROMPT,
            user_prompt=prompt,
            task="query understanding",
            agent="Planner",
        )
    finally:
        emit_progress("Planner", model_label, "query understanding", 1, 1, done=True)
    payload = _extract_json_payload(response)
    return normalize_query_understanding(payload, raw_query, taxonomy)


def build_methodology_terms(methodologies: list[str]) -> list[str]:
    terms = set(term.casefold() for term in _METHOD_PHRASE_TERMS)
    for item in methodologies:
        term = item.casefold().strip()
        if term:
            terms.add(term)
    cleaned = sorted(terms)
    return cleaned


def render_query_artifact(
    understanding: QueryUnderstanding,
    raw_query: str,
    taxonomy: list[str],
    skill_catalog: list[dict[str, object]] | None = None,
    predicted_skills: list[str] | None = None,
    skill_hints: list[str] | None = None,
    web_search_payload: dict[str, object] | None = None,
) -> str:
    payload = {
        "raw_query": raw_query,
        "topic": understanding.topic,
        "normalized_query": understanding.normalized_query,
        "methodologies": understanding.methodologies,
        "confidence": understanding.confidence,
        "rationale": understanding.rationale,
        "constraints": understanding.constraints,
        "audience": understanding.audience,
        "methodology_taxonomy": taxonomy,
        "predicted_skills": predicted_skills or [],
        "skill_hints": skill_hints or [],
        "skill_catalog": skill_catalog or [],
    }
    lines = [
        "# Query + Web Search (editable)",
        "",
        "Edit the JSON blocks below if needed. Keep them valid JSON.",
        "",
        "## Query Understanding (affects planning + skill selection)",
        "",
        "Notes:",
        "- `predicted_skills` is display-only and does not activate skills.",
        "- `skill_hints` activates skills. You can use skill_id, display_name, aliases, or index.",
        "- `constraints` (like timeframe/region/industry) feed planning and KG filtering.",
        "",
        "```json",
        json.dumps(payload, indent=2, ensure_ascii=False),
        "```",
    ]
    if web_search_payload is not None:
        lines.extend(
            [
                "",
                "## Web Search Settings (affects retrieval only)",
                "",
                "Notes:",
                "- `platforms_available` and `platforms_enabled` are informational.",
                "- `preferred_sources` is for future use and is not applied yet.",
                "- Edits are applied to `search_topics`, `modes`, and `search_modes.semantic_rerank` for this run.",
                "",
                "```json",
                json.dumps(web_search_payload, indent=2, ensure_ascii=False),
                "```",
            ]
        )
    return "\n".join(lines) + "\n"


def _extract_json_blocks(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    blocks: list[dict[str, Any]] = []
    for match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE):
        candidate = match.group(1)
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            blocks.append(payload)
    if blocks:
        return blocks
    payload = _extract_json_payload(text)
    if payload:
        return [payload]
    return []


def _is_query_payload(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    if "normalized_query" in payload:
        return True
    return "topic" in payload and "methodologies" in payload and "constraints" in payload


def _is_web_search_payload(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    return "search_topics" in payload or "modes" in payload or "search_modes" in payload


def parse_query_artifact(text: str) -> dict[str, Any]:
    blocks = _extract_json_blocks(text)
    for payload in blocks:
        if _is_query_payload(payload):
            return payload
    if blocks:
        return blocks[0]
    return {}


def parse_web_search_artifact(text: str) -> dict[str, Any]:
    blocks = _extract_json_blocks(text)
    for payload in blocks:
        if _is_web_search_payload(payload):
            return payload
    return {}
