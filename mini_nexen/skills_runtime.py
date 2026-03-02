from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import os
import shlex
import subprocess
import re
import uuid
from pathlib import Path
from typing import Callable, Optional

from . import db
from .config import (
    ARTIFACTS_DIR,
    DEFAULT_ROUNDS,
    DEFAULT_KG_HOPS,
    DEFAULT_PROFILE_TOP_K,
    DEFAULT_TOP_K,
    DEFAULT_OUTLINE_REVIEW_ROUNDS,
    SKILLS_DIR,
    WEB_AUTO_MAX_ROUNDS,
    WEB_EVIDENCE_DEFAULT_DAYS,
    WEB_EXPAND_CONFIDENCE_MIN,
    WEB_EXPAND_CONTRADICTION_CONF_MAX,
    WEB_EXPAND_CONTRADICTION_STALE_DAYS,
    WEB_EXPAND_EVIDENCE_MIN,
    WEB_EXPAND_STALE_DAYS,
    WEB_MAX_NEW_SOURCES,
    WEB_MAX_PER_QUERY,
    WEB_RELEVANCE_THRESHOLD,
    ensure_dirs,
)
from .embeddings import EmbeddingClient, EmbeddingConfig, cosine_similarity
from .kg import (
    KGStore,
    apply_profile_items,
    build_seed_terms,
    detect_contradictions,
    extract_and_store,
    extract_profile_items,
    log_subgraph_summary,
    seed_terms_from_query,
    update_profile_from_mentions,
)
from .llm import (
    LLMClient,
    LLMClientError,
    emit_progress,
    log_task_event,
    log_task_event_quiet,
)
from .planning import (
    PlanDraft,
    build_profile_signals,
    is_ready,
    llm_build_outline,
    llm_draft_plan,
    llm_refine_plan,
    normalize_bracket_tag,
    review_plan_readiness,
    render_plan_md,
)
from .text_utils import score_documents, tokenize
from .web_retrieval import expand_queries, run_web_retrieval
from .query_understanding import (
    DEFAULT_METHOD_TAXONOMY,
    QueryUnderstanding,
    build_methodology_terms,
    infer_query_understanding,
    normalize_query_understanding,
    parse_query_artifact,
    parse_web_search_artifact,
    render_query_artifact,
)


@dataclass
class SkillSpec:
    name: str
    description: str
    inputs: list[str]
    outputs: list[str]
    display_name: str
    aliases: list[str]
    path: Path


@dataclass
class MethodCandidate:
    name: str
    description: str = ""
    steps: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)


@dataclass
class MethodSelection:
    skill_name: str
    method: str
    source: str
    confidence: float | None = None
    steps: list[str] = field(default_factory=list)


@dataclass
class SkillContext:
    topic: str
    raw_topic: str = ""
    normalized_query: str = ""
    output_language: str = "Chinese"
    inferred_methods: list[db.Method] = field(default_factory=list)
    methodology_terms: list[str] = field(default_factory=list)
    methodology_taxonomy: list[str] = field(default_factory=list)
    auto_methods: bool = True
    review_query: bool = False
    interactive: bool = False
    query_artifact_path: Optional[Path] = None
    web_search_artifact_path: Optional[Path] = None
    query_understanding: Optional[QueryUnderstanding] = None
    top_k: int = DEFAULT_TOP_K
    max_rounds: int = DEFAULT_ROUNDS
    round_number: int = 1
    interests: list[db.Interest] = field(default_factory=list)
    methods: list[db.Method] = field(default_factory=list)
    extracted_interests: list[str] = field(default_factory=list)
    profile_summary: list[dict] = field(default_factory=list)
    documents: list[db.Document] = field(default_factory=list)
    kg_fact_cards: list[dict] = field(default_factory=list)
    plan: Optional[PlanDraft] = None
    outline: list[str] = field(default_factory=list)
    plan_md: str = ""
    plan_path: Optional[Path] = None
    notes: list[str] = field(default_factory=list)
    query_hints: list[str] = field(default_factory=list)
    active_skills: list[str] = field(default_factory=list)
    skill_guidance: list[str] = field(default_factory=list)
    skill_hints: list[str] = field(default_factory=list)
    skill_method_index: dict[str, list[MethodCandidate]] = field(default_factory=dict)
    skill_method_selections: dict[str, MethodSelection] = field(default_factory=dict)
    web_enabled: bool = False
    web_forced: bool = False
    web_auto: bool = False
    run_id: int = 0
    web_modes: list[str] = field(default_factory=list)
    web_search_topics: list[str] = field(default_factory=list)
    web_rounds_used: int = 0
    web_max_rounds: int = WEB_AUTO_MAX_ROUNDS
    web_max_results: int = 5
    web_timeout: int = 15
    web_fetch_pages: bool = True
    web_hybrid: bool = True
    web_embed_provider: str | None = None
    web_embed_model: str | None = None
    web_embed_base_url: str | None = None
    web_embed_timeout: int | None = None
    web_embed_api_key: str | None = None
    web_expand_queries: bool = True
    web_max_queries: int = 10
    web_max_new_sources: int = WEB_MAX_NEW_SOURCES
    web_max_per_query: int = WEB_MAX_PER_QUERY
    web_relevance_threshold: float = WEB_RELEVANCE_THRESHOLD
    profile_top_k: int = DEFAULT_PROFILE_TOP_K
    outline_review_rounds: int = DEFAULT_OUTLINE_REVIEW_ROUNDS
    kg_hops: int = DEFAULT_KG_HOPS
    kg_subgraph_stats: dict[str, float | int | str] = field(default_factory=dict)
    kg_updated: bool = False
    llm: Optional[LLMClient] = None


SkillFn = Callable[[SkillContext], SkillContext]


class SkillRegistry:
    def __init__(self, skills_dir: Path = SKILLS_DIR):
        self.skills_dir = skills_dir
        self.skills: dict[str, SkillSpec] = {}

    def load(self) -> None:
        self.skills.clear()
        for skill_path in self.skills_dir.glob("*/SKILL.md"):
            spec = self._parse_skill(skill_path)
            if spec:
                self.skills[spec.name] = spec

    def _parse_skill(self, path: Path) -> Optional[SkillSpec]:
        content = path.read_text(encoding="utf-8")
        if not content.startswith("---"):
            return None
        parts = content.split("---", 2)
        if len(parts) < 3:
            return None
        meta_raw = parts[1]
        meta = {}
        for line in meta_raw.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
        name = meta.get("name")
        if not name:
            return None
        description = meta.get("description", "")
        display_name = meta.get("display_name", name)
        aliases = [item.strip() for item in meta.get("aliases", "").split(",") if item.strip()]
        inputs = [item.strip() for item in meta.get("inputs", "").split(",") if item.strip()]
        outputs = [item.strip() for item in meta.get("outputs", "").split(",") if item.strip()]
        return SkillSpec(
            name=name,
            description=description,
            inputs=inputs,
            outputs=outputs,
            display_name=display_name,
            aliases=aliases,
            path=path,
        )


class SkillRunner:
    def __init__(self, registry: SkillRegistry):
        self.registry = registry
        self.handlers: dict[str, SkillFn] = {}

    def register(self, name: str, fn: SkillFn) -> None:
        self.handlers[name] = fn

    def run(self, name: str, ctx: SkillContext) -> SkillContext:
        if name not in self.handlers:
            raise ValueError(f"Skill '{name}' is not registered")
        if name not in self.registry.skills:
            raise ValueError(f"Skill '{name}' not found in skills registry")
        log_task_event(f"***Skill activated: {name}***")
        return self.handlers[name](ctx)


# Skill implementations

SYSTEMS_ENGINEERING_TRIGGERS = [
    "system design",
    "requirements",
    "architecture",
    "v&v",
    "mbse",
    "icd",
    "trade study",
    "trade-off",
    "trade off",
    "fmea",
    "conops",
    "system integration",
    "dodaf",
    "sysml",
    "三一工程",
    "五看三定",
    "四化设计",
    "系统融合",
    "作战地图",
    "产业投资",
]


def _matches_triggers(texts: list[str], triggers: list[str]) -> bool:
    haystack = " ".join(text for text in texts if text).casefold()
    for trigger in triggers:
        if trigger.casefold() in haystack:
            return True
    return False

def _ensure_method(ctx: SkillContext, name: str, notes: str) -> None:
    if not name:
        return
    existing = {method.method.casefold() for method in ctx.methods if method.method}
    if name.casefold() in existing:
        return
    ctx.methods.append(
        db.Method(
            method_id=f"skill:{uuid.uuid4()}",
            method=name,
            notes=notes,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    )


def _split_front_matter(content: str) -> tuple[list[str], str]:
    if not content.startswith("---"):
        return ([], content)
    parts = content.split("---", 2)
    if len(parts) < 3:
        return ([], content)
    meta_raw = parts[1]
    body = parts[2]
    return (meta_raw.splitlines(), body)


def _parse_list_value(value: str) -> list[str]:
    cleaned = value.strip()
    if not cleaned:
        return []
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    items = [item.strip().strip("'\"") for item in cleaned.split(",") if item.strip()]
    return items


def _split_name_aliases(name: str) -> tuple[str, list[str]]:
    if not name:
        return ("", [])
    pattern = r"^(.*?)\s*[\(（](.*?)[\)）]\s*$"
    match = re.match(pattern, name.strip())
    if match:
        base = match.group(1).strip()
        alias = match.group(2).strip()
        split = _split_outside_parens(alias, " — ") or _split_outside_parens(alias, " - ")
        if split:
            alias = split[0].strip()
        aliases = [alias] if alias else []
        return (base, aliases)
    return (name.strip(), [])


def _split_outside_parens(text: str, sep: str) -> tuple[str, str] | None:
    if not text or not sep:
        return None
    depth = 0
    idx = 0
    while idx <= len(text) - len(sep):
        ch = text[idx]
        if ch in {"(", "（"}:
            depth += 1
        elif ch in {")", "）"} and depth > 0:
            depth -= 1
        if depth == 0 and text.startswith(sep, idx):
            return (text[:idx], text[idx + len(sep) :])
        idx += 1
    return None


def _parse_methods_block(lines: list[str], start_index: int) -> list[MethodCandidate]:
    candidates: list[MethodCandidate] = []
    current: MethodCandidate | None = None
    in_steps = False
    steps_indent: int | None = None
    for line in lines[start_index + 1 :]:
        if not line.strip():
            continue
        if line.lstrip() == line:
            break
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if stripped.startswith("- "):
            item = stripped[2:].strip()
            if in_steps and current and steps_indent is not None and indent > steps_indent:
                current.steps.append(item)
                continue
            if current:
                candidates.append(current)
            current = MethodCandidate(name="", description="", steps=[], aliases=[])
            in_steps = False
            steps_indent = None
            if item.startswith("name:"):
                current.name = item.split(":", 1)[1].strip()
            else:
                current.name = item
            continue
        if current is None:
            continue
        if stripped.startswith("name:"):
            current.name = stripped.split(":", 1)[1].strip()
            in_steps = False
            steps_indent = None
            continue
        if stripped.startswith("description:"):
            current.description = stripped.split(":", 1)[1].strip()
            in_steps = False
            steps_indent = None
            continue
        if stripped.startswith("aliases:"):
            alias_value = stripped.split(":", 1)[1].strip()
            current.aliases = _parse_list_value(alias_value)
            in_steps = False
            steps_indent = None
            continue
        if stripped.startswith("steps:"):
            step_value = stripped.split(":", 1)[1].strip()
            if step_value:
                current.steps = _parse_list_value(step_value)
                in_steps = False
                steps_indent = None
            else:
                in_steps = True
                steps_indent = indent
            continue
    if current:
        candidates.append(current)
    return candidates


def _parse_method_table(lines: list[str], start_index: int) -> tuple[list[MethodCandidate], int]:
    if start_index + 1 >= len(lines):
        return ([], start_index)
    header = lines[start_index].strip()
    divider = lines[start_index + 1].strip()
    if "|" not in header or "|" not in divider:
        return ([], start_index)
    divider_clean = divider.replace("|", "").strip()
    if not divider_clean or not set(divider_clean) <= {"-", ":", " "}:
        return ([], start_index)
    header_cells = [cell.strip() for cell in header.strip("|").split("|")]
    method_idx = None
    desc_idx = None
    for idx, cell in enumerate(header_cells):
        cell_fold = cell.casefold()
        if "method" in cell_fold or "方法" in cell:
            method_idx = idx
        if "description" in cell_fold or "desc" in cell_fold or "说明" in cell or "描述" in cell:
            desc_idx = idx
    if method_idx is None:
        return ([], start_index)
    candidates: list[MethodCandidate] = []
    idx = start_index + 2
    while idx < len(lines):
        row = lines[idx].strip()
        if "|" not in row:
            break
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        if method_idx >= len(cells):
            idx += 1
            continue
        name = cells[method_idx]
        desc = ""
        if desc_idx is not None and desc_idx < len(cells):
            desc = cells[desc_idx]
        if name:
            candidates.append(MethodCandidate(name=name, description=desc))
        idx += 1
    return (candidates, idx - 1)


def _extract_skill_method_candidates(content: str) -> list[MethodCandidate]:
    if not content:
        return []
    meta_lines, body = _split_front_matter(content)
    meta: dict[str, str] = {}
    for line in meta_lines:
        if ":" not in line or line.lstrip() != line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip().casefold().replace("-", "_")] = value.strip()
    tags = _parse_list_value(meta.get("tags", ""))
    is_methodology = meta.get("skill_type", "").casefold() == "methodology" or any(
        tag.casefold() == "methodology" for tag in tags
    )
    candidates_map: dict[str, MethodCandidate] = {}

    def _add_candidate(candidate: MethodCandidate) -> None:
        name = candidate.name.strip()
        if not name:
            return
        split = _split_outside_parens(name, " — ")
        if not split:
            split = _split_outside_parens(name, " - ")
        if split:
            base, extra_desc = split
            if extra_desc and not candidate.description:
                candidate.description = extra_desc.strip()
            name = base.strip()
        name, aliases = _split_name_aliases(name)
        if not name:
            return
        key = name.casefold()
        existing = candidates_map.get(key)
        if existing:
            if candidate.description and not existing.description:
                existing.description = candidate.description
            if candidate.steps and not existing.steps:
                existing.steps = candidate.steps
            existing.aliases = sorted({*existing.aliases, *aliases, *candidate.aliases})
            return
        candidate.name = name
        candidate.aliases = sorted({*aliases, *candidate.aliases})
        candidates_map[key] = candidate

    for idx, line in enumerate(meta_lines):
        if line.strip().startswith("methods:"):
            _, value = line.split(":", 1)
            if value.strip():
                for item in _parse_list_value(value):
                    _add_candidate(MethodCandidate(name=item))
            else:
                for candidate in _parse_methods_block(meta_lines, idx):
                    _add_candidate(candidate)
            break

    if candidates_map or is_methodology:
        lines = body.splitlines()
        in_method_section = False
        method_section_level = None
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            stripped = line.strip()
            if stripped.startswith("#"):
                level = len(stripped.split(" ", 1)[0])
                heading_text = stripped.lstrip("#").strip()
                heading_fold = heading_text.casefold()
                if "method" in heading_fold or "方法" in heading_text or "方法论" in heading_text:
                    in_method_section = True
                    method_section_level = level
                elif method_section_level is not None and level <= method_section_level:
                    in_method_section = False
                    method_section_level = None
            if "|" in stripped:
                table_candidates, new_idx = _parse_method_table(lines, idx)
                if table_candidates:
                    for candidate in table_candidates:
                        _add_candidate(candidate)
                    idx = new_idx + 1
                    continue
            if in_method_section:
                bullet_match = re.match(r"^[-*]\s+\*\*(.+?)\*\*\s*(?:[—:-]\s*(.+))?$", stripped)
                if bullet_match:
                    _add_candidate(MethodCandidate(name=bullet_match.group(1), description=bullet_match.group(2) or ""))
                    idx += 1
                    continue
                heading_match = re.match(r"^#+\s*Method\s*\d*\s*[:：]\s*(.+)$", stripped, flags=re.IGNORECASE)
                if heading_match:
                    _add_candidate(MethodCandidate(name=heading_match.group(1)))
                    idx += 1
                    continue
                inline_match = re.match(r"^Method\s*\d*\s*[:：]\s*(.+)$", stripped, flags=re.IGNORECASE)
                if inline_match:
                    _add_candidate(MethodCandidate(name=inline_match.group(1)))
                    idx += 1
                    continue
            idx += 1
    return list(candidates_map.values())


def _match_candidate_from_text(texts: list[str], candidates: list[MethodCandidate]) -> MethodCandidate | None:
    haystack = " ".join(text for text in texts if text).casefold()
    if not haystack:
        return None
    matches: list[tuple[int, MethodCandidate]] = []
    for candidate in candidates:
        terms = [candidate.name, *candidate.aliases]
        for term in terms:
            if not term:
                continue
            term_fold = term.casefold()
            if term_fold and term_fold in haystack:
                matches.append((len(term_fold), candidate))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]


def _match_candidate_from_methods(methods: list[db.Method], candidates: list[MethodCandidate]) -> MethodCandidate | None:
    if not methods:
        return None
    candidate_map = {candidate.name.casefold(): candidate for candidate in candidates}
    for candidate in candidates:
        for alias in candidate.aliases:
            if alias:
                candidate_map[alias.casefold()] = candidate
    for method in methods:
        if not method.method:
            continue
        match = candidate_map.get(method.method.casefold())
        if match:
            return match
    return None


def _select_skill_method(
    ctx: SkillContext,
    spec: SkillSpec,
    candidates: list[MethodCandidate],
) -> tuple[MethodCandidate | None, str | None, float | None]:
    if not candidates:
        return (None, None, None)
    explicit = _match_candidate_from_text(
        [ctx.raw_topic, ctx.topic, ctx.normalized_query],
        candidates,
    )
    if explicit:
        return (explicit, "user_explicit", 1.0)
    if ctx.llm:
        payload = {
            "skill": spec.display_name or spec.name,
            "query": ctx.topic,
            "raw_query": ctx.raw_topic,
            "normalized_query": ctx.normalized_query,
            "candidates": [
                {
                    "name": candidate.name,
                    "description": candidate.description,
                    "steps": candidate.steps,
                    "aliases": candidate.aliases,
                }
                for candidate in candidates
            ],
            "instructions": [
                "Select the single most appropriate method for the user's task.",
                "Return null if none of the methods apply.",
                "Prefer methods whose steps align with the user's requested workflow.",
                "Respond with JSON only: {\"method\": \"name\" | null, \"confidence\": 0-1, \"rationale\": \"...\"}.",
            ],
        }
        try:
            response = ctx.llm.generate(
                system_prompt="You select the best method from a skill's available methods. Return JSON only.",
                user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
                task="skill method selection",
                agent="Planner",
            )
            selection = _extract_json_payload(response)
        except Exception:
            selection = {}
        method = selection.get("method")
        if isinstance(method, str):
            method_name = method.strip()
            for candidate in candidates:
                if candidate.name.casefold() == method_name.casefold():
                    confidence = selection.get("confidence")
                    try:
                        conf_value = float(confidence)
                    except (TypeError, ValueError):
                        conf_value = None
                    if conf_value is not None and conf_value < 0.35:
                        break
                    return (candidate, "skill_selected", conf_value)
    text = " ".join(item for item in [ctx.topic, ctx.raw_topic, ctx.normalized_query] if item)
    query_tokens = set(tokenize(text))
    best_candidate = None
    best_score = 0
    for candidate in candidates:
        blob = f"{candidate.name} {candidate.description} {' '.join(candidate.steps)}"
        tokens = set(tokenize(blob))
        score = len(query_tokens & tokens)
        if score > best_score:
            best_score = score
            best_candidate = candidate
    if best_candidate and best_score > 0:
        return (best_candidate, "skill_selected", 0.35)
    inferred = _match_candidate_from_methods(ctx.inferred_methods, candidates)
    if inferred:
        return (inferred, "query_inferred", 0.5)
    return (None, None, None)


def _record_method_selection(
    ctx: SkillContext,
    spec: SkillSpec,
    candidate: MethodCandidate,
    source: str,
    confidence: float | None,
) -> None:
    ctx.skill_method_selections[spec.name] = MethodSelection(
        skill_name=spec.name,
        method=candidate.name,
        source=source,
        confidence=confidence,
        steps=candidate.steps,
    )
    notes = f"source={source};skill={spec.display_name or spec.name}"
    if confidence is not None:
        notes = f"{notes};confidence={confidence:.2f}"
    _ensure_method(ctx, candidate.name, notes)


def _auto_select_method_from_skill(ctx: SkillContext, spec: SkillSpec, content: str, texts: list[str]) -> None:
    candidates = _extract_skill_method_candidates(content)
    if not candidates:
        return
    ctx.skill_method_index[spec.name] = candidates
    chosen, source, confidence = _select_skill_method(ctx, spec, candidates)
    if not chosen or not source:
        return
    _record_method_selection(ctx, spec, chosen, source, confidence)


def _progress_line(agent: str, task: str, model: str, current: int, total: int) -> None:
    emit_progress(
        agent,
        model,
        task,
        current,
        total,
        done=current >= total,
    )


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _coerce_date_bound(value: str | None, *, end: bool) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if re.fullmatch(r"\d{4}", text):
        year = int(text)
        if end:
            return datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        return datetime(year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if end:
        return parsed.astimezone(timezone.utc).replace(hour=23, minute=59, second=59)
    return parsed.astimezone(timezone.utc)


def _summarize_evidence_dates(dates: list[datetime]) -> dict[str, object]:
    if not dates:
        return {}
    dates_sorted = sorted(dates)
    now = datetime.now(timezone.utc)
    recent_30 = sum(1 for item in dates if (now - item).days <= 30)
    recent_180 = sum(1 for item in dates if (now - item).days <= 180)
    return {
        "first_seen": dates_sorted[0].isoformat(),
        "last_seen": dates_sorted[-1].isoformat(),
        "recent_30d": recent_30,
        "recent_180d": recent_180,
    }


def _build_contradiction_map(
    store: KGStore,
    claim_ids: set[str],
    claim_sources: dict[str, list[str]],
) -> dict[str, list[dict]]:
    if not claim_ids:
        return {}
    rows = store.get_contradictions_for_claims(claim_ids, limit=200)
    if not rows:
        return {}
    claim_texts: dict[str, str] = {}
    for claim_id in claim_ids:
        text = store.get_claim_text(claim_id)
        if text:
            claim_texts[claim_id] = text
    contra_map: dict[str, list[dict]] = {}
    for row in rows:
        claim_a = row.get("claim_id_a")
        claim_b = row.get("claim_id_b")
        if not claim_a or not claim_b:
            continue
        if claim_a not in claim_ids or claim_b not in claim_ids:
            continue
        confidence = float(row.get("confidence") or 0.0)
        text_a = claim_texts.get(claim_a, "")
        text_b = claim_texts.get(claim_b, "")
        entry_b = {"claim": text_b, "confidence": confidence, "sources": claim_sources.get(claim_b, [])}
        entry_a = {"claim": text_a, "confidence": confidence, "sources": claim_sources.get(claim_a, [])}
        contra_map.setdefault(claim_a, []).append(entry_b)
        contra_map.setdefault(claim_b, []).append(entry_a)
    return contra_map


def _is_stale(value: str | None, days: int) -> bool:
    parsed = _parse_iso_datetime(value)
    if not parsed:
        return False
    delta = datetime.now(timezone.utc) - parsed
    return delta.days >= days


def _should_expand_web(ctx: SkillContext, store: KGStore) -> tuple[bool, list[str]]:
    if ctx.web_forced:
        return True, ["forced"]

    stats = ctx.kg_subgraph_stats or {}
    entity_count = int(stats.get("entity_count", 0) or 0)
    relation_count = int(stats.get("relation_count", 0) or 0)
    claim_count = int(stats.get("claim_count", 0) or 0)
    evidence_total = int(stats.get("evidence_total", 0) or 0)
    evidence_count = int(stats.get("evidence_count", 0) or 0)
    avg_confidence = float(stats.get("avg_confidence", 0.0) or 0.0)
    latest_added_at = stats.get("latest_doc_added_at") or None
    timeframe_specified = bool(stats.get("timeframe_specified"))

    reasons: list[str] = []
    if entity_count == 0 or relation_count == 0:
        reasons.append("kg_empty")
    if claim_count == 0:
        reasons.append("claims_missing")
    if evidence_count < WEB_EXPAND_EVIDENCE_MIN:
        reasons.append("evidence_sparse")
    elif avg_confidence < WEB_EXPAND_CONFIDENCE_MIN:
        reasons.append("evidence_low_conf")
    if evidence_total and evidence_count == 0:
        reasons.append("evidence_out_of_range")
    if (not timeframe_specified) and evidence_count and latest_added_at and _is_stale(latest_added_at, WEB_EXPAND_STALE_DAYS):
        reasons.append("evidence_stale")

    cutoff_iso = None
    if WEB_EXPAND_CONTRADICTION_STALE_DAYS > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=WEB_EXPAND_CONTRADICTION_STALE_DAYS)
        cutoff_iso = cutoff.isoformat()
    contra_count = store.count_contradictions(
        max_confidence=WEB_EXPAND_CONTRADICTION_CONF_MAX,
        older_than=cutoff_iso,
    )
    if contra_count:
        reasons.append(f"contradictions={contra_count}")

    return bool(reasons), reasons


def _extract_json_payload(text: str) -> object:
    if not text:
        return {}
    start_obj = text.find("{")
    end_obj = text.rfind("}")
    start_arr = text.find("[")
    end_arr = text.rfind("]")

    candidate = None
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        candidate = text[start_obj : end_obj + 1]
    elif start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        candidate = text[start_arr : end_arr + 1]
    if not candidate:
        return {}
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {}


def _infer_date_range(label: str | None) -> tuple[str | None, str | None]:
    if not label:
        return (None, None)
    text = label.strip().casefold()
    if not text:
        return (None, None)
    date_range = re.search(
        r"(\d{4}-\d{2}-\d{2})\s*(?:to|through|until|–|-)\s*(\d{4}-\d{2}-\d{2})",
        text,
    )
    if date_range:
        return (date_range.group(1), date_range.group(2))
    year_range = re.search(
        r"(\d{4})\s*(?:to|through|until|–|-)\s*(\d{4})",
        text,
    )
    if year_range:
        return (year_range.group(1), year_range.group(2))
    since_year = re.search(r"(since|from|after)\s+(\d{4})", text)
    if since_year:
        return (since_year.group(2), None)
    until_year = re.search(r"(before|until|through|to)\s+(\d{4})", text)
    if until_year:
        return (None, until_year.group(2))
    last_years = re.search(r"(last|past)\s+(\d+)\s+years?", text)
    if last_years:
        span = int(last_years.group(2))
        if span > 0:
            current_year = datetime.now(timezone.utc).year
            start_year = current_year - span + 1
            return (str(start_year), str(current_year))
    lone_years = re.findall(r"\b(\d{4})\b", text)
    if len(lone_years) == 1:
        return (lone_years[0], lone_years[0])
    return (None, None)


def _filter_methodology_terms(queries: list[str], method_terms: list[str]) -> list[str]:
    if not method_terms:
        return queries
    cleaned = []
    for query in queries:
        text = query
        for term in method_terms:
            if not term:
                continue
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, text, flags=re.IGNORECASE):
                text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip(" ,;:-")
        if not text:
            continue
        if len(text.split()) < 2:
            continue
        cleaned.append(text)
    return cleaned


def _clean_query_list(value: object, method_terms: list[str] | None = None) -> list[str]:
    if isinstance(value, dict):
        value = value.get("queries") or []
    if not isinstance(value, list):
        return []
    cleaned = []
    seen = set()
    for item in value:
        text = str(item).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        cleaned.append(text)
        seen.add(key)
    if method_terms:
        cleaned = _filter_methodology_terms(cleaned, method_terms)
    return cleaned


def expand_queries_with_llm(ctx: SkillContext, query: str, modes: list[str]) -> list[str]:
    if not ctx.llm:
        return []
    model_name = ctx.llm.config.model
    emit_progress("Retriever", model_name, "query expansion", 0, 1, done=False)
    prompt = {
        "query": query,
        "modes": modes,
        "instructions": (
            "Generate 3-6 alternative search queries using synonyms, related terms, and alternate phrasings. "
            "Do not include analysis methodology terms (e.g., benchmarking, SWOT). "
            "Return JSON only as either a list of strings or {\"queries\": [...]}."
        ),
    }
    try:
        response = ctx.llm.generate(
            system_prompt="You generate search query expansions. Return JSON only.",
            user_prompt=json.dumps(prompt, indent=2),
            task="query expansion",
            agent="Retriever",
        )
    finally:
        emit_progress("Retriever", model_name, "query expansion", 1, 1, done=True)
    payload = _extract_json_payload(response)
    return _clean_query_list(payload, ctx.methodology_terms)


def _trim_query(text: str, max_words: int = 8, max_chars: int = 80) -> str:
    words = [word for word in text.split() if word]
    if len(words) > max_words:
        words = words[:max_words]
    trimmed = " ".join(words).strip()
    if len(trimmed) > max_chars:
        trimmed = trimmed[:max_chars].rstrip()
    return trimmed


def _rewrite_gap_queries_with_llm(ctx: SkillContext, gaps: list[str]) -> list[str]:
    if not ctx.llm:
        return []
    model_name = ctx.llm.config.model
    emit_progress("Retriever", model_name, "gap query rewrite", 0, 1, done=False)
    prompt = {
        "gaps": gaps,
        "instructions": (
            "Rewrite each gap into 2-3 short search queries. "
            "Queries should be 2-6 words, no negations, and focused on key entities/relationships. "
            "Avoid analysis methodology terms (e.g., benchmarking, SWOT). "
            "Queries must be in English; translate if needed. "
            "Return JSON only as a flat list of strings."
        ),
    }
    try:
        response = ctx.llm.generate(
            system_prompt="You rewrite research gaps into concise search queries. Return JSON only.",
            user_prompt=json.dumps(prompt, indent=2),
            task="gap query rewrite",
            agent="Retriever",
        )
    finally:
        emit_progress("Retriever", model_name, "gap query rewrite", 1, 1, done=True)
    payload = _extract_json_payload(response)
    return _clean_query_list(payload, ctx.methodology_terms)


def _rewrite_gap_queries_fallback(gaps: list[str], method_terms: list[str] | None = None) -> list[str]:
    cleaned = []
    patterns = [
        r"^the provided documents do not contain any information on\\s+",
        r"^there is no existing literature (connecting|on)\\s+",
        r"^there is no existing literature\\s+",
        r"^there is no\\s+",
        r"^insufficient sources to\\s+",
        r"^no (?:profile signals|sources) available\\s+",
        r"^lack of\\s+",
    ]
    for gap in gaps:
        text = gap.strip()
        if not text:
            continue
        for pattern in patterns:
            if re.match(pattern, text, flags=re.IGNORECASE):
                text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
                break
        text = text.strip(" .:-")
        if not text:
            continue
        cleaned.append(_trim_query(text))
    return _clean_query_list(cleaned, method_terms)


def _rewrite_gap_queries(ctx: SkillContext, gaps: list[str], max_gaps: int = 6) -> list[str]:
    if not gaps:
        return []
    limited = [gap.strip() for gap in gaps if gap.strip()][:max_gaps]
    if not limited:
        return []
    rewritten = _rewrite_gap_queries_with_llm(ctx, limited)
    if rewritten:
        trimmed = [_trim_query(text) for text in rewritten]
        return _clean_query_list(trimmed, ctx.methodology_terms)[: max_gaps * 3]
    return _rewrite_gap_queries_fallback(limited, ctx.methodology_terms)


def _build_inferred_methods(methods: list[str]) -> list[db.Method]:
    created_at = datetime.now(timezone.utc).isoformat()
    inferred = []
    seen = set()
    for method in methods:
        text = (method or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        inferred.append(
            db.Method(
                method_id=str(uuid.uuid4()),
                method=text,
                notes="auto:query_inferred",
                created_at=created_at,
            )
        )
    return inferred


def _apply_query_understanding(ctx: SkillContext, understanding: QueryUnderstanding) -> None:
    ctx.query_understanding = understanding
    ctx.normalized_query = understanding.normalized_query
    if understanding.topic:
        ctx.topic = understanding.topic
    ctx.methodology_terms = build_methodology_terms(understanding.methodologies)
    if ctx.auto_methods:
        ctx.inferred_methods = _build_inferred_methods(understanding.methodologies)


def _build_skill_catalog(registry: SkillRegistry) -> list[dict[str, object]]:
    items = sorted(registry.skills.values(), key=lambda spec: spec.display_name.casefold())
    catalog = []
    for idx, spec in enumerate(items, start=1):
        catalog.append(
            {
                "index": idx,
                "skill_id": spec.name,
                "display_name": spec.display_name,
                "aliases": spec.aliases,
                "description": spec.description,
            }
        )
    return catalog


def _normalize_skill_hints(raw: object, registry: SkillRegistry) -> list[str]:
    if not isinstance(raw, list):
        return []
    ordered = sorted(registry.skills.values(), key=lambda spec: spec.display_name.casefold())
    index_map = {str(idx): spec.name for idx, spec in enumerate(ordered, start=1)}
    alias_map: dict[str, str] = {}
    for spec in registry.skills.values():
        alias_map[spec.name.casefold()] = spec.name
        alias_map[spec.display_name.casefold()] = spec.name
        for alias in spec.aliases:
            alias_map[alias.casefold()] = spec.name
    normalized = []
    seen = set()
    for item in raw:
        text = str(item).strip()
        if not text:
            continue
        if text in index_map:
            name = index_map[text]
        else:
            name = alias_map.get(text.casefold())
        if not name:
            log_task_event(f"Skill hint ignored (unknown): {text}")
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(name)
    return normalized


def _build_allowed_outline_tags(active_skills: list[str]) -> set[str]:
    allowed: set[str] = set()
    for tag in ("關切", "關切證據", "profile", "profile evidence"):
        normalized = normalize_bracket_tag(tag)
        if normalized:
            allowed.add(normalized)
    if not active_skills:
        return allowed
    registry = SkillRegistry()
    registry.load()
    for name in active_skills:
        spec = registry.skills.get(name)
        if not spec:
            continue
        derived_label = name.replace("-", " ").title()
        candidates = [spec.display_name, spec.name, derived_label, *spec.aliases]
        for item in candidates:
            tag = normalize_bracket_tag(str(item))
            if tag:
                allowed.add(tag)
            method_tag = normalize_bracket_tag(f"Methodology Skill: {item}")
            if method_tag:
                allowed.add(method_tag)
    return allowed


def _predict_skills(ctx: SkillContext, registry: SkillRegistry) -> list[str]:
    texts = [ctx.topic, ctx.normalized_query, ctx.raw_topic]
    for method in ctx.methods:
        if method.method:
            texts.append(method.method)
    for method in ctx.inferred_methods:
        if method.method:
            texts.append(method.method)
    predicted = []
    if "systems-engineering" in registry.skills:
        if _matches_triggers(texts, SYSTEMS_ENGINEERING_TRIGGERS):
            predicted.append("systems-engineering")
    return predicted


def _resolve_query_editor() -> list[str]:
    editor = (
        os.getenv("MINI_NEXEN_QUERY_EDITOR")
        or os.getenv("VISUAL")
        or os.getenv("EDITOR")
    )
    if editor:
        return shlex.split(editor)
    return ["code", "--wait"]


def _launch_query_editor(path: Path) -> None:
    command = _resolve_query_editor()
    command = [*command, str(path)]
    subprocess.run(command, check=True)


def _available_web_platforms() -> dict[str, list[str]]:
    platforms = {
        "open": [],
        "forum": [],
        "literature": ["arxiv", "semantic_scholar", "crossref"],
    }
    if os.getenv("BRAVE_SEARCH_API_KEY"):
        platforms["open"].append("brave")
    if os.getenv("TAVILY_API_KEY"):
        platforms["open"].append("tavily")
    if os.getenv("X_API_BEARER_TOKEN"):
        platforms["forum"].append("x")
    if os.getenv("REDDIT_CLIENT_ID") and os.getenv("REDDIT_CLIENT_SECRET") and os.getenv("REDDIT_USER_AGENT"):
        platforms["forum"].append("reddit")
    return platforms


def _render_web_search_artifact(payload: dict[str, object]) -> str:
    return (
        "# Web Search Plan (editable)\n\n"
        "Edit the JSON below if needed. Keep it valid JSON.\n\n"
        "Notes:\n"
        "- `platforms_available` and `platforms_enabled` are informational.\n"
        "- `preferred_sources` is for future use and is not applied yet.\n"
        "- Edits are applied to `search_topics`, `modes`, and `search_modes.semantic_rerank` for this run.\n\n"
        "```json\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}\n"
        "```\n"
    )


def skill_infer_query(ctx: SkillContext) -> SkillContext:
    if not ctx.auto_methods and not ctx.review_query:
        return ctx
    if ctx.query_understanding and ctx.query_artifact_path:
        log_task_event("Query understanding already initialized for this run; skipping.")
        return ctx
    registry = SkillRegistry()
    registry.load()
    raw_query = ctx.raw_topic or ctx.topic
    taxonomy = ctx.methodology_taxonomy or DEFAULT_METHOD_TAXONOMY
    if not ctx.profile_summary:
        store = KGStore()
        summary = store.get_profile_summary(max_signals=ctx.profile_top_k, scope="local")
        if summary:
            ctx.profile_summary = summary
    understanding = infer_query_understanding(
        ctx.llm,
        raw_query,
        taxonomy,
        profile_summary=ctx.profile_summary or None,
    )
    _apply_query_understanding(ctx, understanding)

    ensure_dirs()
    skill_catalog = _build_skill_catalog(registry)
    predicted_skills = _predict_skills(ctx, registry)
    artifact = render_query_artifact(
        understanding,
        raw_query,
        taxonomy,
        skill_catalog=skill_catalog,
        predicted_skills=predicted_skills,
        skill_hints=ctx.skill_hints,
    )
    artifact_path = ARTIFACTS_DIR / datetime.now().strftime("%Y_%m_%d_%H_%M_query.md")
    artifact_path.write_text(artifact, encoding="utf-8")
    ctx.query_artifact_path = artifact_path
    log_task_event(f"Query understanding saved: {artifact_path}")

    if ctx.review_query:
        if ctx.interactive:
            print(f"Query understanding saved to {artifact_path}")
            try:
                _launch_query_editor(artifact_path)
            except FileNotFoundError:
                print("Editor not found. Edit the file manually, then press Enter to continue.")
                input("Press Enter to continue... ")
            except subprocess.CalledProcessError:
                print("Editor exited with a non-zero status. Review the file, then press Enter to continue.")
                input("Press Enter to continue... ")
        else:
            log_task_event("Query review requested but no TTY; continuing without editor.")
        updated_payload = parse_query_artifact(artifact_path.read_text(encoding="utf-8"))
        if updated_payload:
            updated = normalize_query_understanding(updated_payload, raw_query, taxonomy)
            _apply_query_understanding(ctx, updated)
            ctx.skill_hints = _normalize_skill_hints(updated_payload.get("skill_hints"), registry)
            log_task_event("Query understanding updated from reviewed artifact.")
    return ctx


def _build_embedding_config(ctx: SkillContext) -> EmbeddingConfig | None:
    provider = ctx.web_embed_provider
    if not provider and ctx.llm:
        provider = ctx.llm.config.provider
    if not provider:
        return None
    base_url = ctx.web_embed_base_url
    if not base_url and ctx.llm and ctx.llm.config.provider == "lmstudio":
        base_url = ctx.llm.config.base_url
    timeout = ctx.web_embed_timeout or ctx.web_timeout
    api_key = ctx.web_embed_api_key or (ctx.llm.config.api_key if ctx.llm else None)
    return EmbeddingConfig(
        provider=provider,
        model=ctx.web_embed_model,
        base_url=base_url,
        timeout=timeout,
        api_key=api_key,
    )


def _score_web_results(
    query: str,
    results: list[object],
    ctx: SkillContext,
) -> list[tuple[float, object]]:
    if not results:
        return []
    embed_config = _build_embedding_config(ctx) if ctx.web_hybrid else None
    scored: list[tuple[float, object]] = []
    texts = []
    for result in results:
        title = getattr(result, "title", "") or ""
        text = getattr(result, "text", "") or ""
        if len(text) > 2000:
            text = text[:2000]
        texts.append(f"{title}\n{text}".strip())
    if embed_config:
        client = EmbeddingClient(embed_config)
        try:
            embeddings = client.embed_texts([query] + texts)
        except Exception as exc:
            log_task_event(f"Web retrieval scoring: embedding failed, falling back to lexical ({exc})")
            embeddings = []
        if len(embeddings) == len(texts) + 1:
            query_vec = embeddings[0]
            for idx, result in enumerate(results, start=1):
                sim = cosine_similarity(query_vec, embeddings[idx])
                scored.append((sim, result))
            return scored

    query_tokens = tokenize(query)
    doc_texts = [(str(idx), text) for idx, text in enumerate(texts)]
    scores = score_documents(query_tokens, doc_texts)
    for idx, score in scores:
        if idx < len(results):
            scored.append((score, results[idx]))
    return scored


def skill_collect_interests(ctx: SkillContext) -> SkillContext:
    ensure_dirs()
    db.init_db()
    ctx.interests = db.list_interests(limit=20)
    return ctx


def skill_collect_methods(ctx: SkillContext) -> SkillContext:
    ensure_dirs()
    db.init_db()
    ctx.methods = db.list_methods(limit=20)
    if ctx.inferred_methods:
        merged = {method.method.casefold(): method for method in ctx.methods if method.method}
        for method in ctx.inferred_methods:
            key = method.method.casefold()
            if key not in merged:
                merged[key] = method
        ctx.methods = list(merged.values())
    return ctx


def skill_load_profile(ctx: SkillContext) -> SkillContext:
    store = KGStore()
    ctx.extracted_interests = store.get_profile_terms(limit=3)
    if not ctx.profile_summary:
        summary = store.get_profile_summary(max_signals=ctx.profile_top_k, scope="local")
        if summary:
            ctx.profile_summary = summary
        else:
            try:
                signals = build_profile_signals(
                    topic=ctx.topic,
                    interests=ctx.interests,
                    query_hints=ctx.query_hints,
                    llm=ctx.llm,
                    top_k_docs=ctx.top_k,
                    max_signals=ctx.profile_top_k,
                    use_cache=False,
                    cache_result=True,
                )
            except LLMClientError as exc:
                log_task_event(f"Profile summary rebuild failed: {exc}")
                signals = []
            if signals:
                ctx.profile_summary = signals
    return ctx


def skill_retrieve_sources(ctx: SkillContext) -> SkillContext:
    ensure_dirs()
    docs = db.list_documents(limit=200)
    if not docs:
        ctx.documents = []
        return ctx

    query_parts = [ctx.topic]
    for interest in ctx.interests:
        if interest.topic:
            query_parts.append(interest.topic)
    query_parts.extend(ctx.query_hints)
    query = " ".join(part for part in query_parts if part).strip()
    query_tokens = tokenize(query)

    doc_texts = []
    for doc in docs:
        text = db.load_document_text(doc)
        doc_texts.append((doc.doc_id, f"{doc.title}\n{text}"))

    scores = score_documents(query_tokens, doc_texts)
    scores.sort(key=lambda item: item[1], reverse=True)
    selected = []
    for idx, score in scores[: ctx.top_k]:
        if score <= 0:
            continue
        selected.append(docs[idx])
    ctx.documents = selected
    db.mark_documents_used([doc.doc_id for doc in selected])
    log_task_event(
        "Retrieved docs: "
        f"total={len(selected)} "
        f"web={sum(1 for doc in selected if doc.source_type == 'web')} "
        f"file={sum(1 for doc in selected if doc.source_type == 'file')}"
    )
    return ctx


def skill_extract_kg(ctx: SkillContext) -> SkillContext:
    store = KGStore()
    if not ctx.llm:
        ctx.extracted_interests = store.get_profile_terms(limit=3)
        return ctx
    local_docs = []
    for source in ("file", "note", "url"):
        local_docs.extend(db.list_documents_by_source(source, limit=None))
    if not local_docs:
        ctx.extracted_interests = store.get_profile_terms(limit=3)
        return ctx

    extracted = 0
    profiled = 0
    new_doc_ids: list[str] = []
    total_docs = len(local_docs)
    model_name = ctx.llm.config.model if ctx.llm else "unknown"
    if total_docs:
        emit_progress("KGExtractor", model_name, "kg triples", 0, total_docs, done=False)
        emit_progress("Profiler", model_name, "profile extraction", 0, total_docs, done=False)
    for idx, doc in enumerate(local_docs, start=1):
        if store.is_doc_extracted(doc.doc_id):
            _progress_line("KGExtractor", "kg triples", model_name, idx, total_docs)
            emit_progress("Profiler", model_name, "profile extraction", idx, total_docs, done=idx >= total_docs)
            continue
        text = db.load_document_text(doc)
        if not text.strip():
            _progress_line("KGExtractor", "kg triples", model_name, idx, total_docs)
            emit_progress("Profiler", model_name, "profile extraction", idx, total_docs, done=idx >= total_docs)
            continue
        try:
            extracted += extract_and_store(store, ctx.llm, doc.doc_id, text, topic=ctx.topic)
            new_doc_ids.append(doc.doc_id)
        except LLMClientError as exc:
            log_task_event(f"KG extraction skipped doc={doc.doc_id} error={exc}")

        try:
            items = extract_profile_items(ctx.llm, text)
        except LLMClientError as exc:
            log_task_event(f"Profile extraction skipped doc={doc.doc_id} error={exc}")
            items = []
        if items:
            profiled += apply_profile_items(store, doc.doc_id, items)
        _progress_line("KGExtractor", "kg triples", model_name, idx, total_docs)
        emit_progress("Profiler", model_name, "profile extraction", idx, total_docs, done=idx >= total_docs)

    if extracted:
        log_task_event(f"Local KG extraction: added_triples={extracted}")
        ctx.kg_updated = True
    if profiled:
        log_task_event(f"Profile extraction (local): items_added={profiled}")

    if not profiled:
        log_task_event("Profile extraction (local): no items extracted; leaving profile unchanged.")

    if profiled and new_doc_ids:
        update_profile_from_mentions(
            store,
            new_doc_ids,
            limit=10,
        )

    ctx.extracted_interests = store.get_profile_terms(limit=3)
    return ctx


def skill_retrieve_subgraph(ctx: SkillContext) -> SkillContext:
    ctx.kg_fact_cards = []
    store = KGStore()
    query_parts = [ctx.topic]
    for interest in ctx.interests:
        if interest.topic:
            query_parts.append(interest.topic)
    query_parts.extend(ctx.extracted_interests)
    query_parts.extend(ctx.query_hints)
    query = " ".join(part for part in query_parts if part).strip()

    seed_terms = build_seed_terms(
        ctx.topic,
        [interest.topic for interest in ctx.interests if interest.topic] + ctx.extracted_interests,
        ctx.query_hints,
    )
    seed_terms = seed_terms_from_query(query, seed_terms)
    seeds = store.seed_entities_from_terms(seed_terms, limit=12)
    subgraph = store.subgraph(
        [entity.entity_id for entity in seeds],
        hops=max(1, int(ctx.kg_hops or 1)),
        min_confidence=0.3,
    )
    log_subgraph_summary(subgraph)

    timeframe_label = None
    if ctx.query_understanding:
        timeframe_label = ctx.query_understanding.constraints.get("timeframe") or None
    date_from, date_to = _infer_date_range(timeframe_label)
    window_start = _coerce_date_bound(date_from, end=False)
    window_end = _coerce_date_bound(date_to, end=True)
    timeframe_specified = bool(timeframe_label and str(timeframe_label).strip())
    if not window_start and not window_end:
        window_end = datetime.now(timezone.utc)
        window_start = window_end - timedelta(days=WEB_EVIDENCE_DEFAULT_DAYS)
        timeframe_specified = False

    evidence_doc_ids = list(store.evidence_doc_ids([rel.relation_id for rel in subgraph.relations]))
    doc_lookup = {doc.doc_id: doc for doc in db.get_documents_by_ids(evidence_doc_ids)} if evidence_doc_ids else {}

    def _doc_in_window(doc: db.Document) -> bool:
        if not doc:
            return False
        window_value = doc.published_at or doc.added_at
        added = _parse_iso_datetime(window_value)
        if not added:
            return False
        if window_start and added < window_start:
            return False
        if window_end and added > window_end:
            return False
        return True

    evidence_by_relation: dict[str, list[object]] = {}
    in_window_evidence: list[object] = []
    for ev in subgraph.evidence:
        doc = doc_lookup.get(ev.doc_id)
        if not doc or not _doc_in_window(doc):
            continue
        in_window_evidence.append(ev)
        evidence_by_relation.setdefault(ev.relation_id, []).append(ev)

    avg_confidence = 0.0
    if in_window_evidence:
        avg_confidence = sum(ev.confidence for ev in in_window_evidence) / len(in_window_evidence)

    claim_ids: set[str] = set()
    latest_added_at = None
    if in_window_evidence:
        for ev in in_window_evidence:
            doc = doc_lookup.get(ev.doc_id)
            if not doc:
                continue
            parsed = _parse_iso_datetime(doc.added_at)
            if parsed and (latest_added_at is None or parsed > latest_added_at):
                latest_added_at = parsed

    fact_cards: list[dict] = []
    contradiction_rows = []
    if in_window_evidence:
        entity_lookup = {entity.entity_id: entity.name for entity in subgraph.entities}
        claim_sources: dict[str, list[str]] = {}
        for rel in subgraph.relations:
            rel_evidence = evidence_by_relation.get(rel.relation_id, [])
            if not rel_evidence:
                continue
            if rel.claim_id:
                claim_ids.add(rel.claim_id)
            subject = entity_lookup.get(rel.subject_id, rel.subject_id)
            obj = entity_lookup.get(rel.object_id, rel.object_id)
            claim_text = store.get_claim_text(rel.claim_id) if rel.claim_id else None
            statement = claim_text or f"{subject} {rel.predicate} {obj}"
            doc_titles: list[str] = []
            evidence_snippets: list[str] = []
            evidence_conf: list[float] = []
            seen_docs: set[str] = set()
            source_type_counts: dict[str, int] = {}
            dates: list[datetime] = []
            for ev in rel_evidence:
                doc = doc_lookup.get(ev.doc_id)
                if not doc:
                    continue
                evidence_conf.append(float(ev.confidence or 0.0))
                if ev.quote:
                    evidence_snippets.append(ev.quote)
                if doc.doc_id not in seen_docs:
                    seen_docs.add(doc.doc_id)
                    title = doc.title or ""
                    doc_titles.append(title)
                    source_type = (doc.source_type or "").strip().lower() or "unknown"
                    if source_type in {"file", "note", "url"}:
                        bucket = "local"
                    elif source_type == "web":
                        bucket = "web"
                    else:
                        bucket = source_type
                    source_type_counts[bucket] = source_type_counts.get(bucket, 0) + 1
                    if rel.claim_id and title:
                        claim_sources.setdefault(rel.claim_id, [])
                        if title not in claim_sources[rel.claim_id]:
                            claim_sources[rel.claim_id].append(title)
                parsed = _parse_iso_datetime(doc.published_at or doc.added_at)
                if parsed:
                    dates.append(parsed)

            date_stats = _summarize_evidence_dates(dates)
            avg_ev_conf = sum(evidence_conf) / len(evidence_conf) if evidence_conf else 0.0
            scope = "unknown"
            if source_type_counts:
                scope = "mixed"
                if source_type_counts.get("local", 0) and not source_type_counts.get("web", 0):
                    scope = "local_only"
                elif source_type_counts.get("web", 0) and not source_type_counts.get("local", 0):
                    scope = "web_only"
            fact_cards.append(
                {
                    "subject": subject,
                    "predicate": rel.predicate,
                    "object": obj,
                    "claim": claim_text,
                    "statement": statement,
                    "sources": [title for title in doc_titles if title],
                    "source_count": len({title for title in doc_titles if title}),
                    "source_type_counts": source_type_counts,
                    "source_scope": scope,
                    "evidence_snippets": evidence_snippets[:2],
                    "evidence_count": len(rel_evidence),
                    "evidence_confidence_avg": avg_ev_conf,
                    "relation_confidence": rel.confidence,
                    "trend": date_stats,
                    "claim_id": rel.claim_id,
                }
            )

        contradiction_map = _build_contradiction_map(store, claim_ids, claim_sources)
        if contradiction_map:
            for card in fact_cards:
                claim_id = card.get("claim_id")
                if claim_id and claim_id in contradiction_map:
                    card["contradictions"] = contradiction_map[claim_id][:3]
            contradiction_rows = [item for entries in contradiction_map.values() for item in entries]

    fact_cards.sort(
        key=lambda card: (
            int(card.get("source_count") or 0),
            int(card.get("evidence_count") or 0),
            float(card.get("relation_confidence") or 0.0),
        ),
        reverse=True,
    )
    ctx.kg_fact_cards = fact_cards[:60]

    evidence_doc_ids_in_window = {ev.doc_id for ev in in_window_evidence}
    ctx.kg_subgraph_stats = {
        "entity_count": len(subgraph.entities),
        "relation_count": len(subgraph.relations),
        "claim_count": len(claim_ids),
        "evidence_total": len(subgraph.evidence),
        "evidence_count": len(in_window_evidence),
        "fact_card_count": len(ctx.kg_fact_cards),
        "contradiction_count": len(contradiction_rows),
        "avg_confidence": avg_confidence,
        "latest_doc_added_at": latest_added_at.isoformat() if latest_added_at else "",
        "timeframe_from": window_start.isoformat() if window_start else "",
        "timeframe_to": window_end.isoformat() if window_end else "",
        "timeframe_specified": timeframe_specified,
    }

    if evidence_doc_ids_in_window:
        docs = [doc_lookup[doc_id] for doc_id in evidence_doc_ids_in_window if doc_id in doc_lookup]
        docs.sort(key=lambda doc: (doc.published_at or doc.added_at or ""), reverse=True)
        ctx.documents = docs
    else:
        ctx.documents = []

    db.mark_documents_used([doc.doc_id for doc in ctx.documents])
    log_task_event(
        "KG subgraph retrieval: "
        f"docs={len(ctx.documents)} "
        f"web={sum(1 for doc in ctx.documents if doc.source_type == 'web')} "
        f"file={sum(1 for doc in ctx.documents if doc.source_type == 'file')}"
    )
    return ctx


def skill_detect_contradictions(ctx: SkillContext) -> SkillContext:
    if not ctx.kg_updated:
        log_task_event("KG contradiction check skipped: no new KG updates this round.")
        return ctx
    store = KGStore()
    def _progress(current: int, total: int) -> None:
        model_name = ctx.llm.config.model if ctx.llm else "unknown"
        emit_progress(
            "KGContradiction",
            model_name,
            "contradiction check",
            current,
            total,
            done=current >= total,
        )

    updates = detect_contradictions(store, ctx.llm, max_pairs=40, progress=_progress)
    if updates:
        log_task_event(f"Contradictions detected: count={updates}")
    return ctx


def skill_web_retrieve(ctx: SkillContext) -> SkillContext:
    if not ctx.web_enabled:
        return ctx
    if ctx.web_rounds_used >= ctx.web_max_rounds:
        log_task_event(
            f"Web retrieval skipped: max rounds reached ({ctx.web_rounds_used}/{ctx.web_max_rounds})."
        )
        return ctx

    if not ctx.kg_subgraph_stats:
        ctx = skill_retrieve_subgraph(ctx)

    modes = [mode.strip().lower() for mode in (ctx.web_modes or ["open", "lit"])]
    if "tech" in modes:
        modes = ["open" if mode == "tech" else mode for mode in modes]
    if "web" in modes:
        modes = ["open" if mode == "web" else mode for mode in modes]
    if "literature" in modes:
        modes = ["lit" if mode == "literature" else mode for mode in modes]
    interest_queries = []
    seen = set()
    store = KGStore()
    profile_terms = store.get_profile_terms(limit=3)
    seed_topic = ctx.normalized_query or ctx.topic
    for item in (
        [seed_topic]
        + [interest.topic for interest in ctx.interests]
        + profile_terms
        + ctx.query_hints
    ):
        text = (item or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        interest_queries.append(text)
    if ctx.web_search_topics:
        interest_queries = list(ctx.web_search_topics)
    if not interest_queries:
        return ctx

    if ctx.round_number == 1 and not ctx.web_search_topics:
        ensure_dirs()
        platforms_available = _available_web_platforms()
        time_constraint = None
        if ctx.query_understanding:
            time_constraint = ctx.query_understanding.constraints.get("timeframe") or None
        date_from, date_to = _infer_date_range(time_constraint)
        embed_config = _build_embedding_config(ctx)
        semantic_available = False
        if embed_config:
            if embed_config.provider == "gemini" or embed_config.base_url:
                semantic_available = True
        payload = {
            "run_id": ctx.run_id,
            "round": ctx.round_number,
            "search_topics": interest_queries,
            "modes": modes,
            "search_modes": {
                "lexical": True,
                "semantic_rerank": ctx.web_hybrid,
                "semantic_rerank_available": semantic_available,
            },
            "query_expansion": {
                "enabled": ctx.web_expand_queries,
                "max_queries": ctx.web_max_queries,
                "llm_expansion": bool(ctx.llm),
            },
            "date_range": {"from": date_from, "to": date_to, "label": time_constraint},
            "platforms_available": platforms_available,
            "platforms_enabled": {
                "open": "open" in modes,
                "forum": "forum" in modes,
                "literature": "lit" in modes,
            },
            "preferred_sources": [],
        }
        registry = SkillRegistry()
        registry.load()
        raw_query = ctx.raw_topic or ctx.topic
        taxonomy = ctx.methodology_taxonomy or DEFAULT_METHOD_TAXONOMY
        understanding = ctx.query_understanding
        if not understanding:
            understanding = normalize_query_understanding({}, raw_query, taxonomy)
            _apply_query_understanding(ctx, understanding)
        skill_catalog = _build_skill_catalog(registry)
        predicted_skills = _predict_skills(ctx, registry)
        artifact = render_query_artifact(
            understanding,
            raw_query,
            taxonomy,
            skill_catalog=skill_catalog,
            predicted_skills=predicted_skills,
            skill_hints=ctx.skill_hints,
            web_search_payload=payload,
        )
        artifact_path = ctx.query_artifact_path or ARTIFACTS_DIR / datetime.now().strftime("%Y_%m_%d_%H_%M_query.md")
        artifact_path.write_text(artifact, encoding="utf-8")
        ctx.query_artifact_path = artifact_path
        ctx.web_search_artifact_path = artifact_path
        log_task_event(f"Query + web search artifact saved: {artifact_path}")
        if ctx.review_query:
            if ctx.interactive:
                print(f"Query + web search artifact saved to {artifact_path}")
                try:
                    _launch_query_editor(artifact_path)
                except FileNotFoundError:
                    print("Editor not found. Edit the file manually, then press Enter to continue.")
                    input("Press Enter to continue... ")
                except subprocess.CalledProcessError:
                    print("Editor exited with a non-zero status. Review the file, then press Enter to continue.")
                    input("Press Enter to continue... ")
            else:
                log_task_event("Web search plan review requested but no TTY; continuing without editor.")
            updated_text = artifact_path.read_text(encoding="utf-8")
            updated_query_payload = parse_query_artifact(updated_text)
            if updated_query_payload:
                updated = normalize_query_understanding(updated_query_payload, raw_query, taxonomy)
                _apply_query_understanding(ctx, updated)
                ctx.skill_hints = _normalize_skill_hints(updated_query_payload.get("skill_hints"), registry)
                log_task_event("Query understanding updated from reviewed artifact.")
            updated_payload = parse_web_search_artifact(updated_text)
            if isinstance(updated_payload, dict) and updated_payload:
                updated_topics = _clean_query_list(updated_payload.get("search_topics"), ctx.methodology_terms)
                if updated_topics:
                    interest_queries = updated_topics
                updated_modes = updated_payload.get("modes")
                if isinstance(updated_modes, list):
                    cleaned_modes = []
                    for item in updated_modes:
                        text = str(item).strip().lower()
                        if not text:
                            continue
                        if text in {"web", "tech"}:
                            text = "open"
                        if text == "literature":
                            text = "lit"
                        if text not in {"open", "forum", "lit"}:
                            continue
                        if text in cleaned_modes:
                            continue
                        cleaned_modes.append(text)
                    if cleaned_modes:
                        modes = cleaned_modes
                search_modes = updated_payload.get("search_modes")
                if isinstance(search_modes, dict):
                    semantic_flag = search_modes.get("semantic_rerank")
                    if isinstance(semantic_flag, bool):
                        ctx.web_hybrid = semantic_flag
                log_task_event("Web search plan updated from reviewed artifact.")
            else:
                log_task_event("Web search plan review ignored (invalid JSON).")

    if ctx.web_auto:
        should_expand, reasons = _should_expand_web(ctx, store)
        if not should_expand:
            stats = ctx.kg_subgraph_stats or {}
            log_task_event(
                "Web retrieval skipped (auto): "
                f"entities={stats.get('entity_count', 0)} "
                f"relations={stats.get('relation_count', 0)} "
                f"claims={stats.get('claim_count', 0)} "
                f"evidence={stats.get('evidence_count', 0)} "
                f"avg_conf={stats.get('avg_confidence', 0.0):.2f}"
            )
            return ctx
        if reasons == ["forced"]:
            log_task_event("Web retrieval forced by CLI flags.")
        else:
            log_task_event(f"Web retrieval triggered (auto): {', '.join(reasons)}")
    else:
        log_task_event("Web retrieval triggered (default).")

    ctx.web_modes = modes
    if interest_queries:
        ctx.web_search_topics = list(interest_queries)

    log_task_event(
        "Web retrieval: "
        f"modes={','.join(modes) or 'open,lit'} "
        f"max_results={ctx.web_max_results} "
        f"max_queries={ctx.web_max_queries} "
        f"expand={ctx.web_expand_queries} "
        f"per_query={ctx.web_max_per_query} "
        f"max_new={ctx.web_max_new_sources}"
    )

    scored_results: list[tuple[float, object]] = []
    raw_count = 0
    scored_count = 0
    filtered_count = 0
    for query in interest_queries:
        log_task_event_quiet(f"Web retrieval query: {query}")
        extra_queries: list[str] = []
        if ctx.web_expand_queries:
            extra_queries = expand_queries_with_llm(ctx, query, modes)
            expanded = expand_queries(
                query,
                modes,
                max_queries=ctx.web_max_queries,
                extra_queries=extra_queries,
            )
            if expanded:
                log_task_event_quiet(f"Web retrieval expanded queries: {expanded}")
        results = run_web_retrieval(
            query=query,
            modes=modes,
            max_results=ctx.web_max_results,
            timeout=ctx.web_timeout,
            fetch_pages=ctx.web_fetch_pages,
            hybrid=ctx.web_hybrid,
            embed_provider=ctx.web_embed_provider,
            embed_model=ctx.web_embed_model,
            embed_base_url=ctx.web_embed_base_url,
            embed_timeout=ctx.web_embed_timeout,
            embed_api_key=ctx.web_embed_api_key,
            expand_query_flag=ctx.web_expand_queries,
            max_queries=ctx.web_max_queries,
            extra_queries=extra_queries,
        )
        raw_count += len(results)
        scored = _score_web_results(query, results, ctx)
        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            filtered: list[tuple[float, object]] = []
            for score, result in scored:
                if ctx.web_relevance_threshold and ctx.web_hybrid:
                    if score < ctx.web_relevance_threshold:
                        continue
                elif score <= 0:
                    continue
                filtered.append((score, result))
                if len(filtered) >= ctx.web_max_per_query:
                    break
            scored_count += len(scored)
            filtered_count += len(filtered)
            scored_results.extend(filtered)

    ctx.web_rounds_used += 1
    if not scored_results:
        return ctx

    deduped: dict[str, tuple[float, object]] = {}
    for score, result in scored_results:
        url = getattr(result, "url", "") or ""
        if not url:
            continue
        current = deduped.get(url)
        if not current or score > current[0]:
            deduped[url] = (score, result)
    final = sorted(deduped.values(), key=lambda item: item[0], reverse=True)
    final = final[: ctx.web_max_new_sources]

    added = 0
    added_docs: list[db.Document] = []
    skipped_existing = 0
    skipped_content = 0
    skipped_empty = 0
    for score, result in final:
        if not getattr(result, "text", ""):
            skipped_empty += 1
            continue
        tags = ["web", result.source]
        if "lit" in modes or "literature" in modes:
            if result.source in {"arxiv", "semantic_scholar", "crossref"}:
                tags.append("literature")
        if "open" in modes or "web" in modes or "tech" in modes:
            if result.source in {"duckduckgo", "brave", "tavily"}:
                tags.append("open")
        if "forum" in modes:
            if result.source in {"reddit", "x"}:
                tags.append("forum")
        doc, created, dedupe_reason = db.add_document_dedup(
            title=result.title,
            source_type="web",
            source=result.url,
            content_text=result.text,
            tags=tags,
        )
        if not created:
            skipped_existing += 1
            if dedupe_reason == "content_hash":
                skipped_content += 1
            db.update_document_stats(
                doc.doc_id,
                relevance_score=max(0.1, min(1.0, score)),
                last_seen_at=datetime.now(timezone.utc).isoformat(),
                last_seen_run=ctx.run_id,
            )
            continue
        db.update_document_stats(
            doc.doc_id,
            relevance_score=max(0.1, min(1.0, score)),
            last_seen_at=doc.added_at,
            last_seen_run=ctx.run_id,
        )
        added += 1
        added_docs.append(doc)
    log_task_event(
        "Web retrieval summary: "
        f"queries={len(interest_queries)} "
        f"raw={raw_count} "
        f"scored={scored_count} "
        f"accepted={len(final)} "
        f"added={added} "
        f"skipped_existing={skipped_existing} "
        f"skipped_content={skipped_content} "
        f"skipped_empty={skipped_empty} "
        f"filtered={filtered_count}"
    )
    if added_docs:
        extracted = 0
        for doc in added_docs:
            text = db.load_document_text(doc)
            extracted += extract_and_store(store, ctx.llm, doc.doc_id, text, topic=ctx.topic)
        if extracted:
            log_task_event(f"KG extraction (web): added_triples={extracted}")
            ctx.kg_updated = True
    return ctx


def skill_plan_research(ctx: SkillContext) -> SkillContext:
    if not ctx.llm:
        raise ValueError("LLM is required for planning but was not configured.")
    ctx.plan = llm_draft_plan(
        llm=ctx.llm,
        topic=ctx.topic,
        interests=ctx.interests,
        methods=ctx.methods,
        extracted_interests=ctx.extracted_interests,
        documents=ctx.documents,
        round_number=ctx.round_number,
        profile_top_k=ctx.profile_top_k,
        top_k_docs=ctx.top_k,
        kg_fact_cards=ctx.kg_fact_cards,
        output_language=ctx.output_language,
        profile_summary=ctx.profile_summary,
        skill_guidance=ctx.skill_guidance,
        run_id=ctx.run_id,
    )
    return ctx


def skill_refine_plan(ctx: SkillContext) -> SkillContext:
    if not ctx.plan:
        return ctx
    if not ctx.llm:
        raise ValueError("LLM is required for refinement but was not configured.")

    if ctx.llm:
        ready, gaps = review_plan_readiness(ctx.llm, ctx.plan, ctx.output_language)
    else:
        ready, gaps = is_ready(ctx.plan)
    if ready:
        ctx.plan.readiness = "ready"
        return ctx

    ctx.plan.gaps = gaps
    fallback_queries = _rewrite_gap_queries(ctx, gaps)
    ctx.plan = llm_refine_plan(
        llm=ctx.llm,
        plan=ctx.plan,
        documents=ctx.documents,
        interests=ctx.interests,
        methods=ctx.methods,
        extracted_interests=ctx.extracted_interests,
        round_number=ctx.round_number,
        profile_top_k=ctx.profile_top_k,
        top_k_docs=ctx.top_k,
        kg_fact_cards=ctx.kg_fact_cards,
        output_language=ctx.output_language,
        profile_summary=ctx.profile_summary,
        skill_guidance=ctx.skill_guidance,
        run_id=ctx.run_id,
    )
    cleaned_queries = [
        _trim_query(text)
        for text in _clean_query_list(ctx.plan.retrieval_queries, ctx.methodology_terms)
    ]
    ctx.query_hints = cleaned_queries or fallback_queries
    ctx.plan.readiness = "refined"
    return ctx


def skill_build_outline(ctx: SkillContext) -> SkillContext:
    if not ctx.plan:
        return ctx
    if not ctx.llm:
        raise ValueError("LLM is required for outlining but was not configured.")
    allowed_tags = _build_allowed_outline_tags(ctx.active_skills)
    method_steps = {
        name: {
            "method": selection.method,
            "steps": selection.steps,
            "source": selection.source,
        }
        for name, selection in ctx.skill_method_selections.items()
    }
    ctx.outline = llm_build_outline(
        llm=ctx.llm,
        topic=ctx.topic,
        documents=ctx.documents,
        interests=ctx.interests,
        methods=ctx.methods,
        keywords=ctx.plan.keywords,
        kg_fact_cards=ctx.kg_fact_cards,
        output_language=ctx.output_language,
        profile_summary=ctx.profile_summary,
        skill_guidance=ctx.skill_guidance,
        active_skills=ctx.active_skills,
        skill_method_steps=method_steps,
        run_id=ctx.run_id,
        allowed_bracket_tags=allowed_tags,
        profile_review_rounds=ctx.outline_review_rounds,
    )
    return ctx


def skill_persist_plan(ctx: SkillContext) -> SkillContext:
    if not ctx.plan:
        return ctx

    ctx.plan_md = render_plan_md(ctx.plan, ctx.outline, ctx.interests, ctx.methods, llm=ctx.llm)
    return ctx


def build_default_runner() -> SkillRunner:
    registry = SkillRegistry()
    registry.load()

    runner = SkillRunner(registry)

    def skill_systems_engineering(ctx: SkillContext) -> SkillContext:
        spec = registry.skills.get("systems-engineering")
        if not spec:
            return ctx
        if spec.name in ctx.active_skills:
            return ctx
        texts = [ctx.topic, ctx.normalized_query, ctx.raw_topic]
        for method in ctx.methods:
            if method.method:
                texts.append(method.method)
        if not _matches_triggers(texts, SYSTEMS_ENGINEERING_TRIGGERS):
            return ctx
        content = spec.path.read_text(encoding="utf-8")
        ctx.active_skills.append(spec.name)
        ctx.skill_guidance.append(content)
        _auto_select_method_from_skill(ctx, spec, content, texts)
        return ctx
    def skill_apply_skill_hints(ctx: SkillContext) -> SkillContext:
        if not ctx.skill_hints:
            return ctx
        for name in ctx.skill_hints:
            spec = registry.skills.get(name)
            if not spec:
                log_task_event(f"Skill hint not found in registry: {name}")
                continue
            if spec.name in ctx.active_skills:
                continue
            content = spec.path.read_text(encoding="utf-8")
            ctx.active_skills.append(spec.name)
            ctx.skill_guidance.append(content)
            texts = [ctx.topic, ctx.normalized_query, ctx.raw_topic]
            for method in ctx.methods:
                if method.method:
                    texts.append(method.method)
            _auto_select_method_from_skill(ctx, spec, content, texts)
        return ctx
    runner.register("infer_query", skill_infer_query)
    runner.register("collect_interests", skill_collect_interests)
    runner.register("collect_methods", skill_collect_methods)
    runner.register("load_profile", skill_load_profile)
    runner.register("apply_skill_hints", skill_apply_skill_hints)
    runner.register("systems-engineering", skill_systems_engineering)
    runner.register("extract_kg", skill_extract_kg)
    runner.register("web_retrieve", skill_web_retrieve)
    runner.register("detect_contradictions", skill_detect_contradictions)
    runner.register("retrieve_subgraph", skill_retrieve_subgraph)
    runner.register("plan_research", skill_plan_research)
    runner.register("refine_plan", skill_refine_plan)
    runner.register("build_outline", skill_build_outline)
    runner.register("persist_plan", skill_persist_plan)
    return runner
