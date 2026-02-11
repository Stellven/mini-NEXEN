from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import db
from .config import DEFAULT_ROUNDS, DEFAULT_TOP_K, SKILLS_DIR, ensure_dirs
from .llm import LLMClient, log_task_event, log_task_event_quiet
from .planning import (
    PlanDraft,
    is_ready,
    llm_build_outline,
    llm_draft_plan,
    llm_refine_plan,
    render_plan_md,
)
from .text_utils import score_documents, tokenize
from .web_retrieval import expand_queries, run_web_retrieval


@dataclass
class SkillSpec:
    name: str
    description: str
    inputs: list[str]
    outputs: list[str]
    path: Path


@dataclass
class SkillContext:
    topic: str
    top_k: int = DEFAULT_TOP_K
    max_rounds: int = DEFAULT_ROUNDS
    round_number: int = 1
    interests: list[db.Interest] = field(default_factory=list)
    documents: list[db.Document] = field(default_factory=list)
    plan: Optional[PlanDraft] = None
    outline: list[str] = field(default_factory=list)
    plan_md: str = ""
    plan_path: Optional[Path] = None
    notes: list[str] = field(default_factory=list)
    query_hints: list[str] = field(default_factory=list)
    active_skills: list[str] = field(default_factory=list)
    skill_guidance: list[str] = field(default_factory=list)
    web_enabled: bool = False
    web_modes: list[str] = field(default_factory=list)
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
    web_max_queries: int = 4
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
        inputs = [item.strip() for item in meta.get("inputs", "").split(",") if item.strip()]
        outputs = [item.strip() for item in meta.get("outputs", "").split(",") if item.strip()]
        return SkillSpec(name=name, description=description, inputs=inputs, outputs=outputs, path=path)


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


def _clean_query_list(value: object) -> list[str]:
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
    return cleaned


def _expand_queries_with_llm(ctx: SkillContext, query: str, modes: list[str]) -> list[str]:
    if not ctx.llm:
        return []
    prompt = {
        "query": query,
        "modes": modes,
        "instructions": (
            "Generate 3-6 alternative search queries using synonyms, related terms, and alternate phrasings. "
            "Return JSON only as either a list of strings or {\"queries\": [...]}."
        ),
    }
    response = ctx.llm.generate(
        system_prompt="You generate search query expansions. Return JSON only.",
        user_prompt=json.dumps(prompt, indent=2),
        task="query expansion",
        agent="Retriever",
    )
    payload = _extract_json_payload(response)
    return _clean_query_list(payload)

def skill_collect_interests(ctx: SkillContext) -> SkillContext:
    ensure_dirs()
    db.init_db()
    ctx.interests = db.list_interests(limit=20)
    return ctx


def skill_retrieve_sources(ctx: SkillContext) -> SkillContext:
    ensure_dirs()
    docs = db.list_documents(limit=200)
    if not docs:
        ctx.documents = []
        return ctx

    query_tokens = tokenize(ctx.topic)
    for interest in ctx.interests:
        query_tokens.extend(tokenize(interest.topic))
        if interest.notes:
            query_tokens.extend(tokenize(interest.notes))
    for hint in ctx.query_hints:
        query_tokens.extend(tokenize(hint))

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
    return ctx


def skill_web_retrieve(ctx: SkillContext) -> SkillContext:
    if not ctx.web_enabled:
        return ctx

    seen = set()
    query_parts = []
    for item in [ctx.topic] + [interest.topic for interest in ctx.interests]:
        text = (item or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        query_parts.append(text)
    query = " ".join(query_parts).strip()
    if not query:
        return ctx

    log_task_event(
        "Web retrieval: "
        f"modes={','.join(ctx.web_modes) or 'tech,lit'} "
        f"max_results={ctx.web_max_results} "
        f"max_queries={ctx.web_max_queries} "
        f"expand={ctx.web_expand_queries}"
    )
    log_task_event_quiet(f"Web retrieval query: {query}")
    modes = ctx.web_modes or ["tech", "lit"]
    extra_queries: list[str] = []
    if ctx.web_expand_queries:
        extra_queries = _expand_queries_with_llm(ctx, query, modes)
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

    added = 0
    for result in results:
        if not result.text:
            continue
        if db.document_exists(result.url):
            continue
        tags = ["web", result.source]
        if "lit" in modes or "literature" in modes:
            if result.source in {"arxiv", "semantic_scholar", "crossref"}:
                tags.append("literature")
        if "tech" in modes or "web" in modes:
            if result.source == "duckduckgo":
                tags.append("tech")
        db.add_document(
            title=result.title,
            source_type="web",
            source=result.url,
            content_text=result.text,
            tags=tags,
        )
        added += 1
    log_task_event(f"Web retrieval: added {added} sources")
    return ctx


def skill_plan_research(ctx: SkillContext) -> SkillContext:
    if not ctx.llm:
        raise ValueError("LLM is required for planning but was not configured.")
    ctx.plan = llm_draft_plan(
        llm=ctx.llm,
        topic=ctx.topic,
        interests=ctx.interests,
        documents=ctx.documents,
        round_number=ctx.round_number,
        skill_guidance=ctx.skill_guidance,
    )
    return ctx


def skill_refine_plan(ctx: SkillContext) -> SkillContext:
    if not ctx.plan:
        return ctx
    if not ctx.llm:
        raise ValueError("LLM is required for refinement but was not configured.")

    ready, gaps = is_ready(ctx.plan)
    if ready:
        ctx.plan.readiness = "ready"
        return ctx

    ctx.plan.gaps = gaps
    ctx.query_hints = gaps
    ctx.plan = llm_refine_plan(
        llm=ctx.llm,
        plan=ctx.plan,
        documents=ctx.documents,
        interests=ctx.interests,
        round_number=ctx.round_number,
        skill_guidance=ctx.skill_guidance,
    )
    ctx.plan.readiness = "refined"
    return ctx


def skill_build_outline(ctx: SkillContext) -> SkillContext:
    if not ctx.plan:
        return ctx
    if not ctx.llm:
        raise ValueError("LLM is required for outlining but was not configured.")
    ctx.outline = llm_build_outline(
        llm=ctx.llm,
        topic=ctx.topic,
        documents=ctx.documents,
        interests=ctx.interests,
        keywords=ctx.plan.keywords,
        skill_guidance=ctx.skill_guidance,
    )
    return ctx


def skill_persist_plan(ctx: SkillContext) -> SkillContext:
    if not ctx.plan:
        return ctx

    ctx.plan_md = render_plan_md(ctx.plan, ctx.outline, ctx.interests)
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
        texts = [ctx.topic]
        for interest in ctx.interests:
            texts.append(interest.topic)
            if interest.notes:
                texts.append(interest.notes)
        if not _matches_triggers(texts, SYSTEMS_ENGINEERING_TRIGGERS):
            return ctx
        content = spec.path.read_text(encoding="utf-8")
        ctx.active_skills.append(spec.name)
        ctx.skill_guidance.append(content)
        return ctx
    runner.register("collect_interests", skill_collect_interests)
    runner.register("systems-engineering", skill_systems_engineering)
    runner.register("web_retrieve", skill_web_retrieve)
    runner.register("retrieve_sources", skill_retrieve_sources)
    runner.register("plan_research", skill_plan_research)
    runner.register("refine_plan", skill_refine_plan)
    runner.register("build_outline", skill_build_outline)
    runner.register("persist_plan", skill_persist_plan)
    return runner
