from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
import shlex
import subprocess
import re
import uuid
from pathlib import Path
from typing import Callable, Optional

from . import db
from .config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_ROUNDS,
    DEFAULT_TOP_K,
    GRAPH_TOP_CLUSTERS,
    GRAPH_ASSIGN_SIMILARITY_MIN,
    GRAPH_AVG_SIMILARITY_THRESHOLD,
    GRAPH_NOISE_RATIO_THRESHOLD,
    GRAPH_REBUILD_RATIO,
    GRAPH_UNASSIGNED_RATIO_THRESHOLD,
    SKILLS_DIR,
    WEB_MAX_NEW_SOURCES,
    WEB_MAX_PER_QUERY,
    WEB_RELEVANCE_THRESHOLD,
    PLANS_DIR,
    ensure_dirs,
)
from .embeddings import EmbeddingClient, EmbeddingConfig, cosine_similarity
from .graph import GraphManager
from .llm import LLMClient, log_task_event, log_task_event_quiet
from .planning import (
    PlanDraft,
    is_ready,
    llm_build_outline,
    llm_draft_plan,
    llm_refine_plan,
    normalize_bracket_tag,
    render_plan_md,
    merge_doc_chunks,
    select_docs_by_cluster_round_robin,
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
class SkillContext:
    topic: str
    raw_topic: str = ""
    normalized_query: str = ""
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
    min_web_docs: int = 0
    max_rounds: int = DEFAULT_ROUNDS
    round_number: int = 1
    interests: list[db.Interest] = field(default_factory=list)
    methods: list[db.Method] = field(default_factory=list)
    extracted_interests: list[str] = field(default_factory=list)
    documents: list[db.Document] = field(default_factory=list)
    doc_text_overrides: dict[str, str] = field(default_factory=dict)
    plan: Optional[PlanDraft] = None
    outline: list[str] = field(default_factory=list)
    plan_md: str = ""
    plan_path: Optional[Path] = None
    notes: list[str] = field(default_factory=list)
    query_hints: list[str] = field(default_factory=list)
    active_skills: list[str] = field(default_factory=list)
    skill_guidance: list[str] = field(default_factory=list)
    skill_hints: list[str] = field(default_factory=list)
    web_enabled: bool = False
    run_id: int = 0
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
    web_max_queries: int = 10
    web_max_new_sources: int = WEB_MAX_NEW_SOURCES
    web_max_per_query: int = WEB_MAX_PER_QUERY
    web_relevance_threshold: float = WEB_RELEVANCE_THRESHOLD
    graph_chunk_size: int = DEFAULT_CHUNK_SIZE
    graph_chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    graph_rebuild_ratio: float = GRAPH_REBUILD_RATIO
    graph_noise_ratio_threshold: float = GRAPH_NOISE_RATIO_THRESHOLD
    graph_avg_similarity_threshold: float = GRAPH_AVG_SIMILARITY_THRESHOLD
    graph_unassigned_ratio_threshold: float = GRAPH_UNASSIGNED_RATIO_THRESHOLD
    graph_assign_similarity_min: float = GRAPH_ASSIGN_SIMILARITY_MIN
    graph_top_clusters: int = GRAPH_TOP_CLUSTERS
    graph_semantic_labels: bool = True
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
    prompt = {
        "query": query,
        "modes": modes,
        "instructions": (
            "Generate 3-6 alternative search queries using synonyms, related terms, and alternate phrasings. "
            "Do not include analysis methodology terms (e.g., benchmarking, SWOT). "
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
    response = ctx.llm.generate(
        system_prompt="You rewrite research gaps into concise search queries. Return JSON only.",
        user_prompt=json.dumps(prompt, indent=2),
        task="gap query rewrite",
        agent="Retriever",
    )
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
        r"^no (?:themes|sources) available\\s+",
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
    if not active_skills:
        return set()
    registry = SkillRegistry()
    registry.load()
    allowed: set[str] = set()
    for name in active_skills:
        spec = registry.skills.get(name)
        if not spec:
            continue
        candidates = [spec.display_name, spec.name, *spec.aliases]
        for item in candidates:
            tag = normalize_bracket_tag(str(item))
            if tag:
                allowed.add(tag)
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
    understanding = infer_query_understanding(ctx.llm, raw_query, taxonomy)
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
    artifact_path = PLANS_DIR / datetime.now().strftime("%Y_%m_%d_%H_%M_query.md")
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


def _ensure_min_web_docs(
    selected: list[db.Document],
    docs: list[db.Document],
    query_tokens: list[str],
    min_web_docs: int,
    target_k: int,
) -> list[db.Document]:
    if min_web_docs <= 0:
        return selected
    web_available = [doc for doc in docs if doc.source_type == "web"]
    if not web_available:
        return selected
    selected_ids = {doc.doc_id for doc in selected}
    selected_web = [doc for doc in selected if doc.source_type == "web"]
    if len(selected_web) >= min_web_docs:
        return selected

    doc_texts = []
    for doc in docs:
        text = db.load_document_text(doc)
        doc_texts.append((doc.doc_id, f"{doc.title}\n{text}"))
    scores = score_documents(query_tokens, doc_texts)
    scores.sort(key=lambda item: item[1], reverse=True)
    score_map = {docs[idx].doc_id: score for idx, score in scores}
    candidates = [
        docs[idx]
        for idx, score in scores
        if score > 0 and docs[idx].source_type == "web" and docs[idx].doc_id not in selected_ids
    ]

    for candidate in candidates:
        if len(selected_web) >= min_web_docs:
            break
        if candidate.doc_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.doc_id)
        selected_web.append(candidate)

    return selected

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

    graph = GraphManager(
        embed_config=_build_embedding_config(ctx),
        llm=ctx.llm if ctx.graph_semantic_labels else None,
        semantic_labels=ctx.graph_semantic_labels,
        chunk_size=ctx.graph_chunk_size,
        chunk_overlap=ctx.graph_chunk_overlap,
        rebuild_ratio=ctx.graph_rebuild_ratio,
        noise_ratio_threshold=ctx.graph_noise_ratio_threshold,
        avg_similarity_threshold=ctx.graph_avg_similarity_threshold,
        unassigned_ratio_threshold=ctx.graph_unassigned_ratio_threshold,
        assign_similarity_min=ctx.graph_assign_similarity_min,
    )
    graph_result = graph.update_graph(docs)
    if graph_result:
        stats = graph_result.stats
        log_task_event(
            "Graph stats: "
            f"chunks_total={stats.total_chunks} "
            f"chunks_new={graph_result.new_chunks} "
            f"chunks_pruned={graph_result.pruned_chunks} "
            f"clusters={stats.cluster_count} "
            f"assigned={stats.assigned_chunks} "
            f"noise_ratio={stats.noise_ratio:.2f} "
            f"avg_sim={stats.avg_similarity:.2f}"
        )
        if graph_result.rebuild_attempted:
            log_task_event(
                f"Graph rebuild: attempted=yes success={'yes' if graph_result.rebuild_succeeded else 'no'}"
            )
            if graph_result.labels_added:
                added = ", ".join(graph_result.labels_added[:8])
                log_task_event(f"Graph rebuild: labels added ({len(graph_result.labels_added)}): {added}")
            if graph_result.labels_removed:
                removed = ", ".join(graph_result.labels_removed[:8])
                log_task_event(f"Graph rebuild: labels removed ({len(graph_result.labels_removed)}): {removed}")

    ctx.extracted_interests = graph.suggest_interests(ctx.topic, limit=3)

    mapped = graph.map_topic_to_cluster(ctx.topic)
    if mapped:
        cluster_id, label, sim = mapped
        db.add_topic_cluster_map(ctx.topic, cluster_id, sim, ctx.run_id)
        log_task_event(
            f"Topic mapped to cluster: label='{label}' similarity={sim:.2f}"
        )

    cluster_scores = graph.score_clusters(query)
    if cluster_scores:
        top_limit = max(1, int(ctx.graph_top_clusters)) if ctx.graph_top_clusters else 3
        selected_ids = [
            cid for sim, cid, _ in cluster_scores[:top_limit] if sim >= 0.2
        ]
        selected = [item for item in cluster_scores if item[1] in selected_ids]
        remaining = [item for item in cluster_scores if item[1] not in selected_ids]
        ordered = selected + remaining
        lines = [
            f"Cluster scores (query='{query}' total={len(cluster_scores)} selected={len(selected_ids)}):"
        ]
        for sim, cid, label in ordered:
            marker = "*"
            note = " [selected]" if cid in selected_ids else ""
            if cid not in selected_ids:
                marker = " "
            label_text = label or "Unlabeled"
            lines.append(f"{marker} {sim:.2f} | {label_text} | {cid[:8]}{note}")
        log_task_event_quiet("\n".join(lines))
    selected_docs: list[db.Document] = []
    if cluster_scores:
        embed_config = _build_embedding_config(ctx)
        selected_doc_ids, _, _ = select_docs_by_cluster_round_robin(
            query,
            embed_config,
            ctx.graph_top_clusters,
            ctx.top_k,
        )
        if selected_doc_ids:
            doc_lookup = {doc.doc_id: doc for doc in docs}
            selected_docs = [doc_lookup[doc_id] for doc_id in selected_doc_ids if doc_id in doc_lookup]

    if selected_docs:
        selected = _ensure_min_web_docs(
            selected_docs,
            docs,
            query_tokens,
            ctx.min_web_docs,
            ctx.top_k,
        )
        ctx.documents = selected
        ctx.doc_text_overrides, _ = merge_doc_chunks([doc.doc_id for doc in selected])
        db.mark_documents_used([doc.doc_id for doc in selected])
        log_task_event(
            "Retrieved docs: "
            f"total={len(selected)} "
            f"web={sum(1 for doc in selected if doc.source_type == 'web')} "
            f"file={sum(1 for doc in selected if doc.source_type == 'file')}"
        )
        return ctx
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
    selected = _ensure_min_web_docs(
        selected,
        docs,
        query_tokens,
        ctx.min_web_docs,
        ctx.top_k,
    )
    ctx.documents = selected
    ctx.doc_text_overrides, _ = merge_doc_chunks([doc.doc_id for doc in selected])
    db.mark_documents_used([doc.doc_id for doc in selected])
    log_task_event(
        "Retrieved docs: "
        f"total={len(selected)} "
        f"web={sum(1 for doc in selected if doc.source_type == 'web')} "
        f"file={sum(1 for doc in selected if doc.source_type == 'file')}"
    )
    return ctx


def skill_web_retrieve(ctx: SkillContext) -> SkillContext:
    if not ctx.web_enabled:
        return ctx

    modes = [mode.strip().lower() for mode in (ctx.web_modes or ["open", "lit"])]
    if "tech" in modes:
        modes = ["open" if mode == "tech" else mode for mode in modes]
    if "web" in modes:
        modes = ["open" if mode == "web" else mode for mode in modes]
    if "literature" in modes:
        modes = ["lit" if mode == "literature" else mode for mode in modes]
    interest_queries = []
    seen = set()
    cluster_interests: list[str] = []
    graph = GraphManager(
        embed_config=_build_embedding_config(ctx),
        llm=ctx.llm if ctx.graph_semantic_labels else None,
        semantic_labels=ctx.graph_semantic_labels,
        chunk_size=ctx.graph_chunk_size,
        chunk_overlap=ctx.graph_chunk_overlap,
        rebuild_ratio=ctx.graph_rebuild_ratio,
        noise_ratio_threshold=ctx.graph_noise_ratio_threshold,
        avg_similarity_threshold=ctx.graph_avg_similarity_threshold,
        unassigned_ratio_threshold=ctx.graph_unassigned_ratio_threshold,
        assign_similarity_min=ctx.graph_assign_similarity_min,
    )
    cluster_interests = graph.suggest_interests(ctx.topic, limit=3)
    seed_topic = ctx.normalized_query or ctx.topic
    for item in (
        [seed_topic]
        + [interest.topic for interest in ctx.interests]
        + cluster_interests
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
    if not interest_queries:
        return ctx

    if ctx.round_number == 1:
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
        artifact = _render_web_search_artifact(payload)
        artifact_path = PLANS_DIR / datetime.now().strftime("%Y_%m_%d_%H_%M_%S_web_search.md")
        artifact_path.write_text(artifact, encoding="utf-8")
        ctx.web_search_artifact_path = artifact_path
        log_task_event(f"Web search plan saved: {artifact_path}")
        if ctx.review_query:
            if ctx.interactive:
                print(f"Web search plan saved to {artifact_path}")
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
            updated_payload = _extract_json_payload(artifact_path.read_text(encoding="utf-8"))
            if isinstance(updated_payload, dict):
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

    ctx.web_modes = modes

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
    skipped_existing = 0
    skipped_empty = 0
    for score, result in final:
        if not getattr(result, "text", ""):
            skipped_empty += 1
            continue
        if db.document_exists(result.url):
            skipped_existing += 1
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
        doc = db.add_document(
            title=result.title,
            source_type="web",
            source=result.url,
            content_text=result.text,
            tags=tags,
        )
        db.update_document_stats(
            doc.doc_id,
            relevance_score=max(0.1, min(1.0, score)),
            last_seen_at=doc.added_at,
            last_seen_run=ctx.run_id,
        )
        added += 1
    log_task_event(
        "Web retrieval summary: "
        f"queries={len(interest_queries)} "
        f"raw={raw_count} "
        f"scored={scored_count} "
        f"accepted={len(final)} "
        f"added={added} "
        f"skipped_existing={skipped_existing} "
        f"skipped_empty={skipped_empty} "
        f"filtered={filtered_count}"
    )
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
        graph_top_clusters=ctx.graph_top_clusters,
        top_k_docs=ctx.top_k,
        doc_text_overrides=ctx.doc_text_overrides,
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
    fallback_queries = _rewrite_gap_queries(ctx, gaps)
    ctx.plan = llm_refine_plan(
        llm=ctx.llm,
        plan=ctx.plan,
        documents=ctx.documents,
        interests=ctx.interests,
        methods=ctx.methods,
        extracted_interests=ctx.extracted_interests,
        round_number=ctx.round_number,
        graph_top_clusters=ctx.graph_top_clusters,
        top_k_docs=ctx.top_k,
        doc_text_overrides=ctx.doc_text_overrides,
        skill_guidance=ctx.skill_guidance,
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
    ctx.outline = llm_build_outline(
        llm=ctx.llm,
        topic=ctx.topic,
        documents=ctx.documents,
        interests=ctx.interests,
        methods=ctx.methods,
        keywords=ctx.plan.keywords,
        doc_text_overrides=ctx.doc_text_overrides,
        skill_guidance=ctx.skill_guidance,
        active_skills=ctx.active_skills,
        run_id=ctx.run_id,
        allowed_bracket_tags=allowed_tags,
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
        return ctx
    runner.register("infer_query", skill_infer_query)
    runner.register("collect_interests", skill_collect_interests)
    runner.register("collect_methods", skill_collect_methods)
    runner.register("apply_skill_hints", skill_apply_skill_hints)
    runner.register("systems-engineering", skill_systems_engineering)
    runner.register("web_retrieve", skill_web_retrieve)
    runner.register("retrieve_sources", skill_retrieve_sources)
    runner.register("plan_research", skill_plan_research)
    runner.register("refine_plan", skill_refine_plan)
    runner.register("build_outline", skill_build_outline)
    runner.register("persist_plan", skill_persist_plan)
    return runner
