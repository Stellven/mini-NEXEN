from __future__ import annotations

import json
from dataclasses import dataclass, field
import re
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
    min_web_docs: int = 0
    max_rounds: int = DEFAULT_ROUNDS
    round_number: int = 1
    interests: list[db.Interest] = field(default_factory=list)
    methods: list[db.Method] = field(default_factory=list)
    extracted_interests: list[str] = field(default_factory=list)
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
    return _clean_query_list(payload)


def _rewrite_gap_queries_fallback(gaps: list[str]) -> list[str]:
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
    return _clean_query_list(cleaned)


def _rewrite_gap_queries(ctx: SkillContext, gaps: list[str], max_gaps: int = 6) -> list[str]:
    if not gaps:
        return []
    limited = [gap.strip() for gap in gaps if gap.strip()][:max_gaps]
    if not limited:
        return []
    rewritten = _rewrite_gap_queries_with_llm(ctx, limited)
    if rewritten:
        trimmed = [_trim_query(text) for text in rewritten]
        return _clean_query_list(trimmed)[: max_gaps * 3]
    return _rewrite_gap_queries_fallback(limited)


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
        if len(selected) < target_k:
            selected.append(candidate)
            selected_ids.add(candidate.doc_id)
            selected_web.append(candidate)
            continue
        non_web = [doc for doc in selected if doc.source_type != "web"]
        if not non_web:
            break
        non_web.sort(key=lambda doc: score_map.get(doc.doc_id, 0.0))
        to_remove = non_web[0]
        selected.remove(to_remove)
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
    graph_docs = graph.search_documents(query, ctx.top_k, top_clusters=ctx.graph_top_clusters)
    if graph_docs:
        selected = _ensure_min_web_docs(
            graph_docs,
            docs,
            query_tokens,
            ctx.min_web_docs,
            ctx.top_k,
        )
        ctx.documents = selected
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

    log_task_event(
        "Web retrieval: "
        f"modes={','.join(ctx.web_modes) or 'tech,lit'} "
        f"max_results={ctx.web_max_results} "
        f"max_queries={ctx.web_max_queries} "
        f"expand={ctx.web_expand_queries} "
        f"per_query={ctx.web_max_per_query} "
        f"max_new={ctx.web_max_new_sources}"
    )
    modes = ctx.web_modes or ["tech", "lit"]
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
    for item in (
        [ctx.topic]
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

    scored_results: list[tuple[float, object]] = []
    raw_count = 0
    scored_count = 0
    filtered_count = 0
    for query in interest_queries:
        log_task_event_quiet(f"Web retrieval query: {query}")
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
        if "tech" in modes or "web" in modes:
            if result.source == "duckduckgo":
                tags.append("tech")
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
        skill_guidance=ctx.skill_guidance,
    )
    cleaned_queries = [_trim_query(text) for text in _clean_query_list(ctx.plan.retrieval_queries)]
    ctx.query_hints = cleaned_queries or fallback_queries
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
        methods=ctx.methods,
        keywords=ctx.plan.keywords,
        skill_guidance=ctx.skill_guidance,
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
        texts = [ctx.topic]
        for method in ctx.methods:
            if method.method:
                texts.append(method.method)
        if not _matches_triggers(texts, SYSTEMS_ENGINEERING_TRIGGERS):
            return ctx
        content = spec.path.read_text(encoding="utf-8")
        ctx.active_skills.append(spec.name)
        ctx.skill_guidance.append(content)
        return ctx
    runner.register("collect_interests", skill_collect_interests)
    runner.register("collect_methods", skill_collect_methods)
    runner.register("systems-engineering", skill_systems_engineering)
    runner.register("web_retrieve", skill_web_retrieve)
    runner.register("retrieve_sources", skill_retrieve_sources)
    runner.register("plan_research", skill_plan_research)
    runner.register("refine_plan", skill_refine_plan)
    runner.register("build_outline", skill_build_outline)
    runner.register("persist_plan", skill_persist_plan)
    return runner
