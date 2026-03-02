from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import db
from .agents import Orchestrator
from .config import (
    ARTIFACTS_DIR,
    DEFAULT_KG_HOPS,
    DEFAULT_OUTLINE_REVIEW_ROUNDS,
    DEFAULT_PROFILE_TOP_K,
    WEB_ARCHIVE_RUNS_UNUSED,
    WEB_ARCHIVE_SCORE_THRESHOLD,
    WEB_DECAY_PER_RUN,
    ensure_dirs,
)
from .llm import (
    LLMClientError,
    build_client,
    emit_progress,
    load_llm_config,
    log_task_event,
)
from .kg import (
    KGStore,
    apply_profile_items,
    extract_and_store,
    extract_profile_items,
    update_profile_from_mentions,
)
from .planning import build_profile_signals, outline_word_count
from .skills_runtime import SkillContext, build_default_runner


@dataclass
class ResearchResult:
    plan_path: Path
    plan_markdown: str
    outline_word_count: int
    query_artifact_path: Path | None = None
    web_search_artifact_path: Path | None = None


@dataclass
class LocalKGResult:
    local_docs: int
    new_docs: int
    triples_added: int
    profile_rebuilt: bool
    profile_items_added: int


def _list_local_documents() -> list[db.Document]:
    sources = ["file", "note", "url"]
    docs: list[db.Document] = []
    seen: set[str] = set()
    for source in sources:
        for doc in db.list_documents_by_source(source, limit=None, include_archived=True):
            if doc.doc_id in seen:
                continue
            seen.add(doc.doc_id)
            docs.append(doc)
    docs.sort(key=lambda doc: (doc.published_at or doc.added_at or ""), reverse=True)
    return docs


def _rebuild_profile_from_local(
    store: KGStore,
    llm: object,
    local_docs: list[db.Document],
) -> int:
    store.clear_profile()
    extracted = 0
    model_name = getattr(llm, "config", None)
    model_label = model_name.model if model_name else "unknown"
    total_docs = len(local_docs)
    if total_docs:
        emit_progress(
            "Profiler",
            model_label,
            "profile extraction",
            0,
            total_docs,
            done=False,
        )
    for idx, doc in enumerate(local_docs, start=1):
        text = db.load_document_text(doc)
        if not text.strip():
            if total_docs:
                emit_progress(
                    "Profiler",
                    model_label,
                    "profile extraction",
                    idx,
                    total_docs,
                    done=idx >= total_docs,
                )
            continue
        try:
            items = extract_profile_items(llm, text)
        except LLMClientError as exc:
            log_task_event(f"Profile extraction skipped doc={doc.doc_id} error={exc}")
            items = []
        if items:
            extracted += apply_profile_items(store, doc.doc_id, items)
        if total_docs:
            emit_progress(
                "Profiler",
                model_label,
                "profile extraction",
                idx,
                total_docs,
                done=idx >= total_docs,
            )

    if extracted == 0:
        log_task_event("Profile extraction (local): no items extracted; leaving profile empty.")
        return 0

    doc_ids = [doc.doc_id for doc in local_docs]
    if doc_ids:
        update_profile_from_mentions(store, doc_ids, limit=10)

    for interest in db.list_interests(limit=50):
        if not interest.topic:
            continue
        entity_id = store.upsert_entity(interest.topic)
        store.set_profile_edge(entity_id, salience=0.9)

    return extracted


def build_local_kg(
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    discover_model: bool | None = None,
    rebuild_profile: bool = True,
    force_profile_rebuild: bool = False,
) -> LocalKGResult:
    ensure_dirs()
    db.init_db()
    llm_config = load_llm_config(
        provider=provider,
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        discover_model=discover_model,
    )
    if not llm_config:
        raise SystemExit("LLM configuration failed. Check provider/model settings.")

    llm_client = build_client(llm_config)
    store = KGStore()
    local_docs = _list_local_documents()
    if not local_docs:
        return LocalKGResult(
            local_docs=0,
            new_docs=0,
            triples_added=0,
            profile_rebuilt=False,
            profile_items_added=0,
        )

    triples_added = 0
    new_doc_ids: list[str] = []
    total_docs = len(local_docs)
    model_label = llm_client.config.model
    if total_docs:
        emit_progress(
            "KGExtractor",
            model_label,
            "kg triples",
            0,
            total_docs,
            done=False,
        )
    for idx, doc in enumerate(local_docs, start=1):
        if store.is_doc_extracted(doc.doc_id):
            if total_docs:
                emit_progress(
                    "KGExtractor",
                    model_label,
                    "kg triples",
                    idx,
                    total_docs,
                    done=idx >= total_docs,
                )
            continue
        text = db.load_document_text(doc)
        if not text.strip():
            if total_docs:
                emit_progress(
                    "KGExtractor",
                    model_label,
                    "kg triples",
                    idx,
                    total_docs,
                    done=idx >= total_docs,
                )
            continue
        try:
            triples_added += extract_and_store(store, llm_client, doc.doc_id, text)
            new_doc_ids.append(doc.doc_id)
        except LLMClientError as exc:
            log_task_event(f"KG extraction skipped doc={doc.doc_id} error={exc}")
        if total_docs:
            emit_progress(
                "KGExtractor",
                model_label,
                "kg triples",
                idx,
                total_docs,
                done=idx >= total_docs,
            )

    profile_rebuilt = False
    profile_items_added = 0
    if force_profile_rebuild or (rebuild_profile and new_doc_ids):
        profile_rebuilt = True
        profile_items_added = _rebuild_profile_from_local(store, llm_client, local_docs)
        try:
            signals = build_profile_signals(
                topic="",
                interests=[],
                llm=llm_client,
                max_signals=DEFAULT_PROFILE_TOP_K,
                use_cache=False,
                cache_result=True,
            )
            if signals:
                log_task_event(f"Profile signals cached: count={len(signals)}")
        except LLMClientError as exc:
            log_task_event(f"Profile signal caching failed: {exc}")

    log_task_event(
        "Local KG build: "
        f"docs={len(local_docs)} new_docs={len(new_doc_ids)} triples_added={triples_added}"
    )
    if profile_rebuilt:
        log_task_event(f"Profile rebuilt from local docs: items_added={profile_items_added}")

    return LocalKGResult(
        local_docs=len(local_docs),
        new_docs=len(new_doc_ids),
        triples_added=triples_added,
        profile_rebuilt=profile_rebuilt,
        profile_items_added=profile_items_added,
    )


def _plan_filename(now: datetime) -> str:
    return now.strftime("%Y_%m_%d_%H_%M_plan.md")


def run_research(
    topic: str,
    rounds: int,
    top_k: int,
    output_language: str = "Chinese",
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    discover_model: bool | None = None,
    web_enabled: bool = True,
    web_forced: bool = False,
    web_auto: bool = False,
    web_modes: list[str] | None = None,
    web_max_results: int = 5,
    web_timeout: int = 15,
    web_fetch_pages: bool = True,
    web_hybrid: bool = False,
    web_embed_provider: str | None = None,
    web_embed_model: str | None = None,
    web_embed_base_url: str | None = None,
    web_embed_timeout: int | None = None,
    web_embed_api_key: str | None = None,
    web_expand_queries: bool = True,
    web_max_queries: int = 10,
    web_max_new_sources: int = 50,
    web_max_per_query: int = 10,
    web_relevance_threshold: float = 0.25,
    web_max_rounds: int = 3,
    auto_methods: bool = True,
    review_query: bool | None = None,
    interactive: bool = False,
    methodology_taxonomy: list[str] | None = None,
    profile_top_k: int = DEFAULT_PROFILE_TOP_K,
    outline_review_rounds: int = DEFAULT_OUTLINE_REVIEW_ROUNDS,
    kg_hops: int = DEFAULT_KG_HOPS,
) -> ResearchResult:
    ensure_dirs()
    db.init_db()
    run_id = db.increment_research_run()
    decay_result = db.decay_web_documents(
        decay_per_run=WEB_DECAY_PER_RUN,
        archive_threshold=WEB_ARCHIVE_SCORE_THRESHOLD,
        archive_runs_unused=WEB_ARCHIVE_RUNS_UNUSED,
    )
    log_task_event(
        "Decay summary: "
        f"processed={decay_result.updated} "
        f"archived={decay_result.archived} "
        f"run_id={run_id}"
    )

    runner = build_default_runner()
    supervisor = Orchestrator(runner)

    llm_config = load_llm_config(
        provider=provider,
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        discover_model=discover_model,
    )
    llm_client = build_client(llm_config)

    ctx = SkillContext(
        topic=topic,
        raw_topic=topic,
        output_language=output_language,
        max_rounds=rounds,
        top_k=top_k,
        run_id=run_id,
        llm=llm_client,
        auto_methods=auto_methods,
        review_query=bool(review_query),
        interactive=interactive,
        methodology_taxonomy=methodology_taxonomy or [],
        web_enabled=web_enabled,
        web_forced=web_forced,
        web_auto=web_auto,
        web_modes=web_modes or [],
        web_max_results=web_max_results,
        web_timeout=web_timeout,
        web_fetch_pages=web_fetch_pages,
        web_hybrid=web_hybrid,
        web_embed_provider=web_embed_provider,
        web_embed_model=web_embed_model,
        web_embed_base_url=web_embed_base_url,
        web_embed_timeout=web_embed_timeout,
        web_embed_api_key=web_embed_api_key,
        web_expand_queries=web_expand_queries,
        web_max_queries=web_max_queries,
        web_max_new_sources=web_max_new_sources,
        web_max_per_query=web_max_per_query,
        web_relevance_threshold=web_relevance_threshold,
        web_max_rounds=web_max_rounds,
        profile_top_k=profile_top_k,
        outline_review_rounds=outline_review_rounds,
        kg_hops=kg_hops,
    )
    ctx = supervisor.run(ctx)

    plan_md = ctx.plan_md
    now = datetime.now()
    plan_path = ARTIFACTS_DIR / _plan_filename(now)
    plan_path.write_text(plan_md, encoding="utf-8")

    return ResearchResult(
        plan_path=plan_path,
        plan_markdown=plan_md,
        outline_word_count=outline_word_count(ctx.outline),
        query_artifact_path=ctx.query_artifact_path,
        web_search_artifact_path=ctx.web_search_artifact_path,
    )
