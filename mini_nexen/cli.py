from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
import webbrowser

from . import db
import os
import sys

from .config import (
    ARTIFACTS_DIR,
    DEFAULT_KG_HOPS,
    DEFAULT_ROUNDS,
    DEFAULT_PROFILE_TOP_K,
    DEFAULT_TOP_K,
    WEB_AUTO_MAX_ROUNDS,
    WEB_RELEVANCE_THRESHOLD,
    ensure_dirs,
)
from .file_ingest import load_text_from_file
from .llm import LLMClientError, load_llm_config, log_task_event, set_log_echo
from .kg import KGStore, build_full_subgraph, build_subgraph_for_terms, render_dot, render_html
from .web_retrieval import RetrievalRateLimitError
from .research import run_research, build_local_kg
from .seeds import ingest_seed_pack


def _ingest(args: argparse.Namespace) -> None:
    ensure_dirs()
    db.init_db()

    tags = [tag.strip() for tag in (args.tags or "").split(",") if tag.strip()]
    added_docs = []
    skipped = 0

    if args.file and args.url:
        raise SystemExit("Provide either --file or --url (not both).")
    if args.file and args.text and not args.url:
        raise SystemExit("Provide either --file or --text (not both).")

    seed_result = ingest_seed_pack()

    def _has_source(sources: list[str]) -> bool:
        for source in sources:
            if source and db.document_exists(source):
                return True
        return False

    if args.file:
        for raw_path in args.file:
            if not raw_path:
                continue
            file_path = Path(raw_path)
            if not file_path.exists():
                raise SystemExit(f"File not found: {file_path}")
            resolved = str(file_path.resolve())
            if _has_source([resolved, str(file_path)]):
                skipped += 1
                continue
            content = load_text_from_file(file_path)
            title = args.title or file_path.name
            doc, created, _reason = db.add_document_dedup(
                title=title,
                source_type="file",
                source=resolved,
                content_text=content,
                tags=tags,
            )
            if created:
                added_docs.append(doc)
            else:
                skipped += 1

    elif args.url:
        title = args.title or args.url
        content = args.text or ""
        if _has_source([args.url]):
            skipped += 1
        else:
            doc, created, _reason = db.add_document_dedup(
                title=title,
                source_type="url",
                source=args.url,
                content_text=content,
                tags=tags,
            )
            if created:
                added_docs.append(doc)
            else:
                skipped += 1
        if not content:
            print("Note: URL content was not fetched; provide --text to store document content.")

    elif args.text:
        title = args.title or "Personal note"
        content = args.text
        if not content.strip():
            raise SystemExit("Provide non-empty --text for a note.")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        source = f"note:{digest}"
        if _has_source([source]):
            skipped += 1
        else:
            doc, created, _reason = db.add_document_dedup(
                title=title,
                source_type="note",
                source=source,
                content_text=content,
                tags=tags,
            )
            if created:
                added_docs.append(doc)
            else:
                skipped += 1

    elif not args.file and not args.url and not args.text:
        if seed_result.files == 0:
            print("No local files found in data/local files and no --file/--url/--text provided.")
        if seed_result.added == 0:
            print("Local file ingest found no new files to add.")
    else:
        raise SystemExit("Provide --file, --url, or --text to ingest.")

    if added_docs:
        print(f"Ingested documents: {len(added_docs)}")
    if skipped:
        print(f"Skipped duplicates: {skipped}")
    if seed_result.files:
        print(
            "Local file ingest summary: "
            f"files={seed_result.files} added={seed_result.added} skipped={seed_result.skipped}"
        )
    def _kg_rebuild_state() -> tuple[int, int, bool]:
        store = KGStore()
        local_docs: list[db.Document] = []
        for source in ("file", "note", "url"):
            local_docs.extend(db.list_documents_by_source(source, limit=None))
        if not local_docs:
            return 0, 0, True
        missing = sum(1 for doc in local_docs if not store.is_doc_extracted(doc.doc_id))
        profile_empty = not bool(store.get_profile(limit=1))
        return len(local_docs), missing, profile_empty

    new_docs_added = bool(added_docs) or seed_result.added > 0
    local_count, missing_extracts, profile_empty = _kg_rebuild_state()
    needs_rebuild = new_docs_added or missing_extracts > 0 or profile_empty

    if local_count == 0 and not new_docs_added:
        print("No local docs available to build a KG. Skipping rebuild.")
        return
    if not needs_rebuild:
        print("No new docs and KG is up-to-date. Skipping rebuild.")
        return
    if not new_docs_added:
        print(
            "No new docs, but KG needs rebuild: "
            f"local_docs={local_count} missing_extractions={missing_extracts} profile_empty={profile_empty}"
        )

    provider, model = _resolve_llm_choice(args)
    try:
        result = build_local_kg(
            provider=provider,
            model=model,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            discover_model=not args.no_model_discovery,
            rebuild_profile=True,
            force_profile_rebuild=profile_empty and missing_extracts == 0 and not new_docs_added,
        )
    except LLMClientError as exc:
        print(f"LLM error: {exc}")
        raise SystemExit(1) from exc

    print(
        "Local KG build complete. "
        f"local_docs={result.local_docs} "
        f"new_docs={result.new_docs} "
        f"triples_added={result.triples_added}"
    )
    if result.profile_rebuilt:
        print(f"Profile rebuilt from local docs. items_added={result.profile_items_added}")
    else:
        print("Profile rebuild skipped (no local changes).")


def _add_interest(args: argparse.Namespace) -> None:
    ensure_dirs()
    db.init_db()
    topic = (args.topic or args.text or "").strip()
    if not topic:
        raise SystemExit("Provide interest text via --topic or as a positional argument.")
    interest = db.add_interest(topic=topic, notes="")
    print(f"Recorded interest: {interest.interest_id}")


def _add_method(args: argparse.Namespace) -> None:
    ensure_dirs()
    db.init_db()
    method = (args.method or args.text or "").strip()
    if not method:
        raise SystemExit("Provide method text via --method or as a positional argument.")
    recorded = db.add_method(method=method, notes="")
    print(f"Recorded method: {recorded.method_id}")


def _delete_interest(args: argparse.Namespace) -> None:
    ensure_dirs()
    db.init_db()
    deleted = db.delete_interest(args.id)
    if deleted:
        print(f"Deleted interest: {args.id}")
    else:
        print(f"No interest found for id: {args.id}")


def _delete_method(args: argparse.Namespace) -> None:
    ensure_dirs()
    db.init_db()
    deleted = db.delete_method(args.id)
    if deleted:
        print(f"Deleted method: {args.id}")
    else:
        print(f"No method found for id: {args.id}")


def _clear_interests(args: argparse.Namespace) -> None:
    ensure_dirs()
    db.init_db()
    deleted = db.clear_interests()
    print(f"Cleared interests: {deleted}")


def _clear_methods(args: argparse.Namespace) -> None:
    ensure_dirs()
    db.init_db()
    deleted = db.clear_methods()
    print(f"Cleared methods: {deleted}")


def _clear_profile(args: argparse.Namespace) -> None:
    ensure_dirs()
    db.init_db()
    store = KGStore()
    deleted = store.clear_profile()
    print(f"Cleared profile entries: {deleted}")


def _clear_library(args: argparse.Namespace) -> None:
    ensure_dirs()
    db.init_db()
    result = db.clear_library_and_graph(clear_files=True)
    print(
        "Cleared library: "
        f"documents={result['documents']} "
        f"document_stats={result['document_stats']} "
        f"files_removed={result['files_removed']}"
    )
    print(
        "Cleared KG: "
        f"graph_meta={result['graph_meta']} "
        f"kg_entities={result['kg_entities']} "
        f"kg_relations={result['kg_relations']} "
        f"kg_claims={result['kg_claims']} "
        f"kg_evidence={result['kg_evidence']} "
        f"kg_mentions={result['kg_mentions']} "
        f"kg_profiles={result['kg_profiles']} "
        f"kg_contradictions={result['kg_contradictions']} "
        f"kg_doc_state={result['kg_doc_state']}"
    )


def _list_docs(_: argparse.Namespace) -> None:
    ensure_dirs()
    docs = db.list_documents(limit=50)
    if not docs:
        print("No documents in library yet.")
        return
    for doc in docs:
        print(f"{doc.doc_id} | {doc.source_type} | {doc.title} | {doc.source}")


def _list_interests(_: argparse.Namespace) -> None:
    ensure_dirs()
    interests = db.list_interests(limit=50)
    if not interests:
        print("No interests recorded yet.")
        return
    for interest in interests:
        print(f"{interest.interest_id} | {interest.topic}")


def _list_methods(_: argparse.Namespace) -> None:
    ensure_dirs()
    methods = db.list_methods(limit=50)
    if not methods:
        print("No methods recorded yet.")
        return
    for method in methods:
        print(f"{method.method_id} | {method.method}")


def _kg_report(args: argparse.Namespace) -> None:
    ensure_dirs()
    db.init_db()
    store = KGStore()
    limit = max(1, int(args.limit or 10))
    with db._connect() as conn:
        counts = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM kg_entities) AS entities,
                (SELECT COUNT(*) FROM kg_relations) AS relations,
                (SELECT COUNT(*) FROM kg_claims) AS claims,
                (SELECT COUNT(*) FROM kg_evidence) AS evidence,
                (SELECT COUNT(*) FROM kg_mentions) AS mentions,
                (SELECT COUNT(*) FROM kg_user_profile) AS profiles,
                (SELECT COUNT(*) FROM kg_contradictions) AS contradictions
            """
        ).fetchone()
    print("KG Summary:")
    print(
        "entities={entities} relations={relations} claims={claims} "
        "evidence={evidence} mentions={mentions} profiles={profiles} contradictions={contradictions}".format(
            entities=counts["entities"],
            relations=counts["relations"],
            claims=counts["claims"],
            evidence=counts["evidence"],
            mentions=counts["mentions"],
            profiles=counts["profiles"],
            contradictions=counts["contradictions"],
        )
    )

    profile = store.get_profile(limit=limit)
    if profile:
        print("")
        print("Top Profile Signals:")
        for item in profile:
            label = item.get("entity") or ""
            salience = item.get("salience") or 0.0
            print(f"{salience:.2f} | {label}")

    with db._connect() as conn:
        rows = conn.execute(
            """
            SELECT e.name, COUNT(*) AS count
            FROM kg_relations r
            JOIN kg_entities e ON r.subject_id = e.entity_id
            GROUP BY e.name
            ORDER BY count DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    if rows:
        print("")
        print("Top Subject Entities:")
        for row in rows:
            print(f"{row['count']:>3} | {row['name']}")


def _kg_entity_edges(args: argparse.Namespace) -> None:
    ensure_dirs()
    db.init_db()
    store = KGStore()
    min_conf = float(args.min_conf or 0.0)
    entity_id = (args.id or "").strip()
    entity_name = ""
    entity_type = ""
    if entity_id:
        with db._connect() as conn:
            row = conn.execute(
                """
                SELECT entity_id, name, type
                FROM kg_entities
                WHERE user_id = ? AND entity_id = ?
                """,
                (store.user_id, entity_id),
            ).fetchone()
        if not row:
            print(f"Entity id not found: {entity_id}")
            return
        entity_name = row["name"]
        entity_type = row["type"]
    else:
        term = (args.entity or "").strip()
        if not term:
            raise SystemExit("Provide --entity or --id.")
        matches = store.search_entities(term, limit=max(1, int(args.limit or 10)))
        exact = [m for m in matches if m.name.casefold() == term.casefold()]
        if len(exact) == 1:
            chosen = exact[0]
        elif len(matches) == 1:
            chosen = matches[0]
        elif not matches:
            print(f"No entities match: {term}")
            return
        else:
            print("Multiple matches found. Re-run with --id to pick one:")
            for m in matches:
                print(f"{m.entity_id} | {m.name} | {m.type}")
            return
        entity_id = chosen.entity_id
        entity_name = chosen.name
        entity_type = chosen.type

    with db._connect() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN subject_id = ? THEN 1 ELSE 0 END) AS outgoing,
                SUM(CASE WHEN object_id = ? THEN 1 ELSE 0 END) AS incoming
            FROM kg_relations
            WHERE user_id = ?
              AND confidence >= ?
              AND (subject_id = ? OR object_id = ?)
            """,
            (entity_id, entity_id, store.user_id, min_conf, entity_id, entity_id),
        ).fetchone()
        neighbor_rows = conn.execute(
            """
            SELECT DISTINCT
                CASE WHEN subject_id = ? THEN object_id ELSE subject_id END AS neighbor_id
            FROM kg_relations
            WHERE user_id = ?
              AND confidence >= ?
              AND (subject_id = ? OR object_id = ?)
            """,
            (entity_id, store.user_id, min_conf, entity_id, entity_id),
        ).fetchall()

    total = int(row["total"] or 0)
    outgoing = int(row["outgoing"] or 0)
    incoming = int(row["incoming"] or 0)
    neighbors = len(neighbor_rows)

    print(f"Entity: {entity_name} ({entity_type})")
    print(f"Entity ID: {entity_id}")
    print(f"Edges: {total} (outgoing={outgoing}, incoming={incoming}, neighbors={neighbors})")
    if args.show_neighbors and total:
        with db._connect() as conn:
            rows = conn.execute(
                """
                SELECT r.subject_id, r.predicate, r.object_id, r.confidence,
                       es.name AS subject_name, eo.name AS object_name
                FROM kg_relations r
                JOIN kg_entities es ON r.subject_id = es.entity_id
                JOIN kg_entities eo ON r.object_id = eo.entity_id
                WHERE r.user_id = ?
                  AND r.confidence >= ?
                  AND (r.subject_id = ? OR r.object_id = ?)
                ORDER BY r.confidence DESC
                """,
                (store.user_id, min_conf, entity_id, entity_id),
            ).fetchall()
        print("")
        print("Connected Entities:")
        for row in rows:
            direction = "outgoing" if row["subject_id"] == entity_id else "incoming"
            neighbor = row["object_name"] if direction == "outgoing" else row["subject_name"]
            predicate = row["predicate"] or ""
            confidence = float(row["confidence"] or 0.0)
            print(f"{direction} | {predicate} | {neighbor} | c={confidence:.2f}")


def _kg_export_dot(args: argparse.Namespace) -> None:
    ensure_dirs()
    db.init_db()
    store = KGStore()
    min_conf = float(args.min_conf or 0.3)
    limit_edges = max(50, int(args.limit_edges or 200))
    if args.all:
        subgraph = build_full_subgraph(
            store,
            min_confidence=min_conf,
            limit_edges=limit_edges,
        )
    else:
        seeds = _parse_seed_terms(args.seed)
        if not seeds:
            seeds = _default_seed_terms(store, limit=8)
        subgraph = build_subgraph_for_terms(
            store,
            seeds,
            hops=max(1, int(args.hops or 1)),
            min_confidence=min_conf,
            limit_edges=limit_edges,
        )
    profile_edges = store.get_profile(limit=20)
    dot = render_dot(subgraph, user_id=store.user_id, profile_edges=profile_edges)
    out_path = _resolve_export_path(args.out, suffix=".dot")
    out_path.write_text(dot, encoding="utf-8")
    print(f"Wrote DOT: {out_path}")
    if args.open_dot:
        svg_path = out_path.with_suffix(".svg")
        if not shutil.which("dot"):
            print("Graphviz not found (missing 'dot' binary). Install graphviz to auto-render.")
            return
        try:
            subprocess.run(
                ["dot", "-Tsvg", str(out_path), "-o", str(svg_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            print(f"Unable to render DOT to SVG: {detail}")
            return
        print(f"Wrote SVG: {svg_path}")
        try:
            opened = webbrowser.open(svg_path.as_uri())
        except Exception as exc:
            print(f"Unable to open SVG in browser: {exc}")
        else:
            if not opened:
                print("Unable to open SVG in browser (no handler available).")


def _kg_export_html(args: argparse.Namespace) -> None:
    ensure_dirs()
    db.init_db()
    store = KGStore()
    min_conf = float(args.min_conf or 0.3)
    limit_edges = max(50, int(args.limit_edges or 200))
    if args.all:
        subgraph = build_full_subgraph(
            store,
            min_confidence=min_conf,
            limit_edges=limit_edges,
        )
    else:
        seeds = _parse_seed_terms(args.seed)
        if not seeds:
            seeds = _default_seed_terms(store, limit=8)
        subgraph = build_subgraph_for_terms(
            store,
            seeds,
            hops=max(1, int(args.hops or 1)),
            min_confidence=min_conf,
            limit_edges=limit_edges,
        )
    profile_edges = store.get_profile(limit=20)
    html = render_html(subgraph, title="mini-NEXEN KG", user_id=store.user_id, profile_edges=profile_edges)
    out_path = _resolve_export_path(args.out, suffix=".html")
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote HTML: {out_path}")
    if args.open_html:
        try:
            opened = webbrowser.open(out_path.as_uri())
        except Exception as exc:
            print(f"Unable to open HTML in browser: {exc}")
        else:
            if not opened:
                print("Unable to open HTML in browser (no handler available).")


def _parse_seed_terms(items: list[str] | None) -> list[str]:
    if not items:
        return []
    terms: list[str] = []
    for item in items:
        for part in (item or "").split(","):
            text = part.strip()
            if text:
                terms.append(text)
    seen = set()
    unique = []
    for term in terms:
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(term)
    return unique


def _default_seed_terms(store: KGStore, limit: int = 8) -> list[str]:
    profile = store.get_profile(limit=limit)
    if profile:
        return [item["entity"] for item in profile if item.get("entity")][:limit]
    with db._connect() as conn:
        rows = conn.execute(
            """
            SELECT e.name, COUNT(*) AS count
            FROM kg_relations r
            JOIN kg_entities e ON r.subject_id = e.entity_id
            GROUP BY e.name
            ORDER BY count DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [row["name"] for row in rows if row["name"]]


def _resolve_export_path(raw: str | None, suffix: str) -> Path:
    ensure_dirs()
    if raw:
        path = Path(raw)
        return path if path.suffix else path.with_suffix(suffix)
    stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return (ARTIFACTS_DIR / f"kg_export_{stamp}{suffix}").resolve()

def _research(args: argparse.Namespace) -> None:
    ensure_dirs()
    provider, model = _resolve_llm_choice(args)
    llm_config = load_llm_config(
        provider=provider,
        model=model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        discover_model=True,
    )
    if not llm_config:
        raise SystemExit("LLM configuration failed. Check provider/model settings.")

    web_modes = []
    explicit_web = bool(args.web or args.web_open or args.web_forum or args.web_lit)
    if not args.no_web:
        if explicit_web:
            if args.web or args.web_open:
                web_modes.append("open")
            if args.web or args.web_forum:
                web_modes.append("forum")
            if args.web or args.web_lit:
                web_modes.append("lit")
        elif args.web_auto:
            web_modes = ["open", "forum", "lit"]
        else:
            web_modes = ["open", "forum", "lit"]
    web_enabled = bool(web_modes)
    web_forced = explicit_web and not args.no_web
    web_auto = args.web_auto and not args.no_web
    web_hybrid = web_enabled and not args.web_no_hybrid
    if args.web_hybrid:
        web_hybrid = True
    embed_provider = provider if web_enabled else None
    embed_model = None
    if web_enabled:
        embed_provider, embed_model = _resolve_embed_choice(provider, args)
    embed_base_url = (
        args.web_embed_base_url
        or os.getenv("MINI_NEXEN_EMBED_BASE_URL")
        or os.getenv("LMSTUDIO_BASE_URL")
    )

    llm_model_label = _format_lmstudio_model(llm_config.model) if llm_config.provider == "lmstudio" else llm_config.model
    print(f"LLM enabled: provider={llm_config.provider}, model={llm_model_label}")
    if llm_config.provider == "gemini":
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            print("Warning: GEMINI_API_KEY not found. Gemini may fail unless ADC is configured.")
    if llm_config.provider == "lmstudio":
        print(f"LM Studio base URL: {llm_config.base_url}")

    if web_enabled:
        embed_label = _format_embed_model(embed_provider, embed_model or args.web_embed_model)
        print(f"Embeddings enabled: provider={embed_provider}, model={embed_label}")
        if embed_provider == "lmstudio" and embed_base_url:
            print(f"LM Studio embedding base URL: {embed_base_url}")
    else:
        print("Embeddings: disabled (web retrieval off)")

    interactive = sys.stdin.isatty()
    auto_methods = True
    review_query = True

    log_task_event("--------- Task Starts ----------")
    log_task_event(f"Topic: {args.topic}")
    log_task_event(f"Provider: {llm_config.provider} | Model: {llm_config.model}")
    try:
        result = run_research(
            topic=args.topic,
            rounds=args.rounds,
            top_k=args.top_k,
            output_language=args.language,
            provider=provider,
            model=model,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            discover_model=True,
            web_enabled=web_enabled,
            web_forced=web_forced,
            web_auto=web_auto,
            web_modes=web_modes,
            web_max_results=args.web_max_results,
            web_timeout=args.web_timeout,
            web_fetch_pages=not args.web_no_fetch,
            web_hybrid=web_hybrid,
            web_embed_provider=embed_provider,
            web_embed_model=embed_model or args.web_embed_model,
            web_embed_base_url=args.web_embed_base_url,
            web_embed_timeout=args.web_embed_timeout,
            web_embed_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            web_expand_queries=not args.web_no_expand,
            web_max_queries=args.web_max_queries,
            web_max_new_sources=args.web_max_new,
            web_max_per_query=args.web_max_per_query,
            web_relevance_threshold=args.web_relevance_threshold,
            web_max_rounds=args.web_max_rounds,
            auto_methods=auto_methods,
            review_query=review_query,
            interactive=interactive,
            profile_top_k=args.profile_top_k,
            kg_hops=args.kg_hops,
        )
        if result.query_artifact_path:
            if result.web_search_artifact_path == result.query_artifact_path:
                print(f"Query + web search artifact: {result.query_artifact_path}")
            else:
                print(f"Query understanding artifact: {result.query_artifact_path}")
        if result.web_search_artifact_path and result.web_search_artifact_path != result.query_artifact_path:
            print(f"Web search plan artifact: {result.web_search_artifact_path}")
        print(f"Saved plan: {result.plan_path}")
        # print(result.plan_markdown)
        print(
            "Research outline completed. "
            f"Saved in {result.plan_path.parent} "
            f"(outline words: {result.outline_word_count})"
        )
    except RetrievalRateLimitError as exc:
        print(
            "Retrieval error: rate limit exceeded. "
            f"{exc.label} attempts={exc.attempts} elapsed={exc.elapsed:.0f}s"
        )
        raise SystemExit(1) from exc
    except LLMClientError as exc:
        print(f"LLM error: {exc}")
        raise SystemExit(1) from exc
    finally:
        log_task_event("---------- Task Ends ----------")


def _resolve_llm_choice(args: argparse.Namespace) -> tuple[str, str]:
    env_provider = os.getenv("MINI_NEXEN_PROVIDER")
    env_model = os.getenv("MINI_NEXEN_MODEL")

    if args.provider and args.model:
        return args.provider, args.model

    if not sys.stdin.isatty():
        if env_provider and env_model:
            return env_provider, env_model
        raise SystemExit(
            "LLM not configured and no TTY available. Set MINI_NEXEN_PROVIDER and MINI_NEXEN_MODEL "
            "or pass --provider/--model."
        )

    if env_provider and env_model:
        keep = _prompt_yes_no(
            f"Use existing LLM config provider={env_provider}, model={env_model}?", default=True
        )
        if keep:
            return env_provider, env_model

    provider = _prompt_provider(env_provider)
    model_default = env_model if env_provider == provider else None
    model = _prompt_model(provider, model_default)
    return provider, model


def _env_truthy(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _env_optional_bool(name: str) -> bool | None:
    if name not in os.environ:
        return None
    value = os.getenv(name, "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def _resolve_embed_choice(provider: str, args: argparse.Namespace) -> tuple[str | None, str | None]:
    if args.web_embed_model:
        return provider, args.web_embed_model

    env_model = os.getenv("MINI_NEXEN_EMBED_MODEL")
    if env_model:
        return provider, env_model

    if not sys.stdin.isatty():
        if provider == "gemini":
            return "gemini", "gemini-embedding-001"
        return provider, None

    options = [
        ("gemini", "gemini-embedding-001", "Gemini embedding"),
        ("lmstudio", None, "LM Studio embedding (auto-detect)"),
    ]
    print("Select embedding model:")
    for idx, (_, model, label) in enumerate(options, start=1):
        model_label = model or "auto-detect"
        print(f"{idx}. {model_label} ({label})")
    while True:
        choice = input("Enter number: ").strip()
        if not choice:
            return options[0][0], options[0][1]
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(options):
                selected_provider, selected_model, _ = options[index - 1]
                return selected_provider, selected_model
        print("Invalid selection.")


def _format_lmstudio_model(model: str | None) -> str:
    if not model or model == "your-local-model":
        return "auto-detect"
    return model


def _format_embed_model(provider: str | None, model: str | None) -> str:
    if provider == "gemini":
        return model or "gemini-embedding-001"
    if provider == "lmstudio":
        return _format_lmstudio_model(model)
    return model or "auto-detect"


def _prompt_yes_no(message: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{message} {suffix} ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer y or n.")


def _prompt_provider(default: str | None = None) -> str:
    options = ["gemini", "lmstudio"]
    print("Select LLM provider:")
    for idx, option in enumerate(options, start=1):
        marker = " (default)" if option == default else ""
        print(f"{idx}. {option}{marker}")
    while True:
        choice = input("Enter number: ").strip()
        if not choice and default in options:
            return default
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(options):
                return options[index - 1]
        print("Invalid selection.")


def _prompt_model(provider: str, default: str | None = None) -> str:
    if provider == "gemini":
        options = [
            "gemini-3-pro-preview",
            "gemini-3-flash-preview",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "custom",
        ]
        print("Select Gemini model:")
        for idx, option in enumerate(options, start=1):
            marker = " (default)" if option == default else ""
            print(f"{idx}. {option}{marker}")
        while True:
            choice = input("Enter number: ").strip()
            if not choice and default:
                return default
            if choice.isdigit():
                index = int(choice)
                if 1 <= index <= len(options):
                    selected = options[index - 1]
                    if selected == "custom":
                        return _prompt_custom_model(default)
                    return selected
            print("Invalid selection.")

    if provider == "lmstudio":
        options = ["auto-detect", "custom"]
        default_option = "auto-detect" if not default or default == "your-local-model" else "custom"
        print("Select LM Studio model:")
        for idx, option in enumerate(options, start=1):
            marker = " (default)" if option == default_option else ""
            print(f"{idx}. {option}{marker}")
        while True:
            choice = input("Enter number: ").strip()
            if not choice:
                if default_option == "custom":
                    return default or "your-local-model"
                return "your-local-model"
            if choice.isdigit():
                index = int(choice)
                if 1 <= index <= len(options):
                    selected = options[index - 1]
                    if selected == "custom":
                        return _prompt_custom_model(default)
                    return "your-local-model"
            print("Invalid selection.")

    raise SystemExit(f"Unsupported provider: {provider}")


def _prompt_custom_model(default: str | None = None) -> str:
    while True:
        prompt = "Model name"
        if default:
            prompt += f" [{default}]"
        prompt += ": "
        value = input(prompt).strip()
        if value:
            return value
        if default:
            return default
        print("Model name is required.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="mini-NEXEN research agent",
        epilog=(
            "LLM required: export MINI_NEXEN_PROVIDER=gemini|lmstudio and MINI_NEXEN_MODEL=<model>. "
            "Gemini requires GEMINI_API_KEY. LM Studio uses LMSTUDIO_BASE_URL."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser(
        "ingest",
        help="Ingest local content (including data/local files) and rebuild the local KG + profile",
    )
    ingest.add_argument("--verbose", action="store_true", help="Echo LLM log events to stdout")
    ingest.add_argument("--quiet", action="store_true", help="Disable log echoing")
    ingest.add_argument("--file", action="append", help="Path to a local file (repeatable)")
    ingest.add_argument("--url", help="URL to record")
    ingest.add_argument("--text", help="Inline text content")
    ingest.add_argument("--title", help="Custom title")
    ingest.add_argument("--tags", help="Comma-separated tags")
    ingest.add_argument("--provider", choices=["gemini", "lmstudio"], help="LLM provider")
    ingest.add_argument("--model", help="Model name (provider-specific)")
    ingest.add_argument("--base-url", help="Override base URL (LM Studio only)")
    ingest.add_argument("--temperature", type=float, help="Sampling temperature")
    ingest.add_argument("--max-tokens", type=int, help="Max tokens to generate")
    ingest.add_argument(
        "--no-model-discovery",
        action="store_true",
        help="Disable LM Studio model discovery (use configured model name as-is)",
    )
    ingest.set_defaults(func=_ingest)

    interest = sub.add_parser("interest", help="Record an interest topic")
    interest.add_argument("text", nargs="?", help="Interest text (if --topic omitted)")
    interest.add_argument("--topic", help="Interest topic text")
    interest.set_defaults(func=_add_interest)

    method = sub.add_parser("method", help="Record an analysis method/approach")
    method.add_argument("text", nargs="?", help="Method text (if --method omitted)")
    method.add_argument("--method", help="Method text")
    method.set_defaults(func=_add_method)

    del_interest = sub.add_parser("delete-interest", help="Delete a single interest by id")
    del_interest.add_argument("--id", required=True, help="Interest id to remove")
    del_interest.set_defaults(func=_delete_interest)

    del_method = sub.add_parser("delete-method", help="Delete a single method by id")
    del_method.add_argument("--id", required=True, help="Method id to remove")
    del_method.set_defaults(func=_delete_method)

    clear_interests = sub.add_parser("clear-interests", help="Delete all interests")
    clear_interests.set_defaults(func=_clear_interests)

    clear_methods = sub.add_parser("clear-methods", help="Delete all methods")
    clear_methods.set_defaults(func=_clear_methods)

    clear_profile = sub.add_parser("clear-profile", help="Delete all profile signals")
    clear_profile.set_defaults(func=_clear_profile)

    clear_library = sub.add_parser("clear-library", help="Delete all documents + graph data")
    clear_library.set_defaults(func=_clear_library)

    list_docs = sub.add_parser("list-docs", help="List documents")
    list_docs.set_defaults(func=_list_docs)

    list_interests = sub.add_parser("list-interests", help="List interests")
    list_interests.set_defaults(func=_list_interests)

    list_methods = sub.add_parser("list-methods", help="List methods")
    list_methods.set_defaults(func=_list_methods)

    kg_report = sub.add_parser("kg-report", help="Summarize the local knowledge graph")
    kg_report.add_argument("--limit", type=int, default=10, help="Limit for top lists (default: 10)")
    kg_report.set_defaults(func=_kg_report)

    kg_entity_edges = sub.add_parser("kg-entity-edges", help="Count edges connected to an entity")
    kg_entity_edges.add_argument("--entity", help="Entity name to search")
    kg_entity_edges.add_argument("--id", help="Entity id (exact)")
    kg_entity_edges.add_argument(
        "--min-conf",
        type=float,
        default=0.0,
        help="Min relation confidence (default: 0.0)",
    )
    kg_entity_edges.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max entity matches to show when searching (default: 10)",
    )
    kg_entity_edges.add_argument(
        "--show-neighbors",
        action="store_true",
        help="List connected entities and predicates",
    )
    kg_entity_edges.set_defaults(func=_kg_entity_edges)

    kg_export_dot = sub.add_parser("kg-export-dot", help="Export a KG subgraph as DOT")
    kg_export_dot.add_argument("--seed", action="append", help="Seed term (repeat or comma-separated)")
    kg_export_dot.add_argument(
        "--hops",
        type=int,
        default=DEFAULT_KG_HOPS,
        help=f"Subgraph hops (default: {DEFAULT_KG_HOPS})",
    )
    kg_export_dot.add_argument("--min-conf", type=float, default=0.3, help="Min relation confidence")
    kg_export_dot.add_argument("--limit-edges", type=int, default=200, help="Max edges to export")
    kg_export_dot.add_argument(
        "--all",
        action="store_true",
        help="Export the full KG (ignores --seed/--hops)",
    )
    kg_export_dot.add_argument("--out", help="Output path (.dot)")
    kg_export_dot.add_argument(
        "--no-open",
        action="store_false",
        dest="open_dot",
        help="Do not render/open the exported DOT after writing",
    )
    kg_export_dot.set_defaults(open_dot=True)
    kg_export_dot.set_defaults(func=_kg_export_dot)

    kg_export_html = sub.add_parser("kg-export-html", help="Export a KG subgraph as HTML")
    kg_export_html.add_argument("--seed", action="append", help="Seed term (repeat or comma-separated)")
    kg_export_html.add_argument(
        "--hops",
        type=int,
        default=DEFAULT_KG_HOPS,
        help=f"Subgraph hops (default: {DEFAULT_KG_HOPS})",
    )
    kg_export_html.add_argument("--min-conf", type=float, default=0.3, help="Min relation confidence")
    kg_export_html.add_argument("--limit-edges", type=int, default=200, help="Max edges to export")
    kg_export_html.add_argument(
        "--all",
        action="store_true",
        help="Export the full KG (ignores --seed/--hops)",
    )
    kg_export_html.add_argument("--out", help="Output path (.html)")
    kg_export_html.add_argument(
        "--no-open",
        action="store_false",
        dest="open_html",
        help="Do not open the exported HTML after writing",
    )
    kg_export_html.set_defaults(open_html=True)
    kg_export_html.set_defaults(func=_kg_export_html)

    research = sub.add_parser("research", help="Generate a research plan")
    research.add_argument("--verbose", action="store_true", help="Echo LLM log events to stdout")
    research.add_argument("--quiet", action="store_true", help="Disable log echoing")
    research.add_argument("--topic", required=True)
    research.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    research.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    research.add_argument(
        "--kg-hops",
        type=int,
        default=DEFAULT_KG_HOPS,
        help=f"Subgraph hops for planning (default: {DEFAULT_KG_HOPS})",
    )
    research.add_argument(
        "--language",
        default="Chinese",
        help="Output language for plan and outline generation (default: Chinese)",
    )
    research.add_argument("--provider", choices=["gemini", "lmstudio"], help="LLM provider")
    research.add_argument("--model", help="Model name (provider-specific)")
    research.add_argument("--base-url", help="Override base URL (LM Studio only)")
    research.add_argument("--temperature", type=float, help="Sampling temperature")
    research.add_argument("--max-tokens", type=int, help="Max tokens to generate")
    research.add_argument("--web", action="store_true", help="Enable web retrieval (open + forum + literature)")
    research.add_argument(
        "--web-auto",
        action="store_true",
        help="Enable web retrieval on-demand (only when KG expansion criteria are met).",
    )
    research.add_argument(
        "--web-open",
        "--web-tech",
        dest="web_open",
        action="store_true",
        help="Enable open-web retrieval (Brave/Google/Tavily).",
    )
    research.add_argument(
        "--web-forum",
        dest="web_forum",
        action="store_true",
        help="Enable forum retrieval (X/Reddit).",
    )
    research.add_argument("--web-lit", action="store_true", help="Enable literature retrieval")
    research.add_argument("--no-web", action="store_true", help="Disable web retrieval (default: on)")
    research.add_argument(
        "--theme-top-k",
        dest="profile_top_k",
        type=int,
        default=argparse.SUPPRESS,
        help="Deprecated; use --profile-top-k",
    )
    research.add_argument(
        "--profile-top-k",
        dest="profile_top_k",
        type=int,
        default=DEFAULT_PROFILE_TOP_K,
        help="Number of top profile signals to include in the plan output (default: 10)",
    )
    research.add_argument("--web-max-results", type=int, default=5, help="Max results per source")
    research.add_argument("--web-timeout", type=int, default=15, help="Web fetch timeout (seconds)")
    research.add_argument("--web-no-fetch", action="store_true", help="Skip fetching full pages")
    research.add_argument("--web-hybrid", action="store_true", help="Force semantic reranking on")
    research.add_argument("--web-no-hybrid", action="store_true", help="Disable semantic reranking")
    research.add_argument("--web-embed-model", help="Embedding model name for reranking (auto-detects if omitted)")
    research.add_argument("--web-embed-base-url", help="Embedding base URL (defaults to LMSTUDIO_BASE_URL)")
    research.add_argument("--web-embed-timeout", type=int, help="Embedding timeout (seconds)")
    research.add_argument("--web-no-expand", action="store_true", help="Disable query expansion")
    research.add_argument("--web-max-queries", type=int, default=10, help="Max expanded queries (default: 10)")
    research.add_argument("--web-max-new", type=int, default=200, help="Max new sources per run (default: 200)")
    research.add_argument(
        "--web-max-per-query",
        "--web-max-per-interest",
        dest="web_max_per_query",
        type=int,
        default=10,
        help="Max sources per query seed (default: 10)",
    )
    research.add_argument(
        "--web-relevance-threshold",
        type=float,
        default=WEB_RELEVANCE_THRESHOLD,
        help="Minimum relevance score when reranking (default: 0.25)",
    )
    research.add_argument(
        "--web-max-rounds",
        type=int,
        default=WEB_AUTO_MAX_ROUNDS,
        help="Maximum web retrieval rounds per task (default: 3).",
    )
    research.set_defaults(func=_research)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    env_verbose = _env_optional_bool("MINI_NEXEN_VERBOSE")
    if env_verbose is not None:
        set_log_echo(env_verbose)
    else:
        set_log_echo(True)
    args.func(args)


if __name__ == "__main__":
    main()
