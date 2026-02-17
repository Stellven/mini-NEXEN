from __future__ import annotations

import argparse
from pathlib import Path

from . import db
import os
import sys

from .config import DEFAULT_MIN_WEB_DOCS, DEFAULT_ROUNDS, DEFAULT_TOP_K, GRAPH_TOP_CLUSTERS, ensure_dirs
from .file_ingest import load_text_from_file
from .llm import LLMClientError, load_llm_config, log_task_event, set_log_echo
from .web_retrieval import RetrievalRateLimitError
from .research import run_research


def _ingest(args: argparse.Namespace) -> None:
    ensure_dirs()
    db.init_db()

    tags = [tag.strip() for tag in (args.tags or "").split(",") if tag.strip()]

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            raise SystemExit(f"File not found: {file_path}")
        content = load_text_from_file(file_path)
        title = args.title or file_path.name
        doc = db.add_document(
            title=title,
            source_type="file",
            source=str(file_path),
            content_text=content,
            tags=tags,
        )
        print(f"Ingested file document: {doc.doc_id}")
        return

    if args.url:
        title = args.title or args.url
        content = args.text or ""
        doc = db.add_document(
            title=title,
            source_type="url",
            source=args.url,
            content_text=content,
            tags=tags,
        )
        print(f"Recorded URL document: {doc.doc_id}")
        if not content:
            print("Note: URL content was not fetched; provide --text to store highlights.")
        return

    if args.text:
        title = args.title or "Personal note"
        doc = db.add_document(
            title=title,
            source_type="note",
            source="user",
            content_text=args.text,
            tags=tags,
        )
        print(f"Recorded note: {doc.doc_id}")
        return

    raise SystemExit("Provide --file, --url, or --text to ingest.")


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
    if not args.yes:
        raise SystemExit("Refusing to clear interests without --yes")
    deleted = db.clear_interests()
    print(f"Cleared interests: {deleted}")


def _clear_methods(args: argparse.Namespace) -> None:
    ensure_dirs()
    db.init_db()
    if not args.yes:
        raise SystemExit("Refusing to clear methods without --yes")
    deleted = db.clear_methods()
    print(f"Cleared methods: {deleted}")


def _clear_library(args: argparse.Namespace) -> None:
    ensure_dirs()
    db.init_db()
    if not args.yes:
        raise SystemExit("Refusing to clear library without --yes")
    result = db.clear_library_and_graph(clear_files=True)
    print(
        "Cleared library + graph: "
        f"documents={result['documents']} "
        f"document_stats={result['document_stats']} "
        f"chunks={result['chunks']} "
        f"clusters={result['clusters']} "
        f"topic_cluster_map={result['topic_cluster_map']} "
        f"graph_meta={result['graph_meta']} "
        f"files_removed={result['files_removed']}"
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


def _research(args: argparse.Namespace) -> None:
    ensure_dirs()
    provider, model = _resolve_llm_choice(args)
    llm_config = load_llm_config(
        provider=provider,
        model=model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        discover_model=not args.no_model_discovery,
    )
    if not llm_config:
        raise SystemExit("LLM configuration failed. Check provider/model settings.")

    web_modes = []
    if not args.no_web:
        if args.web or args.web_tech or args.web_lit:
            if args.web or args.web_tech:
                web_modes.append("tech")
            if args.web or args.web_lit:
                web_modes.append("lit")
        else:
            web_modes = ["tech", "lit"]
    web_enabled = bool(web_modes)
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

    log_task_event("--------- Task Starts ----------")
    log_task_event(f"Topic: {args.topic}")
    log_task_event(f"Provider: {llm_config.provider} | Model: {llm_config.model}")
    try:
        result = run_research(
            topic=args.topic,
            rounds=args.rounds,
            top_k=args.top_k,
            min_web_docs=args.min_web_docs,
            provider=provider,
            model=model,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            discover_model=not args.no_model_discovery,
            web_enabled=web_enabled,
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
            ingest_seeds=args.ingest_seeds,
            auto_interest=args.auto_interest,
            graph_semantic_labels=not args.no_graph_semantic_labels or args.graph_semantic_labels,
            graph_top_clusters=args.graph_top_clusters,
        )
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
            "gemini-2.0-pro",
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
    parser.add_argument("--verbose", action="store_true", help="Echo LLM log events to stdout")
    parser.add_argument("--quiet", action="store_true", help="Disable log echoing")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Add a file, URL, or note to the library")
    ingest.add_argument("--verbose", action="store_true", help="Echo LLM log events to stdout")
    ingest.add_argument("--quiet", action="store_true", help="Disable log echoing")
    ingest.add_argument("--file", help="Path to a text file")
    ingest.add_argument("--url", help="URL to record")
    ingest.add_argument("--text", help="Inline text content")
    ingest.add_argument("--title", help="Custom title")
    ingest.add_argument("--tags", help="Comma-separated tags")
    ingest.set_defaults(func=_ingest)

    interest = sub.add_parser("interest", help="Record an interest topic")
    interest.add_argument("--verbose", action="store_true", help="Echo LLM log events to stdout")
    interest.add_argument("--quiet", action="store_true", help="Disable log echoing")
    interest.add_argument("text", nargs="?", help="Interest text (if --topic omitted)")
    interest.add_argument("--topic", help="Interest topic text")
    interest.set_defaults(func=_add_interest)

    method = sub.add_parser("method", help="Record an analysis method/approach")
    method.add_argument("--verbose", action="store_true", help="Echo LLM log events to stdout")
    method.add_argument("--quiet", action="store_true", help="Disable log echoing")
    method.add_argument("text", nargs="?", help="Method text (if --method omitted)")
    method.add_argument("--method", help="Method text")
    method.set_defaults(func=_add_method)

    del_interest = sub.add_parser("delete-interest", help="Delete a single interest by id")
    del_interest.add_argument("--verbose", action="store_true", help="Echo LLM log events to stdout")
    del_interest.add_argument("--quiet", action="store_true", help="Disable log echoing")
    del_interest.add_argument("--id", required=True, help="Interest id to remove")
    del_interest.set_defaults(func=_delete_interest)

    del_method = sub.add_parser("delete-method", help="Delete a single method by id")
    del_method.add_argument("--verbose", action="store_true", help="Echo LLM log events to stdout")
    del_method.add_argument("--quiet", action="store_true", help="Disable log echoing")
    del_method.add_argument("--id", required=True, help="Method id to remove")
    del_method.set_defaults(func=_delete_method)

    clear_interests = sub.add_parser("clear-interests", help="Delete all interests")
    clear_interests.add_argument("--verbose", action="store_true", help="Echo LLM log events to stdout")
    clear_interests.add_argument("--quiet", action="store_true", help="Disable log echoing")
    clear_interests.add_argument("--yes", action="store_true", help="Confirm deletion")
    clear_interests.set_defaults(func=_clear_interests)

    clear_methods = sub.add_parser("clear-methods", help="Delete all methods")
    clear_methods.add_argument("--verbose", action="store_true", help="Echo LLM log events to stdout")
    clear_methods.add_argument("--quiet", action="store_true", help="Disable log echoing")
    clear_methods.add_argument("--yes", action="store_true", help="Confirm deletion")
    clear_methods.set_defaults(func=_clear_methods)

    clear_library = sub.add_parser("clear-library", help="Delete all documents + graph data")
    clear_library.add_argument("--verbose", action="store_true", help="Echo LLM log events to stdout")
    clear_library.add_argument("--quiet", action="store_true", help="Disable log echoing")
    clear_library.add_argument("--yes", action="store_true", help="Confirm deletion")
    clear_library.set_defaults(func=_clear_library)

    list_docs = sub.add_parser("list-docs", help="List documents")
    list_docs.add_argument("--verbose", action="store_true", help="Echo LLM log events to stdout")
    list_docs.add_argument("--quiet", action="store_true", help="Disable log echoing")
    list_docs.set_defaults(func=_list_docs)

    list_interests = sub.add_parser("list-interests", help="List interests")
    list_interests.add_argument("--verbose", action="store_true", help="Echo LLM log events to stdout")
    list_interests.add_argument("--quiet", action="store_true", help="Disable log echoing")
    list_interests.set_defaults(func=_list_interests)

    list_methods = sub.add_parser("list-methods", help="List methods")
    list_methods.add_argument("--verbose", action="store_true", help="Echo LLM log events to stdout")
    list_methods.add_argument("--quiet", action="store_true", help="Disable log echoing")
    list_methods.set_defaults(func=_list_methods)

    research = sub.add_parser("research", help="Generate a research plan")
    research.add_argument("--verbose", action="store_true", help="Echo LLM log events to stdout")
    research.add_argument("--quiet", action="store_true", help="Disable log echoing")
    research.add_argument("--topic", required=True)
    research.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    research.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    research.add_argument(
        "--min-web-docs",
        type=int,
        default=DEFAULT_MIN_WEB_DOCS,
        help="Minimum web documents to include in planning (default: 4)",
    )
    research.add_argument("--provider", choices=["gemini", "lmstudio"], help="LLM provider")
    research.add_argument("--model", help="Model name (provider-specific)")
    research.add_argument("--base-url", help="Override base URL (LM Studio only)")
    research.add_argument("--temperature", type=float, help="Sampling temperature")
    research.add_argument("--max-tokens", type=int, help="Max tokens to generate")
    research.add_argument("--web", action="store_true", help="Enable web retrieval (tech + literature)")
    research.add_argument("--web-tech", action="store_true", help="Enable tech/news/forums retrieval")
    research.add_argument("--web-lit", action="store_true", help="Enable literature retrieval")
    research.add_argument("--no-web", action="store_true", help="Disable web retrieval (default: on)")
    research.add_argument(
        "--ingest", "--ingest-seeds", dest="ingest_seeds", action="store_true", help="Ingest seed files"
    )
    research.add_argument("--auto-interest", action="store_true", help="Add research topic to interests")
    research.add_argument(
        "--graph-semantic-labels",
        action="store_true",
        help="Use LLM to label clusters (default: on)",
    )
    research.add_argument(
        "--no-graph-semantic-labels",
        action="store_true",
        help="Disable LLM cluster labels",
    )
    research.add_argument(
        "--graph-top-clusters",
        type=int,
        default=GRAPH_TOP_CLUSTERS,
        help="Number of top clusters to use for retrieval (default: 3)",
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
    research.add_argument("--web-max-new", type=int, default=50, help="Max new sources per run (default: 50)")
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
        default=0.25,
        help="Minimum relevance score when reranking (default: 0.25)",
    )
    research.add_argument(
        "--no-model-discovery",
        action="store_true",
        help="Disable LM Studio model discovery (use configured model name as-is)",
    )
    research.set_defaults(func=_research)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "quiet", False):
        set_log_echo(False)
    else:
        env_verbose = _env_optional_bool("MINI_NEXEN_VERBOSE")
        if env_verbose is not None:
            set_log_echo(env_verbose or bool(getattr(args, "verbose", False)))
        else:
            set_log_echo(True)
    args.func(args)


if __name__ == "__main__":
    main()
