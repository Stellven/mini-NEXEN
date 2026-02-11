from __future__ import annotations

import argparse
from pathlib import Path

from . import db
import os
import sys

from .config import DEFAULT_ROUNDS, DEFAULT_TOP_K, ensure_dirs
from .llm import LLMClientError, load_llm_config, log_task_event, set_log_echo
from .research import run_research


def _ingest(args: argparse.Namespace) -> None:
    ensure_dirs()
    db.init_db()

    tags = [tag.strip() for tag in (args.tags or "").split(",") if tag.strip()]

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            raise SystemExit(f"File not found: {file_path}")
        content = file_path.read_text(encoding="utf-8")
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
    interest = db.add_interest(topic=args.topic, notes=args.notes or "")
    print(f"Recorded interest: {interest.interest_id}")


def _delete_interest(args: argparse.Namespace) -> None:
    ensure_dirs()
    db.init_db()
    deleted = db.delete_interest(args.id)
    if deleted:
        print(f"Deleted interest: {args.id}")
    else:
        print(f"No interest found for id: {args.id}")


def _clear_interests(args: argparse.Namespace) -> None:
    ensure_dirs()
    db.init_db()
    if not args.yes:
        raise SystemExit("Refusing to clear interests without --yes")
    deleted = db.clear_interests()
    print(f"Cleared interests: {deleted}")


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
        notes = f" ({interest.notes})" if interest.notes else ""
        print(f"{interest.interest_id} | {interest.topic}{notes}")


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

    print(f"LLM enabled: provider={llm_config.provider}, model={llm_config.model}")
    if llm_config.provider == "gemini":
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            print("Warning: GEMINI_API_KEY not found. Gemini may fail unless ADC is configured.")
    if llm_config.provider == "lmstudio":
        print(f"LM Studio base URL: {llm_config.base_url}")

    log_task_event("--------- Task Starts ----------")
    log_task_event(f"Topic: {args.topic}")
    log_task_event(f"Provider: {llm_config.provider} | Model: {llm_config.model}")
    try:
        result = run_research(
            topic=args.topic,
            rounds=args.rounds,
            top_k=args.top_k,
            provider=provider,
            model=model,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            discover_model=not args.no_model_discovery,
        )
        print(f"Saved plan: {result.plan_path}")
        # print(result.plan_markdown)
        print(f"Research outline completed. Saved in {result.plan_path.parent}")
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
            "gemini-2.0-flash",
            "gemini-2.0-pro",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
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
        options = ["your-local-model", "custom"]
        print("Select LM Studio model:")
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
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Add a file, URL, or note to the library")
    ingest.add_argument("--verbose", action="store_true", help="Echo LLM log events to stdout")
    ingest.add_argument("--file", help="Path to a text file")
    ingest.add_argument("--url", help="URL to record")
    ingest.add_argument("--text", help="Inline text content")
    ingest.add_argument("--title", help="Custom title")
    ingest.add_argument("--tags", help="Comma-separated tags")
    ingest.set_defaults(func=_ingest)

    interest = sub.add_parser("interest", help="Record an interest topic")
    interest.add_argument("--verbose", action="store_true", help="Echo LLM log events to stdout")
    interest.add_argument("--topic", required=True)
    interest.add_argument("--notes")
    interest.set_defaults(func=_add_interest)

    del_interest = sub.add_parser("delete-interest", help="Delete a single interest by id")
    del_interest.add_argument("--verbose", action="store_true", help="Echo LLM log events to stdout")
    del_interest.add_argument("--id", required=True, help="Interest id to remove")
    del_interest.set_defaults(func=_delete_interest)

    clear_interests = sub.add_parser("clear-interests", help="Delete all interests")
    clear_interests.add_argument("--verbose", action="store_true", help="Echo LLM log events to stdout")
    clear_interests.add_argument("--yes", action="store_true", help="Confirm deletion")
    clear_interests.set_defaults(func=_clear_interests)

    list_docs = sub.add_parser("list-docs", help="List documents")
    list_docs.add_argument("--verbose", action="store_true", help="Echo LLM log events to stdout")
    list_docs.set_defaults(func=_list_docs)

    list_interests = sub.add_parser("list-interests", help="List interests")
    list_interests.add_argument("--verbose", action="store_true", help="Echo LLM log events to stdout")
    list_interests.set_defaults(func=_list_interests)

    research = sub.add_parser("research", help="Generate a research plan")
    research.add_argument("--verbose", action="store_true", help="Echo LLM log events to stdout")
    research.add_argument("--topic", required=True)
    research.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    research.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    research.add_argument("--provider", choices=["gemini", "lmstudio"], help="LLM provider")
    research.add_argument("--model", help="Model name (provider-specific)")
    research.add_argument("--base-url", help="Override base URL (LM Studio only)")
    research.add_argument("--temperature", type=float, help="Sampling temperature")
    research.add_argument("--max-tokens", type=int, help="Max tokens to generate")
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
    set_log_echo(bool(getattr(args, "verbose", False)))
    args.func(args)


if __name__ == "__main__":
    main()
