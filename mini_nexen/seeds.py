from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import db
from .config import LOCAL_FILES_DIR
from .file_ingest import load_text_from_file
from .llm import log_task_event


@dataclass
class SeedIngestResult:
    added: int
    skipped: int
    files: int


def _derive_tags(path: Path) -> list[str]:
    tags = [tag for tag in path.stem.split("_") if tag]
    tags.append("seed")
    seen = set()
    output = []
    for tag in tags:
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(tag)
    return output


def _derive_title(content: str, path: Path) -> str:
    default_title = path.stem.replace("_", " ").title()
    if not content:
        return default_title
    first_line = content.splitlines()[0].strip()
    if first_line.lower().startswith("title:"):
        title = first_line.split(":", 1)[1].strip()
        return title or default_title
    return default_title


def ingest_seed_pack(seed_dir: Path | None = None) -> SeedIngestResult:
    db.init_db()
    seed_root = seed_dir or LOCAL_FILES_DIR
    if not seed_root.exists() or not seed_root.is_dir():
        return SeedIngestResult(added=0, skipped=0, files=0)

    allowed = {".txt", ".md", ".markdown", ".pdf", ".docx"}
    files = sorted([path for path in seed_root.iterdir() if path.suffix.lower() in allowed])
    added = 0
    skipped = 0
    for path in files:
        if path.name == "seed_urls.txt":
            skipped += 1
            continue
        resolved = str(path.resolve())
        raw = str(path)
        if db.document_exists(resolved) or db.document_exists(raw):
            skipped += 1
            continue
        content = load_text_from_file(path)
        title = _derive_title(content, path)
        tags = _derive_tags(path)
        _doc, created, _reason = db.add_document_dedup(
            title=title,
            source_type="file",
            source=resolved,
            content_text=content,
            tags=tags,
        )
        if created:
            added += 1
        else:
            skipped += 1

    if added or skipped:
        log_task_event(
            f"Local file ingest: files={len(files)} added={added} skipped={skipped} dir={seed_root}"
        )
    return SeedIngestResult(added=added, skipped=skipped, files=len(files))
