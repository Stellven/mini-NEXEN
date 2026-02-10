from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LIBRARY_DIR = DATA_DIR / "library"
PLANS_DIR = PROJECT_ROOT / "plans"
DB_PATH = DATA_DIR / "mini_nexen.sqlite3"
SKILLS_DIR = PROJECT_ROOT / "skills"
LLM_LOG_PATH = DATA_DIR / "llm_calls.log"

DEFAULT_TOP_K = 8
DEFAULT_ROUNDS = 2


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
