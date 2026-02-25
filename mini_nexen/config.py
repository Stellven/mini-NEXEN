from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LIBRARY_DIR = DATA_DIR / "library"
PLANS_DIR = PROJECT_ROOT / "plans"
DB_PATH = DATA_DIR / "mini_nexen.sqlite3"
SKILLS_DIR = PROJECT_ROOT / "skills"
TASK_LOG_PATH = DATA_DIR / "task_events.log"

DEFAULT_TOP_K = 3
DEFAULT_MIN_WEB_DOCS = 10
DEFAULT_ROUNDS = 2

# Planning defaults
DEFAULT_THEME_TOP_K = 10

WEB_MAX_NEW_SOURCES = 200
WEB_MAX_PER_QUERY = 10
WEB_RELEVANCE_THRESHOLD = 0.25

# Web expansion gating
WEB_EXPAND_EVIDENCE_MIN = 4
WEB_EXPAND_CONFIDENCE_MIN = 0.75
WEB_EXPAND_STALE_DAYS = 180
WEB_EXPAND_CONTRADICTION_CONF_MAX = 0.6
WEB_EXPAND_CONTRADICTION_STALE_DAYS = 180
WEB_EVIDENCE_DEFAULT_DAYS = 180
WEB_AUTO_MAX_ROUNDS = 3

WEB_DECAY_PER_RUN = 0.95
WEB_ARCHIVE_SCORE_THRESHOLD = 0.2
WEB_ARCHIVE_RUNS_UNUSED = 20


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
