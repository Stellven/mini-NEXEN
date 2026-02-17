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

# Graph + retrieval defaults
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
GRAPH_REBUILD_RATIO = 0.15
GRAPH_NOISE_RATIO_THRESHOLD = 0.35
GRAPH_AVG_SIMILARITY_THRESHOLD = 0.2
GRAPH_UNASSIGNED_RATIO_THRESHOLD = 0.4
GRAPH_ASSIGN_SIMILARITY_MIN = 0.3
GRAPH_TOP_CLUSTERS = 10

WEB_MAX_NEW_SOURCES = 5
WEB_MAX_PER_QUERY = 10
WEB_RELEVANCE_THRESHOLD = 0.25

WEB_DECAY_PER_RUN = 0.95
WEB_ARCHIVE_SCORE_THRESHOLD = 0.2
WEB_ARCHIVE_RUNS_UNUSED = 20


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
