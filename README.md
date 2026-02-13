# mini-NEXEN

Minimal, local-first research backend for generating research plans and detailed outlines from a personal library of documents and URLs.

**What It Does**
- Maintains a personal library of documents, URLs, and notes.
- Tracks user interests (NotebookLM-style memory).
- Builds a lightweight topic graph from document chunks and embeddings.
- Uses cluster-aware local retrieval plus optional external web sources (guardrails on by default).
- Generates a research plan + detailed outline and saves it as `YYYY_MM_DD_HH_MM_plan.md`.

**Project Structure**
- `mini_nexen/`: Core logic (agents, planning, storage).
- `skills/`: Skills definitions (`SKILL.md` per skill).
- `data/`: SQLite DB + logs + stored documents.
- `data/library/`: Stored text copies of ingested content.
- `data/seeds/`: Seed documents and URLs for quick bootstrapping.
- `plans/`: Generated research plans.

**Environment Setup**
The project expects a conda environment named `mini-nexen`.

```bash
conda env create -f environment.yml
conda activate mini-nexen
```

Installed dependencies (via conda in `mini-nexen`):
- `numpy`
- `scipy`
- `scikit-learn`

Installed dependencies (via pip in `mini-nexen`):
- `requests`
- `google-genai`
- `hdbscan`
- `pypdf` (PDF ingestion)
- `python-docx` (DOCX ingestion)

LLM providers:
- Gemini requires `GEMINI_API_KEY` or `GOOGLE_API_KEY`.
- LM Studio requires a local server with an OpenAI-compatible endpoint.

**Usage**

**Minimal command (prompts you to select an LLM if not configured):**
```bash
python -m mini_nexen.cli research --topic "Agentic research systems"
```

**Configure backend LLMs**

Gemini (API key required):
```bash
export MINI_NEXEN_PROVIDER=gemini
export GEMINI_API_KEY="your-key"
export MINI_NEXEN_MODEL="gemini-2.5-flash"
python -m mini_nexen.cli research --topic "Agentic research systems"
```

LM Studio (local server):
```bash
export MINI_NEXEN_PROVIDER=lmstudio
export MINI_NEXEN_MODEL="your-local-model"
export LMSTUDIO_BASE_URL="http://localhost:1234/v1"
python -m mini_nexen.cli research --topic "Agentic research systems"
```

LM Studio setup checklist:
- Start the LM Studio server.
- Ensure a model is loaded.
- Confirm the OpenAI-compatible endpoint is reachable: `GET /v1/models`.

**Additional arguments (grouped by functionality)**

**Ingestion**
- `ingest --file` / `--url` / `--text`: Create a document record and store raw text in `data/library/`.
- `ingest --title`: Set a custom title.
- `ingest --tags`: Comma-separated tags stored in the DB.
- `research --ingest` (or `--ingest-seeds`): Ingest new `.txt`, `.md`, `.markdown`, `.pdf`, `.docx` files from `data/seeds/`.

**Agentic workflow**
- `research --rounds`: Number of plan refinement rounds (default: 2).
- `research --auto-interest`: Add the research topic to stored interests (default: off).

**Graph + Retrieval**
- Graph build is automatic during `research` runs (chunking + clustering + rebuilds based on quality metrics).
- Web retrieval is enabled by default (tech + literature). Use `--no-web` to disable.
- `--web` / `--web-tech` / `--web-lit`: Choose web sources.
- `--web-max-results`: Max results per source (default: 5).
- `--web-timeout`: Web fetch timeout in seconds (default: 15).
- `--web-no-fetch`: Skip fetching full pages (store metadata/abstracts only).
- `--web-hybrid`: Force semantic reranking on.
- `--web-no-hybrid`: Disable semantic reranking.
- `--web-embed-model`: Embedding model for reranking (auto-detects if omitted).
- `--web-embed-base-url`: Embedding endpoint (defaults to `LMSTUDIO_BASE_URL`).
- `--web-embed-timeout`: Embedding timeout in seconds.
- `--web-no-expand`: Disable query expansion (LLM expansion only).
- `--web-max-queries`: Max expanded queries (default: 10).
- `--web-max-new`: Max new sources per run (default: 50).
- `--web-max-per-query`: Cap results per query seed (default: 10).
- `--web-relevance-threshold`: Filter low-relevance results when reranking (default: 0.25).
- `--top-k`: Max local documents pulled into planning (default: 8).
- `--no-graph-semantic-labels`: Disable LLM-based cluster labels.

**Planning**
- `--temperature`: Sampling temperature for the LLM.
- `--max-tokens`: Max tokens to generate.

**Outline**
- Outline is generated from the plan automatically; no explicit flags.

**Model selection**
- `--provider` / `--model` / `--base-url`: Override LLM provider/model per run.
- `--no-model-discovery`: Disable LM Studio model discovery.

**Logging**
- `--verbose`: Echo log events to stdout.
- `--quiet`: Disable log echoing.
- `MINI_NEXEN_VERBOSE=1|0`: Default logging behavior.

**Defaults & Diagnostics (what’s on/off by default)**
Defaults (can be changed via flags):
- Web retrieval is **on** by default (tech + literature). Use `--no-web` to disable.
- Query expansion is **on** by default (LLM-based only). Use `--web-no-expand` to disable.
- Semantic reranking is **on** by default when web retrieval is enabled. Use `--web-no-hybrid` to disable.
- Seed ingestion is **off** by default. Use `--ingest` on `research` to ingest `data/seeds/`.
- Auto-adding research topics to interests is **off** by default. Use `--auto-interest` to enable.
- LM Studio model discovery is **on** by default. Use `--no-model-discovery` to disable.
- Semantic cluster labeling is **on** by default when an LLM is available. Use `--no-graph-semantic-labels` to disable.
- Task event logging to `data/task_events.log` is always **on**; CLI echoing is on unless `--quiet` or `MINI_NEXEN_VERBOSE=0`.

Development / diagnostic behaviors to be aware of:
- Topic-to-cluster mappings are written to `topic_cluster_map` for traceability.
- Web source decay + archiving runs at the start of each `research` run (based on run count).
- Graph rebuilds are triggered automatically by quality thresholds and growth ratio.
- Per-source web retrieval success/failure and counts are logged in `data/task_events.log`.

**Outputs**
- Plan + outline saved in `plans/YYYY_MM_DD_HH_MM_plan.md`.
- Ingested document text stored in `data/library/<doc_id>.txt`.
- Metadata, interests, and graph stored in `data/mini_nexen.sqlite3`.
- Task event log written to `data/task_events.log`.
- Web sources decay per research run and may be archived if unused; local user documents are retained.
- Research topics are mapped to clusters and stored in `topic_cluster_map` for traceability.

**Graph & Debugging**

Check cluster sizes:
```bash
sqlite3 data/mini_nexen.sqlite3 "SELECT label, size, updated_at FROM clusters ORDER BY size DESC LIMIT 20;"
```

Check cluster quality metrics:
```bash
sqlite3 data/mini_nexen.sqlite3 "SELECT value FROM graph_meta WHERE key='metrics';"
```

See unassigned chunk ratio:
```bash
sqlite3 data/mini_nexen.sqlite3 "SELECT COUNT(*) AS total, SUM(CASE WHEN cluster_id IS NULL THEN 1 ELSE 0 END) AS unassigned FROM chunks;"
```

Inspect top chunks for a cluster:
```bash
sqlite3 data/mini_nexen.sqlite3 "SELECT text FROM chunks WHERE cluster_id = (SELECT cluster_id FROM clusters ORDER BY size DESC LIMIT 1) LIMIT 5;"
```

Inspect topic-to-cluster mappings:
```bash
sqlite3 data/mini_nexen.sqlite3 "SELECT topic, cluster_id, similarity, run_id, created_at FROM topic_cluster_map ORDER BY created_at DESC LIMIT 10;"
```

**List/Manage Stored Data**
```bash
python -m mini_nexen.cli list-docs
python -m mini_nexen.cli list-interests
python -m mini_nexen.cli delete-interest --id "<INTEREST_ID>"
python -m mini_nexen.cli clear-interests --yes
```

Clearing interests removes only the `interests` table entries; it does not delete documents, chunks, or clusters.

**Seed Pack**
```bash
python -m mini_nexen.cli ingest --file data/seeds/agentic_research_overview.txt --tags "agentic,overview"
python -m mini_nexen.cli ingest --file data/seeds/retrieval_planning_notes.txt --tags "retrieval,planning"
python -m mini_nexen.cli ingest --file data/seeds/evaluation_metrics_cheatsheet.txt --tags "evaluation,metrics"
```

See `data/seeds/seed_urls.txt` for suggested URLs to ingest later with short excerpts.

**Skills**
Skills live in `skills/<skill_name>/SKILL.md`. The runtime loads these at startup and only executes registered skills.
