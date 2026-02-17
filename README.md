# mini-NEXEN

Minimal, local-first research backend for generating research plans and detailed outlines from a personal library of documents and URLs.

**What It Does**
- Maintains a personal library of documents, URLs, and notes in SQLite.
- Tracks user interests and analysis methods as lightweight memory.
- Builds a topic graph from document chunks and embeddings.
- Retrieves local sources with cluster-aware ranking and optional web retrieval.
- Generates a multi-round research plan and a long-form outline.

**Quick Start**
1. Create and activate the environment.
```bash
conda env create -f environment.yml
conda activate mini-nexen
```
2. Run a research pass (you will be prompted to pick a provider/model if not configured).
```bash
python -m mini_nexen.cli research --topic "Agentic research systems"
```

**Environment Setup (Detailed)**
1. Create the conda environment from `environment.yml`.
```bash
conda env create -f environment.yml
```
2. Activate the environment.
```bash
conda activate mini-nexen
```
3. Sanity-check the install.
```bash
python -m mini_nexen.cli --help
```

**Build / Installation**
- There is no build step; this is a pure Python project. Run commands from the repo root with `python -m mini_nexen.cli ...`.

**Docker**
1. Build the image.
```bash
docker build -t mini-nexen .
```
2. Run a research pass (mount `data/` and `plans/` so outputs persist).
```bash
docker run --rm -it \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/plans:/app/plans" \
  -e GEMINI_API_KEY="your-key" \
  mini-nexen research --topic "Agentic research systems"
```
Notes:
- For LM Studio running on the host, use `--network=host` (Linux) or set `LMSTUDIO_BASE_URL` to `http://host.docker.internal:1234/v1`.
- If you want interactive provider/model prompts, keep `-it`.

**Docker Compose**
1. Configure environment variables (optional).
```bash
cp .env.example .env
```
2. Build and run (shows help by default).
```bash
docker compose up --build
```
3. Run a research pass.
```bash
docker compose run --rm mini-nexen research --topic "Agentic research systems"
```
Notes:
- Edit `.env` to set Gemini/LM Studio keys and model defaults.
- Data and plans are persisted to `./data` and `./plans`.

**Setup**
- Python 3.11 is required.
- Dependencies are defined in `environment.yml`. Conda packages include `numpy`, `scipy`, `scikit-learn`, `hdbscan`, `requests`. Pip packages include `google-genai`, `pypdf`, `python-docx`.
- Optional external services: Gemini requires `GEMINI_API_KEY` or `GOOGLE_API_KEY`. LM Studio requires an OpenAI-compatible endpoint (default `http://localhost:1234/v1`).

**Configuration**
LLM configuration is resolved in this order: CLI flags, environment variables, then interactive prompt (if TTY is available).

LLM env vars:
- `MINI_NEXEN_PROVIDER`: `gemini` or `lmstudio`
- `MINI_NEXEN_MODEL`: model name (see below)
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`: Gemini auth
- `LMSTUDIO_BASE_URL`: LM Studio endpoint (default `http://localhost:1234/v1`)
- `LMSTUDIO_API_KEY`: LM Studio auth if required

LLM defaults:
- Gemini default model: `gemini-2.5-flash`
- LM Studio default model: `local-model` (if you set `MINI_NEXEN_MODEL` to `your-local-model` and keep discovery on, the client will auto-detect a model)
- Temperature default: `0.2`
- Max tokens default: `2048`

Embedding configuration (used by web reranking and the local graph):
- `MINI_NEXEN_EMBED_PROVIDER`: `gemini` or `lmstudio` (optional)
- `MINI_NEXEN_EMBED_MODEL`: embedding model name (optional)
- `MINI_NEXEN_EMBED_BASE_URL`: embedding endpoint (optional)
- `LMSTUDIO_BASE_URL`: fallback embedding endpoint when provider is `lmstudio`

Logging configuration:
- `MINI_NEXEN_VERBOSE=1|0` sets default log echoing.
- `--verbose` forces log echoing.
- `--quiet` disables log echoing.

**Data Locations**
- `data/mini_nexen.sqlite3`: metadata, interests, graph, and stats.
- `data/library/`: stored text copies of ingested content.
- `data/seeds/`: optional user-provided seed files (if you use `--ingest`).
- `data/task_events.log`: structured log of retrieval/LLM/graph events.
- `plans/`: generated plans as `YYYY_MM_DD_HH_MM_plan.md`.

**Pipeline (Research Command)**
1. Resolve LLM config from flags/env/interactive prompts.
2. Ensure data directories and initialize the SQLite DB.
3. Increment the research run counter and optionally ingest seed files from `data/seeds/` (if you maintain your own).
4. Decay and archive existing web documents based on usage and relevance.
5. Optionally add the current topic to the interests table.
6. Start planning rounds (default 2).
7. Load stored interests and methods.
8. If the systems-engineering skill triggers on keywords, inject its guidance into the planning context.
9. If web retrieval is enabled, build query seeds from topic + interests + graph suggestions and run retrieval.
10. Web retrieval searches DuckDuckGo (tech) and arXiv/Semantic Scholar/Crossref (literature), optionally expands queries with the LLM, fetches pages, and reranks results with embeddings.
11. Web results are stored as new documents with relevance scores.
12. Update the graph: chunk documents, embed missing chunks, cluster (HDBSCAN then KMeans fallback), and update metrics.
13. Map the topic to the closest cluster and record it in `topic_cluster_map`.
14. Select local docs via cluster round-robin (top clusters + top_k); if clusters are unavailable, fall back to lexical scoring.
15. Merge the most similar chunks per doc for LLM context and enforce a minimum count of web docs if requested.
16. Draft a research plan with the LLM using interests, methods, extracted themes, and selected docs.
17. If readiness checks fail (gaps or not enough sources), refine the plan and generate new query hints for the next round.
18. Build a long-form outline (target 1000-2000 words) and render the final plan markdown.
19. Save `plans/YYYY_MM_DD_HH_MM_plan.md` and log summary stats.

**Retrieval & Graph Details**
- Chunking uses sentence splits, default chunk size `500` tokens with `100` tokens of overlap.
- Each chunk is embedded and stored in `chunks.embedding_json`.
- Clustering uses HDBSCAN when available; if it produces no clusters, KMeans is used; if dependencies are missing, a single-cluster fallback is used.
- Rebuilds trigger when there are no clusters and at least 5 chunks, when new chunks exceed 15% of total, when quality metrics degrade, or when archived chunks are pruned.
- Incremental updates assign new chunks to the nearest centroid when similarity is at least `0.3`.
- Cluster labels are generated by the LLM when enabled and available.
- Retrieval ranks clusters by query embedding similarity and selects docs with a round-robin across clusters.

**CLI Reference**

`research` generates a plan and outline.

Common behavior:
- If no provider/model is configured and stdin is not a TTY, the command exits with an error.
- If no `--web*` flags are given, web retrieval runs with both `tech` and `lit` modes.

Research flags:
| Flag | Meaning | Default |
| --- | --- | --- |
| `--topic` | Research topic | required |
| `--rounds` | Planning rounds | `2` |
| `--top-k` | Max local docs fed into planning | `3` |
| `--min-web-docs` | Minimum web docs to include from the library | `10` |
| `--provider` | LLM provider | env or prompt |
| `--model` | LLM model | env or prompt |
| `--base-url` | LM Studio base URL | `LMSTUDIO_BASE_URL` or default |
| `--temperature` | LLM sampling temperature | `0.2` |
| `--max-tokens` | LLM max tokens | `2048` |
| `--web` | Enable tech + literature retrieval | off (enabled implicitly if no web flags and not `--no-web`) |
| `--web-tech` | Enable tech/news retrieval | off |
| `--web-lit` | Enable literature retrieval | off |
| `--no-web` | Disable web retrieval | off |
| `--ingest` / `--ingest-seeds` | Ingest `data/seeds/` before retrieval (user-provided) | off |
| `--auto-interest` | Add topic to interests table | off |
| `--graph-semantic-labels` | Force LLM-based cluster labels on | on by default |
| `--no-graph-semantic-labels` | Disable LLM-based cluster labels | off |
| `--graph-top-clusters` | Clusters used for retrieval | `10` |
| `--web-max-results` | Max results per source | `5` |
| `--web-timeout` | Web fetch timeout (seconds) | `15` |
| `--web-no-fetch` | Skip fetching full pages | off |
| `--web-hybrid` | Force semantic reranking on | on by default when web enabled |
| `--web-no-hybrid` | Disable semantic reranking | off |
| `--web-embed-model` | Embedding model for reranking | auto-detect |
| `--web-embed-base-url` | Embedding base URL | `MINI_NEXEN_EMBED_BASE_URL` or `LMSTUDIO_BASE_URL` |
| `--web-embed-timeout` | Embedding timeout (seconds) | `--web-timeout` |
| `--web-no-expand` | Disable LLM query expansion | off |
| `--web-max-queries` | Max expanded queries | `10` |
| `--web-max-new` | Max new sources added per run | `200` |
| `--web-max-per-query` / `--web-max-per-interest` | Max results per query seed | `10` |
| `--web-relevance-threshold` | Filter threshold when reranking | `0.25` |
| `--no-model-discovery` | Disable LM Studio model discovery | off |
| `--verbose` | Echo log events to stdout | off |
| `--quiet` | Disable log echoing | off |

`ingest` adds local content to the library.
| Flag | Meaning | Default |
| --- | --- | --- |
| `--file` | Path to `.txt`, `.md`, `.markdown`, `.pdf`, `.docx` | required (one of file/url/text) |
| `--url` | Record a URL (no fetch unless `--text` provided) | optional |
| `--text` | Inline content for URL or personal note | optional |
| `--title` | Custom title | inferred |
| `--tags` | Comma-separated tags | none |

`interest` records an interest.
- Use positional text or `--topic`.

`method` records an analysis method.
- Use positional text or `--method`.

Other commands:
- `list-docs`, `list-interests`, `list-methods`
- `delete-interest --id <ID>`
- `delete-method --id <ID>`
- `clear-interests --yes`
- `clear-methods --yes`
- `clear-library --yes` (also deletes stored files in `data/library/`)

**Defaults in `mini_nexen/config.py`**
These are library-level defaults and can be changed in code. CLI defaults may override some of them.
- `DEFAULT_TOP_K = 3`
- `DEFAULT_MIN_WEB_DOCS = 10`
- `DEFAULT_ROUNDS = 2`
- `DEFAULT_CHUNK_SIZE = 500`
- `DEFAULT_CHUNK_OVERLAP = 100`
- `GRAPH_REBUILD_RATIO = 0.15`
- `GRAPH_NOISE_RATIO_THRESHOLD = 0.35`
- `GRAPH_AVG_SIMILARITY_THRESHOLD = 0.2`
- `GRAPH_UNASSIGNED_RATIO_THRESHOLD = 0.4`
- `GRAPH_ASSIGN_SIMILARITY_MIN = 0.3`
- `GRAPH_TOP_CLUSTERS = 10`
- `WEB_MAX_NEW_SOURCES = 200`
- `WEB_MAX_PER_QUERY = 10`
- `WEB_RELEVANCE_THRESHOLD = 0.25`
- `WEB_DECAY_PER_RUN = 0.95`
- `WEB_ARCHIVE_SCORE_THRESHOLD = 0.2`
- `WEB_ARCHIVE_RUNS_UNUSED = 20`

To modify these defaults:
- Use CLI flags for per-run changes.
- Use env vars for provider/model/logging changes.
- Edit `mini_nexen/config.py` for graph and retrieval constants not exposed via CLI.

**Plan Output Structure**
The generated markdown includes:
- Scope, key questions, keywords, gaps, and notes.
- Source type summaries for selected docs and the entire library.
- Manually added interests and locally extracted theme clusters.
- A numbered research outline.

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
