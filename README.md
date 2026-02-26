# mini-NEXEN

Minimal, local-first research backend for generating research plans and detailed outlines from a personal library of documents and URLs.

**What It Does**
- Maintains a personal library of documents, URLs, and notes in SQLite.
- Builds a knowledge graph (entities, relations, claims, evidence) from local + web sources.
- Extracts a user profile (interest/intent/focus/attention) from local docs.
- Retrieves evidence with KG subgraph expansion and optional web retrieval.
- Generates a multi-round research plan and a long-form outline grounded in evidence.

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
Web retrieval is **on by default**. Disable it with `--no-web` or narrow it with `--web-open`, `--web-forum`, or `--web-lit`.

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
2. Run a research pass (mount `data/` and `artifacts/` so outputs persist).
```bash
docker run --rm -it \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -e GEMINI_API_KEY="your-key" \
  -e BRAVE_SEARCH_API_KEY="your-key" \
  -e TAVILY_API_KEY="your-key" \
  -e X_API_BEARER_TOKEN="your-key" \
  -e REDDIT_CLIENT_ID="your-id" \
  -e REDDIT_CLIENT_SECRET="your-secret" \
  -e REDDIT_USER_AGENT="mini-nexen/0.1" \
  mini-nexen research --topic "Agentic research systems"
```
Example (equivalent to local CLI):
```bash
docker run --rm -it \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -e GEMINI_API_KEY="your-key" \
  -e BRAVE_SEARCH_API_KEY="your-key" \
  -e TAVILY_API_KEY="your-key" \
  -e X_API_BEARER_TOKEN="your-key" \
  -e REDDIT_CLIENT_ID="your-id" \
  -e REDDIT_CLIENT_SECRET="your-secret" \
  -e REDDIT_USER_AGENT="mini-nexen/0.1" \
  mini-nexen research --topic "AI agent 驱动的 workflow 最近技术分析报告" --web-lit
```
Notes:
- For LM Studio running on the host, use `--network=host` (Linux) or set `LMSTUDIO_BASE_URL` to `http://host.docker.internal:1234/v1`.
- If you want interactive provider/model prompts, keep `-it`.
- `--review-query` requires an editor inside the container; set `MINI_NEXEN_QUERY_EDITOR` to an available editor or skip review.

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
- Edit `.env` to set Gemini/LM Studio keys, Brave Search API key, and optional `MINI_NEXEN_QUERY_EDITOR`.
- Data and artifacts are persisted to `./data` and `./artifacts`.

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
- `MINI_NEXEN_LLM_TIMEOUT`: per-request timeout in seconds (default `60`)

Web retrieval env vars:
- `BRAVE_SEARCH_API_KEY`: Enables API-based open-web search via Brave Search API.
- `TAVILY_API_KEY`: Tavily Search API key.
- `X_API_BEARER_TOKEN`: X API bearer token for recent search.
- `REDDIT_CLIENT_ID`: Reddit app client ID for search.
- `REDDIT_CLIENT_SECRET`: Reddit app client secret for search.
- `REDDIT_USER_AGENT`: User agent string for Reddit API requests.

Web retrieval modes:
- `open`: Open-web search (Brave/Tavily). `--web-open` (alias: `--web-tech`).
- `forum`: Forum/social sources (X, Reddit). `--web-forum`.
- `lit`: Literature sources (arXiv, Semantic Scholar, Crossref). `--web-lit`.
- Aliases accepted in query artifacts or programmatic calls: `web` and `tech` map to `open`, `literature` maps to `lit`.

Web retrieval behavior:
- Always-on by default unless `--no-web` is set.
- `--web-auto`: Run web retrieval only when KG expansion criteria are met.

Query review env vars:
- `MINI_NEXEN_QUERY_EDITOR`: Editor command to open the query artifact (defaults to `code --wait`).

Skill hints (in query artifact):
- `predicted_skills` is display-only.
- `skill_hints` activates skills and accepts skill_id, display_name, aliases, or catalog index.

LLM defaults:
- Gemini default model: `gemini-2.5-flash`
- LM Studio default model: `local-model` (if you set `MINI_NEXEN_MODEL` to `your-local-model` and keep discovery on, the client will auto-detect a model)
- Temperature default: `0.2`
- Max tokens default: `2048`

Embedding configuration (used by web reranking and KG extraction support):
- `MINI_NEXEN_EMBED_PROVIDER`: `gemini` or `lmstudio` (optional)
- `MINI_NEXEN_EMBED_MODEL`: embedding model name (optional)
- `MINI_NEXEN_EMBED_BASE_URL`: embedding endpoint (optional)
- `LMSTUDIO_BASE_URL`: fallback embedding endpoint when provider is `lmstudio`

Logging configuration:
- `MINI_NEXEN_VERBOSE=1|0` sets default log echoing.
- Progress updates (lines with `(...)%`) overwrite in-place when stdout is a TTY.

**Data Locations**
- `data/mini_nexen.sqlite3`: metadata, interests, KG, and stats.
- `data/library/`: stored text copies of ingested content.
- `data/local files/`: optional batch folder for local documents ingested by the `ingest` command.
- `data/task_events.log`: structured log of retrieval/LLM/graph events.
- `artifacts/`: generated plans (`YYYY_MM_DD_HH_MM_plan.md`) and query understanding artifacts (`YYYY_MM_DD_HH_MM_query.md`). Query artifacts include predicted skills (display-only) and `skill_hints` for forced activation.

**Pipeline (Research Command)**
1. Resolve LLM config from flags/env/interactive prompts.
2. Ensure data directories and initialize the SQLite DB.
3. Increment the research run counter.
4. Decay and archive existing web documents based on usage and relevance.
5. Infer topic + analysis methodologies from the query and save a reviewable query-understanding artifact (includes predicted skills and `skill_hints` overrides).
6. Start planning rounds (default 2).
7. Load stored interests, methods, and local profile signals.
8. If the systems-engineering skill triggers on keywords, inject its guidance into the planning context.
9. Retrieve a KG subgraph (default 2 hops) using topic + interests + profile signals + query hints, then build KG fact cards and provenance stats.
10. If web retrieval is enabled (default) and KG signals indicate gaps, run retrieval to expand the KG with new web sources.
11. Detect contradictions in the KG when LLM is configured.
12. Draft/refine the plan and build the outline using KG fact cards and the KG source index for provenance.
13. Save `artifacts/YYYY_MM_DD_HH_MM_plan.md` and log summary stats.

**Retrieval & KG Details**
- KG entities are deduped by canonical name; claims are deduped by normalized text.
- Relations store predicates + confidence; evidence links relations/claims back to sources.
- Contradiction detection flags claim pairs with shared subject/predicate and conflicting objects.
- KG subgraph retrieval seeds from query + profile and expands 2 hops (default) to fetch evidence docs.

**KG Visualization**
Export a subgraph as DOT (Graphviz) or HTML (interactive):
```bash
python -m mini_nexen.cli kg-export-dot --seed "agentic science" --hops 2 --out ./kg.dot
python -m mini_nexen.cli kg-export-html --seed "agentic science" --hops 2 --out ./kg.html
```
If you omit `--seed`, it uses top profile terms or top subject entities as seeds.
To render DOT:
```bash
dot -Tpng kg.dot -o kg.png
```
Open `kg.html` in a browser to explore the interactive graph.

**CLI Reference**

`research` generates a plan and outline.

KG utilities:
```bash
python -m mini_nexen.cli kg-report --limit 10
python -m mini_nexen.cli kg-export-dot --seed "agentic science" --hops 2
python -m mini_nexen.cli kg-export-html --seed "agentic science" --hops 2
```


Common behavior:
- If no provider/model is configured and stdin is not a TTY, the command exits with an error.
- If no `--web*` flags are given, web retrieval runs with `open + forum + lit` by default.

**KG Reporting**
```bash
python -m mini_nexen.cli kg-report --limit 10
```

Research flags:
| Flag | Meaning | Default |
| --- | --- | --- |
| `--topic` | Research topic | required |
| `--rounds` | Planning rounds | `2` |
| `--top-k` | Doc limit hint for plan metadata | `3` |
| `--kg-hops` | KG subgraph hops used for planning | `2` |
| `--provider` | LLM provider | env or prompt |
| `--model` | LLM model | env or prompt |
| `--base-url` | LM Studio base URL | `LMSTUDIO_BASE_URL` or default |
| `--temperature` | LLM sampling temperature | `0.2` |
| `--max-tokens` | LLM max tokens | `2048` |
| `--web` | Enable open + forum + literature retrieval | off (enabled implicitly if no web flags and not `--no-web`) |
| `--web-open` | Enable open-web retrieval (Brave/Google/Tavily) | off |
| `--web-forum` | Enable forum retrieval (X/Reddit) | off |
| `--web-lit` | Enable literature retrieval | off |
| `--no-web` | Disable web retrieval | off |
| `--review-query` | Pause to review inferred query understanding | auto (TTY) |
| `--no-review-query` | Skip the query understanding review step | off |
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

`ingest` adds local content to the library, updates the local KG, and rebuilds the user profile when local KG changes.
Duplicate sources are skipped automatically.
Because it rebuilds the KG, it requires LLM configuration (provider/model + API key when applicable).
The command also scans `data/local files/` and ingests any new files found there.
If you pass no `--file/--url/--text`, it only processes `data/local files/`.
| Flag | Meaning | Default |
| --- | --- | --- |
| `--file` | Path to `.txt`, `.md`, `.markdown`, `.pdf`, `.docx` (repeatable) | required (one of file/url/text) |
| `--url` | Record a URL (no fetch unless `--text` provided) | optional |
| `--text` | Inline content for URL or personal note | optional |
| `--title` | Custom title | inferred |
| `--tags` | Comma-separated tags | none |
| `--provider` | LLM provider for KG/profile extraction | env or prompt |
| `--model` | LLM model for KG/profile extraction | env or prompt |
| `--base-url` | LM Studio base URL | `LMSTUDIO_BASE_URL` or default |
| `--temperature` | LLM sampling temperature | `0.2` |
| `--max-tokens` | LLM max tokens | `2048` |
| `--no-model-discovery` | Disable LM Studio model discovery | off |

`interest` records an interest.
- Use positional text or `--topic`.

`method` records an analysis method.
- Use positional text or `--method`.

Other commands:
- `list-docs`, `list-interests`, `list-methods`
- `delete-interest --id <ID>`
- `delete-method --id <ID>`
- `clear-interests`
- `clear-methods`
- `clear-library` (also deletes stored files in `data/library/`)

**Defaults in `mini_nexen/config.py`**
These are library-level defaults and can be changed in code. CLI defaults may override some of them.
- `DEFAULT_TOP_K = 3`
- `DEFAULT_ROUNDS = 2`
- `DEFAULT_KG_HOPS = 2`
- `DEFAULT_PROFILE_TOP_K = 10`
- `WEB_MAX_NEW_SOURCES = 200`
- `WEB_MAX_PER_QUERY = 10`
- `WEB_RELEVANCE_THRESHOLD = 0.25`
- `WEB_DECAY_PER_RUN = 0.95`
- `WEB_ARCHIVE_SCORE_THRESHOLD = 0.2`
- `WEB_ARCHIVE_RUNS_UNUSED = 20`

To modify these defaults:
- Use CLI flags for per-run changes.
- Use env vars for provider/model/logging changes.
- Edit `mini_nexen/config.py` for planning and retrieval constants not exposed via CLI.

**Plan Output Structure**
The generated markdown includes:
- Scope, key questions, keywords, gaps, and notes.
- Source type summaries for selected docs and the entire library.
- Manually added interests and KG evidence summaries.
- A numbered research outline.

**KG & Debugging**
Check KG summary:
```bash
python -m mini_nexen.cli kg-report
```

Use `kg-report` plus the KG export commands (`kg-export-dot`, `kg-export-html`) to inspect evidence density and provenance.
