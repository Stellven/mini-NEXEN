# mini-NEXEN

Minimal, agentic research backend for generating research plans and detailed outlines from a personal library of documents and URLs.

## What this does
- Maintains a personal library of documents, URLs, and notes.
- Tracks user interests (NotebookLM-style memory).
- Runs a multi-agent pipeline driven by skills.
- Generates a research plan + detailed outline and saves it as `YYYY_MM_DD_HH_MM_plan.md`.
- Uses LLM-backed planning and outlining (Gemini or LM Studio required).

## Structure
- `mini_nexen/`: Core logic (agents, planning, storage).
- `skills/`: Oh-my-opencode-style skills (`SKILL.md` per skill).
- `data/`: SQLite database and stored documents.
- `data/library/`: Stored text copies of ingested content.
- `data/seeds/`: Seed documents and URLs for quick bootstrapping.
- `plans/`: Generated research plans.

## Environment
The project expects a conda environment named `mini-nexen`. A local environment file is provided.

```bash
conda env create -f environment.yml
conda activate mini-nexen
```

If your environment cannot access external channels, you may need to configure local channels or mirrors before creating the env.

Installed dependencies (via pip in `mini-nexen`):
- `requests`
- `google-genai`

LM Studio is required only if you choose the `lmstudio` provider.
Gemini requires an API key (`GEMINI_API_KEY` or `GOOGLE_API_KEY`).

## Usage
Ingest documents and URLs:

```bash
python -m mini_nexen.cli ingest --file /path/to/doc.txt --title "Doc title" --tags "ml, research"
python -m mini_nexen.cli ingest --url "https://example.com/report" --title "Report" --text "Key excerpt"
python -m mini_nexen.cli ingest --text "Personal note about the topic" --title "My notes"
```

Record interests:

```bash
python -m mini_nexen.cli interest --topic "Agentic research" --notes "Focus on planning and retrieval"
```

Generate a research plan:

```bash
python -m mini_nexen.cli research --topic "Agentic research systems" --rounds 2 --top-k 8
```

Enable internet retrieval (tech/blog/news/forums + literature):

```bash
python -m mini_nexen.cli research --topic "Agentic research systems" --web --rounds 2 --top-k 8
```

Internet retrieval options:
- `--web` enables both tech and literature sources.
- `--web-tech` enables tech/blog/news/forums sources only.
- `--web-lit` enables literature sources only.
- `--web-max-results` controls max results per source (default: 5).
- `--web-timeout` sets the fetch timeout in seconds (default: 15).
- `--web-no-fetch` skips fetching full pages (stores only metadata/abstracts when available).
- `--web-hybrid` forces semantic reranking on.
- `--web-no-hybrid` disables semantic reranking.
- Semantic reranking defaults to on when web retrieval is enabled and uses LM Studio embeddings if available.
- `--web-embed-model` sets the embedding model (auto-detected from LM Studio if omitted).
- `--web-embed-base-url` sets the embedding endpoint (defaults to `LMSTUDIO_BASE_URL`).
- `--web-embed-timeout` sets the embedding timeout in seconds.
- `--web-no-expand` disables query expansion (enabled by default).
- `--web-max-queries` sets the max expanded queries (default: 4).

Current sources:
- Tech/web: DuckDuckGo HTML results.
- Literature: arXiv, Semantic Scholar, Crossref.

Outputs:
- Plan + outline saved in `plans/YYYY_MM_DD_HH_MM_plan.md` (outline is under `## Detailed Outline`).
- Ingested document text stored in `data/library/<doc_id>.txt`.
- Metadata and interests stored in `data/mini_nexen.sqlite3`.
- LLM call log written to `data/llm_calls.log` (task boundaries, per-agent messages, rate limit events).

### LLM-backed agents
The planner and outliner run with an LLM. Choose either Gemini or LM Studio.

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
- If you are running in WSL and LM Studio is on Windows, enable “Serve on Local Network”
  in LM Studio and use the Windows IP address in `LMSTUDIO_BASE_URL`.

WSL convenience (auto-detect Windows host IP from the WSL default gateway):
```bash
export LMSTUDIO_BASE_URL="http://$(ip route | awk '/default/ {print $3}'):1234/v1"
```

Persist that setting for new shells:
```bash
echo 'export LMSTUDIO_BASE_URL="http://$(ip route | awk '\''/default/ {print $3}'\''):1234/v1"' >> ~/.bashrc
```

You can also override provider/model per run:
```bash
python -m mini_nexen.cli research --topic "Agentic research systems" --provider gemini --model gemini-2.5-flash
```

Additional CLI overrides: `--base-url` (LM Studio), `--temperature`, `--max-tokens`.
Log echoing is enabled by default. Use `--quiet` to disable.
Use `--verbose` to explicitly enable (it can be placed before or after the subcommand).
You can also set `MINI_NEXEN_VERBOSE=1` (or `0`) to control the default.

Embedding selection:
- When web retrieval is enabled, the CLI will prompt for an embedding model based on the selected provider.
- Gemini defaults to `gemini-embedding-001`.
- LM Studio defaults to auto-detecting an embedding model from `/v1/models`.
- LM Studio LLM selection uses `auto-detect` (internally `your-local-model`) or a custom model name.
LM Studio model discovery can be disabled with `--no-model-discovery`.

If no provider is set, the CLI prompts you to select a provider and model. If env vars are already set, it asks to confirm or change them.

Note: The system performs retrieval only from the local library (no internet fetching).

Outline format:
- Generated outlines are research plans (not report outlines).
- Target length is 1000-2000 words with 1-2 sub-layers under each major step.
  The length is a soft target; the system reports the final word count.

List stored data:

```bash
python -m mini_nexen.cli list-docs
python -m mini_nexen.cli list-interests
```

Manage interests:

```bash
python -m mini_nexen.cli delete-interest --id "<INTEREST_ID>"
python -m mini_nexen.cli clear-interests --yes
```

Interest IDs are UUIDs shown by `list-interests`.

Seed pack:

```bash
python -m mini_nexen.cli ingest --file data/seeds/agentic_research_overview.txt --tags "agentic,overview"
python -m mini_nexen.cli ingest --file data/seeds/retrieval_planning_notes.txt --tags "retrieval,planning"
python -m mini_nexen.cli ingest --file data/seeds/evaluation_metrics_cheatsheet.txt --tags "evaluation,metrics"
```

See `data/seeds/seed_urls.txt` for suggested URLs to ingest later with short excerpts.

## Skills
Skills live in `skills/<skill_name>/SKILL.md`. The agent runtime loads these at startup and only executes registered skills. This keeps the agent behavior explicit and auditable.
