# Multi-Agent Generative AI Pipeline

An autonomous multi-agent LLM pipeline for schema-agnostic, data-conditioned
generation with targeted iterative refinement. It infers the schema from the
input data at runtime, extracts the conventions that data follows, generates
new records matching them, validates the result against ground truth measured
from the source, and repairs what it can before deciding whether a retry is
needed.

Three run modes are included:

- **Full pipeline** — the 8-agent system (`main.py`)
- **Single-agent baseline** — one LLM call, no RAG, no repair, no retry (`main.py --baseline`)
- **Naïve-retry ablation** — full pipeline with Agent 7's targeted retry replaced by a naïve full-batch restart, for RQ3-style comparisons (`main_ablation.py`)

## Project Structure

```
multi_agent_pipeline/
├── main.py                      CLI entry point — full pipeline & baseline
├── main_ablation.py             CLI entry point — naive-retry ablation
├── config.py                    Settings, backends, and model profiles
├── requirements.txt             Python dependencies
├── setup_and_run.sh             One-shot setup script (macOS/Linux)
├── api_server.py                FastAPI REST server
├── .env.example                 Environment variable template
│
├── agents/                      The 8 pipeline agents
│   ├── __init__.py
│   ├── base_agent.py            Abstract base class shared by all agents
│   ├── agent1_ingestion.py      Data ingestion + RAG chunking
│   ├── agent2_schema.py         Schema inspection (cached)
│   ├── agent3_tasks.py          Task recognition from user instructions
│   ├── agent4_features.py       FAISS retrieval + ground-truth extraction
│   ├── agent5_generator.py      Batched generation + deterministic repair
│   ├── agent6_validator.py      Rule tier + LLM-judge validation
│   ├── agent7_retry.py          Targeted retry with stall detection
│   ├── agent7_naive_retry.py    Naive full-restart retry (ablation only)
│   ├── agent8_writer.py         Output + audit log writer
│   └── agent8_writer_ablation.py  Ablation-variant writer (naive_retry_run_*)
│
├── baseline/                    Single-agent baseline arm
│   ├── __init__.py
│   ├── single_agent_baseline.py     Baseline generator + shared scoring
│   └── single_agent_generator.py    Standalone baseline generator
│
├── core/                        Orchestration and shared infrastructure
│   ├── __init__.py
│   ├── pipeline.py              Full 8-agent orchestrator
│   ├── ablation_pipeline.py     Naive-retry ablation orchestrator
│   ├── baseline_pipeline.py     Single-agent baseline orchestrator
│   ├── llm_client.py            Unified client for all backends
│   └── message_bus.py           Shared pub/sub context store
│
├── data/                        Input datasets
│   ├── all_US_recipes_combined.json
│   └── BG_2026.04.22.recipe.en.json
│
├── docker/
│   ├── docker-compose.yaml
│   └── Dockerfile
│
├── logs/                        Runtime logs (created automatically)
├── outputs/                     Pipeline outputs (created automatically)
├── temp/                        Scratch space for uploaded files
│
├── tools/
│   ├── __init__.py
│   └── python_runtime.py        Sandboxed Python execution helper
│
├── utils/
│   ├── __init__.py
│   ├── file_loader.py           JSON/CSV/XLSX/PDF/DOCX loader + RAG sampling
│   └── logger.py                Logging setup (loguru + rich)
│
└── README.md
```

## Setup

### macOS / Linux

```bash
git clone <repository-url>
cd multi_agent_pipeline
chmod +x setup_and_run.sh
./setup_and_run.sh --setup-only
```

Manual equivalent:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then open `.env` and add the API key(s) for any hosted backend you plan to
use (Ollama needs no key).

### Windows

```powershell
git clone <repository-url>
cd multi_agent_pipeline
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Then open `.env` in a text editor and add the API key(s) for any hosted
backend you plan to use.

`setup_and_run.sh` is a bash script and does not run natively on Windows.
Use the manual steps above, or run the script under WSL or Git Bash.

### Ollama (local backend, optional)

If you plan to use the local Ollama backend, install it separately from
[ollama.com](https://ollama.com), then pull a model:

```bash
ollama pull qwen2.5:7b-instruct
```

## Running

All three modes are invoked from the same two entry points. Run
`python main.py --list-model-profiles` to see every available model profile
name before choosing one.

### 1. Full pipeline (8 agents)

```bash
python main.py \
  --input "data/BG_2026.04.22.recipe.en.json" \
  --context "Use the provided JSON file of Bulgarian recipes and generate 15 new different unique Bulgarian breakfast cuisine recipes in JSON file format which follows the input JSON data schema. Avoid the duplicate recipes." \
  --model-profile "qwen2.5:7b-instruct"
```

```bash
python main.py \
  --input "data/all_US_recipes_combined.json" \
  --context "Use the provided JSON file of US recipes and generate 15 new different unique Italian breakfast cuisine recipes in JSON file format which follows the input JSON data schema. Avoid the duplicate recipes." \
  --model-profile "Qwen/Qwen3-8B"
```

### 2. Single-agent baseline

Add `--baseline` to run the same task through the one-shot baseline arm
instead of the full pipeline. It is scored by the same validator against the
same ground truth, so its `audit_log.json` is directly comparable to a
full-pipeline run on the same input and context.

```bash
python main.py \
  --input "data/BG_2026.04.22.recipe.en.json" \
  --context "Use the provided JSON file of Bulgarian recipes and generate 15 new different unique Bulgarian breakfast cuisine recipes in JSON file format which follows the input JSON data schema. Avoid the duplicate recipes." \
  --model-profile "qwen2.5:7b-instruct" \
  --baseline
```

### 3. Naïve-retry ablation

Uses the separate `main_ablation.py` entry point. Agent 7's targeted retry
logic is replaced with a naïve full-batch restart on every failure; every
other agent is unmodified.

```bash
python main_ablation.py \
  --input "data/BG_2026.04.22.recipe.en.json" \
  --context "Use the provided JSON file of Bulgarian recipes and generate 15 new different unique Bulgarian breakfast cuisine recipes in JSON file format which follows the input JSON data schema. Avoid the duplicate recipes." \
  --model-profile "qwen2.5:7b-instruct"
```

```bash
python main_ablation.py \
  --input "data/all_US_recipes_combined.json" \
  --context "Use the provided JSON file of US recipes and generate 15 new different unique Italian breakfast cuisine recipes in JSON file format which follows the input JSON data schema. Avoid the duplicate recipes." \
  --model-profile "qwen2.5:7b-instruct"
```

All three modes write to `outputs/<prefix>_<timestamp>_<run_id>/` as
`output.json` plus `audit_log.json` (verdict, retry count, per-agent
timings). The prefix is `run_` for the full pipeline, `baseline_run_` for
the baseline, and `naive_retry_run_` for the ablation.

## CLI Reference

```
--input, -i             Input data file (JSON/CSV/XLSX/TXT/PDF/DOCX)
--context, -c           Plain-language task instructions
--model-profile         Named profile — sets backend + model together
--backend                ollama | anthropic | huggingface | groq | nvidia
--model, -m              Override the model directly for the active backend
--baseline               Run the single-agent baseline instead of the pipeline
--list-model-profiles    Print all available model profiles and exit
```

`main_ablation.py` uses `--input`, `--context`, and `--model-profile` only.

## Agents

| # | Agent | Responsibility |
|---|-------|-----------------|
| 1 | Data Ingestion | Loads JSON/CSV/XLSX/TXT/PDF/DOCX; stratified sampling for large files |
| 2 | Schema Inspector | Infers entity type and domain (cached; bypassed on deep retry) |
| 3 | Task Recognizer | Parses the instruction into a task spec (type, format, quantity, constraints) |
| 4 | Feature Extractor | RAG (FAISS + sentence-transformers) plus programmatic ground-truth extraction |
| 5 | Output Generator | Batched generation with deterministic repair before validity checks |
| 6 | Quality Validator | Deterministic rule checks, then an LLM judge filtered against ground truth |
| 7 | Increment Retry | Chooses between a simple retry and a deep retry, with stall detection |
| 8 | Output Writer | Writes `output.json` and `audit_log.json` to a timestamped run directory |

Agents share state through `core/message_bus.py`. `core/pipeline.py` runs the
generation/validation loop until validation passes or `MAX_RETRIES` is
reached, then force-writes the best available result.

## Backends

| Backend | Requirement |
|---------|-------------|
| Ollama | Local, no API key. Default. |
| Hugging Face | `HUGGINGFACE_API_TOKEN` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Groq | `GROQ_API_KEY` |
| NVIDIA NIM | `NVIDIA_API_KEY` |

API keys are read from the environment only. Copy `.env.example` to `.env`
and set them there — never commit `.env`.

## REST API

```bash
python api_server.py
```

Runs on `http://localhost:8000`, docs at `/docs`.

## Docker

```bash
docker compose -f docker/docker-compose.yaml up --build
docker compose -f docker/docker-compose.yaml up pipeline
```

## Requirements

- Python 3.10+ (3.12 recommended)
- ~4 GB disk for dependencies (`sentence-transformers`, `faiss-cpu`)
- GPU optional; hosted backends need none
- `huggingface_hub>=0.29` required for the Hugging Face backend

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| `Cannot verify Ollama models` | Ollama isn't running |
| Truncated or unparseable generations | Raise `AGENT5_BATCH_TOKENS_CEILING` in `.env` |
| Force-written output at max retries | See `validation_verdict.critical_issues` in `audit_log.json` |
| Slow first run | `sentence-transformers` downloads its embedding model once |
