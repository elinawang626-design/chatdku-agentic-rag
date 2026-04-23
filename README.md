# chatdku-agentic-rag

This repository is a compact, interview-ready implementation of the ChatDKU candidate task built around the provided advising documents.

It focuses on:

- document ingestion from PDF and DOCX
- chunking with source and page metadata
- three tools: vector search, keyword search, and internet search
- a bilingual interface (`en` and `zh`)
- source-grounded answers with document names and page numbers
- a clean upgrade path to DSPy and local OpenAI-compatible LLM serving

## Current scope

This version is designed to be honest, runnable, and easy to explain in an interview setting.

Implemented now:

- `keyword search`: BM25-style scoring implemented locally
- `vector search`: deterministic hashed embedding search implemented locally
- `internet search`: optional DuckDuckGo HTML search through the Python standard library
- `agent`: routes across tools, builds grounded context, and formats a bilingual answer
- `LLM seam`: optional local OpenAI-compatible generation layer for final answer synthesis

Not yet implemented as hard dependencies:

- DSPy runtime modules
- real model-backed embeddings
- required local model serving with `vLLM` or `SGLang`

The repository is structured so these can be added without rewriting the retrieval pipeline.

## Why this design

This gives a working end-to-end mini RAG pipeline over the candidate documents while keeping the codebase small and inspectable. The project is also structured so you can later swap in:

- a real embedding model such as `BAAI/bge-small-en-v1.5`
- a local OpenAI-compatible LLM served by `SGLang` or `vLLM`
- real DSPy modules instead of the fallback orchestration path

## Repository layout

```text
chatdku-agentic-rag/
├── README.md
├── .gitignore
├── pyproject.toml
├── scripts/
│   ├── ingest.py
│   ├── run_eval.py
│   └── smoke_test.py
├── data/
│   └── eval/
│       └── sample_eval.json
└── src/
    └── chatdku_rag/
        ├── __init__.py
        ├── agent.py
        ├── cli.py
        ├── ingest.py
        ├── internet.py
        ├── llm.py
        ├── models.py
        ├── retrievers.py
        └── utils.py
```

## Starting point

The main entry point is:

```bash
python -m chatdku_rag.cli ask --lang en "How many credits do I need to graduate?"
```

Or in Chinese:

```bash
python -m chatdku_rag.cli ask --lang zh "毕业需要多少学分？"
```

## Dependencies

Minimum tested dependencies:

- `pypdf`
- `python-docx`

Optional dependencies for the full intended setup:

- `dspy`
- `vllm` or `sglang`
- `sentence-transformers`

Install example:

```bash
pip install -e .
```

If you want final answers synthesized by a locally served model, expose an OpenAI-compatible endpoint and set:

```bash
export CHATDKU_LLM_BASE_URL="http://localhost:8000/v1"
export CHATDKU_LLM_MODEL="Qwen/Qwen3-8B"
export CHATDKU_LLM_API_KEY="EMPTY"
```

## Ingest the candidate documents

The ingestion script can either use the original local file paths from the task or accept explicit paths:

- `/Users/elina/Downloads/ChatDKU Candidate Task Documents/Advising FAQ (12-19-24 Update).docx`
- `/Users/elina/Downloads/ChatDKU Candidate Task Documents/ug_bulletin_2025-2026.pdf`

Build the local index:

```bash
python scripts/ingest.py
```

Or pass files manually:

```bash
python scripts/ingest.py \
  --input "/path/to/Advising FAQ (12-19-24 Update).docx" \
  --input "/path/to/ug_bulletin_2025-2026.pdf"
```

This writes a serialized corpus to:

```text
data/index.json
```

## Run a query

English:

```bash
python -m chatdku_rag.cli ask --lang en "Can I take classes while on leave of absence?"
```

Chinese:

```bash
python -m chatdku_rag.cli ask --lang zh "休学期间可以在别的学校上课吗？"
```

To force internet search in the tool mix:

```bash
python -m chatdku_rag.cli ask --lang en --allow-internet "What is DKU?"
```

When the local LLM env vars are configured, the agent will use retrieved evidence to produce a more natural final answer. Otherwise it falls back to extractive synthesis.

## Evaluation

A small evaluation set is included to compare retrieval modes:

```bash
python scripts/run_eval.py
```

The script reports hit-rate style retrieval metrics for:

- keyword only
- vector only
- hybrid

Basic smoke test:

```bash
python scripts/smoke_test.py
```

## Suggested next upgrade

The fastest path to fully matching the original task spec is:

1. Serve a local instruction model with `vLLM` or `SGLang`
2. Replace hashed vectors with a real embedding model
3. Add DSPy routing / answer-generation modules
4. Record empirical comparisons across at least two LLMs and two embedding models
5. Add a minimal web UI for bilingual interaction

## How to upgrade this into the full task target

### 1. Real local LLM

Serve a model with `vLLM` or `SGLang`, then point the agent to an OpenAI-compatible endpoint. The project is already structured around a generation seam in `agent.py`.

Recommended models from the posting:

- `Qwen/Qwen3-8B`
- `meta-llama/Llama-3.2-3B-Instruct`
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`

### 2. Real embeddings

Replace the hashed vectorizer with a model-backed embedder such as:

- `BAAI/bge-small-en-v1.5`
- `Qwen/Qwen3-Embedding-0.6B`
- `google/embeddinggemma-300m`

### 3. DSPy

If `dspy` is available, you can swap the fallback planner/router with DSPy signatures or modules. The code includes a clean boundary for that integration instead of mixing orchestration into retrieval code.

## Notes on this version

- This version is fully source-grounded over the provided local docs.
- It is designed to be easy to explain in an interview.
- In the current environment, large-model serving and DSPy packages were not preinstalled, so the implementation keeps those as clean extension points rather than pretending they are already available.
- The generated local index is intentionally excluded from version control; rebuild it with `python scripts/ingest.py`.
