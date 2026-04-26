# chatdku-agentic-rag

This repository implements a bilingual mini agentic RAG system for the ChatDKU AI Engineer candidate task over the provided advising document set.

## Implemented requirements

The current repository now includes all core building blocks requested in the PDF:

- document ingestion from PDF and DOCX
- chunking with source metadata and page numbers
- keyword search
- vector search with real embedding models
- internet search
- bilingual query interface (`en` and `zh`)
- source-grounded answers with document name and page number
- local LLM serving via `vLLM`
- DSPy-based routing and answer generation
- empirical evaluation across multiple embedding models and multiple locally served LLMs

## Repository layout

```text
chatdku-agentic-rag/
├── README.md
├── .gitignore
├── pyproject.toml
├── results/
│   ├── empirical_eval.md
│   └── local_llm_eval_observed.md
├── scripts/
│   ├── ingest.py
│   ├── run_eval.py
│   └── smoke_test.py
├── data/
│   └── eval/
│       ├── llm_eval_configs.example.json
│       └── sample_eval.json
└── src/
    └── chatdku_rag/
        ├── __init__.py
        ├── agent.py
        ├── cli.py
        ├── dspy_program.py
        ├── ingest.py
        ├── internet.py
        ├── llm.py
        ├── models.py
        ├── retrievers.py
        └── utils.py
```

## Installation

Core dependencies:

```bash
pip install -e .
```

For local model serving:

```bash
pip install -e .[serving]
```

## Ingest the document set

Build the local index from the provided candidate documents:

```bash
python scripts/ingest.py \
  --input "/path/to/Advising FAQ (12-19-24 Update).docx" \
  --input "/path/to/ug_bulletin_2025-2026.pdf"
```

This writes:

```text
data/index.json
```

The ingestion step is intentionally explicit. The repository does not hardcode any machine-specific absolute paths.

## Start a local vLLM server

Example with a small open model that fits on Apple silicon CPU:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key local \
  --generation-config vllm
```

Another tested comparison model:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key local \
  --generation-config vllm
```

## Configure DSPy + local LLM access

```bash
export CHATDKU_LLM_BASE_URL="http://127.0.0.1:8000/v1"
export CHATDKU_LLM_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
export CHATDKU_LLM_API_KEY="local"
```

## Run the system

English:

```bash
python -m chatdku_rag.cli ask \
  --lang en \
  --embedding-model BAAI/bge-small-en-v1.5 \
  "How many credits do I need to graduate?"
```

Chinese:

```bash
python -m chatdku_rag.cli ask \
  --lang zh \
  --embedding-model BAAI/bge-small-en-v1.5 \
  "休学期间可以在别的学校上课吗？"
```

## DSPy layer

The DSPy implementation lives in [src/chatdku_rag/dspy_program.py](src/chatdku_rag/dspy_program.py) and is used for:

- routing between local-only and internet-supplemented answering
- grounded answer generation from retrieved evidence

## Real embedding models tested

The repository now supports and evaluates:

- `BAAI/bge-small-en-v1.5`
- `sentence-transformers/all-MiniLM-L6-v2`

A lightweight `hash` backend remains available as a fallback baseline.

## Empirical evaluation

Run:

```bash
python scripts/run_eval.py \
  --input "/path/to/Advising FAQ (12-19-24 Update).docx" \
  --input "/path/to/ug_bulletin_2025-2026.pdf"
```

The latest regenerated retrieval, embedding, and local-LLM results are stored in [results/empirical_eval.md](results/empirical_eval.md).

### Embedding comparison

- `hybrid-hash`: 100.00% retrieval hit rate
- `hybrid-bge-small-en-v1.5`: 100.00% retrieval hit rate
- `hybrid-all-MiniLM-L6-v2`: 100.00% retrieval hit rate

### Local LLM comparison via vLLM

- `Qwen/Qwen2.5-0.5B-Instruct` with DSPy + `bge-small-en-v1.5`: 100.00% retrieval hit rate, 66.67% answer keyword hit rate
- `Qwen/Qwen2.5-1.5B-Instruct` with DSPy + `bge-small-en-v1.5`: 100.00% retrieval hit rate, 83.33% answer keyword hit rate

The 1.5B model was materially slower on CPU in this local Apple silicon environment.

To reproduce multiple local-LLM rows in one run, start separate OpenAI-compatible servers on different ports and pass a config file:

```bash
python scripts/run_eval.py \
  --input "/path/to/Advising FAQ (12-19-24 Update).docx" \
  --input "/path/to/ug_bulletin_2025-2026.pdf" \
  --llm-configs data/eval/llm_eval_configs.example.json
```

The earlier archived run notes are preserved in [results/local_llm_eval_observed.md](results/local_llm_eval_observed.md).

## Verification

```bash
python scripts/smoke_test.py \
  --input "/path/to/Advising FAQ (12-19-24 Update).docx" \
  --input "/path/to/ug_bulletin_2025-2026.pdf"
```

## Notes

- The project was tested on Apple silicon with `vLLM` CPU serving.
- The task PDF recommends larger models like `Qwen3-8B` and `DeepSeek-R1-Distill-Llama-8B`; on this local machine, smaller open models were used to complete the end-to-end local-serving requirement in a practical way.
- The retrieval and answer-generation stack is source-grounded over the provided local documents, including FAQ page numbers inferred from the document's table of contents.
