from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from chatdku_rag.agent import ChatDKUAgent
from chatdku_rag.ingest import build_corpus, load_corpus, save_corpus
from chatdku_rag.llm import OpenAICompatConfig
from chatdku_rag.models import EvalExample
from chatdku_rag.retrievers import HashEmbedder, HybridSearcher, KeywordSearcher, SentenceTransformerEmbedder, VectorSearcher


def load_examples(path: Path) -> list[EvalExample]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [EvalExample(**row) for row in rows]


def load_llm_configs(path: Path) -> list[OpenAICompatConfig]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [OpenAICompatConfig.from_dict(row) for row in rows]


def ensure_index(index_path: Path, inputs: list[Path]) -> None:
    if index_path.exists():
        return
    if not inputs:
        raise SystemExit(
            "Missing data/index.json. Run `python scripts/ingest.py --input ...` first or pass `--input` here."
        )
    chunks = build_corpus(inputs)
    save_corpus(chunks, index_path)


def evaluate_retriever(searcher, examples: list[EvalExample]) -> float:
    hits = 0
    for row in examples:
        results = searcher.search(row.question, limit=3)
        documents = {hit.chunk.doc_name for hit in results}
        if row.expected_document in documents:
            hits += 1
    return hits / len(examples)


def evaluate_agent(agent: ChatDKUAgent, examples: list[EvalExample]) -> tuple[float, float]:
    retrieval_hits = 0
    answer_hits = 0
    for row in examples:
        result = agent.answer(row.question, language=row.language, allow_internet=False)
        documents = {source["document"] for source in result.sources}
        if row.expected_document in documents:
            retrieval_hits += 1
        answer_lower = result.answer.lower()
        if all(keyword.lower() in answer_lower for keyword in row.expected_answer_keywords):
            answer_hits += 1
    return retrieval_hits / len(examples), answer_hits / len(examples)


def write_markdown_report(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = [
        "# Empirical Evaluation",
        "",
        "| Label | Retrieval Hit Rate | Answer Keyword Hit Rate | Avg Latency (ms) |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        output.append(
            f"| {row['label']} | {row['retrieval']} | {row['answer']} | {row['latency_ms']} |"
        )
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval and local-LLM evaluation.")
    parser.add_argument(
        "--input",
        action="append",
        dest="inputs",
        default=[],
        help="Path to a source PDF or DOCX. Used to rebuild the index if missing.",
    )
    parser.add_argument("--index", default=None, help="Optional path to an index file. Defaults to data/index.json.")
    parser.add_argument(
        "--examples",
        default=None,
        help="Optional path to the evaluation examples JSON. Defaults to data/eval/sample_eval.json.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional output report path. Defaults to results/empirical_eval.md.",
    )
    parser.add_argument(
        "--llm-configs",
        default=None,
        help="Optional JSON file containing multiple OpenAI-compatible model configs to evaluate sequentially.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    index_path = Path(args.index).expanduser() if args.index else root / "data" / "index.json"
    examples_path = Path(args.examples).expanduser() if args.examples else root / "data" / "eval" / "sample_eval.json"
    report_path = Path(args.report).expanduser() if args.report else root / "results" / "empirical_eval.md"
    inputs = [Path(value).expanduser() for value in args.inputs]

    ensure_index(index_path, inputs)
    corpus = load_corpus(index_path)
    examples = load_examples(examples_path)

    retrieval_configs = {
        "keyword": KeywordSearcher(corpus),
        "vector-hash": VectorSearcher(corpus, embedder=HashEmbedder()),
        "hybrid-hash": HybridSearcher(corpus, embedder=HashEmbedder()),
        "vector-bge-small-en-v1.5": VectorSearcher(
            corpus,
            embedder=SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5"),
        ),
        "hybrid-bge-small-en-v1.5": HybridSearcher(
            corpus,
            embedder=SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5"),
        ),
        "vector-all-MiniLM-L6-v2": VectorSearcher(
            corpus,
            embedder=SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2"),
        ),
        "hybrid-all-MiniLM-L6-v2": HybridSearcher(
            corpus,
            embedder=SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2"),
        ),
    }

    markdown_rows: list[dict[str, str]] = []
    for name, searcher in retrieval_configs.items():
        started = time.perf_counter()
        retrieval = evaluate_retriever(searcher, examples)
        latency_ms = ((time.perf_counter() - started) / len(examples)) * 1000
        print(f"{name}: retrieval={retrieval:.2%}, avg_latency_ms={latency_ms:.1f}")
        markdown_rows.append(
            {
                "label": name,
                "retrieval": f"{retrieval:.2%}",
                "answer": "N/A",
                "latency_ms": f"{latency_ms:.1f}",
            }
        )

    llm_configs: list[OpenAICompatConfig] = []
    if args.llm_configs:
        llm_configs.extend(load_llm_configs(Path(args.llm_configs).expanduser()))
    else:
        env_config = OpenAICompatConfig.from_env()
        if env_config is not None:
            llm_configs.append(env_config)

    for llm_config in llm_configs:
        embedding_model = os.getenv("CHATDKU_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        agent = ChatDKUAgent(
            index_path=index_path,
            embedding_model_name=embedding_model,
            llm_config=llm_config,
        )
        started = time.perf_counter()
        retrieval, answer = evaluate_agent(agent, examples)
        latency_ms = ((time.perf_counter() - started) / len(examples)) * 1000
        label = f"langchain+{llm_config.display_name()}+{embedding_model}"
        print(f"{label}: retrieval={retrieval:.2%}, answer={answer:.2%}, avg_latency_ms={latency_ms:.1f}")
        markdown_rows.append(
            {
                "label": label,
                "retrieval": f"{retrieval:.2%}",
                "answer": f"{answer:.2%}",
                "latency_ms": f"{latency_ms:.1f}",
            }
        )

    write_markdown_report(report_path, markdown_rows)


if __name__ == "__main__":
    main()
