from __future__ import annotations

import json
import os
import time
from pathlib import Path

from chatdku_rag.agent import ChatDKUAgent
from chatdku_rag.ingest import load_corpus
from chatdku_rag.models import EvalExample
from chatdku_rag.retrievers import HashEmbedder, HybridSearcher, KeywordSearcher, SentenceTransformerEmbedder, VectorSearcher


def load_examples(path: Path) -> list[EvalExample]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [EvalExample(**row) for row in rows]


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
        started = time.perf_counter()
        result = agent.answer(row.question, language=row.language, allow_internet=False)
        _ = time.perf_counter() - started
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
    root = Path(__file__).resolve().parents[1]
    corpus = load_corpus(root / "data" / "index.json")
    examples = load_examples(root / "data" / "eval" / "sample_eval.json")

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

    if os.getenv("CHATDKU_LLM_BASE_URL") and os.getenv("CHATDKU_LLM_MODEL"):
        agent = ChatDKUAgent(embedding_model_name="BAAI/bge-small-en-v1.5")
        started = time.perf_counter()
        retrieval, answer = evaluate_agent(agent, examples)
        latency_ms = ((time.perf_counter() - started) / len(examples)) * 1000
        label = f"dspy+{os.getenv('CHATDKU_LLM_MODEL')}+bge-small-en-v1.5"
        print(f"{label}: retrieval={retrieval:.2%}, answer={answer:.2%}, avg_latency_ms={latency_ms:.1f}")
        markdown_rows.append(
            {
                "label": label,
                "retrieval": f"{retrieval:.2%}",
                "answer": f"{answer:.2%}",
                "latency_ms": f"{latency_ms:.1f}",
            }
        )

    write_markdown_report(root / "results" / "empirical_eval.md", markdown_rows)


if __name__ == "__main__":
    main()
