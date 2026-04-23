from __future__ import annotations

import json
from pathlib import Path

from chatdku_rag.ingest import load_corpus
from chatdku_rag.retrievers import HybridSearcher, KeywordSearcher, VectorSearcher


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    corpus = load_corpus(root / "data" / "index.json")
    examples = json.loads((root / "data" / "eval" / "sample_eval.json").read_text(encoding="utf-8"))

    searchers = {
        "keyword": KeywordSearcher(corpus),
        "vector": VectorSearcher(corpus),
        "hybrid": HybridSearcher(corpus),
    }

    for name, searcher in searchers.items():
        hits = 0
        for row in examples:
            results = searcher.search(row["question"], limit=3)
            documents = {hit.chunk.doc_name for hit in results}
            if row["expected_document"] in documents:
                hits += 1
        total = len(examples)
        print(f"{name}: {hits}/{total} = {hits / total:.2%}")


if __name__ == "__main__":
    main()
