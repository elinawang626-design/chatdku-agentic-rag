from __future__ import annotations

import argparse
from pathlib import Path

from chatdku_rag.agent import ChatDKUAgent
from chatdku_rag.ingest import build_corpus, save_corpus


def ensure_index(index_path: Path, inputs: list[Path]) -> None:
    if index_path.exists():
        return
    if not inputs:
        raise SystemExit(
            "Missing data/index.json. Run `python scripts/ingest.py --input ...` first or pass `--input` here."
        )
    chunks = build_corpus(inputs)
    save_corpus(chunks, index_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a basic end-to-end smoke test.")
    parser.add_argument(
        "--input",
        action="append",
        dest="inputs",
        default=[],
        help="Path to a source PDF or DOCX. Used to rebuild the index if missing.",
    )
    parser.add_argument(
        "--index",
        default=None,
        help="Optional path to an index file. Defaults to data/index.json.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    index_path = Path(args.index).expanduser() if args.index else root / "data" / "index.json"
    inputs = [Path(value).expanduser() for value in args.inputs]
    ensure_index(index_path, inputs)

    agent = ChatDKUAgent(index_path=index_path, embedding_model_name="hash")
    result = agent.answer("How many credits do I need to graduate?", language="en")
    if not result.sources:
        raise SystemExit("Smoke test failed: no sources returned.")
    if "136" not in result.answer and "credit" not in result.answer.lower():
        raise SystemExit("Smoke test failed: graduation-credit answer was not grounded.")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
