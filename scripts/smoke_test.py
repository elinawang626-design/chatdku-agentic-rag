from __future__ import annotations

from pathlib import Path

from chatdku_rag.agent import ChatDKUAgent


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    index_path = root / "data" / "index.json"
    if not index_path.exists():
        raise SystemExit("Missing data/index.json. Run `python scripts/ingest.py` first.")

    agent = ChatDKUAgent(index_path=index_path, embedding_model_name="hash")
    result = agent.answer("How many credits do I need to graduate?", language="en")
    if not result.sources:
        raise SystemExit("Smoke test failed: no sources returned.")
    if "136" not in result.answer and "credit" not in result.answer.lower():
        raise SystemExit("Smoke test failed: graduation-credit answer was not grounded.")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
