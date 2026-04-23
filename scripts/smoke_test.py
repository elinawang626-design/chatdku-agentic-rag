from pathlib import Path

from chatdku_rag.agent import ChatDKUAgent


def main() -> None:
    index_path = Path(__file__).resolve().parents[1] / "data" / "index.json"
    if not index_path.exists():
        raise SystemExit("Missing data/index.json. Run `python scripts/ingest.py` first.")

    agent = ChatDKUAgent(index_path=index_path)
    result = agent.answer("How many credits do I need to graduate?", language="en")
    if not result.sources:
        raise SystemExit("Smoke test failed: no sources returned.")
    if "136" not in result.answer and "credit" not in result.answer.lower():
        raise SystemExit("Smoke test failed: answer does not look grounded in graduation-credit policy.")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
