from __future__ import annotations

import argparse
import json

from .agent import ChatDKUAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="ChatDKU mini bilingual RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the local index")
    ask_parser.add_argument("question", help="Question to answer")
    ask_parser.add_argument("--lang", choices=["en", "zh"], default="en")
    ask_parser.add_argument("--allow-internet", action="store_true")

    args = parser.parse_args()

    if args.command == "ask":
        agent = ChatDKUAgent()
        result = agent.answer(args.question, language=args.lang, allow_internet=args.allow_internet)
        print(result.answer)
        print("\nSources:")
        print(json.dumps(result.sources, ensure_ascii=False, indent=2))
        print("\nTool trace:")
        print(json.dumps(result.tool_trace, ensure_ascii=False))


if __name__ == "__main__":
    main()
