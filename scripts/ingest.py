import argparse
from pathlib import Path

from chatdku_rag.ingest import build_corpus, save_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local document index.")
    parser.add_argument(
        "--input",
        action="append",
        dest="inputs",
        default=None,
        help="Path to a PDF or DOCX file. May be passed multiple times.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    input_paths = [Path(value).expanduser() for value in args.inputs] if args.inputs else None
    chunks = build_corpus(input_paths)
    output_path = root / "data" / "index.json"
    save_corpus(chunks, output_path)
    print(f"Indexed {len(chunks)} chunks into {output_path}")


if __name__ == "__main__":
    main()
