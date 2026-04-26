import argparse
from pathlib import Path

from chatdku_rag.ingest import build_corpus, save_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local document index.")
    parser.add_argument(
        "--input",
        action="append",
        dest="inputs",
        default=[],
        help="Path to a PDF or DOCX file. May be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output index path. Defaults to data/index.json.",
    )
    args = parser.parse_args()

    if not args.inputs:
        raise SystemExit("Pass at least one `--input` path to build the index.")

    root = Path(__file__).resolve().parents[1]
    input_paths = [Path(value).expanduser() for value in args.inputs]
    output_path = Path(args.output).expanduser() if args.output else root / "data" / "index.json"
    chunks = build_corpus(input_paths)
    save_corpus(chunks, output_path)
    print(f"Indexed {len(chunks)} chunks into {output_path}")


if __name__ == "__main__":
    main()
