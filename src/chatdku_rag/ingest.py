from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from pypdf import PdfReader

from .models import Chunk
from .utils import normalize_whitespace


DEFAULT_INPUTS = [
    Path("/Users/elina/Downloads/ChatDKU Candidate Task Documents/Advising FAQ (12-19-24 Update).docx"),
    Path("/Users/elina/Downloads/ChatDKU Candidate Task Documents/ug_bulletin_2025-2026.pdf"),
]


def _chunk_text(doc_name: str, text: str, page: int | None, size: int = 1200, overlap: int = 150) -> list[Chunk]:
    clean = normalize_whitespace(text)
    if not clean:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(clean):
        end = min(len(clean), start + size)
        body = clean[start:end]
        if end < len(clean):
            last_space = body.rfind(" ")
            if last_space > size // 2:
                body = body[:last_space]
                end = start + last_space
        chunk_id = f"{doc_name}:{page or 0}:{index}"
        chunks.append(Chunk(chunk_id=chunk_id, doc_name=doc_name, page=page, text=body.strip()))
        if end >= len(clean):
            break
        start = max(start + 1, end - overlap)
        index += 1
    return chunks


def extract_pdf(path: Path) -> list[Chunk]:
    reader = PdfReader(str(path))
    chunks: list[Chunk] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        chunks.extend(_chunk_text(path.name, text, page_index))
    return chunks


def extract_docx(path: Path) -> list[Chunk]:
    doc = Document(str(path))
    text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    return _chunk_text(path.name, text, None)


def build_corpus(paths: list[Path] | None = None) -> list[Chunk]:
    inputs = paths or DEFAULT_INPUTS
    corpus: list[Chunk] = []
    for path in inputs:
        if not path.exists():
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            corpus.extend(extract_pdf(path))
        elif suffix == ".docx":
            corpus.extend(extract_docx(path))
    return corpus


def save_corpus(chunks: list[Chunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [chunk.to_dict() for chunk in chunks]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_corpus(path: Path) -> list[Chunk]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(**row) for row in rows]
