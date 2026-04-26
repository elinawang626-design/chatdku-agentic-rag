from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from docx import Document
from pypdf import PdfReader

from .models import Chunk
from .utils import normalize_whitespace


TOC_LINE_RE = re.compile(r"^(?P<title>.+?)\s+(\.{2,}\s*)?(?P<page>\d{1,3})$")


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


def _normalize_heading(text: str) -> str:
    lowered = normalize_whitespace(text).lower()
    return re.sub(r"\s+", " ", lowered).strip(" :.-")


def _parse_toc_page_map(paragraphs: list[str]) -> tuple[dict[str, int], int]:
    page_map: dict[str, int] = {}
    body_start = 0
    found_entries = 0

    for index, raw in enumerate(paragraphs):
        text = normalize_whitespace(raw)
        if not text:
            continue
        match = TOC_LINE_RE.match(text)
        if match:
            title = _normalize_heading(match.group("title"))
            page = int(match.group("page"))
            if title:
                page_map[title] = page
                found_entries += 1
            continue
        if found_entries >= 10:
            body_start = index
            break
    return page_map, body_start


def _parse_toc_page_map_from_textutil(path: Path) -> dict[str, int]:
    try:
        completed = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}

    page_map: dict[str, int] = {}
    found_entries = 0
    for raw in completed.stdout.splitlines():
        text = normalize_whitespace(raw)
        if not text:
            continue
        match = TOC_LINE_RE.match(text)
        if not match:
            if found_entries >= 10:
                break
            continue
        title = _normalize_heading(match.group("title"))
        page = int(match.group("page"))
        if title:
            page_map[title] = page
            found_entries += 1
    return page_map


def extract_docx(path: Path) -> list[Chunk]:
    doc = Document(str(path))
    paragraphs = [paragraph.text for paragraph in doc.paragraphs]
    page_map, body_start = _parse_toc_page_map(paragraphs)
    if not page_map:
        page_map = _parse_toc_page_map_from_textutil(path)
        body_start = 0

    if not page_map:
        text = "\n".join(paragraphs)
        return _chunk_text(path.name, text, None)

    chunks: list[Chunk] = []
    current_page: int | None = min(page_map.values())
    buffer: list[str] = []

    def flush_buffer() -> None:
        nonlocal buffer
        if not buffer:
            return
        chunks.extend(_chunk_text(path.name, "\n".join(buffer), current_page))
        buffer = []

    for raw in paragraphs[body_start:]:
        text = normalize_whitespace(raw)
        if not text:
            continue
        heading_key = _normalize_heading(text)
        mapped_page = page_map.get(heading_key)
        if mapped_page is not None and mapped_page != current_page:
            flush_buffer()
            current_page = mapped_page
        buffer.append(text)

    flush_buffer()
    return chunks


def build_corpus(paths: list[Path] | None = None) -> list[Chunk]:
    if not paths:
        raise ValueError("No input files provided. Pass one or more PDF/DOCX paths.")

    corpus: list[Chunk] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Input document not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            corpus.extend(extract_pdf(path))
        elif suffix == ".docx":
            corpus.extend(extract_docx(path))
        else:
            raise ValueError(f"Unsupported input type: {path}")
    return corpus


def save_corpus(chunks: list[Chunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [chunk.to_dict() for chunk in chunks]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_corpus(path: Path) -> list[Chunk]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(**row) for row in rows]
