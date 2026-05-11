#!/usr/bin/env python3
"""
cortex.py - Offline chat app with built-in indexer + RAG.

Drag-and-drop or upload PDF / EPUB / DOCX / TXT / MD files. Cortex
extracts, chunks, embeds, and caches them locally. Attach indexed
documents to conversations to ground answers in their content.

Cache lives at:
    Linux:   ~/.config/cortex/library/
    macOS:   ~/Library/Application Support/cortex/library/
    Windows: %APPDATA%/cortex/library/

Setup:
    pip install fastapi uvicorn ollama numpy python-multipart \
                pypdf ebooklib beautifulsoup4 python-docx
    ollama pull qwen2.5:7b
    ollama pull nomic-embed-text
    python cortex.py

Configure via environment:
    CORTEX_MODEL=qwen2.5:7b              # default; bump to :14b or :32b on bigger GPUs
    CORTEX_EMBED_MODEL=nomic-embed-text
    CORTEX_HOST=127.0.0.1
    CORTEX_PORT=8000
    CORTEX_TOP_K=6
    CORTEX_LIBRARY=/custom/library/path     # auto by platform if unset
    CORTEX_SMARTREADER_CACHE=/path           # set to also read SmartReader caches
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import pickle
import re
import sqlite3
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Callable, Iterable

import numpy as np
import ollama
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel


# === Configuration ========================================================
# === Model tiers ==========================================================
# Cortex ships with three model tiers in a single executable. Users select
# at runtime via the UI dropdown; the choice persists across launches via
# a small JSON file alongside the conversation DB.
#
# Each tier has:
#   id:            stable identifier used by the UI and config
#   ollama_name:   the actual Ollama model tag to call
#   label:         display name in the UI
#   description:   tooltip / detail line
#   tier:          numeric capability ranking (1=lite, 2=standard, 3=research)
#                  used by mode gating to hide scaffolds the model can't handle
#   recommended_top_k: per-tier retrieval default

MODEL_TIERS: dict[str, dict] = {
    "lite": {
        "ollama_name": "qwen2.5:7b",
        "label": "Lite (7B)",
        "description": "Fast, fits 8 GB VRAM. Good for quick lookups and RAG-grounded queries.",
        "tier": 1,
        "recommended_top_k": 4,
    },
    "standard": {
        "ollama_name": "qwen2.5:14b",
        "label": "Standard (14B)",
        "description": "Balanced. Needs ~10 GB VRAM. Stronger reasoning than 7B at usable speed.",
        "tier": 2,
        "recommended_top_k": 6,
    },
    "research": {
        "ollama_name": "qwen2.5:32b-instruct-q4_K_L",
        "label": "Research (32B Q4_K_L)",
        "description": "Highest precision. Needs 24 GB VRAM for full speed, or 32+ GB system RAM for slow CPU offload.",
        "tier": 3,
        "recommended_top_k": 6,
    },
}

DEFAULT_TIER = os.environ.get("CORTEX_DEFAULT_TIER", "lite")
if DEFAULT_TIER not in MODEL_TIERS:
    DEFAULT_TIER = "lite"


# Runtime state for the active model. Persisted to disk so the user's
# choice survives restarts. The legacy CORTEX_MODEL env var still works
# for power users — it overrides the tier system entirely with a raw
# Ollama model name.
_ACTIVE_TIER: str = DEFAULT_TIER
_OVERRIDE_MODEL: str | None = os.environ.get("CORTEX_MODEL")  # if set, bypasses tiers


def _state_file() -> Path:
    return LIBRARY_DIR.parent / "cortex_state.json"


def _load_active_tier() -> None:
    """Read the persisted tier choice from disk, fall back to default."""
    global _ACTIVE_TIER
    if _OVERRIDE_MODEL:
        return  # env override wins, ignore persisted state
    try:
        f = _state_file()
        if f.exists():
            data = json.loads(f.read_text())
            tier = data.get("active_tier")
            if tier in MODEL_TIERS:
                _ACTIVE_TIER = tier
    except Exception:
        pass


def _save_active_tier(tier: str) -> None:
    """Persist the user's tier choice."""
    try:
        f = _state_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"active_tier": tier}))
    except Exception as e:
        print(f"WARN: could not persist tier: {e}", file=sys.stderr)


def active_model_name() -> str:
    """Return the Ollama model name to call right now."""
    if _OVERRIDE_MODEL:
        return _OVERRIDE_MODEL
    return MODEL_TIERS[_ACTIVE_TIER]["ollama_name"]


def active_tier_info() -> dict:
    """Return descriptive info about the currently active model."""
    if _OVERRIDE_MODEL:
        return {
            "id": "custom",
            "ollama_name": _OVERRIDE_MODEL,
            "label": _OVERRIDE_MODEL,
            "description": "Custom model set via CORTEX_MODEL env var.",
            "tier": 3,  # treat custom as full-capability for mode gating
            "recommended_top_k": int(os.environ.get("CORTEX_TOP_K", "6")),
        }
    return {"id": _ACTIVE_TIER, **MODEL_TIERS[_ACTIVE_TIER]}


EMBED_MODEL = os.environ.get("CORTEX_EMBED_MODEL", "nomic-embed-text")
HOST = os.environ.get("CORTEX_HOST", "127.0.0.1")
PORT = int(os.environ.get("CORTEX_PORT", "8000"))


def active_top_k() -> int:
    """Resolve the retrieval count for the currently active model.
    Honors CORTEX_TOP_K override if set, otherwise uses the tier's recommendation."""
    env_override = os.environ.get("CORTEX_TOP_K")
    if env_override:
        try:
            return int(env_override)
        except ValueError:
            pass
    return active_tier_info()["recommended_top_k"]


CHUNK_SIZE = 1000        # characters per chunk
CHUNK_OVERLAP = 200      # overlap between adjacent chunks
EMBED_TRUNCATE = 500     # SmartReader-compatible: embed first N chars only


def _cortex_library_dir() -> Path:
    override = os.environ.get("CORTEX_LIBRARY")
    if override:
        return Path(override)
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", "")) / "cortex" / "library"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "cortex" / "library"
    return Path.home() / ".config" / "cortex" / "library"


def _smartreader_cache_dir() -> Path | None:
    """SmartReader cache integration is opt-in.

    To enable, set the CORTEX_SMARTREADER_CACHE environment variable to the
    path of an existing SmartReader cache directory. Cortex then exposes
    those caches as read-only library entries tagged `sr`.

    Auto-detection was removed in 1.2: SmartReader caches lack the hierarchical
    section summaries Cortex now produces, and surfacing them by default
    created confusing duplicates with Cortex-native caches of the same books.
    Users migrating from SmartReader should re-index in Cortex to take
    advantage of structure-aware indexing.
    """
    override = os.environ.get("CORTEX_SMARTREADER_CACHE")
    if not override:
        return None
    p = Path(override)
    if not p.exists():
        print(
            f"WARN: CORTEX_SMARTREADER_CACHE={override} but path does not exist",
            file=sys.stderr,
        )
        return None
    return p


LIBRARY_DIR = _cortex_library_dir()
SMARTREADER_DIR = _smartreader_cache_dir()
DB_PATH = LIBRARY_DIR.parent / "conversations.db"


# === Chunk class ==========================================================
class TextChunk:
    """Compatible with SmartReader's pickled cache format.
    Note: no __slots__ on purpose — SmartReader's TextChunk doesn't use slots,
    so its pickles carry a __dict__ that needs to land on a regular class."""

    def __init__(self, text="", page_number=0, chunk_id=0, embedding=None, metadata=None):
        self.text = text
        self.page_number = page_number
        self.chunk_id = chunk_id
        self.embedding = embedding
        self.metadata = metadata if metadata is not None else {}


class _CompatUnpickler(pickle.Unpickler):
    """Remap any TextChunk class path (SmartReader's, ours) to ours."""
    def find_class(self, module: str, name: str):
        if name == "TextChunk":
            return TextChunk
        return super().find_class(module, name)


def load_pickle_chunks(path: Path) -> list[TextChunk]:
    with open(path, "rb") as f:
        chunks = _CompatUnpickler(f).load()
    return [c for c in chunks if getattr(c, "embedding", None)]


def save_pickle_chunks(path: Path, chunks: list[TextChunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(chunks, f)


# === Text extractors per format ==========================================
SUPPORTED_EXT = {".pdf", ".epub", ".docx", ".txt", ".md", ".markdown"}


def extract_pdf(path: Path) -> Iterable[tuple[int, str]]:
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader  # fallback
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            yield i, text


def extract_epub(path: Path) -> Iterable[tuple[int, str]]:
    from ebooklib import epub, ITEM_DOCUMENT
    from bs4 import BeautifulSoup
    book = epub.read_epub(str(path))
    chapter_idx = 0
    for item in book.get_items():
        if item.get_type() != ITEM_DOCUMENT:
            continue
        chapter_idx += 1
        soup = BeautifulSoup(item.get_content(), "html.parser")
        # Drop scripts/styles, get readable text
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            yield chapter_idx, text


def extract_docx(path: Path) -> Iterable[tuple[int, str]]:
    from docx import Document
    doc = Document(str(path))
    # Group paragraphs into pseudo-pages of ~3000 chars to keep
    # citation page numbers meaningful instead of one-per-paragraph.
    PAGE_BUDGET = 3000
    page = 1
    buf: list[str] = []
    size = 0
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        buf.append(text)
        size += len(text) + 1
        if size >= PAGE_BUDGET:
            yield page, "\n\n".join(buf)
            page += 1
            buf, size = [], 0
    if buf:
        yield page, "\n\n".join(buf)


def extract_text(path: Path) -> Iterable[tuple[int, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    PAGE_BUDGET = 3000
    page = 1
    for start in range(0, len(text), PAGE_BUDGET):
        chunk = text[start:start + PAGE_BUDGET]
        if chunk.strip():
            yield page, chunk
            page += 1


def extract(path: Path) -> Iterable[tuple[int, str]]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        yield from extract_pdf(path)
    elif ext == ".epub":
        yield from extract_epub(path)
    elif ext == ".docx":
        yield from extract_docx(path)
    elif ext in {".txt", ".md", ".markdown"}:
        yield from extract_text(path)
    else:
        raise ValueError(f"unsupported file type: {ext}")


# === Chunking =============================================================
def chunk_pages(pages: Iterable[tuple[int, str]],
                chunk_size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    chunk_id = 0
    for page_num, page_text in pages:
        paragraphs = re.split(r"\n\s*\n", page_text)
        current = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # If a single paragraph is larger than chunk_size, hard-split it
            while len(para) > chunk_size:
                head, para = para[:chunk_size], para[chunk_size - overlap:]
                if current:
                    chunks.append(TextChunk(current.strip(), page_num, chunk_id))
                    chunk_id += 1
                    current = ""
                chunks.append(TextChunk(head, page_num, chunk_id))
                chunk_id += 1
            if len(current) + len(para) + 2 < chunk_size:
                current += para + "\n\n"
            else:
                if current.strip():
                    chunks.append(TextChunk(current.strip(), page_num, chunk_id))
                    chunk_id += 1
                # Keep tail-of-current as overlap context
                tail = current[-overlap:] if len(current) >= overlap else current
                current = tail + para + "\n\n"
        if current.strip():
            chunks.append(TextChunk(current.strip(), page_num, chunk_id))
            chunk_id += 1
    return chunks


# === Embedding ============================================================
def embed_text(text: str) -> list[float] | None:
    try:
        resp = ollama.embeddings(model=EMBED_MODEL, prompt=text[:EMBED_TRUNCATE])
        return resp["embedding"]
    except Exception as e:
        print(f"WARN: embed failed: {e}", file=sys.stderr)
        return None


def embed_query(text: str) -> np.ndarray | None:
    try:
        resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    except Exception as e:
        print(f"WARN: embed query failed: {e}", file=sys.stderr)
        return None
    vec = np.array(resp["embedding"], dtype=np.float32)
    n = np.linalg.norm(vec) + 1e-10
    return vec / n


# === Indexing pipeline ====================================================
INDEX_JOBS: dict[str, dict] = {}   # book_id -> {status, progress, total, message}


def cache_path_for(file_hash: str, title: str) -> Path:
    safe = re.sub(r"[^\w\s.-]", "", title).replace(" ", "_")[:60]
    return LIBRARY_DIR / f"{safe}_{file_hash[:8]}.pkl"


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


# === Hierarchical summarization ===========================================
# At index time, group chunks into sections (~10-20 chunks each, aligned
# to page/chapter boundaries when possible) and produce an LLM summary
# of each section. At query time, when chunks are retrieved we also
# fetch their section summaries — this gives the model both detail
# (verbatim chunks) and context (what the surrounding section is about),
# which addresses the most common chunk-retrieval failure: technically
# correct answer that misses the larger argument.
#
# Summaries are stored separately from chunks so:
#   - existing caches remain loadable without re-indexing
#   - summaries can be regenerated without redoing embeddings
#   - the format change is incremental, not breaking
#
# Summarization is conditional: skipped on small documents where the
# whole thing fits in context anyway. Defaults are tuned for the Lite
# tier; higher tiers are fine with the same numbers.

SUMMARIZE_MIN_CHUNKS = 30        # don't summarize tiny documents
SUMMARIZE_SECTION_SIZE = 15      # ~15 chunks per section (~15000 chars)
SUMMARIZE_MAX_INPUT_CHARS = 12000  # cap section size handed to the LLM
SUMMARIZE_TARGET_WORDS = 120     # asked-for summary length


class Section:
    """A contiguous group of chunks plus an LLM-produced summary.
    `title` is set when the section corresponds to a known structural unit
    (PDF outline entry, parsed TOC entry); empty for fallback page-count sections."""
    def __init__(self, section_id: int, chunk_ids: list[int],
                 page_range: tuple[int, int], summary: str = "",
                 title: str = ""):
        self.section_id = section_id
        self.chunk_ids = chunk_ids
        self.page_range = page_range
        self.summary = summary
        self.title = title

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "chunk_ids": self.chunk_ids,
            "page_range": list(self.page_range),
            "summary": self.summary,
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Section":
        return cls(
            d["section_id"],
            list(d["chunk_ids"]),
            tuple(d["page_range"]),
            d.get("summary", ""),
            d.get("title", ""),
        )


def summaries_path_for(cache_path: Path) -> Path:
    """Companion path for a chunk cache's summaries file."""
    return cache_path.with_suffix(".summaries.json")


def load_summaries(cache_path: Path) -> list[Section]:
    """Load summaries for a cache. Returns empty list if not yet summarized
    (older caches, small documents, or summarization skipped)."""
    p = summaries_path_for(cache_path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return [Section.from_dict(s) for s in data.get("sections", [])]
    except Exception as e:
        print(f"WARN: could not load summaries {p}: {e}", file=sys.stderr)
        return []


def save_summaries(cache_path: Path, sections: list[Section]) -> None:
    p = summaries_path_for(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"sections": [s.to_dict() for s in sections]}, indent=2),
        encoding="utf-8",
    )


# --- Document structure extraction ----------------------------------------
# Books usually have explicit chapter/section structure. When we can recover
# it, the resulting sections are far more coherent than fixed-size chunking.
# Three approaches, tried in order:
#   1. PDF outline / bookmarks (built into well-made PDFs)
#   2. Table-of-contents page parsing (regex on text)
#   3. Fallback: fixed-size page-aligned chunking (the original strategy)

TOC_MIN_ENTRIES = 4         # need at least N entries to trust a structure
TOC_MIN_SECTION_CHUNKS = 4  # below this, merge with neighbor


def extract_pdf_outline(pdf_path: Path) -> list[tuple[str, int]]:
    """Read the PDF's built-in outline (bookmarks). Returns a flat list of
    (title, 1-indexed page number) entries, or [] if no usable outline exists.

    PDF outlines are trees; we flatten to a list. Sub-entries become their
    own sections at the same level — which is what we want for retrieval
    purposes (chapter and its subsections are both queryable units).
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            return []

    try:
        reader = PdfReader(str(pdf_path))
        outline = reader.outline
    except Exception as e:
        print(f"WARN: could not read PDF outline: {e}", file=sys.stderr)
        return []

    entries: list[tuple[str, int]] = []

    def _walk(items, depth: int = 0) -> None:
        for item in items:
            if isinstance(item, list):
                _walk(item, depth + 1)
                continue
            try:
                title = getattr(item, "title", None) or ""
                title = str(title).strip()
                if not title:
                    continue
                # Resolve the page reference to a 1-indexed page number
                page_num = reader.get_destination_page_number(item)  # 0-indexed
                if page_num is None:
                    continue
                entries.append((title, page_num + 1))
            except Exception:
                continue

    try:
        _walk(outline)
    except Exception as e:
        print(f"WARN: outline walk failed: {e}", file=sys.stderr)
        return []

    # Deduplicate consecutive entries pointing at the same page
    deduped: list[tuple[str, int]] = []
    for title, page in entries:
        if deduped and deduped[-1][1] == page:
            continue
        deduped.append((title, page))
    return deduped


# Regex patterns for matching TOC lines. The structure varies but most
# have "Chapter N Title ... page" or "N.N Title ... page" formats.
# We match conservatively to avoid false positives in body text.
_TOC_LINE_PATTERNS = [
    # "Chapter 3   The Brain   47"
    re.compile(r"^\s*(?:Chapter|Ch\.?|Part|Section)\s+(\d+|[IVXLCDM]+)\s+(.+?)\s+(\d{1,4})\s*$",
               re.IGNORECASE),
    # "3.2  Synaptic plasticity   142"
    re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.+?)\s+(\d{1,4})\s*$"),
    # "Introduction ............. 1"  (dot leaders)
    re.compile(r"^\s*([^\d\.][^\.]{2,80}?)\s*\.{2,}\s*(\d{1,4})\s*$"),
    # "The Limbic System    73"  (loose form, last-resort)
    re.compile(r"^\s*([A-Z][^\d]{4,80}?)\s{2,}(\d{1,4})\s*$"),
]


def parse_toc_from_text(pages: list[tuple[int, str]]) -> list[tuple[str, int]]:
    """Find a TOC page by name, parse it for chapter/page entries.

    Looks for pages near the front of the document whose text contains the
    word "Contents" or "Table of Contents." Then parses lines using the
    patterns above. Returns [] if no plausible TOC is found.
    """
    # Search the first 30 pages for a TOC marker
    toc_pages: list[str] = []
    in_toc = False
    for page_num, text in pages[:30]:
        head = text.strip().split("\n", 3)[0:3]
        head_text = " ".join(head).lower()
        if not in_toc and ("contents" in head_text or "table of contents" in head_text):
            in_toc = True
        if in_toc:
            toc_pages.append(text)
            # TOC usually doesn't span more than 5-8 pages; stop if we find
            # what looks like chapter content (paragraph-shaped text)
            line_count = len([l for l in text.split("\n") if l.strip()])
            avg_line_len = (sum(len(l) for l in text.split("\n")) / max(1, line_count))
            if avg_line_len > 80 and len(toc_pages) >= 2:
                break
            if len(toc_pages) >= 8:
                break

    if not toc_pages:
        return []

    # Apply patterns to each line
    entries: list[tuple[str, int]] = []
    for page_text in toc_pages:
        for line in page_text.split("\n"):
            line = line.strip()
            if not line or len(line) < 6:
                continue
            for pat in _TOC_LINE_PATTERNS:
                m = pat.match(line)
                if not m:
                    continue
                groups = m.groups()
                # Last group is always the page number
                try:
                    page = int(groups[-1])
                except ValueError:
                    break
                # Title is everything before the page number
                title_parts = [g for g in groups[:-1] if g]
                title = " ".join(title_parts).strip()
                # Reject obvious false positives
                if not title or page < 1 or page > 9999:
                    break
                if len(title) > 120:  # probably matched a sentence, not a TOC line
                    break
                entries.append((title, page))
                break

    # Sanity: TOC pages should monotonically increase
    cleaned: list[tuple[str, int]] = []
    last_page = 0
    for title, page in entries:
        if page < last_page:
            # Out-of-order — this isn't really a TOC entry, skip
            continue
        cleaned.append((title, page))
        last_page = page

    return cleaned


def sections_from_structure(chunks: list[TextChunk],
                            structure: list[tuple[str, int]]
                            ) -> list[Section]:
    """Given a list of (chapter_title, start_page) entries, build sections
    by assigning each chunk to the chapter that contains its page.

    Tiny sections (fewer than TOC_MIN_SECTION_CHUNKS chunks) are merged
    backward into the previous section — these are usually preface entries,
    appendix subsections, or TOC parsing artifacts that we don't want as
    standalone summarizable units.
    """
    if not chunks or not structure:
        return []

    # Sort structure by page number; the outline order isn't always page-order
    # (e.g. some books list appendices before bibliography)
    structure = sorted(structure, key=lambda t: t[1])

    # Build sections by walking chunks and assigning to the latest chapter
    # whose start_page is <= the chunk's page.
    sections: list[Section] = []
    current_chunks: list[int] = []
    current_title = ""
    current_start_page = chunks[0].page_number
    current_struct_idx = -1

    def _flush(end_page: int) -> None:
        nonlocal current_chunks
        if not current_chunks:
            return
        sections.append(Section(
            section_id=len(sections),
            chunk_ids=current_chunks.copy(),
            page_range=(current_start_page, end_page),
            title=current_title,
        ))
        current_chunks = []

    last_page = chunks[0].page_number
    for chunk in chunks:
        # Which structural entry does this chunk fall under?
        target_idx = current_struct_idx
        for i, (_, start_page) in enumerate(structure):
            if start_page <= chunk.page_number:
                target_idx = i
            else:
                break
        # Crossed into a new chapter — flush the previous section
        if target_idx != current_struct_idx:
            _flush(last_page)
            current_struct_idx = target_idx
            if target_idx >= 0:
                current_title = structure[target_idx][0]
                current_start_page = structure[target_idx][1]
            else:
                current_title = ""  # pre-first-chapter material
                current_start_page = chunk.page_number
        current_chunks.append(chunk.chunk_id)
        last_page = chunk.page_number
    _flush(last_page)

    # Merge tiny sections backward — TOC parsing often produces single-page
    # entries for things like "About the Author" that aren't worth their own
    # summary. The threshold of TOC_MIN_SECTION_CHUNKS is conservative.
    merged: list[Section] = []
    for section in sections:
        if (merged
                and len(section.chunk_ids) < TOC_MIN_SECTION_CHUNKS
                and len(merged[-1].chunk_ids) + len(section.chunk_ids) < SUMMARIZE_SECTION_SIZE * 2):
            prev = merged[-1]
            prev.chunk_ids.extend(section.chunk_ids)
            prev.page_range = (prev.page_range[0], section.page_range[1])
            # Keep the earlier title — usually the parent chapter
        else:
            merged.append(section)

    # Renumber section_ids after merging
    for i, s in enumerate(merged):
        s.section_id = i

    return merged


def build_sections(chunks: list[TextChunk],
                   src_path: Path | None = None,
                   pages: list[tuple[int, str]] | None = None,
                   target_chunks_per_section: int = SUMMARIZE_SECTION_SIZE,
                   ) -> list[Section]:
    """Build sections for a document, preferring structural cues over
    fixed-size chunking.

    Strategy in order of preference:
      1. If src_path is a PDF and has a usable outline → use that
      2. If pages were provided and contain a parseable TOC → use that
      3. Fallback: page-aligned fixed-size grouping (the original strategy)
    """
    if not chunks:
        return []

    # Attempt 1: PDF outline
    if src_path and src_path.suffix.lower() == ".pdf":
        structure = extract_pdf_outline(src_path)
        if len(structure) >= TOC_MIN_ENTRIES:
            sections = sections_from_structure(chunks, structure)
            if sections:
                print(f"  Used PDF outline: {len(sections)} sections", file=sys.stderr)
                return sections

    # Attempt 2: TOC page parsing
    if pages:
        structure = parse_toc_from_text(pages)
        if len(structure) >= TOC_MIN_ENTRIES:
            sections = sections_from_structure(chunks, structure)
            if sections:
                print(f"  Parsed TOC: {len(sections)} sections", file=sys.stderr)
                return sections

    # Attempt 3: fallback to original fixed-size strategy
    return _build_sections_fixed(chunks, target_chunks_per_section)


def _build_sections_fixed(chunks: list[TextChunk],
                          target_chunks_per_section: int,
                          ) -> list[Section]:
    """Original fixed-size, page-aligned section builder. Used as fallback
    when no structural cues are recoverable."""
    if not chunks:
        return []
    sections: list[Section] = []
    buf: list[int] = []
    buf_start_page = chunks[0].page_number
    last_page = chunks[0].page_number

    for chunk in chunks:
        # If buffer is full enough AND we just crossed a page boundary,
        # close the section. The page-boundary check avoids splitting
        # arbitrary content mid-page.
        if (len(buf) >= target_chunks_per_section
                and chunk.page_number != last_page):
            sections.append(Section(
                section_id=len(sections),
                chunk_ids=buf.copy(),
                page_range=(buf_start_page, last_page),
            ))
            buf = []
            buf_start_page = chunk.page_number
        buf.append(chunk.chunk_id)
        last_page = chunk.page_number

    if buf:
        sections.append(Section(
            section_id=len(sections),
            chunk_ids=buf.copy(),
            page_range=(buf_start_page, last_page),
        ))
    return sections


def summarize_section(section_text: str, doc_title: str) -> str:
    """Ask the active LLM to summarize a section of the document.
    Returns empty string on failure — caller can decide whether to retry."""
    # Truncate input to avoid blowing out context on small models.
    truncated = section_text[:SUMMARIZE_MAX_INPUT_CHARS]
    if len(section_text) > SUMMARIZE_MAX_INPUT_CHARS:
        truncated += "\n\n[...section continues; this is a partial view...]"

    system = (
        "You are summarizing a section of a longer document for a retrieval "
        "system. Your summary will be shown alongside specific excerpts from "
        "this section when a user asks a question that touches it. The "
        "summary's job is to give context — what this section is fundamentally "
        "about, what it argues or describes, and how it connects to the "
        "broader document. Be concrete; mention specific concepts, methods, "
        "or claims by name. Avoid generic phrasing like 'this section "
        "discusses various topics'."
    )
    user = (
        f"Document title: {doc_title}\n\n"
        f"Section content:\n{truncated}\n\n"
        f"Produce a {SUMMARIZE_TARGET_WORDS}-word summary of what this "
        f"section is about. Output only the summary text, no preamble."
    )

    try:
        resp = ollama.chat(
            model=active_model_name(),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={"temperature": 0.3},  # lower than chat — we want consistency
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        print(f"WARN: summarization failed: {e}", file=sys.stderr)
        return ""


def summarize_book(chunks: list[TextChunk], doc_title: str,
                   progress: Callable[[int, int, str], None],
                   progress_start: int = 96, progress_end: int = 99,
                   src_path: Path | None = None,
                   pages: list[tuple[int, str]] | None = None,
                   ) -> list[Section]:
    """Build sections and summarize each one. Updates progress in the
    given range. Returns sections with summaries filled in (may be empty
    strings on failure — we don't fail the whole index if summarization
    has trouble).

    src_path and pages are passed to build_sections so it can use structural
    cues (PDF outline, TOC page) when available; without them, falls back
    to fixed-size page-aligned grouping."""
    sections = build_sections(chunks, src_path=src_path, pages=pages)
    if not sections:
        return []

    progress(progress_start, 100, f"Summarizing {len(sections)} sections...")
    chunks_by_id = {c.chunk_id: c for c in chunks}

    for i, section in enumerate(sections):
        # Assemble section text from its chunks
        section_text = "\n\n".join(
            chunks_by_id[cid].text
            for cid in section.chunk_ids
            if cid in chunks_by_id
        )
        section.summary = summarize_section(section_text, doc_title)
        # Smooth progress update across the summarization range
        if i % 2 == 0 or i == len(sections) - 1:
            span = max(1, progress_end - progress_start)
            pct = progress_start + int((i + 1) / len(sections) * span)
            progress(pct, 100, f"Summarizing {i + 1}/{len(sections)} sections...")

    return sections


def index_file(src_path: Path, book_id: str,
               progress: Callable[[int, int, str], None],
               display_title: str | None = None) -> Path:
    """Run extraction → chunking → embedding → save. Updates progress.

    `display_title` is what the user sees in the library — typically derived
    from the original upload filename, NOT from src_path (which is a temp
    file with a UUID prefix). Falls back to src_path.stem if not provided.
    """
    title = display_title or src_path.stem
    progress(0, 100, f"Reading {src_path.name}...")
    pages = list(extract(src_path))
    if not pages:
        raise RuntimeError("no extractable text")
    progress(10, 100, f"Chunking {len(pages)} sections...")
    chunks = chunk_pages(pages)
    if not chunks:
        raise RuntimeError("no chunks produced")
    total = len(chunks)
    progress(15, 100, f"Embedding {total} chunks...")
    for i, chunk in enumerate(chunks):
        emb = embed_text(chunk.text)
        if emb is None:
            raise RuntimeError(f"embedding failed at chunk {i}")
        chunk.embedding = emb
        if i % 10 == 0 or i == total - 1:
            pct = 15 + int((i + 1) / total * 80)
            progress(pct, 100, f"Embedding {i + 1}/{total} chunks...")
    progress(96, 100, "Saving cache...")
    fh = file_hash(src_path)
    out = cache_path_for(fh, title)
    save_pickle_chunks(out, chunks)

    # Hierarchical summarization — only worth doing for documents large enough
    # that section-level context actually helps. Small documents fit in
    # context anyway.
    skip_summaries = os.environ.get("CORTEX_SKIP_SUMMARIES", "").lower() in ("1", "true", "yes")
    if not skip_summaries and len(chunks) >= SUMMARIZE_MIN_CHUNKS:
        try:
            sections = summarize_book(chunks, title, progress,
                                      progress_start=96, progress_end=99,
                                      src_path=src_path, pages=pages)
            save_summaries(out, sections)
        except Exception as e:
            print(f"WARN: summarization failed for {title}: {e}", file=sys.stderr)

    progress(100, 100, "Done")
    return out


async def run_index_job(book_id: str, src_path: Path,
                        display_title: str | None = None) -> None:
    job = INDEX_JOBS[book_id]

    def progress(cur: int, tot: int, msg: str) -> None:
        job["progress"] = cur
        job["total"] = tot
        job["message"] = msg

    try:
        # Run blocking work off the event loop
        out_path = await asyncio.to_thread(
            index_file, src_path, book_id, progress, display_title,
        )
        job["status"] = "done"
        job["cache_path"] = str(out_path)
        # Refresh in-memory book registry so the new book is queryable
        discover_books()
    except Exception as e:
        job["status"] = "error"
        job["message"] = str(e)
    finally:
        # Clean up the temp upload
        try:
            src_path.unlink(missing_ok=True)
        except Exception:
            pass


# === In-memory book index =================================================
class Book:
    def __init__(self, book_id: str, title: str, path: Path, source: str):
        self.id = book_id
        self.title = title
        self.path = path
        self.source = source     # "cortex" or "smartreader"
        self.chunks: list[TextChunk] = []
        self.matrix: np.ndarray | None = None
        self.sections: list[Section] = []
        # chunk_id -> section_id, built once at load time for O(1) lookup
        self._chunk_to_section: dict[int, int] = {}

    def load(self) -> None:
        self.chunks = load_pickle_chunks(self.path)
        if not self.chunks:
            self.matrix = None
            return
        embs = np.array([c.embedding for c in self.chunks], dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
        self.matrix = embs / norms
        # Load hierarchical summaries if available — older caches won't have them
        self.sections = load_summaries(self.path)
        self._chunk_to_section = {
            cid: s.section_id
            for s in self.sections
            for cid in s.chunk_ids
        }

    def section_for_chunk(self, chunk_id: int) -> Section | None:
        sid = self._chunk_to_section.get(chunk_id)
        if sid is None or sid >= len(self.sections):
            return None
        return self.sections[sid]

    def unload(self) -> None:
        self.chunks = []
        self.matrix = None
        self.sections = []
        self._chunk_to_section = {}


_BOOKS: dict[str, Book] = {}


_LEADING_HEX_PREFIX = re.compile(r"^[0-9a-f]{8}_", re.IGNORECASE)


def _title_from_smartreader_stem(stem: str) -> str:
    if stem.endswith("_enhanced"):
        stem = stem[: -len("_enhanced")]
    if len(stem) > 9 and stem[-9] == "_":
        stem = stem[:-9]
    return stem.replace("_", " ").strip()


def _title_from_cortex_stem(stem: str) -> str:
    # Strip trailing 8-char hash separator (our cache filename suffix).
    if len(stem) > 9 and stem[-9] == "_":
        stem = stem[:-9]
    # Strip leading 8-hex-char prefix that older Cortex builds accidentally
    # captured from temporary upload filenames. Harmless on new caches.
    stem = _LEADING_HEX_PREFIX.sub("", stem)
    return stem.replace("_", " ").strip()


def discover_books() -> None:
    """Scan Cortex library + optional SmartReader cache."""
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)

    found_ids: set[str] = set()
    for pkl in sorted(LIBRARY_DIR.glob("*.pkl")):
        bid = "cx:" + pkl.stem
        found_ids.add(bid)
        if bid not in _BOOKS:
            _BOOKS[bid] = Book(bid, _title_from_cortex_stem(pkl.stem), pkl, "cortex")

    if SMARTREADER_DIR and SMARTREADER_DIR.exists():
        for pkl in sorted(SMARTREADER_DIR.glob("*.pkl")):
            bid = "sr:" + pkl.stem
            found_ids.add(bid)
            if bid not in _BOOKS:
                _BOOKS[bid] = Book(bid, _title_from_smartreader_stem(pkl.stem), pkl, "smartreader")

    # Drop books whose cache file vanished
    for stale in [bid for bid in _BOOKS if bid not in found_ids]:
        _BOOKS.pop(stale, None)


def ensure_loaded(book_id: str) -> Book | None:
    book = _BOOKS.get(book_id)
    if book is None:
        return None
    if book.matrix is None:
        try:
            book.load()
        except Exception as e:
            print(f"WARN: load {book.path.name}: {e}", file=sys.stderr)
            return None
    return book


def retrieve(book_ids: list[str], query: str, top_k: int | None = None) -> list[dict]:
    """Retrieve top-K chunks across all attached books.

    When multiple books are attached, this guarantees each book contributes
    at least one chunk to the result set (assuming it has any chunks at all).
    Without this guarantee, a single book that happens to score slightly
    higher on a query can monopolize all K slots, leaving the other books
    invisible to the model — which is the cause of the "model only cites
    the first book" symptom.

    Algorithm:
      1. Score every chunk in every attached book against the query.
      2. Reserve `min_per_book` slots per book (taking each book's top
         chunks for its reserved share).
      3. Fill remaining slots from the global pool, skipping anything
         already reserved.
      4. Re-sort the final result by score so the highest-relevance
         chunk appears first regardless of source.
    """
    if not book_ids:
        return []
    if top_k is None:
        top_k = active_top_k()
    qvec = embed_query(query)
    if qvec is None:
        return []

    # Per-book scored chunks: {book_id: [(score, Book, chunk), ...]}
    # Carrying the Book reference (not just title) lets us look up section
    # summaries downstream without a second discover_books call.
    by_book: dict[str, list[tuple[float, Book, TextChunk]]] = {}
    for bid in book_ids:
        book = ensure_loaded(bid)
        if not book or book.matrix is None:
            continue
        scores = book.matrix @ qvec
        order = np.argsort(-scores)
        by_book[bid] = [
            (float(scores[i]), book, book.chunks[i])
            for i in order
        ]

    if not by_book:
        return []

    def _to_result(s: float, book: Book, chunk: TextChunk) -> dict:
        """Pack a scored chunk into the result dict, attaching section
        context if hierarchical summaries are available for this book."""
        out = {
            "score": s,
            "book": book.title,
            "page": chunk.page_number,
            "text": chunk.text,
        }
        section = book.section_for_chunk(chunk.chunk_id)
        if section and section.summary:
            out["section_summary"] = section.summary
            out["section_pages"] = list(section.page_range)
            if section.title:
                out["section_title"] = section.title
        return out

    # Single-book case: just take the top-K. No reservation needed.
    if len(by_book) == 1:
        only = next(iter(by_book.values()))
        result = only[:top_k]
        return [_to_result(s, book, chunk) for s, book, chunk in result]

    # Multi-book case: reserve at least 1 chunk per book, then fill.
    n_books = len(by_book)
    min_per_book = max(1, top_k // (n_books * 2))
    min_per_book = min(min_per_book, top_k // n_books)

    reserved: list[tuple[float, Book, TextChunk]] = []
    seen: set[int] = set()
    for bid, scored in by_book.items():
        for entry in scored[:min_per_book]:
            reserved.append(entry)
            seen.add(id(entry[2]))

    # Build the global pool of remaining candidates and fill the rest.
    remaining_slots = max(0, top_k - len(reserved))
    global_pool: list[tuple[float, Book, TextChunk]] = []
    for scored in by_book.values():
        for entry in scored:
            if id(entry[2]) in seen:
                continue
            global_pool.append(entry)
    global_pool.sort(key=lambda t: t[0], reverse=True)

    final = reserved + global_pool[:remaining_slots]
    final.sort(key=lambda t: t[0], reverse=True)

    return [_to_result(s, book, chunk) for s, book, chunk in final]


# === Database =============================================================
def _db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _db_init() -> None:
    with _db() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attachments (
                conversation_id TEXT NOT NULL,
                book_id TEXT NOT NULL,
                attached_at REAL NOT NULL,
                PRIMARY KEY (conversation_id, book_id)
            );
            CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_att_conv ON attachments(conversation_id);
        """)


def get_attachments(cid: str) -> list[str]:
    with _db() as c:
        rows = c.execute(
            "SELECT book_id FROM attachments WHERE conversation_id = ?", (cid,)
        ).fetchall()
    return [r["book_id"] for r in rows]


# === Prompts ==============================================================
SYSTEM_BASE = """You are a thoughtful, capable AI assistant running locally on the user's machine. You have no internet access; your training data has a fixed cutoff.

Be direct and substantive. Skip preambles. Match formatting to content — clean prose by default, structure only when it helps. Treat the user as a capable adult; no condescension. When you don't know something, say so plainly. For technical tasks, prefer working code and concrete examples.

Since you run offline, never claim to look something up. If asked about current events or time-sensitive data, say it's outside your reach."""

SYSTEM_RAG_SUFFIX = """

The user has attached source documents to this conversation. The system has retrieved the most relevant excerpts for their question. Use them as primary grounding material.

Some excerpts are preceded by a [Section context — Title, pp. X-Y] block — this is a high-level summary of the surrounding section, provided to help you understand what argument or topic the specific excerpts are situated in. Use section context to interpret excerpts correctly, but cite specific excerpts (with their page numbers), not the section summary, when making concrete claims.

Rules:
- Cite specific pages when you draw on the excerpts. Format: [Title, p. N].
- For non-PDF sources, "p. N" maps to chapter (EPUB) or section (DOCX/text) — cite it the same way.
- If the excerpts don't contain the answer, say so plainly. Do not invent citations or fill gaps with general knowledge presented as if it came from the source.
- You may use general knowledge to frame or connect what the excerpts say, but be explicit when doing so.
- If excerpts contradict each other, point that out instead of forcing a synthesis.
- When MULTIPLE sources are attached, draw from all of them — not just the first one you read. The retrieved excerpts come from different books and may cover complementary aspects of the question. Cite each source you use explicitly. If you only end up citing one source, briefly note whether the other sources had relevant material or simply didn't address the question."""


def format_rag_context(chunks: list[dict]) -> str:
    if not chunks:
        return ""

    # When multiple distinct sources are present, surface that explicitly at
    # the top so the model doesn't anchor on the first source it sees.
    sources = sorted({c["book"] for c in chunks})
    if len(sources) > 1:
        header = (
            f"Retrieved excerpts from {len(sources)} attached sources: "
            f"{', '.join(sources)}.\n"
            "Each excerpt below is labeled with its source — consider all of them.\n\n"
        )
    else:
        header = "Retrieved excerpts from attached sources:\n\n"

    # Group chunks by (book, section_pages) so a single section summary
    # appears once even if multiple chunks from that section are retrieved.
    # The model gets section context up front, then the specific excerpts —
    # this is the "hierarchical context" pattern: zoom out, then zoom in.
    from collections import defaultdict
    groups: dict[tuple, list[tuple[int, dict]]] = defaultdict(list)
    for i, c in enumerate(chunks, 1):
        sec_key = (
            c["book"],
            tuple(c.get("section_pages", [])) if c.get("section_summary") else None,
        )
        groups[sec_key].append((i, c))

    blocks = []
    for (book, section_pages), members in groups.items():
        # If this group has a section summary, show it once at the top
        first = members[0][1]
        section_summary = first.get("section_summary", "")
        section_title = first.get("section_title", "")
        if section_summary and section_pages:
            sp_start, sp_end = section_pages
            page_label = (
                f"p. {sp_start}" if sp_start == sp_end
                else f"pp. {sp_start}–{sp_end}"
            )
            # Include chapter title when we have one — gives the model
            # the structural cue ("this is Chapter 3: Synaptic Plasticity")
            # in addition to the prose summary.
            if section_title:
                header_line = f"[Section context — {book}, '{section_title}', {page_label}]"
            else:
                header_line = f"[Section context — {book}, {page_label}]"
            blocks.append(f"{header_line}\n{section_summary}")
        for idx, c in members:
            blocks.append(
                f"[Excerpt {idx}] {c['book']}, p. {c['page']} "
                f"(relevance {c['score']:.2f})\n{c['text'].strip()}"
            )

    return header + "\n\n---\n\n".join(blocks)


# === Reasoning modes ======================================================
# Modes prepend a scaffold instruction to the user's question. The scaffold
# forces the model to produce structured intermediate output before its
# prose answer — this dramatically improves the quality of complex queries
# on small/medium local models, which otherwise tend to skip steps and
# anchor on the first piece of evidence they read.
#
# Each mode has:
#   id:          stable identifier used by the UI and API
#   label:       short name for the UI dropdown
#   description: tooltip text explaining when to use it
#   scaffold:    instruction injected into the user message
#
# To add a new mode, append an entry. The frontend picks them up automatically
# via /api/modes.

MODES: dict[str, dict] = {
    "default": {
        "label": "Default",
        "description": "Direct answer with no extra scaffolding. Best for simple lookups, factual questions, and casual conversation.",
        "min_tier": 1,
        "scaffold": "",
    },
    "compare": {
        "label": "Compare",
        "description": "Forces a structured comparison before the prose answer. Use for 'A vs B', 'best approach', 'tradeoffs', or any question with multiple options.",
        "min_tier": 1,
        "scaffold": (
            "This question involves comparing options or weighing tradeoffs. "
            "Before your prose answer, produce a markdown comparison table:\n"
            "  - Rows = the options being compared\n"
            "  - Columns = the criteria that matter for this question\n"
            "  - Cells = concrete ratings, values, or short notes (not just 'good/bad')\n"
            "Then write a prose synthesis below the table that draws conclusions "
            "from it. Cite sources where applicable. If a cell can't be filled "
            "from your knowledge or the attached sources, write 'unknown' rather "
            "than guessing."
        ),
    },
    "process": {
        "label": "Process",
        "description": "Forces an explicit state/step layout before the prose explanation. Use for 'how does X work', 'what's the flow', biological pathways, algorithms, or system dynamics.",
        "min_tier": 2,
        "scaffold": (
            "This question is about a process, system, or sequence over time. "
            "Before your prose answer, produce a structured outline:\n"
            "  1. List the distinct states, stages, or steps involved\n"
            "  2. For each: what triggers entry into it, what happens during it, "
            "what causes the transition out\n"
            "  3. Note any feedback loops or branching points explicitly\n"
            "Format as a numbered list or small table. Then write the prose "
            "explanation below, referring back to the structure you laid out. "
            "Cite sources where applicable."
        ),
    },
    "cross_source": {
        "label": "Cross-source",
        "description": "Forces explicit cross-referencing across all attached documents before the answer. Use when multiple books are attached and you want them all considered. Best on 14B+ models.",
        "min_tier": 2,
        "scaffold": (
            "Multiple sources are attached. Before your prose answer, produce a "
            "cross-reference table:\n"
            "  - Rows = the key claims relevant to the question\n"
            "  - Columns = each attached source\n"
            "  - Cells = which source supports the claim (with page), 'silent' "
            "if a source doesn't address it, or 'contradicts' with a brief note "
            "if a source disagrees\n"
            "Then write the prose answer, drawing from ALL sources that have "
            "something to contribute. If a source ended up being silent on every "
            "relevant claim, say so explicitly at the end of your answer."
        ),
    },
    "critique": {
        "label": "Critique",
        "description": "Forces a structured strengths/weaknesses analysis before recommendations. Use for reviewing a plan, paper, code design, or proposal. Best on 14B+ models.",
        "min_tier": 2,
        "scaffold": (
            "This question asks for evaluation or critique. Before your prose "
            "response, produce a structured analysis:\n"
            "  - Strengths: what works, with specific reasons\n"
            "  - Weaknesses: what's flawed or risky, with specific reasons\n"
            "  - Unknowns: what you can't evaluate without more information\n"
            "  - Recommendations: concrete changes, ordered by importance\n"
            "Use bullet points with at least one specific example per item. "
            "Avoid vague critique like 'could be clearer' — say what specifically "
            "is unclear and why. Then write a prose summary below."
        ),
    },
}

DEFAULT_MODE = "default"


def apply_mode(user_content: str, mode_id: str) -> str:
    """Wrap a user message with the chosen mode's scaffold instruction."""
    mode = MODES.get(mode_id) or MODES[DEFAULT_MODE]
    scaffold = mode.get("scaffold", "")
    if not scaffold:
        return user_content
    # The scaffold goes BEFORE the question so the model reads instructions
    # first, then sees the question through that lens.
    return f"{scaffold}\n\n---\n\n{user_content}"


# === API schemas ==========================================================
class ChatRequest(BaseModel):
    conversation_id: str | None = None
    content: str
    mode: str = DEFAULT_MODE


class AttachRequest(BaseModel):
    book_id: str


# === FastAPI ==============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    _db_init()
    _load_active_tier()
    discover_books()
    info = active_tier_info()
    print(f"Cortex: model={active_model_name()} (tier: {info['label']}) embed={EMBED_MODEL}", file=sys.stderr)
    print(f"  library: {LIBRARY_DIR}", file=sys.stderr)
    if SMARTREADER_DIR:
        print(f"  smartreader: {SMARTREADER_DIR} (read-only)", file=sys.stderr)
    print(f"  discovered {len(_BOOKS)} indexed document(s)", file=sys.stderr)
    try:
        ollama.list()
    except Exception as e:
        print(f"WARN: Ollama not reachable ({e})", file=sys.stderr)
    # Start the idle watchdog — it exits the process if the browser tab
    # has been closed for ~30 seconds.
    watchdog_task = asyncio.create_task(_idle_watchdog())
    try:
        yield
    finally:
        watchdog_task.cancel()


app = FastAPI(lifespan=lifespan, title="Cortex")


@app.get("/api/model")
def model_info():
    info = active_tier_info()
    return {
        "model": active_model_name(),
        "tier_id": info["id"],
        "tier_label": info["label"],
        "tier_description": info["description"],
        "tier_rank": info["tier"],
        "embed_model": EMBED_MODEL,
        "library_dir": str(LIBRARY_DIR),
        "smartreader_dir": str(SMARTREADER_DIR) if SMARTREADER_DIR else None,
        "supported_ext": sorted(SUPPORTED_EXT),
        "override_active": _OVERRIDE_MODEL is not None,
    }


@app.get("/api/model/tiers")
def list_tiers():
    """Return the available model tiers for the UI dropdown.
    The current active tier is also returned so the UI can highlight it."""
    info = active_tier_info()
    return {
        "active": info["id"],
        "override_active": _OVERRIDE_MODEL is not None,
        "tiers": [
            {
                "id": tid,
                "ollama_name": t["ollama_name"],
                "label": t["label"],
                "description": t["description"],
                "tier": t["tier"],
            }
            for tid, t in MODEL_TIERS.items()
        ],
    }


class TierSwitchRequest(BaseModel):
    tier: str


def _list_installed_ollama_models() -> list[str]:
    """Return the list of installed Ollama model names.

    Different versions of the Ollama Python client return different shapes:
      - Older versions: dict with 'models' list, each model has 'name' key
      - Newer versions: ListResponse object with .models attribute, each
        Model has .model attribute (note: 'model', not 'name')
      - Some versions populate both fields, some only one
    This helper tries every shape so the rest of the code doesn't have to care.
    """
    try:
        raw = ollama.list()
    except Exception as e:
        print(f"WARN: ollama.list() failed: {e}", file=sys.stderr)
        return []

    # Find the 'models' container — could be a dict key or an attribute
    if hasattr(raw, "models"):
        models = raw.models
    elif isinstance(raw, dict):
        models = raw.get("models", [])
    else:
        return []

    names: list[str] = []
    for m in models:
        # Try dict access first (works for older clients and dict-shaped items)
        name = ""
        if isinstance(m, dict):
            name = m.get("model") or m.get("name") or ""
        else:
            # Object form — try both attribute names
            name = (
                getattr(m, "model", None)
                or getattr(m, "name", None)
                or ""
            )
        if name:
            names.append(str(name))
    return names


@app.get("/api/ollama/installed")
def ollama_installed_models():
    """List models currently installed in the local Ollama. Used by the
    tier menu to show which tiers are ready to use vs. need to be pulled."""
    return {"names": _list_installed_ollama_models()}


@app.post("/api/model/switch")
def switch_tier(body: TierSwitchRequest):
    """Switch the active model tier. Verifies the model is actually pulled
    in Ollama before committing, so users get a clear error rather than a
    cryptic failure mid-query."""
    global _ACTIVE_TIER
    if _OVERRIDE_MODEL is not None:
        raise HTTPException(
            400,
            "CORTEX_MODEL env var is set — tier switching disabled. "
            "Unset CORTEX_MODEL to use the tier UI.",
        )
    if body.tier not in MODEL_TIERS:
        raise HTTPException(404, f"unknown tier: {body.tier}")

    target_name = MODEL_TIERS[body.tier]["ollama_name"]

    # Verify the target model exists in the local Ollama before switching.
    # Match is intentionally lenient about formatting variations Ollama
    # introduces (case, trailing :latest, .gguf suffix) but strict about
    # the actual model identity — we don't want "qwen2.5:14b" to match
    # "qwen2.5:7b" just because they share the "qwen2.5" prefix.
    installed_names = _list_installed_ollama_models()

    def _norm(s: str) -> str:
        s = s.lower().strip()
        if s.endswith(".gguf"):
            s = s[:-5]
        if s.endswith(":latest"):
            s = s[:-7]
        return s

    target_norm = _norm(target_name)
    # If the configured name has no tag (e.g. just "qwen2.5-32b-q4kl"),
    # an installed copy with any tag is acceptable. If it has a tag,
    # the tag must match.
    target_has_tag = ":" in target_norm

    def _is_match(installed: str) -> bool:
        n = _norm(installed)
        if n == target_norm:
            return True
        if not target_has_tag and n.startswith(target_norm + ":"):
            return True
        return False

    match = any(_is_match(n) for n in installed_names)

    if not match:
        installed_summary = ", ".join(installed_names[:6]) or "(none detected)"
        if len(installed_names) > 6:
            installed_summary += f", ... and {len(installed_names) - 6} more"
        raise HTTPException(
            400,
            f"Model '{target_name}' is not installed in Ollama. "
            f"Pull it first: ollama pull {target_name}\n\n"
            f"Currently installed: {installed_summary}",
        )

    _ACTIVE_TIER = body.tier
    _save_active_tier(body.tier)
    print(f"Switched to tier '{body.tier}' (model: {target_name})", file=sys.stderr)
    return {"ok": True, "active": body.tier, "model": target_name}


@app.get("/api/modes")
def list_modes():
    """Return the available reasoning modes for the UI dropdown.
    Includes the active model's tier rank so the frontend can dim or hide
    modes the current model isn't strong enough to handle well."""
    info = active_tier_info()
    return {
        "default": DEFAULT_MODE,
        "current_tier_rank": info["tier"],
        "modes": [
            {
                "id": mid,
                "label": m["label"],
                "description": m["description"],
                "min_tier": m.get("min_tier", 1),
            }
            for mid, m in MODES.items()
        ],
    }


# === Shutdown / heartbeat =================================================
# The browser is the only "client" of this server. When the user closes the
# tab, we want the process to exit so port 8000 doesn't stay held and so the
# next launch isn't blocked. Two mechanisms cooperate:
#   1. The browser sends a heartbeat every few seconds while the tab is open.
#   2. A background task watches the heartbeat and exits if it goes silent
#      for too long.
# A direct /api/shutdown endpoint also exists for explicit "quit" actions.

import signal as _signal
_LAST_HEARTBEAT: float = time.time()
_IDLE_TIMEOUT_SEC: float = 30.0    # exit if no heartbeat for this long


@app.post("/api/heartbeat")
def heartbeat():
    """Browser pings this every few seconds while the tab is open."""
    global _LAST_HEARTBEAT
    _LAST_HEARTBEAT = time.time()
    return {"ok": True}


@app.post("/api/shutdown")
def shutdown():
    """Explicit shutdown — used by the JS unload handler."""
    print("Shutdown requested by client.", file=sys.stderr)
    # Give FastAPI a moment to send the response before we exit.
    threading.Timer(0.3, lambda: os.kill(os.getpid(), _signal.SIGTERM)).start()
    return {"ok": True}


async def _idle_watchdog():
    """Background task that exits the process if heartbeats stop arriving."""
    global _LAST_HEARTBEAT
    # Reset on startup so we don't immediately trigger before the browser opens.
    _LAST_HEARTBEAT = time.time()
    grace_sec = 60.0   # extra time after startup before watchdog can fire
    started_at = time.time()
    while True:
        await asyncio.sleep(5)
        if time.time() - started_at < grace_sec:
            continue
        if time.time() - _LAST_HEARTBEAT > _IDLE_TIMEOUT_SEC:
            print(
                f"No browser heartbeat for {_IDLE_TIMEOUT_SEC}s — shutting down.",
                file=sys.stderr,
            )
            os.kill(os.getpid(), _signal.SIGTERM)
            return


@app.get("/api/library")
def list_library():
    discover_books()
    results = []
    for b in sorted(_BOOKS.values(), key=lambda x: x.title.lower()):
        # Cheap check: does the summaries companion file exist?
        has_summaries = summaries_path_for(b.path).exists()
        results.append({
            "id": b.id, "title": b.title, "source": b.source,
            "size_mb": round(b.path.stat().st_size / (1024 * 1024), 1),
            "has_summaries": has_summaries,
        })
    return results


@app.delete("/api/library/{book_id}")
def delete_book(book_id: str):
    book = _BOOKS.get(book_id)
    if not book:
        raise HTTPException(404, "book not found")
    if book.source != "cortex":
        raise HTTPException(400, "cannot delete imported SmartReader caches")
    try:
        book.path.unlink(missing_ok=True)
        # Also remove the summaries companion file if it exists
        summaries_path_for(book.path).unlink(missing_ok=True)
    except Exception as e:
        raise HTTPException(500, f"delete failed: {e}")
    _BOOKS.pop(book_id, None)
    with _db() as c:
        c.execute("DELETE FROM attachments WHERE book_id = ?", (book_id,))
    return {"ok": True}


@app.post("/api/library/{book_id}/summarize")
async def summarize_existing(book_id: str):
    """Generate hierarchical summaries for a previously-indexed book.
    Used to retroactively add summaries to caches indexed before this feature
    existed, or to regenerate summaries after switching to a stronger model."""
    book = _BOOKS.get(book_id)
    if not book:
        raise HTTPException(404, "book not found")
    if book.source != "cortex":
        raise HTTPException(400, "cannot modify imported SmartReader caches")

    # Make sure the book's chunks are loaded
    book = ensure_loaded(book_id)
    if not book or not book.chunks:
        raise HTTPException(500, "book has no loadable chunks")

    job_id = "sum:" + uuid.uuid4().hex[:12]
    INDEX_JOBS[job_id] = {
        "status": "running", "progress": 0, "total": 100,
        "message": "Queued...", "filename": f"Summarize: {book.title}",
    }

    def _do_summarize() -> None:
        job = INDEX_JOBS[job_id]
        def progress(cur: int, tot: int, msg: str) -> None:
            job["progress"] = cur
            job["total"] = tot
            job["message"] = msg
        try:
            progress(5, 100, "Building sections...")
            sections = summarize_book(
                book.chunks, book.title, progress,
                progress_start=10, progress_end=98,
            )
            save_summaries(book.path, sections)
            # Refresh the in-memory book so the new summaries take effect
            book.sections = sections
            book._chunk_to_section = {
                cid: s.section_id for s in sections for cid in s.chunk_ids
            }
            job["status"] = "done"
            progress(100, 100, "Done")
        except Exception as e:
            job["status"] = "error"
            job["message"] = str(e)

    asyncio.create_task(asyncio.to_thread(_do_summarize))
    return {"job_id": job_id}


@app.post("/api/library/upload")
async def upload_document(file: UploadFile = File(...)):
    name = file.filename or "uploaded"
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise HTTPException(400, f"unsupported file type: {ext}")

    # Save upload to a temp location inside the library dir
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = LIBRARY_DIR / "_uploads"
    tmp_dir.mkdir(exist_ok=True)
    safe_name = re.sub(r"[^\w\s.-]", "", name).strip().replace(" ", "_")
    tmp_path = tmp_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"
    with open(tmp_path, "wb") as out:
        while True:
            buf = await file.read(1 << 20)
            if not buf:
                break
            out.write(buf)

    book_id = "job:" + uuid.uuid4().hex[:12]
    INDEX_JOBS[book_id] = {
        "status": "running", "progress": 0, "total": 100,
        "message": "Queued...", "filename": name,
    }
    # Pass the original (cleaned) filename stem as the display title.
    # Without this, src_path.stem keeps the UUID prefix from tmp_path
    # and the library shows titles like "abc12345 mybook" instead of "mybook".
    display_title = Path(safe_name).stem
    asyncio.create_task(run_index_job(book_id, tmp_path, display_title))
    return {"book_id": book_id}


@app.get("/api/library/jobs")
def list_jobs():
    return INDEX_JOBS


@app.get("/api/library/jobs/{job_id}")
def job_status(job_id: str):
    job = INDEX_JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


@app.get("/api/conversations")
def list_conversations():
    with _db() as c:
        rows = c.execute(
            "SELECT id, title, updated_at FROM conversations "
            "ORDER BY updated_at DESC LIMIT 100"
        ).fetchall()
    return [dict(r) for r in rows]


def _attachment_summary(cid: str) -> list[dict]:
    out = []
    for bid in get_attachments(cid):
        b = _BOOKS.get(bid)
        if b:
            out.append({"id": b.id, "title": b.title, "source": b.source})
    return out


@app.get("/api/conversations/{cid}")
def get_conversation(cid: str):
    with _db() as c:
        conv = c.execute("SELECT * FROM conversations WHERE id = ?", (cid,)).fetchone()
        if not conv:
            raise HTTPException(404, "conversation not found")
        msgs = c.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id",
            (cid,),
        ).fetchall()
    return {
        "id": cid, "title": conv["title"],
        "messages": [dict(m) for m in msgs],
        "attachments": _attachment_summary(cid),
    }


@app.delete("/api/conversations/{cid}")
def delete_conversation(cid: str):
    with _db() as c:
        c.execute("DELETE FROM messages WHERE conversation_id = ?", (cid,))
        c.execute("DELETE FROM attachments WHERE conversation_id = ?", (cid,))
        c.execute("DELETE FROM conversations WHERE id = ?", (cid,))
    return {"ok": True}


@app.post("/api/conversations/{cid}/attach")
def attach_book(cid: str, body: AttachRequest):
    if body.book_id not in _BOOKS:
        discover_books()
    if body.book_id not in _BOOKS:
        raise HTTPException(404, "book not found in library")
    with _db() as c:
        exists = c.execute("SELECT 1 FROM conversations WHERE id = ?", (cid,)).fetchone()
        if not exists:
            now = time.time()
            c.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (cid, "(new conversation)", now, now),
            )
        c.execute(
            "INSERT OR IGNORE INTO attachments (conversation_id, book_id, attached_at) "
            "VALUES (?, ?, ?)",
            (cid, body.book_id, time.time()),
        )
    ensure_loaded(body.book_id)
    return {"ok": True, "attachments": _attachment_summary(cid)}


@app.delete("/api/conversations/{cid}/attach/{book_id}")
def detach_book(cid: str, book_id: str):
    with _db() as c:
        c.execute(
            "DELETE FROM attachments WHERE conversation_id = ? AND book_id = ?",
            (cid, book_id),
        )
    return {"ok": True, "attachments": _attachment_summary(cid)}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    cid = req.conversation_id or uuid.uuid4().hex[:12]
    now = time.time()

    with _db() as c:
        exists = c.execute("SELECT 1 FROM conversations WHERE id = ?", (cid,)).fetchone()
        if not exists:
            title = (req.content[:60] + "…") if len(req.content) > 60 else req.content
            c.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (cid, title, now, now),
            )
        elif req.content:
            row = c.execute("SELECT title FROM conversations WHERE id = ?", (cid,)).fetchone()
            if row and row["title"] == "(new conversation)":
                title = (req.content[:60] + "…") if len(req.content) > 60 else req.content
                c.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, cid))

        history_rows = c.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id",
            (cid,),
        ).fetchall()
        history = [{"role": r["role"], "content": r["content"]} for r in history_rows]
        c.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (cid, "user", req.content, now),
        )

    book_ids = get_attachments(cid)
    rag_chunks: list[dict] = retrieve(book_ids, req.content, active_top_k()) if book_ids else []

    # Resolve the mode. If the user picked Default but multiple sources are
    # attached AND retrieval pulled from more than one, AND the active model
    # is strong enough (tier 2+), nudge toward cross_source. On the Lite tier
    # we leave it alone — 7B can't reliably fill a cross-reference table.
    requested_mode = req.mode if req.mode in MODES else DEFAULT_MODE
    distinct_sources = {c["book"] for c in rag_chunks}
    current_tier_rank = active_tier_info()["tier"]
    effective_mode = requested_mode
    if (requested_mode == DEFAULT_MODE
            and len(distinct_sources) > 1
            and current_tier_rank >= MODES["cross_source"]["min_tier"]):
        effective_mode = "cross_source"

    # Build the question with the mode scaffold applied.
    # Guard against picking a mode the active model can't handle (e.g. user
    # switched from Standard to Lite mid-conversation while Compare was selected).
    chosen_mode_def = MODES.get(effective_mode, MODES[DEFAULT_MODE])
    if chosen_mode_def.get("min_tier", 1) > current_tier_rank:
        effective_mode = DEFAULT_MODE
    scaffolded_question = apply_mode(req.content, effective_mode)

    if rag_chunks:
        system_prompt = SYSTEM_BASE + SYSTEM_RAG_SUFFIX
        user_message = f"{format_rag_context(rag_chunks)}\n\n---\n\nQuestion: {scaffolded_question}"
    else:
        system_prompt = SYSTEM_BASE
        user_message = scaffolded_question

    full_messages = (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": user_message}]
    )

    # Capture model name once at request time so a mid-stream switch doesn't
    # affect this in-flight query.
    model_for_request = active_model_name()

    async def event_stream() -> AsyncIterator[str]:
        yield _sse({
            "type": "meta", "conversation_id": cid, "model": model_for_request,
            "mode": {
                "requested": requested_mode,
                "effective": effective_mode,
                "auto_promoted": requested_mode != effective_mode,
                "label": MODES.get(effective_mode, {}).get("label", effective_mode),
            },
            "rag": {
                "active": bool(rag_chunks),
                "chunks": [
                    {"book": c["book"], "page": c["page"], "score": round(c["score"], 3)}
                    for c in rag_chunks
                ],
            },
        })
        full: list[str] = []
        try:
            client = ollama.AsyncClient()
            async for chunk in await client.chat(
                model=model_for_request, messages=full_messages, stream=True
            ):
                text = chunk.get("message", {}).get("content", "")
                if text:
                    full.append(text)
                    yield _sse({"type": "token", "content": text})
        except Exception as e:
            yield _sse({"type": "error", "message": str(e)})
            return

        full_text = "".join(full)
        with _db() as c:
            c.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (cid, "assistant", full_text, time.time()),
            )
            c.execute("UPDATE conversations SET updated_at = ? WHERE id = ?",
                      (time.time(), cid))
        yield _sse({"type": "done"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


# === HTML UI ==============================================================
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cortex</title>
<style>
:root {
  --bg:#0a0e14; --surface:#11151c; --surface-2:#1a1f29;
  --text:#c5d0d9; --text-dim:#6b7785;
  --accent:#5dd9ff; --accent-2:#ff9d5d; --accent-dim:#2a4a55;
  --border:#1f2530; --success:#67e480; --danger:#ff6b6b;
}
* { box-sizing:border-box; margin:0; padding:0; }
html, body {
  height:100%; font-family:'JetBrains Mono','SF Mono',Menlo,Consolas,monospace;
  font-size:14px; background:var(--bg); color:var(--text);
  -webkit-font-smoothing:antialiased;
}
.app { display:flex; height:100vh; }
.sidebar {
  width:320px; background:var(--surface); border-right:1px solid var(--border);
  display:flex; flex-direction:column; flex-shrink:0;
}
.sidebar header { padding:16px; border-bottom:1px solid var(--border); }
.sidebar h1 {
  font-size:16px; letter-spacing:0.08em; color:var(--accent);
  margin-bottom:6px; font-weight:600;
}
.model-info { display:flex; align-items:center; gap:8px; font-size:12px; color:var(--text-dim); }
.model-selector {
  display:flex; align-items:center; gap:8px; font-size:12px; color:var(--text-dim);
  cursor:pointer; padding:4px 6px; border-radius:4px; margin:-4px -6px;
  transition:background 0.15s;
  user-select:none;
}
.model-selector:hover { background:var(--surface-2); color:var(--text); }
.model-chevron { margin-left:auto; opacity:0.6; font-size:10px; }
.tier-menu {
  position:absolute; top:60px; left:12px; right:12px; z-index:50;
  background:var(--surface-2); border:1px solid var(--border); border-radius:6px;
  padding:4px; box-shadow:0 8px 24px rgba(0,0,0,0.4);
}
.tier-option {
  padding:8px 10px; border-radius:4px; cursor:pointer; font-size:12px;
  color:var(--text-dim); transition:all 0.1s;
}
.tier-option:hover { background:var(--surface); color:var(--text); }
.tier-option.active { background:var(--accent-dim); color:var(--accent); }
.tier-option .tier-label { font-weight:600; display:block; margin-bottom:2px; color:inherit; }
.tier-option .tier-desc { font-size:10.5px; opacity:0.75; line-height:1.4; }
.tier-option .tier-pulled { font-size:9px; color:var(--success); margin-left:6px; }
.tier-option .tier-not-pulled { font-size:9px; color:var(--accent-2); margin-left:6px; }
.tier-option.locked { opacity:0.5; cursor:not-allowed; }
.tier-option.locked:hover { background:transparent; }
.tier-switching {
  position:fixed; inset:0; background:rgba(10,14,20,0.85); z-index:200;
  display:flex; align-items:center; justify-content:center;
  font-size:14px; color:var(--accent); letter-spacing:0.04em;
}
.dot { width:8px; height:8px; border-radius:50%; background:var(--success);
  box-shadow:0 0 8px rgba(103,228,128,0.5); }
#new-chat {
  margin:12px; padding:9px 12px; background:transparent;
  border:1px solid var(--border); color:var(--text); font-family:inherit;
  font-size:13px; cursor:pointer; border-radius:4px; text-align:left;
  transition:all 0.15s;
}
#new-chat:hover { border-color:var(--accent-dim); color:var(--accent); }
.section-label {
  display:flex; align-items:center; justify-content:space-between;
  font-size:10px; letter-spacing:0.14em; color:var(--text-dim);
  padding:14px 16px 6px; font-weight:600;
}
.section-label .add-btn {
  background:none; border:none; color:var(--accent-2); cursor:pointer;
  font-size:14px; padding:0 4px; opacity:0.7;
}
.section-label .add-btn:hover { opacity:1; }
.scrollarea { flex:1; overflow-y:auto; min-height:0; }
.split { display:flex; flex-direction:column; flex:1; min-height:0; }
.split > .scrollarea { flex:1 1 50%; }
.split > .divider { height:1px; background:var(--border); margin:4px 12px; }

#conversation-list, #library-list { list-style:none; padding:0 8px 8px; }
#conversation-list li, #library-list li {
  padding:7px 10px; border-radius:4px; cursor:pointer; font-size:12.5px;
  color:var(--text-dim); margin-bottom:1px; display:flex;
  align-items:center; justify-content:space-between; gap:6px;
}
#conversation-list li:hover, #library-list li:hover {
  background:var(--surface-2); color:var(--text);
}
#conversation-list li.active { background:var(--surface-2); color:var(--accent); }
#conversation-list li span, #library-list li span.title {
  flex:1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.delete, .attach-btn {
  visibility:hidden; background:none; border:none; color:var(--text-dim);
  cursor:pointer; font-size:14px; padding:0 4px; line-height:1;
}
#conversation-list li:hover .delete,
#library-list li:hover .attach-btn,
#library-list li:hover .delete-book { visibility:visible; }
.delete:hover, .delete-book:hover { color:var(--danger); }
.attach-btn:hover { color:var(--accent-2); }
#library-list li.attached {
  background:rgba(255,157,93,0.08); color:var(--accent-2);
}
#library-list li.attached .attach-btn { visibility:visible; color:var(--accent-2); }
#library-list li .source-tag {
  font-size:9px; opacity:0.5; padding:0 4px; letter-spacing:0.06em;
}
#library-list li .delete-book {
  visibility:hidden; background:none; border:none; color:var(--text-dim);
  cursor:pointer; font-size:13px; padding:0 4px;
}
#library-list .indexing {
  font-size:11px; color:var(--accent-2); padding:6px 10px;
  background:var(--surface-2); border-radius:4px; margin:0 2px 4px;
}
#library-list .indexing .bar {
  height:3px; background:var(--surface); margin-top:4px; border-radius:2px;
  overflow:hidden;
}
#library-list .indexing .bar > span {
  display:block; height:100%; background:var(--accent-2);
  transition:width 0.3s;
}
.empty-library {
  font-size:11.5px; color:var(--text-dim); padding:8px 16px 16px;
  line-height:1.6; text-align:center;
}

.chat { flex:1; display:flex; flex-direction:column; min-width:0; position:relative; }
.attachments-bar {
  padding:10px 24px; border-bottom:1px solid var(--border);
  display:none; flex-wrap:wrap; gap:6px; align-items:center;
  background:var(--surface);
}
.attachments-bar.visible { display:flex; }
.attachments-bar .label {
  font-size:10px; letter-spacing:0.14em; color:var(--text-dim);
  margin-right:4px; font-weight:600;
}
.chip {
  background:var(--surface-2); border:1px solid var(--accent-dim);
  color:var(--accent-2); padding:3px 8px; border-radius:3px;
  font-size:11.5px; display:inline-flex; align-items:center; gap:6px;
}
.chip .x { cursor:pointer; opacity:0.6; }
.chip .x:hover { opacity:1; color:var(--danger); }

.messages { flex:1; overflow-y:auto; padding:24px 0 8px; }
.message {
  max-width:820px; margin:0 auto 24px; padding:0 24px;
  display:flex; gap:14px; align-items:flex-start;
}
.message .role {
  font-size:10px; letter-spacing:0.12em; color:var(--text-dim);
  width:56px; padding-top:3px; flex-shrink:0; font-weight:600;
}
.message.user .role { color:var(--accent); }
.message.system-info .role { color:var(--accent-2); }
.message .content { flex:1; line-height:1.65; word-wrap:break-word; min-width:0; }
.message .content > *:last-child { margin-bottom:0; }
.message .content pre {
  background:var(--surface); border:1px solid var(--border); border-radius:4px;
  padding:12px 14px; margin:10px 0; overflow-x:auto; font-size:13px; line-height:1.5;
}
.message .content pre code { color:var(--text); }
.message .content :not(pre) > code {
  background:var(--surface-2); padding:1px 6px; border-radius:3px;
  color:var(--accent); font-size:0.92em;
}
.message .content h1, .message .content h2, .message .content h3 {
  margin:18px 0 8px; color:var(--text); font-weight:600;
}
.message .content h1 { font-size:18px; }
.message .content h2 { font-size:16px; }
.message .content h3 { font-size:14px; }
.message .content strong { color:var(--text); font-weight:600; }
.rag-info {
  font-size:11px; color:var(--text-dim); margin:6px 0 12px;
  padding:8px 10px; background:var(--surface); border-left:2px solid var(--accent-2);
  border-radius:0 4px 4px 0;
}
.rag-info b { color:var(--accent-2); font-weight:600; }
.rag-info .src { display:block; margin-top:3px; }

.input-area { border-top:1px solid var(--border); background:var(--surface); padding:14px 24px 18px; }
.mode-bar {
  max-width:820px; margin:0 auto 10px; display:flex; flex-wrap:wrap; gap:6px;
}
.mode-pill {
  background:transparent; border:1px solid var(--border); color:var(--text-dim);
  padding:4px 10px; border-radius:14px; font-family:inherit; font-size:11px;
  cursor:pointer; transition:all 0.15s; letter-spacing:0.04em;
}
.mode-pill:hover { color:var(--text); border-color:var(--accent-dim); }
.mode-pill.active {
  background:var(--accent-dim); border-color:var(--accent); color:var(--accent);
}
.mode-pill .mode-help {
  display:inline-block; margin-left:4px; opacity:0.5; cursor:help;
}
.mode-promoted {
  font-size:11px; color:var(--accent-2); font-style:italic;
  padding:2px 0 6px;
}
.input-wrap {
  max-width:820px; margin:0 auto; display:flex; gap:10px; align-items:flex-end;
}
#input {
  flex:1; background:var(--surface-2); border:1px solid var(--border);
  border-radius:6px; padding:10px 12px; color:var(--text);
  font-family:inherit; font-size:14px; resize:none;
  min-height:40px; max-height:240px; outline:none; line-height:1.5;
}
#input:focus { border-color:var(--accent-dim); }
#send {
  background:var(--accent); color:var(--bg); border:none;
  padding:10px 20px; border-radius:6px; font-family:inherit;
  font-size:13px; font-weight:600; cursor:pointer; height:40px;
  letter-spacing:0.06em;
}
#send:hover { opacity:0.85; }
#send:disabled { opacity:0.4; cursor:not-allowed; }

.empty-state {
  text-align:center; color:var(--text-dim); padding-top:80px;
  max-width:540px; margin:0 auto;
}
.empty-state h2 {
  color:var(--accent); font-size:18px; margin-bottom:10px;
  letter-spacing:0.04em; font-weight:600;
}
.empty-state p { font-size:13px; line-height:1.7; margin-bottom:8px; }
.error { color:var(--danger); }
.cursor {
  display:inline-block; width:8px; height:14px; background:var(--accent);
  vertical-align:text-bottom; margin-left:1px; animation:blink 1s steps(2) infinite;
}
@keyframes blink { to { opacity:0; } }

.dropzone-overlay {
  position:absolute; inset:0; background:rgba(10,14,20,0.92);
  border:2px dashed var(--accent-2); display:none;
  align-items:center; justify-content:center; z-index:100;
  font-size:18px; color:var(--accent-2); pointer-events:none;
  letter-spacing:0.04em;
}
.dropzone-overlay.active { display:flex; }

::-webkit-scrollbar { width:8px; height:8px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:4px; }
::-webkit-scrollbar-thumb:hover { background:var(--text-dim); }
.hint { font-size:11px; color:var(--text-dim); max-width:820px; margin:6px auto 0; padding:0 24px; }
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <header>
      <h1>Cortex</h1>
      <div class="model-selector" id="model-selector" title="Click to switch model">
        <span class="dot"></span>
        <span id="model-name">loading…</span>
        <span class="model-chevron">▾</span>
      </div>
      <div class="tier-menu" id="tier-menu" style="display:none;"></div>
    </header>
    <button id="new-chat">+ new chat</button>
    <div class="split">
      <div class="scrollarea">
        <div class="section-label">CONVERSATIONS</div>
        <ul id="conversation-list"></ul>
      </div>
      <div class="divider"></div>
      <div class="scrollarea">
        <div class="section-label">
          LIBRARY
          <button class="add-btn" id="upload-btn" title="Add document">＋</button>
        </div>
        <ul id="library-list"></ul>
        <div class="empty-library" id="library-empty" style="display:none;">
          No documents yet.<br><br>
          Click ＋ to upload, or drop PDF/EPUB/DOCX/TXT/MD anywhere.
        </div>
      </div>
    </div>
    <input type="file" id="file-input" multiple accept=".pdf,.epub,.docx,.txt,.md,.markdown" style="display:none;">
  </aside>
  <main class="chat">
    <div class="dropzone-overlay" id="dropzone">drop files to index</div>
    <div id="attachments-bar" class="attachments-bar">
      <span class="label">SOURCES</span>
      <div id="attachments-chips"></div>
    </div>
    <div id="messages" class="messages"></div>
    <div class="input-area">
      <div class="mode-bar" id="mode-bar"></div>
      <div class="input-wrap">
        <textarea id="input" rows="1" placeholder="Ask anything. Drop files to index. Attach books from the sidebar to query them. Enter to send, Shift+Enter for newline."></textarea>
        <button id="send">SEND</button>
      </div>
      <div class="hint">offline · no data leaves this machine</div>
    </div>
  </main>
</div>
<script>
const state = {
  currentConversationId: null, isStreaming: false,
  attachments: [], library: [], jobs: {},
  modes: [], currentMode: 'default',
  tiers: [], activeTier: null, currentTierRank: 1, overrideActive: false,
};
const $ = (s) => document.querySelector(s);

function escapeHtml(t) {
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}
function parseMarkdown(text) {
  const codeBlocks = [];
  text = text.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    const idx = codeBlocks.length; codeBlocks.push(escapeHtml(code.replace(/\n+$/,'')));
    return '\u0000CODE'+idx+'\u0000';
  });
  const inlineCode = [];
  text = text.replace(/`([^`\n]+)`/g, (_, code) => {
    const idx = inlineCode.length; inlineCode.push(escapeHtml(code));
    return '\u0000INLINE'+idx+'\u0000';
  });
  text = escapeHtml(text);
  text = text.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  text = text.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  text = text.replace(/^# (.+)$/gm, '<h1>$1</h1>');
  text = text.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
  text = text.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, '$1<em>$2</em>');
  text = text.replace(/\n/g, '<br>');
  text = text.replace(/\u0000CODE(\d+)\u0000/g, (_, idx) => '<pre><code>'+codeBlocks[idx]+'</code></pre>');
  text = text.replace(/\u0000INLINE(\d+)\u0000/g, (_, idx) => '<code>'+inlineCode[idx]+'</code>');
  return text;
}
function appendMessage(role, content) {
  const messages = $('#messages');
  const empty = messages.querySelector('.empty-state');
  if (empty) empty.remove();
  const div = document.createElement('div');
  div.className = 'message ' + role;
  const label = role==='user' ? 'YOU' : (role==='system-info' ? 'RAG' : 'AI');
  div.innerHTML = '<div class="role">'+label+'</div><div class="content">'+parseMarkdown(content)+'</div>';
  messages.appendChild(div);
  scrollToBottom();
  return div;
}
function scrollToBottom() { const m=$('#messages'); m.scrollTop=m.scrollHeight; }
function showEmptyState() {
  const hasLib = state.library.length > 0;
  const note = hasLib
    ? 'Click ⊕ next to a book in the sidebar to ground answers in its content.'
    : 'Drop a PDF, EPUB, DOCX, or text file anywhere to index it.';
  $('#messages').innerHTML =
    '<div class="empty-state"><h2>local. offline. yours.</h2>'
    + '<p>Ask anything. Conversations are saved locally to SQLite.</p>'
    + '<p style="color:var(--accent-2);">'+note+'</p></div>';
}
async function loadConversations() {
  const res = await fetch('/api/conversations');
  const conversations = await res.json();
  const list = $('#conversation-list');
  list.innerHTML = '';
  for (const conv of conversations) {
    const li = document.createElement('li');
    li.dataset.id = conv.id;
    if (conv.id === state.currentConversationId) li.classList.add('active');
    const span = document.createElement('span');
    span.textContent = conv.title;
    span.onclick = () => loadConversation(conv.id);
    const del = document.createElement('button');
    del.className='delete'; del.textContent='×'; del.title='delete';
    del.onclick = (e) => { e.stopPropagation(); deleteConversation(conv.id); };
    li.appendChild(span); li.appendChild(del);
    list.appendChild(li);
  }
}
async function loadLibrary() {
  const res = await fetch('/api/library');
  state.library = await res.json();
  renderLibrary();
}
function renderLibrary() {
  const list = $('#library-list');
  const empty = $('#library-empty');
  list.innerHTML = '';
  // Render in-progress jobs first
  for (const [jid, job] of Object.entries(state.jobs)) {
    if (job.status === 'done' || job.status === 'error_acked') continue;
    const div = document.createElement('div');
    div.className = 'indexing';
    const filename = escapeHtml(job.filename || 'document');
    if (job.status === 'error') {
      div.innerHTML = filename + ' <span style="color:var(--danger)">— ' + escapeHtml(job.message || 'failed') + '</span>';
    } else {
      const pct = Math.round((job.progress/job.total)*100);
      div.innerHTML = filename + ' — ' + escapeHtml(job.message || '...') + ' (' + pct + '%)<div class="bar"><span style="width:' + pct + '%"></span></div>';
    }
    list.appendChild(div);
  }
  if (!state.library.length && !Object.keys(state.jobs).length) {
    empty.style.display = 'block';
    return;
  }
  empty.style.display = 'none';
  const attachedIds = new Set(state.attachments.map(a => a.id));
  for (const book of state.library) {
    const li = document.createElement('li');
    if (attachedIds.has(book.id)) li.classList.add('attached');
    const title = document.createElement('span');
    title.className='title';
    const sourceTag = book.source !== 'cortex' ? ' <span class="source-tag">'+book.source+'</span>' : '';
    title.innerHTML = escapeHtml(book.title) + sourceTag;
    title.title = book.title + ' (' + book.size_mb + ' MB)';
    title.onclick = () => toggleAttach(book.id);
    const delBtn = document.createElement('button');
    delBtn.className = 'delete-book';
    delBtn.textContent = '×';
    delBtn.title = book.source === 'cortex' ? 'delete from library' : 'managed by SmartReader';
    if (book.source !== 'cortex') delBtn.style.display = 'none';
    delBtn.onclick = (e) => { e.stopPropagation(); deleteBook(book.id, book.title); };
    const attachBtn = document.createElement('button');
    attachBtn.className='attach-btn';
    attachBtn.textContent = attachedIds.has(book.id) ? '✕' : '⊕';
    attachBtn.title = attachedIds.has(book.id) ? 'detach' : 'attach to current chat';
    attachBtn.onclick = (e) => { e.stopPropagation(); toggleAttach(book.id); };
    li.appendChild(title); li.appendChild(delBtn); li.appendChild(attachBtn);
    list.appendChild(li);
  }
}
async function deleteBook(bookId, title) {
  if (!confirm('Delete "' + title + '" from library? This removes the index cache but not the original file.')) return;
  const res = await fetch('/api/library/' + encodeURIComponent(bookId), {method:'DELETE'});
  if (!res.ok) { console.error('delete failed'); return; }
  state.attachments = state.attachments.filter(a => a.id !== bookId);
  await loadLibrary();
  renderAttachmentsBar();
}
function ensureConversationId() {
  if (!state.currentConversationId) {
    state.currentConversationId = 'c' + Math.random().toString(36).slice(2, 14);
  }
  return state.currentConversationId;
}
async function toggleAttach(bookId) {
  const cid = ensureConversationId();
  const isAttached = state.attachments.some(a => a.id === bookId);
  if (isAttached) {
    const res = await fetch('/api/conversations/'+cid+'/attach/'+encodeURIComponent(bookId), {method:'DELETE'});
    state.attachments = (await res.json()).attachments || [];
  } else {
    const res = await fetch('/api/conversations/'+cid+'/attach', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({book_id: bookId}),
    });
    if (!res.ok) return;
    state.attachments = (await res.json()).attachments || [];
  }
  renderLibrary(); renderAttachmentsBar();
  await loadConversations();
}
function renderAttachmentsBar() {
  const bar = $('#attachments-bar'); const chips = $('#attachments-chips');
  chips.innerHTML = '';
  if (!state.attachments.length) { bar.classList.remove('visible'); return; }
  bar.classList.add('visible');
  for (const a of state.attachments) {
    const chip = document.createElement('span');
    chip.className='chip';
    chip.innerHTML = escapeHtml(a.title) + ' <span class="x" title="detach">✕</span>';
    chip.querySelector('.x').onclick = () => toggleAttach(a.id);
    chips.appendChild(chip);
  }
}
async function loadConversation(id) {
  const res = await fetch('/api/conversations/'+id);
  if (!res.ok) return;
  const data = await res.json();
  state.currentConversationId = id;
  state.attachments = data.attachments || [];
  $('#messages').innerHTML='';
  for (const m of data.messages) appendMessage(m.role, m.content);
  if (!data.messages.length) showEmptyState();
  renderLibrary(); renderAttachmentsBar();
  await loadConversations();
}
async function deleteConversation(id) {
  await fetch('/api/conversations/'+id, {method:'DELETE'});
  if (state.currentConversationId === id) newChat();
  await loadConversations();
}
function newChat() {
  state.currentConversationId = null;
  state.attachments = [];
  showEmptyState();
  renderLibrary(); renderAttachmentsBar();
  loadConversations();
  $('#input').focus();
}
function renderRagInfo(rag) {
  if (!rag || !rag.active || !rag.chunks?.length) return;
  const messages = $('#messages');
  const div = document.createElement('div');
  div.className='message system-info';
  let html = '<div class="role">RAG</div><div class="content"><div class="rag-info">';
  html += '<b>retrieved '+rag.chunks.length+' excerpt(s):</b>';
  for (const c of rag.chunks) {
    html += '<span class="src">→ '+escapeHtml(c.book)+', p. '+c.page+' <span style="opacity:0.6">'+c.score.toFixed(3)+'</span></span>';
  }
  html += '</div></div>';
  div.innerHTML = html;
  messages.appendChild(div); scrollToBottom();
}
async function loadTiers() {
  try {
    const res = await fetch('/api/model/tiers');
    const data = await res.json();
    state.tiers = data.tiers || [];
    state.activeTier = data.active;
    state.overrideActive = data.override_active || false;
  } catch (e) {
    console.error('failed to load tiers', e);
  }
  renderActiveModelBadge();
}
function renderActiveModelBadge() {
  const nameEl = $('#model-name');
  if (!nameEl) return;
  const active = state.tiers.find(t => t.id === state.activeTier);
  if (state.overrideActive) {
    nameEl.textContent = state.activeTier || 'custom';
  } else if (active) {
    nameEl.textContent = active.label;
  } else {
    nameEl.textContent = 'no model';
  }
}
function toggleTierMenu(force) {
  const menu = $('#tier-menu');
  if (!menu) return;
  const willOpen = force === undefined ? menu.style.display === 'none' : force;
  if (state.overrideActive && willOpen) {
    alert('CORTEX_MODEL env var is set — model switching is disabled.\nUnset CORTEX_MODEL to use the tier picker.');
    return;
  }
  menu.style.display = willOpen ? 'block' : 'none';
  if (willOpen) renderTierMenu();
}
async function renderTierMenu() {
  const menu = $('#tier-menu');
  if (!menu) return;
  // Fetch installed Ollama models so we can mark each tier as pulled or not.
  let installed = new Set();
  try {
    const res = await fetch('/api/ollama/installed');
    const data = await res.json();
    for (const n of data.names || []) installed.add(n);
  } catch (e) { /* non-fatal */ }
  menu.innerHTML = '';
  for (const tier of state.tiers) {
    const opt = document.createElement('div');
    const isActive = tier.id === state.activeTier;
    // Lenient name matching: normalize :latest, .gguf, case differences,
    // but require the tag to match if the configured name has one.
    const norm = (s) => {
      let v = (s || '').toLowerCase().trim();
      if (v.endsWith('.gguf')) v = v.slice(0, -5);
      if (v.endsWith(':latest')) v = v.slice(0, -7);
      return v;
    };
    const targetNorm = norm(tier.ollama_name);
    const targetHasTag = targetNorm.includes(':');
    const isPulled = Array.from(installed).some(n => {
      const nNorm = norm(n);
      if (nNorm === targetNorm) return true;
      if (!targetHasTag && nNorm.startsWith(targetNorm + ':')) return true;
      return false;
    });
    opt.className = 'tier-option' + (isActive ? ' active' : '');
    const pulledTag = isPulled
      ? '<span class="tier-pulled">✓ installed</span>'
      : '<span class="tier-not-pulled">⚠ run: ollama pull ' + escapeHtml(tier.ollama_name) + '</span>';
    opt.innerHTML =
      '<div class="tier-label">' + escapeHtml(tier.label) + pulledTag + '</div>' +
      '<div class="tier-desc">' + escapeHtml(tier.description) + '</div>';
    if (!isActive) {
      opt.onclick = () => switchTier(tier.id);
    }
    menu.appendChild(opt);
  }
}
async function switchTier(tierId) {
  // Show a brief overlay so the user knows the switch is in progress.
  const overlay = document.createElement('div');
  overlay.className = 'tier-switching';
  overlay.textContent = 'switching model...';
  document.body.appendChild(overlay);
  try {
    const res = await fetch('/api/model/switch', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({tier: tierId}),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert('Switch failed: ' + (err.detail || res.status));
      return;
    }
    state.activeTier = tierId;
    renderActiveModelBadge();
    toggleTierMenu(false);
    // Refresh modes since tier changed (some may now be available/unavailable).
    await loadModes();
  } catch (e) {
    alert('Switch error: ' + e.message);
  } finally {
    overlay.remove();
  }
}
async function loadModes() {
  try {
    const res = await fetch('/api/modes');
    const data = await res.json();
    state.modes = data.modes || [];
    state.currentTierRank = data.current_tier_rank || 1;
    if (data.default && !state.currentMode) state.currentMode = data.default;
    // If the previously-selected mode is now locked behind a higher tier,
    // fall back to default so we don't silently send a question with a
    // scaffold the model can't handle.
    const cur = state.modes.find(m => m.id === state.currentMode);
    if (cur && cur.min_tier > state.currentTierRank) {
      state.currentMode = data.default;
    }
  } catch (e) {
    console.error('failed to load modes', e);
  }
  renderModeBar();
}
function renderModeBar() {
  const bar = $('#mode-bar');
  if (!bar) return;
  bar.innerHTML = '';
  for (const mode of state.modes) {
    const locked = mode.min_tier > (state.currentTierRank || 1);
    if (locked) continue;  // hide unavailable modes entirely
    const btn = document.createElement('button');
    btn.className = 'mode-pill' + (mode.id === state.currentMode ? ' active' : '');
    btn.textContent = mode.label;
    btn.title = mode.description;
    btn.onclick = () => {
      state.currentMode = mode.id;
      renderModeBar();
    };
    bar.appendChild(btn);
  }
}
function renderModePromotion(modeMeta) {
  const messages = $('#messages');
  const div = document.createElement('div');
  div.className = 'message system-info';
  const requested = state.modes.find(m => m.id === modeMeta.requested);
  const effective = state.modes.find(m => m.id === modeMeta.effective);
  const reqLabel = requested ? requested.label : modeMeta.requested;
  const effLabel = effective ? effective.label : modeMeta.effective;
  div.innerHTML = '<div class="role">MODE</div><div class="content"><div class="mode-promoted">' +
    'Multiple sources attached — switched from <b>' + escapeHtml(reqLabel) +
    '</b> to <b>' + escapeHtml(effLabel) + '</b> for this question.</div></div>';
  messages.appendChild(div); scrollToBottom();
}
async function sendMessage() {
  if (state.isStreaming) return;
  const input = $('#input'); const content = input.value.trim();
  if (!content) return;
  input.value=''; input.style.height='auto';
  state.isStreaming = true; $('#send').disabled = true;
  appendMessage('user', content);
  const aiDiv = appendMessage('assistant', '');
  const aiContent = aiDiv.querySelector('.content');
  aiContent.innerHTML = '<span class="cursor"></span>';
  let fullText = ''; let ragShown = false;
  try {
    const response = await fetch('/api/chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        conversation_id: state.currentConversationId,
        content: content,
        mode: state.currentMode,
      }),
    });
    if (!response.ok) throw new Error('HTTP '+response.status);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream:true});
      const events = buffer.split('\n\n');
      buffer = events.pop();
      for (const event of events) {
        if (!event.startsWith('data: ')) continue;
        let data; try { data = JSON.parse(event.slice(6)); } catch { continue; }
        if (data.type === 'meta') {
          state.currentConversationId = data.conversation_id;
          if (data.mode && data.mode.auto_promoted) {
            renderModePromotion(data.mode);
          }
          if (!ragShown && data.rag) {
            const messages = $('#messages');
            messages.removeChild(aiDiv);
            renderRagInfo(data.rag);
            messages.appendChild(aiDiv);
            ragShown = true;
          }
        } else if (data.type === 'token') {
          fullText += data.content;
          aiContent.innerHTML = parseMarkdown(fullText) + '<span class="cursor"></span>';
          scrollToBottom();
        } else if (data.type === 'done') {
          aiContent.innerHTML = parseMarkdown(fullText);
          await loadConversations();
        } else if (data.type === 'error') {
          aiContent.innerHTML = '<span class="error">Error: '+escapeHtml(data.message)+'</span>';
        }
      }
    }
  } catch (e) {
    aiContent.innerHTML = '<span class="error">Connection error: '+escapeHtml(e.message)+'</span>';
  } finally {
    state.isStreaming = false;
    $('#send').disabled = false;
    $('#input').focus();
  }
}

// === Upload + drag-and-drop ==============================================
async function uploadFiles(fileList) {
  for (const file of fileList) {
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res = await fetch('/api/library/upload', {method:'POST', body: fd});
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        alert('Upload failed: ' + (err.detail || res.status));
        continue;
      }
      const data = await res.json();
      state.jobs[data.book_id] = {
        status:'running', progress:0, total:100,
        message:'Queued...', filename: file.name,
      };
    } catch (e) {
      alert('Upload error: ' + e.message);
    }
  }
  renderLibrary();
}
async function pollJobs() {
  const activeIds = Object.entries(state.jobs)
    .filter(([_, j]) => j.status === 'running')
    .map(([id, _]) => id);
  if (!activeIds.length) return;
  const all = await (await fetch('/api/library/jobs')).json();
  let needsLibraryReload = false;
  for (const [jid, job] of Object.entries(all)) {
    if (state.jobs[jid] || activeIds.includes(jid)) {
      const prevStatus = state.jobs[jid]?.status;
      state.jobs[jid] = {...state.jobs[jid], ...job};
      if (job.status === 'done' && prevStatus !== 'done') {
        needsLibraryReload = true;
        // Auto-clean done jobs after a short delay
        setTimeout(() => { delete state.jobs[jid]; renderLibrary(); }, 2000);
      }
      if (job.status === 'error' && prevStatus !== 'error') {
        // Keep error visible; ack on next click
        setTimeout(() => { delete state.jobs[jid]; renderLibrary(); }, 8000);
      }
    }
  }
  if (needsLibraryReload) await loadLibrary();
  renderLibrary();
}

let dragDepth = 0;
function setupDragDrop() {
  const overlay = $('#dropzone');
  document.body.addEventListener('dragenter', (e) => {
    if (!e.dataTransfer?.types?.includes('Files')) return;
    dragDepth++; overlay.classList.add('active');
  });
  document.body.addEventListener('dragleave', () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) overlay.classList.remove('active');
  });
  document.body.addEventListener('dragover', (e) => { e.preventDefault(); });
  document.body.addEventListener('drop', (e) => {
    e.preventDefault();
    dragDepth = 0; overlay.classList.remove('active');
    if (e.dataTransfer?.files?.length) uploadFiles(e.dataTransfer.files);
  });
}

async function init() {
  await loadTiers();
  await loadLibrary();
  await loadModes();
  showEmptyState();
  await loadConversations();
  $('#new-chat').onclick = newChat;
  $('#send').onclick = sendMessage;
  $('#upload-btn').onclick = () => $('#file-input').click();
  $('#file-input').onchange = (e) => {
    if (e.target.files.length) uploadFiles(e.target.files);
    e.target.value = '';
  };
  $('#model-selector').onclick = (e) => {
    e.stopPropagation();
    toggleTierMenu();
  };
  // Close menu when clicking outside
  document.addEventListener('click', (e) => {
    const menu = $('#tier-menu');
    if (!menu || menu.style.display === 'none') return;
    if (!menu.contains(e.target) && !$('#model-selector').contains(e.target)) {
      toggleTierMenu(false);
    }
  });
  const input = $('#input');
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 240) + 'px';
  });
  input.focus();
  setupDragDrop();
  setInterval(pollJobs, 1000);
  setInterval(loadLibrary, 30000);

  // === Heartbeat + clean shutdown ========================================
  // While the tab is open, ping the server every 10 seconds. The server's
  // watchdog uses this to know the user hasn't closed the tab. If we miss
  // pings for ~30s the server shuts itself down so port 8000 is freed.
  fetch('/api/heartbeat', {method:'POST'}).catch(() => {});
  setInterval(() => {
    fetch('/api/heartbeat', {method:'POST'}).catch(() => {});
  }, 10000);

  // When the tab closes (window close, navigation away, refresh), tell the
  // server explicitly. sendBeacon is the right tool — it fires reliably
  // during unload where regular fetch() may be cancelled by the browser.
  window.addEventListener('pagehide', () => {
    try {
      navigator.sendBeacon('/api/shutdown');
    } catch (e) { /* best-effort */ }
  });
}
init();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


# === Entry point ==========================================================
if __name__ == "__main__":
    print(f"Cortex starting on http://{HOST}:{PORT}", file=sys.stderr)
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
