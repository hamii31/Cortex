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
MODEL = os.environ.get("CORTEX_MODEL", "qwen2.5:32b-instruct-q4_K_L")
EMBED_MODEL = os.environ.get("CORTEX_EMBED_MODEL", "nomic-embed-text")
HOST = os.environ.get("CORTEX_HOST", "127.0.0.1")
PORT = int(os.environ.get("CORTEX_PORT", "8000"))

# Number of retrieved chunks per query. Smaller models choke on long
# context — 4 is a safer default for 7B-class models, 6 works well for
# larger or partially-offloaded models that have more headroom.
TOP_K = int(os.environ.get("CORTEX_TOP_K", "6"))

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
    """Optional: also read from SmartReader's cache if it's set."""
    override = os.environ.get("CORTEX_SMARTREADER_CACHE")
    if override:
        return Path(override)
    # Auto-detect: if SmartReader cache exists at standard path, use it too
    if sys.platform == "win32":
        sr = Path(os.environ.get("APPDATA", "")) / "SmartReader" / "cache"
    elif sys.platform == "darwin":
        sr = Path.home() / "Library" / "Application Support" / "SmartReader" / "cache"
    else:
        sr = Path.home() / ".config" / "smartreader" / "cache"
    return sr if sr.exists() else None


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

    def load(self) -> None:
        self.chunks = load_pickle_chunks(self.path)
        if not self.chunks:
            self.matrix = None
            return
        embs = np.array([c.embedding for c in self.chunks], dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
        self.matrix = embs / norms

    def unload(self) -> None:
        self.chunks = []
        self.matrix = None


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


def retrieve(book_ids: list[str], query: str, top_k: int = TOP_K) -> list[dict]:
    if not book_ids:
        return []
    qvec = embed_query(query)
    if qvec is None:
        return []
    pool: list[tuple[float, str, TextChunk]] = []
    for bid in book_ids:
        book = ensure_loaded(bid)
        if not book or book.matrix is None:
            continue
        scores = book.matrix @ qvec
        k = min(top_k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        for i in idx:
            pool.append((float(scores[i]), book.title, book.chunks[i]))
    pool.sort(key=lambda t: t[0], reverse=True)
    pool = pool[:top_k]
    return [
        {"score": s, "book": title, "page": chunk.page_number, "text": chunk.text}
        for s, title, chunk in pool
    ]


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

Rules:
- Cite specific pages when you draw on the excerpts. Format: [Title, p. N].
- For non-PDF sources, "p. N" maps to chapter (EPUB) or section (DOCX/text) — cite it the same way.
- If the excerpts don't contain the answer, say so plainly. Do not invent citations or fill gaps with general knowledge presented as if it came from the source.
- You may use general knowledge to frame or connect what the excerpts say, but be explicit when doing so.
- If excerpts contradict each other, point that out instead of forcing a synthesis."""


def format_rag_context(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(
            f"[Excerpt {i}] {c['book']}, p. {c['page']} (relevance {c['score']:.2f})\n"
            f"{c['text'].strip()}"
        )
    return "Retrieved excerpts from attached sources:\n\n" + "\n\n---\n\n".join(blocks)


# === API schemas ==========================================================
class ChatRequest(BaseModel):
    conversation_id: str | None = None
    content: str


class AttachRequest(BaseModel):
    book_id: str


# === FastAPI ==============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    _db_init()
    discover_books()
    print(f"Cortex: model={MODEL} embed={EMBED_MODEL}", file=sys.stderr)
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
    return {
        "model": MODEL,
        "embed_model": EMBED_MODEL,
        "library_dir": str(LIBRARY_DIR),
        "smartreader_dir": str(SMARTREADER_DIR) if SMARTREADER_DIR else None,
        "supported_ext": sorted(SUPPORTED_EXT),
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
    return [
        {
            "id": b.id, "title": b.title, "source": b.source,
            "size_mb": round(b.path.stat().st_size / (1024 * 1024), 1),
        }
        for b in sorted(_BOOKS.values(), key=lambda x: x.title.lower())
    ]


@app.delete("/api/library/{book_id}")
def delete_book(book_id: str):
    book = _BOOKS.get(book_id)
    if not book:
        raise HTTPException(404, "book not found")
    if book.source != "cortex":
        raise HTTPException(400, "cannot delete imported SmartReader caches")
    try:
        book.path.unlink(missing_ok=True)
    except Exception as e:
        raise HTTPException(500, f"delete failed: {e}")
    _BOOKS.pop(book_id, None)
    # Also remove any attachments to this book
    with _db() as c:
        c.execute("DELETE FROM attachments WHERE book_id = ?", (book_id,))
    return {"ok": True}


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
    rag_chunks: list[dict] = retrieve(book_ids, req.content, TOP_K) if book_ids else []

    if rag_chunks:
        system_prompt = SYSTEM_BASE + SYSTEM_RAG_SUFFIX
        user_message = f"{format_rag_context(rag_chunks)}\n\n---\n\nQuestion: {req.content}"
    else:
        system_prompt = SYSTEM_BASE
        user_message = req.content

    full_messages = (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": user_message}]
    )

    async def event_stream() -> AsyncIterator[str]:
        yield _sse({
            "type": "meta", "conversation_id": cid, "model": MODEL,
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
                model=MODEL, messages=full_messages, stream=True
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
      <div class="model-info">
        <span class="dot"></span>
        <span id="model-name">loading…</span>
      </div>
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
      body: JSON.stringify({conversation_id: state.currentConversationId, content: content}),
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
  try {
    const res = await fetch('/api/model');
    const data = await res.json();
    $('#model-name').textContent = data.model;
  } catch {
    $('#model-name').textContent = 'connection error';
  }
  await loadLibrary();
  showEmptyState();
  await loadConversations();
  $('#new-chat').onclick = newChat;
  $('#send').onclick = sendMessage;
  $('#upload-btn').onclick = () => $('#file-input').click();
  $('#file-input').onchange = (e) => {
    if (e.target.files.length) uploadFiles(e.target.files);
    e.target.value = '';
  };
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
