# Cortex

A fully offline chat app for local LLMs via Ollama, with built-in document indexing and retrieval-augmented generation. Drop a PDF, EPUB, DOCX, or text file into the window, attach it to a conversation, and query it with a model of your choice running on your own hardware. No cloud, no telemetry, no internet required after setup.

![offline](https://img.shields.io/badge/offline-yes-67e480?style=flat-square) ![ollama](https://img.shields.io/badge/runtime-ollama-5dd9ff?style=flat-square) ![rag](https://img.shields.io/badge/RAG-built--in-ff9d5d?style=flat-square)

## What it is

Cortex is a single-file FastAPI app with an embedded HTML UI. It runs a chat interface against any Ollama model, persists conversations to local SQLite, and includes a complete document indexing pipeline so you can ground answers in your own books, papers, and notes.

The name reflects what the app does: it acts as an external cortex — memory (your indexed documents) and reasoning (a local LLM) brought together so you can think through complex material without anything leaving the machine. The default model is `qwen2.5:7b`.

This project pairs naturally with [SmartReader](https://github.com/hamii31/SmartReader) — Cortex reads SmartReader's pickle caches automatically, so books indexed in either app are queryable in Cortex. Cortex is the more recent, more complete tool: it does everything SmartReader does plus chat-style conversations, multi-document attachment, and a wider range of input formats.

## Features

- **Fully offline** — once Ollama and the models are installed, no internet is needed.
- **Single 32B model** — strong reasoning quality from a model that fits a single 24 GB GPU.
- **Built-in document indexer** — drag and drop PDF, EPUB, DOCX, TXT, or Markdown.
- **Drag-and-drop UI** — drop files anywhere on the window to start indexing.
- **Multi-document RAG** — attach multiple sources to a conversation; retrieval merges across them.
- **Citation-aware prompting** — the model is instructed to cite `[Title, p. N]` and not to invent.
- **SmartReader compatibility** — reads existing SmartReader caches read-only.
- **Streaming responses** — token-by-token output via Server-Sent Events.
- **Persistent history** — conversations and attachments saved to local SQLite.
- **Per-source retrieval visibility** — each AI response shows which excerpts were retrieved.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Cortex (FastAPI)                       │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   Indexer    │    │  Retrieval   │    │     Chat     │   │
│  │              │    │              │    │              │   │
│  │  PDF/EPUB/   │───▶│  Embed query │──▶│ qwen2.5 model│   │
│  │  DOCX/TXT    │    │  Top-K via   │    │    + RAG     │   │
│  │      ↓       │    │  cosine sim  │    │  context     │   │
│  │  Chunk +     │    │      ↓       │    │      ↓       │   │
│  │  embed       │    │  Inject      │    │  Stream to   │   │
│  │      ↓       │    │  excerpts    │    │  browser     │   │
│  │  .pkl cache  │    │              │    │              │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│         │                    │                    │         │
│         └────────────────────┴────────────────────┘         │
│                              │                              │
│                              ▼                              │
│                    ┌──────────────────┐                     │
│                    │  Ollama (local)  │                     │
│                    │  qwen2.5         │                     │
│                    │  nomic-embed-text│                     │
│                    └──────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

The retrieval pipeline:

1. Document arrives via upload or drag-and-drop.
2. Format-specific extractor pulls text with location metadata (PDF page numbers, EPUB chapter index, DOCX/text pseudo-pages of ~3000 characters).
3. Text is split into overlapping ~1000-character chunks with 200-character overlap.
4. Each chunk is embedded via `nomic-embed-text` (768-dim, embedding the first 500 chars of each chunk).
5. Chunks are pickled to a local cache directory.
6. At query time, the question is embedded and cosine similarity ranks all chunks across all attached documents. Top-K are merged by score.
7. The 32B model receives the retrieved excerpts as a system-attached context block alongside instructions to cite specific pages.

## Requirements

- **Ollama** ([install](https://ollama.com))
- **GPU/RAM**, depending on the model you want to run:
  - **8 GB VRAM** for the default `qwen2.5:7b` (the typical mid-range laptop GPU)
  - **16 GB VRAM** for `qwen2.5:14b` (noticeably better reasoning)
  - **24 GB VRAM** for `qwen2.5:32b` (the practical ceiling for consumer hardware)
  - Lower VRAM still works via Ollama's CPU/GPU split, but expect slow generation
- **~5 GB disk** for default models (chat + embedder)
- Python 3.10+ only required if running from source

## Installation

### Option A: Download the executable (recommended)

1. Download the latest `Cortex.exe` (Windows) or `Cortex` binary (Linux/macOS) from the [Releases page](https://github.com/hamii31/Cortex/releases).
2. Install Ollama from [ollama.com](https://ollama.com) if you haven't already.
3. Pull the default models the first time:
   ```bash
   ollama pull qwen2.5:7b            # ~4.7 GB, the default chat model
   ollama pull nomic-embed-text      # ~270 MB, for retrieval
   ```
4. Double-click the executable. Cortex starts a local server and opens your default browser to the chat UI. If Ollama isn't running, you'll get a dialog explaining what to do.

A log file at `cortex.log` next to the executable captures any errors — useful when filing bug reports.

### Option B: Run from source

For developers, contributors, or anyone who wants to modify Cortex.

```bash
# 1. Clone the repo
git clone https://github.com/hamii31/Cortex.git
cd Cortex

# 2. Install Ollama (see ollama.com), then pull models:
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# 3. Install Python dependencies
pip install fastapi uvicorn ollama numpy python-multipart \
            pypdf ebooklib beautifulsoup4 python-docx

# 4. Run the chat server directly
python cortex.py

# Or run via the launcher (auto-opens browser, checks Ollama):
python cortex_launcher.py
```

Open [http://localhost:8000](http://localhost:8000) in any browser if it doesn't open automatically.

### Option C: Build your own executable

If you want to package Cortex yourself — for a custom build, a different platform, or a fork — use the included build script:

```bash
pip install pyinstaller
python build_executable.py
```

The result lands in `dist/Cortex.exe` (Windows), `dist/Cortex` (Linux), or `dist/Cortex.app` (macOS). PyInstaller doesn't cross-compile, so build on the target platform you want to ship.

Build options:

```bash
python build_executable.py --debug      # keeps the console window visible (useful for diagnosing issues)
python build_executable.py --onedir     # folder distribution, faster cold start, larger to ship
```

Place an `icon.ico` (Windows), `icon.icns` (macOS), or `icon.png` (Linux) alongside the build script and it'll be bundled automatically.

## Usage

### Indexing a document

Drag any supported file into the Cortex window. A progress bar appears in the sidebar showing extraction → chunking → embedding → caching. When finished, the document slides into your library and is immediately queryable.

Supported formats:

| Format | Page semantics | Library tag |
|---|---|---|
| `.pdf` | Real PDF page numbers | `q4` |
| `.epub` | Chapter index (no real pages exist in EPUB) | `q4` |
| `.docx` | Pseudo-pages of ~3000 characters | `q4` |
| `.txt`, `.md` | Pseudo-pages of ~3000 characters | `q4` |
| SmartReader cache | Whatever SmartReader stored | `sr` (read-only) |

### Asking a question

1. Click **+ new chat** in the sidebar (or just start typing).
2. Click **⊕** next to one or more books in the **LIBRARY** section to attach them.
3. Type your question and press Enter.

When sources are attached, Cortex retrieves the top relevant excerpts and shows them in a small **RAG** panel above the AI's response. The model is instructed to cite specific pages and to say "the source doesn't cover this" rather than fabricating.

If no documents are attached, Cortex behaves as a normal offline chat — no retrieval runs, no excerpts are injected.

### Multi-document queries

Attach several books at once. Retrieval runs across all of them in parallel and merges the top-K results by similarity score. Useful for questions that span sources — for example, attaching a textbook plus a recent paper plus your own lecture notes.

## Configuration

Configure via environment variables before launching:

| Variable | Default | Notes |
|---|---|---|
| `CORTEX_MODEL` | `qwen2.5:7b` | Any Ollama model. Bump to `qwen2.5:14b` on 16 GB GPUs or `qwen2.5:32b` on 24 GB GPUs for noticeably stronger reasoning. `llama3.3:70b` is the high end for 40+ GB hardware. |
| `CORTEX_EMBED_MODEL` | `nomic-embed-text` | Embedding model used for both indexing and retrieval. Must be available in Ollama. |
| `CORTEX_HOST` | `127.0.0.1` | Set to `0.0.0.0` to expose to your local network (no auth — be careful). |
| `CORTEX_PORT` | `8000` | HTTP port. |
| `CORTEX_TOP_K` | `4` | Number of chunks retrieved per query, merged across all attached books. Bump to 6–8 for larger models with more context room. |
| `CORTEX_LIBRARY` | platform-specific | Override the library cache directory. |
| `CORTEX_SMARTREADER_CACHE` | auto-detected | Path to a SmartReader cache to also expose. Set explicitly to override. |

### Default cache locations

| OS | Library directory | Conversation DB |
|---|---|---|
| Linux | `~/.config/cortex/library/` | `~/.config/cortex/conversations.db` |
| macOS | `~/Library/Application Support/cortex/library/` | `~/Library/Application Support/cortex/conversations.db` |
| Windows | `%APPDATA%\cortex\library\` | `%APPDATA%\cortex\conversations.db` |

If a SmartReader cache exists at its standard location, Cortex auto-detects and exposes it as read-only library entries tagged `sr`.

## API reference

Cortex exposes a small REST API. Use it from scripts, other tools, or to integrate Cortex's library into your own pipelines.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/model` | Active model and config info |
| `GET` | `/api/library` | List indexed documents |
| `POST` | `/api/library/upload` | Upload a document for indexing (multipart/form-data, field `file`) |
| `GET` | `/api/library/jobs` | All current/recent indexing jobs and their status |
| `GET` | `/api/library/jobs/{job_id}` | Single job status |
| `DELETE` | `/api/library/{book_id}` | Remove a document from the library (Cortex-managed only, not SmartReader imports) |
| `GET` | `/api/conversations` | List recent conversations |
| `GET` | `/api/conversations/{cid}` | Get conversation messages and attachments |
| `DELETE` | `/api/conversations/{cid}` | Delete conversation |
| `POST` | `/api/conversations/{cid}/attach` | Attach a book to a conversation (body: `{"book_id": "..."}`) |
| `DELETE` | `/api/conversations/{cid}/attach/{book_id}` | Detach a book |
| `POST` | `/api/chat` | Send a message; returns SSE stream of tokens (body: `{"conversation_id": "...", "content": "..."}`) |

## Troubleshooting

### Windows: "Unknown publisher" warning when launching

This is normal for unsigned executables. Click **More info → Run anyway**. Code signing requires a paid certificate (~$200–400/year) which isn't worth it for a personal project; the SmartReader README has the same caveat.

### Antivirus blocks Cortex.exe

PyInstaller-packed executables sometimes trip antivirus heuristics. False positive — whitelist `Cortex.exe` in your antivirus settings, or build from source if you'd rather verify the code yourself.

### Cortex.exe opens and immediately closes (Windows)

Run `Cortex.exe` from a terminal so you can see the error, or check `cortex.log` next to the executable. The most common cause is Ollama not being installed or not running. Install [Ollama](https://ollama.com), make sure it's started, then relaunch Cortex.

If you see an error about `sys.stderr` or `isatty`, you have an old build — rebuild with the latest `cortex_launcher.py`.

### `Connection error` in the UI

Ollama isn't running. Start it: `ollama serve` (it's usually already running as a service after install).

### `'TextChunk' object has no attribute '__dict__'` when loading SmartReader caches

You're running an older version of `cortex.py`. Update — the fix removes `__slots__` from the compatibility class so SmartReader's pickled instances deserialize correctly.

### Indexing is very slow

Embedding throughput is the bottleneck and runs through Ollama. A 900-page book takes 15–30 minutes on a single GPU. This is the same speed SmartReader achieves — it's not Cortex overhead, it's just how long that many embedding calls take. Index in the background and avoid running large chat queries simultaneously.

### CUDA error 500 / "shared object initialization failed"

The model is too big for your GPU. Either use a smaller model (`CORTEX_MODEL=qwen2.5:7b`) or restart Ollama and let it auto-tune the GPU/CPU split. The 7B default fits 8 GB VRAM cleanly; 14B needs 16 GB and 32B needs 24 GB. If you've recently crashed the runner, restart Ollama (right-click the tray icon → Quit, then start again) before retrying — the GPU context can stay in a bad state until Ollama is fully restarted.

### Out of memory on the chat model

Switch to a smaller model: `CORTEX_MODEL=qwen2.5:7b cortex.exe` (or just rely on the default). For very tight VRAM, even `gemma2:2b` works. Quality drops but the app is identical.

### Citations point to wrong pages

For non-PDF formats, "page" is a soft concept (chapter for EPUB, ~3000-char section for DOCX/text). The model is told this in the system prompt but the citation grammar still reads as "p. N" for consistency. If you want stricter citations for a DOCX, convert it to PDF first and re-index.

### Retrieval misses obviously relevant content

The default embedding truncates each chunk to 500 characters before embedding (matches SmartReader's behavior so caches are interchangeable). For technical documents with key information past the first 500 characters of a chunk, this can hurt recall. If you want full-text embedding, change `EMBED_TRUNCATE = 500` to `EMBED_TRUNCATE = 2000` near the top of the file and re-index. Be aware: SmartReader caches will no longer be query-equivalent if you do this.

### Kill process listening on port 8000

Until I have fixed this bug, which occurs after closing the executable, you should utilize the following commands to terminate the process running on port 8000:

#### LINUX:
##### Find the PID and stop it (replace port and PID as needed):
```bash
sudo ss -ltnp '( sport = :8000 )'
```

##### then kill the PID shown (replace <PID>)
```bash
sudo kill <PID>        # graceful
sudo kill -9 <PID>     # forceful
```

##### Alternatively kill by command name (replace pattern):
```bash
sudo pkill -f 'pattern'
```

#### WINDOWS:
##### Find what's listening on port 8000
```bash
Get-NetTCPConnection -LocalPort 8000 | Select-Object OwningProcess
```

##### Then kill that PID:
```bash
Stop-Process -Id <PID> -Force
```

## Privacy and data handling

- All processing is local. No data is sent to any external service.
- Ollama runs on `localhost`. Verify with `ss -tlnp | grep 11434` (Linux) or by disconnecting from the network and confirming queries still work.
- Uploaded documents live temporarily in `<library>/_uploads/` during indexing and are deleted after the cache is written. Only the embedded `.pkl` cache persists.
- Conversation history is plain SQLite — readable, exportable, deletable.
- The default bind is `127.0.0.1`, not exposed to your local network. If you set `CORTEX_HOST=0.0.0.0`, anyone on your LAN can hit the API; there is no authentication.

## License

MIT.

## Acknowledgments

- [Ollama](https://ollama.com) for the local LLM runtime.
- [SmartReader](https://github.com/hamii31/SmartReader) for establishing the offline RAG pattern this project extends.
- The Qwen, Llama, and DeepSeek teams for open-weight models that make this practical.
- [nomic-embed-text](https://www.nomic.ai/blog/posts/nomic-embed-text-v1) for embeddings.
