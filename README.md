# Cortex

<p align="center">
  <img src="docs/Screenshot 2026-05-11 085611.png" width="800" alt="Cortex chat interface with RAG citations">
</p>

A fully offline chat app for local LLMs via Ollama, with structure-aware document indexing, retrieval-augmented generation, runtime model switching, and structured reasoning modes. Drop a PDF, EPUB, DOCX, or text file into the window, attach it to a conversation, and query it with the model that best fits your hardware. No cloud, no telemetry, no internet required after setup.

![offline](https://img.shields.io/badge/offline-yes-67e480?style=flat-square) ![ollama](https://img.shields.io/badge/runtime-ollama-5dd9ff?style=flat-square) ![rag](https://img.shields.io/badge/RAG-hierarchical-ff9d5d?style=flat-square) ![modes](https://img.shields.io/badge/modes-5-c084fc?style=flat-square) [![DOI](https://zenodo.org/badge/1231694841.svg)](https://doi.org/10.5281/zenodo.20069615) [![Downloads](https://img.shields.io/github/downloads/hamii31/Cortex/total?style=flat-square)](https://github.com/hamii31/Cortex/releases)

## What it is

Cortex is a single-file FastAPI app with an embedded HTML UI. It runs a chat interface against any Ollama model, persists conversations to local SQLite, and includes a complete document indexing pipeline so you can ground answers in your own books, papers, and notes.

The name reflects what the app does: it acts as an external cortex — memory (your indexed documents) and reasoning (a local LLM) brought together so you can think through complex material without anything leaving the machine.

**Cortex works best with well-structured documents** — books, textbooks, dissertations, peer-reviewed papers, and other content that has explicit chapter and section structure. At index time, Cortex reads each document's authored structure (PDF outlines, table-of-contents pages, EPUB chapters) and uses it to produce hierarchical summaries that the model sees alongside specific excerpts during retrieval. The result is answers that are grounded in both detail (verbatim passages) and context (what section the passage comes from and what that section is about). Unstructured documents still work, but the depth advantage of Cortex over naive RAG diminishes the less structure the source has.

## Features

- **Fully offline** — once Ollama and the models are installed, no internet is needed.
- **Three model tiers, one executable** — switch between 7B (fast), 14B (balanced), and 32B Q4_K_L (research-grade) at runtime via the sidebar dropdown. Your choice persists across launches.
- **Structure-aware indexing** — at index time, Cortex reads each document's authored structure (PDF outline, TOC, EPUB chapters) and produces section-level summaries that the model sees alongside specific excerpts during retrieval. Falls back to fixed-size sections when no structure is recoverable.
- **Hierarchical retrieval context** — retrieved excerpts arrive with their section summary attached, giving the model both detail and surrounding context.
- **Reasoning modes** — five structured prompt scaffolds (Default, Compare, Process, Cross-source, Critique) that force the model to produce organized intermediate output before its prose answer. Modes auto-disable on smaller models that can't handle them well.
- **Multi-source RAG with guaranteed coverage** — when multiple documents are attached, retrieval reserves slots per source so no book gets ignored, and the prompt explicitly instructs the model to use all attached sources.
- **Built-in document indexer** — drag and drop PDF, EPUB, DOCX, TXT, or Markdown into the window.
- **Citation-aware prompting** — the model is instructed to cite `[Title, p. N]` and not to invent.
- **Streaming responses** — token-by-token output via Server-Sent Events.
- **Persistent history** — conversations and attachments saved to local SQLite.

## Architecture

The indexing and retrieval pipeline:

1. **Document arrives** via upload or drag-and-drop.
2. **Format-specific extractor** pulls text with location metadata (PDF page numbers, EPUB chapter index, DOCX/text pseudo-pages of ~3000 characters).
3. **Structure extraction** attempts to recover the document's authored structure: PDF outline → TOC page parsing → fall back to fixed-size sections.
4. **Chunking** splits text into overlapping ~1000-character chunks with 200-character overlap.
5. **Embedding** of each chunk via `nomic-embed-text` (768-dim).
6. **Hierarchical summarization** runs the active LLM over each section to produce a concise summary stored alongside the embedding cache.
7. **Retrieval** at query time: the question is embedded and cosine similarity ranks all chunks across all attached documents. Per-book minimum slots are reserved when multiple sources are attached.
8. **Context assembly** packs retrieved excerpts with their section summaries into a structured prompt; the model receives both detail and context.
9. **Generation** with whatever reasoning mode is active produces the final response with citations.

## Requirements

- **Ollama** ([install](https://ollama.com))
- **VRAM**, depending on which tier you want to use:
  - **8 GB** for the Lite tier (7B model)
  - **12 GB** for the Standard tier (14B model)
  - **24 GB** for the Research tier (32B Q4_K_L)
  - Lower VRAM still works via Ollama's CPU/GPU split, but expect slow generation
- **~5 GB disk** for Lite-only setup; ~14 GB for Lite + Standard; ~34 GB for all three tiers
- Python 3.10+ only required if running from source

## Installation

### Option A: Download the executable (recommended)

1. Download `Cortex.exe` (Windows) from the [Releases page](https://github.com/hamii31/Cortex/releases).
2. Install [Ollama](https://ollama.com) if you haven't already.
3. Pull the embedder and at least one model tier:

   ```bash
   ollama pull nomic-embed-text                  # required for retrieval (~270 MB)

   # Then pull whichever tier(s) you want — you can install all three:
   ollama pull qwen2.5:7b                        # Lite tier (~4.7 GB)
   ollama pull qwen2.5:14b                       # Standard tier (~9 GB)

   # For the Research tier, pull the high-fidelity Q4_K_L variant:
   ollama pull hf.co/bartowski/Qwen2.5-32B-Instruct-GGUF:Qwen2.5-32B-Instruct-Q4_K_L.gguf
   ```

4. Double-click `Cortex.exe`. Cortex starts a local server and opens your default browser to the chat UI.
5. Click the model name in the top-left of the sidebar to open the tier dropdown. Pick whichever model you want to use.

A log file at `cortex.log` next to the executable captures any errors — useful when filing bug reports.

#### About the model tiers

| Tier | Model | Min VRAM | Best for |
|---|---|---|---|
| **Lite** | `qwen2.5:7b` | 8 GB | Daily use, quick lookups, fast response. RAG-grounded queries are strong; pure reasoning is the weakest of the three. |
| **Standard** | `qwen2.5:14b` | 12 GB | The balanced sweet spot. Strong reasoning at usable speed. Recommended default if you have the VRAM. |
| **Research** | `qwen2.5:32b` Q4_K_L | 24 GB (or 32+ GB system RAM) | Best precision, especially on technical and academic content. Uses the Q4_K_L quantization, which preserves higher precision (Q6_K) on token embeddings and the output projection — sharper handling of specialized terminology and rare tokens. |

The tier dropdown shows install status for each one (`✓ installed` or `⚠ run: ollama pull ...`).

### Option B: Run from source

```bash
git clone https://github.com/hamii31/Cortex.git
cd Cortex
ollama pull nomic-embed-text
ollama pull qwen2.5:7b   # or another tier
pip install fastapi uvicorn ollama numpy python-multipart \
            pypdf ebooklib beautifulsoup4 python-docx
python cortex.py
```

Open [http://localhost:8000](http://localhost:8000) if it doesn't open automatically.

### Option C: Build your own executable

```bash
pip install pyinstaller
python build_executable.py
```

The result lands in `dist/Cortex.exe` (Windows), `dist/Cortex` (Linux), or `dist/Cortex.app` (macOS). PyInstaller doesn't cross-compile, so build on the target platform.

## Usage

### Switching models

Click the model name in the top-left of the sidebar to open the tier dropdown. Pick any tier with `✓ installed` to switch to it. Cortex updates the active model immediately and refreshes the available reasoning modes.

### Reasoning modes

Above the chat input is a row of mode pills. Click one to set the mode for your next message.

| Mode | What it does | Best for | Min tier |
|---|---|---|---|
| **Default** | No scaffold — direct answer | Simple lookups, factual questions | Lite |
| **Compare** | Forces a markdown comparison table before prose | "A vs B", tradeoffs, "best approach" questions | Lite |
| **Process** | Forces explicit state/step layout before prose | "How does X work", pathways, algorithms, system dynamics | Standard |
| **Cross-source** | Forces a cross-reference table across attached documents | Multi-document queries where you want all sources considered | Standard |
| **Critique** | Forces structured strengths/weaknesses analysis | Reviewing a plan, paper, code design, or proposal | Standard |

When multiple documents are attached and the active model is Standard or Research, Cortex silently promotes Default-mode queries to Cross-source mode. Modes that require strong instruction-following are hidden on the Lite tier — the 7B can't reliably produce the structured scaffolds those modes need.

### Indexing a document

Drag any supported file into the Cortex window. A progress bar appears in the sidebar showing extraction → chunking → embedding → summarizing → caching. When finished, the document slides into your library and is immediately queryable.

**Cortex prefers documents with explicit structure.** During indexing, the system attempts to recover each document's authored chapter and section layout in this order:

1. **PDF outline (bookmarks tree)** — most modern PDF books and papers have this. Best results.
2. **Table-of-contents page parsing** — for PDFs without outlines, Cortex parses common TOC formats from text.
3. **Fixed-size fallback** — for documents with no recoverable structure, sections are formed by grouping ~15 chunks at page boundaries.

| Format | Page semantics | Structure source |
|---|---|---|
| `.pdf` | Real PDF page numbers | Outline → TOC page → fallback |
| `.epub` | Chapter index | EPUB spine (chapters used directly as section units) |
| `.docx` | Pseudo-pages of ~3000 characters | Fallback only |
| `.txt`, `.md` | Pseudo-pages of ~3000 characters | Fallback only |

The log file (`cortex.log`) reports which strategy was used for each book — useful when you want to verify a textbook was properly structured. Documents with chapter-level structure (well-made PDF books, modern academic papers) produce noticeably better retrieval context than unstructured material.

### Asking a question

1. Click **+ new chat** in the sidebar (or just start typing).
2. Click **⊕** next to one or more books in the **LIBRARY** section to attach them.
3. Pick a reasoning mode if appropriate.
4. Type your question and press Enter.

When sources are attached, Cortex retrieves the top relevant excerpts and shows them in a small **RAG** panel above the AI's response. If a retrieved chunk belongs to a section with a summary, the section summary is shown to the model as context before the verbatim excerpts. The model is instructed to cite specific pages (not the summary) and to say "the source doesn't cover this" rather than fabricating.

If no documents are attached, Cortex behaves as a normal offline chat — no retrieval runs, no excerpts are injected.

## Library separation from SmartReader

Cortex originated as an extension of [SmartReader](https://github.com/hamii31/SmartReader), and earlier versions auto-detected SmartReader's cache directory at launch. **This was removed in 1.2.** SmartReader caches no longer appear in Cortex's library unless you explicitly opt in.

The reason: SmartReader caches were produced before Cortex's structure-aware indexing existed and lack the hierarchical section summaries that make Cortex queries work well. Surfacing them by default created duplicates with Cortex-native caches of the same books, with strictly inferior retrieval context.

**Recommended migration:** re-index any books you care about in Cortex by dragging the original files into the window. The new caches will use structure-aware sections and produce better answers.

**Opt-in compatibility:** if you still want SmartReader cache access (for example, you have indexed books that you don't have the original files for), set the environment variable:

```bash
CORTEX_SMARTREADER_CACHE=/path/to/SmartReader/cache
```

On Windows the default SmartReader location is `%APPDATA%\SmartReader\cache`. When set, SmartReader-indexed books appear in the library tagged `sr` and are read-only — Cortex won't modify them.

## Configuration

Configure via environment variables before launching:

| Variable | Default | Notes |
|---|---|---|
| `CORTEX_DEFAULT_TIER` | `lite` | Which tier is active on first launch (`lite`, `standard`, or `research`). Subsequent launches honor your last selection in the UI. |
| `CORTEX_MODEL` | (unset) | Power-user override: set to any Ollama model name to bypass the tier system entirely. The UI selector becomes disabled when this is set. |
| `CORTEX_EMBED_MODEL` | `nomic-embed-text` | Embedding model used for both indexing and retrieval. |
| `CORTEX_HOST` | `127.0.0.1` | Set to `0.0.0.0` to expose to your local network (no auth — be careful). |
| `CORTEX_PORT` | `8000` | HTTP port. |
| `CORTEX_TOP_K` | tier-dependent (4 for Lite, 6 for Standard/Research) | Number of chunks retrieved per query. |
| `CORTEX_LIBRARY` | platform-specific | Override the library cache directory. |
| `CORTEX_SMARTREADER_CACHE` | (unset, opt-in) | Path to a SmartReader cache directory to expose alongside Cortex's library. |
| `CORTEX_SKIP_SUMMARIES` | (unset) | Set to `1` to skip hierarchical summarization at index time. Speeds indexing but loses section context at query time. |

### Default cache locations

| OS | Library directory |
|---|---|
| Linux | `~/.config/cortex/library/` |
| macOS | `~/Library/Application Support/cortex/library/` |
| Windows | `%APPDATA%\cortex\library\` |

`conversations.db`, `cortex_state.json`, and `cortex.log` live in the parent of the library directory.

## API reference

Cortex exposes a small REST API. Use it from scripts, other tools, or to integrate Cortex's library into your own pipelines.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/model` | Active model and config info |
| `GET` | `/api/model/tiers` | List available tiers and which is active |
| `POST` | `/api/model/switch` | Switch active tier (body: `{"tier": "lite"}`) |
| `GET` | `/api/modes` | List available reasoning modes (filtered by current tier) |
| `GET` | `/api/ollama/installed` | List models currently installed in Ollama |
| `GET` | `/api/library` | List indexed documents |
| `POST` | `/api/library/upload` | Upload a document for indexing |
| `POST` | `/api/library/{book_id}/summarize` | Retroactively generate hierarchical summaries for an existing cache |
| `GET` | `/api/library/jobs` | All current/recent indexing jobs and their status |
| `DELETE` | `/api/library/{book_id}` | Remove a document from the library (Cortex-managed only) |
| `GET` | `/api/conversations` | List recent conversations |
| `GET` | `/api/conversations/{cid}` | Get conversation messages and attachments |
| `DELETE` | `/api/conversations/{cid}` | Delete conversation |
| `POST` | `/api/conversations/{cid}/attach` | Attach a book |
| `DELETE` | `/api/conversations/{cid}/attach/{book_id}` | Detach a book |
| `POST` | `/api/chat` | Send a message; returns SSE stream of tokens |
| `POST` | `/api/heartbeat` | Browser keepalive |
| `POST` | `/api/shutdown` | Explicit clean shutdown |

## Troubleshooting

### Windows: "Unknown publisher" warning when launching

Normal for unsigned executables. Click **More info → Run anyway**.

### Antivirus blocks Cortex.exe

PyInstaller-packed executables sometimes trip antivirus heuristics. False positive — whitelist `Cortex.exe` or build from source.

### Cortex.exe opens and immediately closes (Windows)

Run from a terminal so you can see the error, or check `cortex.log` next to the executable. The most common cause is Ollama not being installed or not running.

### `Connection error` in the UI

Ollama isn't running. Start it: `ollama serve`.

### "Model 'X' is not installed" when switching tiers

The error message lists what is currently installed. Pull the missing tier with the command shown.

### Indexing is slow on large books

For a 900-page textbook, indexing takes ~20-40 minutes total: embedding the chunks (15-30 min) plus summarizing sections (5-10 min on Lite, longer on Research). This is dominated by Ollama's per-call inference speed. Index in the background and avoid running large chat queries at the same time. To skip summarization (saves the section-summary step), set `CORTEX_SKIP_SUMMARIES=1` before launching.

### CUDA error 500 / "shared object initialization failed"

The model is too big for your GPU. Switch to a smaller tier in the dropdown. If you've recently crashed the Ollama runner, restart it (right-click the tray icon → Quit, then start again) — the GPU context can stay in a bad state until Ollama is fully restarted.

### Mode pill doesn't appear on Lite tier

Process, Cross-source, and Critique modes require Standard or Research tier. Switch up in the dropdown to access them.

### Section summaries are missing for an older indexed book

Re-index the book by dragging the original file in again, or call `POST /api/library/{book_id}/summarize` to generate summaries retroactively (without re-embedding). Note: retroactive summarization can't recover document structure (PDF outline, TOC), so the summaries will use fixed-size sections.

### Retrieval misses obviously relevant content

The default embedding truncates each chunk to 500 characters. For technical documents with key info past the first 500 characters of a chunk, change `EMBED_TRUNCATE = 500` to `EMBED_TRUNCATE = 2000` near the top of the file and re-index.

### Kill process listening on port 8000

The process should auto-shutdown ~30 seconds after you close the browser tab. If it doesn't:

```bash
# Linux
sudo ss -ltnp '( sport = :8000 )'
sudo kill <PID>

# Windows (PowerShell)
Get-NetTCPConnection -LocalPort 8000 | Select-Object OwningProcess
Stop-Process -Id <PID> -Force
```

## Privacy and data handling

- All processing is local. No data is sent to any external service.
- Ollama runs on `localhost`. Verify with `ss -tlnp | grep 11434` (Linux) or by disconnecting from the network and confirming queries still work.
- Uploaded documents live temporarily in `<library>/_uploads/` during indexing and are deleted after the cache is written. Only the embedded `.pkl` cache and the `.summaries.json` companion persist.
- Conversation history is plain SQLite — readable, exportable, deletable.
- The default bind is `127.0.0.1`, not exposed to your local network. If you set `CORTEX_HOST=0.0.0.0`, anyone on your LAN can hit the API; there is no authentication.

## License

MIT.

## Acknowledgments

- [Ollama](https://ollama.com) for the local LLM runtime.
- [SmartReader](https://github.com/hamii31/SmartReader) — the predecessor project that established the offline RAG pattern Cortex extends.
- The Qwen team for open-weight models that make this practical.
- [bartowski](https://huggingface.co/bartowski) for high-quality GGUF quantizations.
- [nomic-embed-text](https://www.nomic.ai/blog/posts/nomic-embed-text-v1) for embeddings.
