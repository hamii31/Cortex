# Cortex

<p align="center">
  <img src="docs/Screenshot 2026-05-09 113007.png" width="800" alt="Cortex chat interface with RAG citations">
</p>

A fully offline chat app for local LLMs via Ollama, with built-in document indexing, retrieval-augmented generation, runtime model switching, and structured reasoning modes. Drop a PDF, EPUB, DOCX, or text file into the window, attach it to a conversation, and query it with the model that best fits your hardware. No cloud, no telemetry, no internet required after setup.

![offline](https://img.shields.io/badge/offline-yes-67e480?style=flat-square) ![ollama](https://img.shields.io/badge/runtime-ollama-5dd9ff?style=flat-square) ![rag](https://img.shields.io/badge/RAG-built--in-ff9d5d?style=flat-square) ![modes](https://img.shields.io/badge/modes-5-c084fc?style=flat-square) [![DOI](https://zenodo.org/badge/1231694841.svg)](https://doi.org/10.5281/zenodo.20069615) [![Downloads](https://img.shields.io/github/downloads/hamii31/Cortex/total?style=flat-square)](https://github.com/hamii31/Cortex/releases)

## What it is

Cortex is a single-file FastAPI app with an embedded HTML UI. It runs a chat interface against any Ollama model, persists conversations to local SQLite, and includes a complete document indexing pipeline so you can ground answers in your own books, papers, and notes.

The name reflects what the app does: it acts as an external cortex — memory (your indexed documents) and reasoning (a local LLM) brought together so you can think through complex material without anything leaving the machine.

This project pairs naturally with [SmartReader](https://github.com/hamii31/SmartReader) — Cortex reads SmartReader's pickle caches automatically, so books indexed in either app are queryable in Cortex.

## Features

- **Fully offline** — once Ollama and the models are installed, no internet is needed.
- **Three model tiers, one executable** — switch between 7B (fast), 14B (balanced), and 32B Q4_K_L (research-grade) at runtime via the sidebar dropdown. Your choice persists across launches.
- **Reasoning modes** — five structured prompt scaffolds (Default, Compare, Process, Cross-source, Critique) that force the model to produce organized intermediate output before its prose answer. Modes auto-disable on smaller models that can't handle them well.
- **Multi-source RAG with guaranteed coverage** — when multiple documents are attached, retrieval reserves slots per source so no book gets ignored, and the prompt explicitly instructs the model to use all attached sources.
- **Built-in document indexer** — drag and drop PDF, EPUB, DOCX, TXT, or Markdown into the window.
- **Citation-aware prompting** — the model is instructed to cite `[Title, p. N]` and not to invent.
- **SmartReader compatibility** — reads existing SmartReader caches read-only.
- **Streaming responses** — token-by-token output via Server-Sent Events.
- **Persistent history** — conversations and attachments saved to local SQLite.

## Architecture

The retrieval pipeline:

1. Document arrives via upload or drag-and-drop.
2. Format-specific extractor pulls text with location metadata (PDF page numbers, EPUB chapter index, DOCX/text pseudo-pages of ~3000 characters).
3. Text is split into overlapping ~1000-character chunks with 200-character overlap.
4. Each chunk is embedded via `nomic-embed-text` (768-dim, embedding the first 500 chars of each chunk).
5. Chunks are pickled to a local cache directory.
6. At query time, the question is embedded and cosine similarity ranks all chunks across all attached documents. Per-book minimum slots are reserved when multiple sources are attached, then the rest fill competitively.
7. The active model receives the retrieved excerpts plus the chosen mode's scaffold instruction, then generates a structured response with citations.

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
5. Click the model name in the top-left of the sidebar to open the tier dropdown. Pick whichever model you want to use for this session — your choice persists across launches.

A log file at `cortex.log` next to the executable captures any errors — useful when filing bug reports.

#### About the model tiers

| Tier | Model | Min VRAM | Best for |
|---|---|---|---|
| **Lite** | `qwen2.5:7b` | 8 GB | Daily use, quick lookups, fast response. RAG-grounded queries are strong; pure reasoning is the weakest of the three. |
| **Standard** | `qwen2.5:14b` | 12 GB | The balanced sweet spot. Strong reasoning at usable speed. Recommended default if you have the VRAM. |
| **Research** | `qwen2.5:32b` Q4_K_L | 24 GB (or 32+ GB system RAM) | Best precision, especially on technical and academic content. Uses the Q4_K_L quantization, which preserves higher precision (Q6_K) on token embeddings and the output projection — sharper handling of specialized terminology and rare tokens. ~1 GB larger than standard Q4_K_M. |

The tier dropdown shows install status for each one (`✓ installed` or `⚠ run: ollama pull ...`), so you can see at a glance which are ready. If you don't have the VRAM for the bigger tiers, install just the Lite tier and ignore the rest — Cortex won't try to load tiers you didn't pull.

### Option B: Run from source

For developers, contributors, or anyone who wants to modify Cortex.

```bash
# 1. Clone the repo
git clone https://github.com/hamii31/Cortex.git
cd Cortex

# 2. Install Ollama (see ollama.com), then pull the embedder and at least
#    one model — same as Option A.
ollama pull nomic-embed-text
ollama pull qwen2.5:7b   # or another tier

# 3. Install Python dependencies
pip install fastapi uvicorn ollama numpy python-multipart \
            pypdf ebooklib beautifulsoup4 python-docx

# 4. Run
python cortex.py
# Or run via the launcher (auto-opens browser, checks Ollama):
python cortex_launcher.py
```

Open [http://localhost:8000](http://localhost:8000) if it doesn't open automatically.

### Option C: Build your own executable

```bash
pip install pyinstaller
python build_executable.py
```

The result lands in `dist/Cortex.exe` (Windows), `dist/Cortex` (Linux), or `dist/Cortex.app` (macOS). PyInstaller doesn't cross-compile, so build on the target platform.

Build options:

```bash
python build_executable.py --debug    # keeps console visible (useful for diagnostics)
python build_executable.py --onedir   # folder distribution, faster cold start
```

Place an `icon.ico` (Windows), `icon.icns` (macOS), or `icon.png` (Linux) alongside the build script and it'll be bundled automatically.

## Usage

### Switching models

Click the model name in the top-left corner of the sidebar (next to the green dot). A dropdown opens showing all three tiers with their descriptions and install status. Click any tier to switch to it — Cortex updates the active model immediately and refreshes the available reasoning modes.

Switching is instant from Cortex's perspective, but Ollama needs a moment to load the new model into VRAM the first time you use it after a switch (this happens during the first query, not during the switch itself).

### Reasoning modes

Above the chat input is a row of mode pills. Click one to set the mode for your next message. The active pill is highlighted; the choice is sticky across messages until you change it.

| Mode | What it does | Best for | Min tier |
|---|---|---|---|
| **Default** | No scaffold — direct answer | Simple lookups, factual questions | Lite |
| **Compare** | Forces a markdown comparison table before prose | "A vs B", tradeoffs, "best approach" questions | Lite |
| **Process** | Forces explicit state/step layout before prose | "How does X work", pathways, algorithms, system dynamics | Standard |
| **Cross-source** | Forces a cross-reference table across attached documents | Multi-document queries where you want all sources considered | Standard |
| **Critique** | Forces structured strengths/weaknesses analysis | Reviewing a plan, paper, code design, or proposal | Standard |

**Auto-promotion to Cross-source.** When multiple documents are attached and the active model is Standard or Research, Cortex silently promotes Default-mode queries to Cross-source. A small "MODE" notice appears in the chat showing what happened. This is opt-out — if you explicitly pick a non-default mode, that wins.

**Why mode availability depends on model tier.** Smaller models can't reliably fill out structured scaffolds — a 7B model attempting a Cross-source table produces malformed output ~30-50% of the time. Modes that require strong instruction-following are hidden on the Lite tier so you don't get an inconsistent experience.

### Indexing a document

Drag any supported file into the Cortex window. A progress bar appears in the sidebar showing extraction → chunking → embedding → caching. When finished, the document slides into your library and is immediately queryable.

| Format | Page semantics | Source |
|---|---|---|
| `.pdf` | Real PDF page numbers | Cortex (default, no tag shown) |
| `.epub` | Chapter index (no real pages exist in EPUB) | Cortex |
| `.docx` | Pseudo-pages of ~3000 characters | Cortex |
| `.txt`, `.md` | Pseudo-pages of ~3000 characters | Cortex |
| SmartReader cache | Whatever SmartReader stored | `sr` (read-only) |

### Asking a question

1. Click **+ new chat** in the sidebar (or just start typing).
2. Click **⊕** next to one or more books in the **LIBRARY** section to attach them.
3. Pick a reasoning mode if appropriate.
4. Type your question and press Enter.

When sources are attached, Cortex retrieves the top relevant excerpts and shows them in a small **RAG** panel above the AI's response. Multi-source queries reserve slots per book to ensure no source gets ignored. The model is instructed to cite specific pages and to say "the source doesn't cover this" rather than fabricating.

If no documents are attached, Cortex behaves as a normal offline chat — no retrieval runs, no excerpts are injected.

## Configuration

Configure via environment variables before launching:

| Variable | Default | Notes |
|---|---|---|
| `CORTEX_DEFAULT_TIER` | `lite` | Which tier is active on first launch (`lite`, `standard`, or `research`). Subsequent launches honor your last selection in the UI. |
| `CORTEX_MODEL` | (unset) | Power-user override: set to any Ollama model name to bypass the tier system entirely. The UI selector becomes disabled when this is set. |
| `CORTEX_EMBED_MODEL` | `nomic-embed-text` | Embedding model used for both indexing and retrieval. Must be available in Ollama. |
| `CORTEX_HOST` | `127.0.0.1` | Set to `0.0.0.0` to expose to your local network (no auth — be careful). |
| `CORTEX_PORT` | `8000` | HTTP port. |
| `CORTEX_TOP_K` | tier-dependent (4 for Lite, 6 for Standard/Research) | Number of chunks retrieved per query, merged across all attached books. |
| `CORTEX_LIBRARY` | platform-specific | Override the library cache directory. |
| `CORTEX_SMARTREADER_CACHE` | auto-detected | Path to a SmartReader cache to also expose. |

### Default cache locations

| OS | Library directory |
|---|---|
| Linux | `~/.config/cortex/library/` |
| macOS | `~/Library/Application Support/cortex/library/` |
| Windows | `%APPDATA%\cortex\library\` |

`conversations.db` and `cortex_state.json` (which holds the persisted tier choice) live in the parent of the library directory. If a SmartReader cache exists at its standard location, Cortex auto-detects and exposes it as read-only library entries tagged `sr`.

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
| `POST` | `/api/library/upload` | Upload a document for indexing (multipart/form-data, field `file`) |
| `GET` | `/api/library/jobs` | All current/recent indexing jobs and their status |
| `DELETE` | `/api/library/{book_id}` | Remove a document from the library (Cortex-managed only) |
| `GET` | `/api/conversations` | List recent conversations |
| `GET` | `/api/conversations/{cid}` | Get conversation messages and attachments |
| `DELETE` | `/api/conversations/{cid}` | Delete conversation |
| `POST` | `/api/conversations/{cid}/attach` | Attach a book (body: `{"book_id": "..."}`) |
| `DELETE` | `/api/conversations/{cid}/attach/{book_id}` | Detach a book |
| `POST` | `/api/chat` | Send a message (body: `{"conversation_id": "...", "content": "...", "mode": "..."}`); returns SSE stream of tokens |
| `POST` | `/api/heartbeat` | Browser keepalive (no body) |
| `POST` | `/api/shutdown` | Explicit clean shutdown |

## Troubleshooting

### Windows: "Unknown publisher" warning when launching

This is normal for unsigned executables. Click **More info → Run anyway**. Code signing requires a paid certificate (~$200–400/year) which isn't worth it for a personal project.

### Antivirus blocks Cortex.exe

PyInstaller-packed executables sometimes trip antivirus heuristics. False positive — whitelist `Cortex.exe` in your antivirus settings, or build from source.

### Cortex.exe opens and immediately closes (Windows)

Run `Cortex.exe` from a terminal so you can see the error, or check `cortex.log` next to the executable. The most common cause is Ollama not being installed or not running.

### `Connection error` in the UI

Ollama isn't running. Start it: `ollama serve` (it's usually already running as a service after install).

### "Model 'X' is not installed" when switching tiers

The error message lists what is currently installed. Pull the missing tier with the command shown, then refresh the tier dropdown. If the model IS installed but Cortex doesn't see it, restart Cortex — model lists are cached briefly.

### `'TextChunk' object has no attribute '__dict__'` when loading SmartReader caches

You're running an older version of `cortex.py`. Update to the latest release.

### Indexing is very slow

Embedding throughput is the bottleneck and runs through Ollama. A 900-page book takes 15–30 minutes on a single GPU. Index in the background and avoid running large chat queries simultaneously.

### CUDA error 500 / "shared object initialization failed"

The model is too big for your GPU. Switch to a smaller tier in the dropdown (Lite for 8 GB GPUs, Standard for 16 GB). If you've recently crashed the Ollama runner, restart it (right-click the tray icon → Quit, then start again) — the GPU context can stay in a bad state until Ollama is fully restarted.

For the Research tier on partial-offload hardware (8–16 GB VRAM, 32+ GB system RAM): expect 2–5 tokens/sec. This is normal — Ollama is running most of the model on CPU. Switch to Standard if the speed is unacceptable.

### Mode pill doesn't appear on Lite tier

Process, Cross-source, and Critique modes require Standard or Research tier. They're hidden on Lite because the 7B model can't reliably produce the structured scaffolds those modes require. Switch to a higher tier in the dropdown to access them.

### Citations point to wrong pages

For non-PDF formats, "page" is a soft concept (chapter for EPUB, ~3000-char section for DOCX/text). Convert DOCX to PDF and re-index for stricter citations.

### Retrieval misses obviously relevant content

The default embedding truncates each chunk to 500 characters before embedding (matches SmartReader's behavior so caches are interchangeable). For technical documents with key information past the first 500 characters of a chunk, this can hurt recall. Change `EMBED_TRUNCATE = 500` to `EMBED_TRUNCATE = 2000` near the top of the file and re-index. Note: SmartReader caches will no longer be query-equivalent if you do this.

### Kill process listening on port 8000

If the process gets stuck, terminate it manually:

#### LINUX:
```bash
sudo ss -ltnp '( sport = :8000 )'
sudo kill <PID>        # graceful
sudo kill -9 <PID>     # forceful
```

#### WINDOWS:
```powershell
Get-NetTCPConnection -LocalPort 8000 | Select-Object OwningProcess
Stop-Process -Id <PID> -Force
```

The process should auto-shutdown ~30 seconds after you close the browser tab via the heartbeat watchdog, but if it doesn't, the commands above force it.

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
- The Qwen team for open-weight models that make this practical.
- [bartowski](https://huggingface.co/bartowski) for high-quality GGUF quantizations.
- [nomic-embed-text](https://www.nomic.ai/blog/posts/nomic-embed-text-v1) for embeddings.
