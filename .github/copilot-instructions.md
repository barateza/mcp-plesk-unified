# Plesk Unified MCP Server - AI Agent Instructions

## Project Overview
Plesk Unified is a Model Context Protocol (MCP) server that indexes and retrieves Plesk documentation via semantic search with vector embeddings and intelligent reranking. It aggregates documentation from multiple sources (API, CLI, Admin Guide, PHP stubs, JS SDK) into a unified knowledge base powered by LanceDB.

## Architecture & Components

### Core Stack
- **Server Framework**: FastMCP (Model Context Protocol)
- **Embeddings**: BAAI/bge-m3 (multilingual, bidirectional)
- **Reranking**: BAAI/bge-reranker-base (cross-encoder for result ranking)
- **Vector DB**: LanceDB with Apache Arrow backend
- **Parsing**: BeautifulSoup4 for HTML, raw text for PHP/JS

### Data Flow
1. **Sources** → Multiple Plesk documentation sources (5 categories) with optional Git auto-cloning
2. **Parsing** → Category-specific parsers extract text, title, breadcrumb hierarchy
3. **Chunking** → Variable chunk sizes by type (HTML: 3000, PHP: 6000, JS: 5000 chars)
4. **Embedding** → BAAI/bge-m3 generates vectors (auto-downloader model on first run)
5. **Storage** → LanceDB table with `UnifiedSchema` (vector, text, title, filename, category, breadcrumb)
6. **Retrieval** → Query search → rerank → top-5 results with relevance scores

### Key Files
- [server.py](server.py) - Main entry point; contains all tools, parsers, indexing logic
- [main.py](main.py) - Placeholder entrypoint
- [pyproject.toml](pyproject.toml) - Dependencies (fastmcp, lancedb, sentence-transformers)

## Development Workflows

### Running the Server
```bash
source .venv/bin/activate
python server.py
```
Server initializes on startup: downloads missing sources, builds embeddings, starts MCP server. Models auto-download (~2GB total) on first run.

### Adding New Documentation Sources
Edit `SOURCES` list in [server.py](server.py#L32-L50). Structure:
```python
{"path": KB_DIR / "category", "cat": "category", "type": "html|php|js", "repo_url": "optional git url"}
```
- **type** determines parser and chunk size
- **repo_url** triggers automatic Git cloning if source missing

### Category Defaults
- `html`: 3000-char chunks, TOC-based metadata extraction
- `php`: 6000-char chunks, raw parsing
- `js`: 5000-char chunks, raw parsing

### Git Integration for Auto-Cloning
Sources with `repo_url` trigger automatic Git cloning if the source directory is missing or empty:

**Behavior:**
- If `path` exists and is non-empty: skips cloning
- If `path` missing or empty: clones repo from `repo_url`
- Cloning failures are silent; category is skipped with report entry

## MCP Tool API Reference

### `refresh_knowledge(target_category, reset_db)`
**Purpose**: Indexes Plesk documentation into LanceDB.

**Parameters:**
- `target_category` (str, default: `"all"`) - Category to index. Choices: `"guide"`, `"cli"`, `"api"`, `"php-stubs"`, `"js-sdk"`, or `"all"`
- `reset_db` (bool, default: `False`) - Wipe database on start. Use `True` **only** for first run or full reindex

**Returns:** String report with per-category status

**Behavior:**
- On `reset_db=False`: Skips already-indexed files by filename (incremental mode)
- On `reset_db=True`: Clears entire database before indexing
- Logs progress via stderr (`[LOG]` prefixed messages)
- Silent failures on individual file parse errors

### `search_plesk_unified(query, category)`
**Purpose**: Semantic search with reranking over indexed documentation.

**Parameters:**
- `query` (str, required) - Natural language search query
- `category` (str, optional) - Filter to single category, or `None` for all

**Returns:** String with up to 5 formatted results with relevance scores (0.0–1.0)

**Performance:** ~100–500ms per query

## Key Patterns & Conventions

### Batch Processing
- Files processed in batches of 10 (`BATCH_SIZE_FILES`)
- Documents added to LanceDB in batches to prevent memory overload
- Tracks existing files by category to support incremental indexing

### HTML Metadata Extraction
TOC (table of contents) from `toc.json` provides breadcrumb hierarchy and titles. Falls back to page `<title>` tag if TOC unavailable. Breadcrumbs prepended as context to document text for better semantic understanding.

### Reranking Strategy
- Initial search retrieves 25 results fast via vector similarity
- CrossEncoderReranker reduces to top-5 with refined relevance scores
- Relevance score returned to users for transparency

### Error Handling
- Silent failures on parse errors (returns `None`)
- Missing sources skipped with report entry
- Encoding errors ignored (`encoding="utf-8", errors="ignore"`)

## Common Tasks

### Index New Documentation
```python
# From SOURCES, ensure `path` exists and `repo_url` is set if Git repo
# Then run:
python server.py  # Auto-initializes on startup

# Or trigger via tool:
mcp.refresh_knowledge(target_category="cli", reset_db=False)
```

### Debug Search Quality
- Enable logging to stderr: `print(f"[LOG]...", file=sys.stderr)`
- Logs appear in MCP server terminal
- Check `_relevance_score` in results for reranking confidence

### Extend Parsers
- Add new `parse_*` function following signature: `(file_path, [toc_metadata]) → (title, breadcrumb, text)`
- Update file type routing in `refresh_knowledge` loop
- Update chunk size for type in `SOURCES` metadata

## Real Usage Examples

### Query Flow: Search & Interpret Results
```python
results = search_plesk_unified(
    query="How do I create a custom extension button?",
    category="api"
)

# Output structure (first result):
# === API | Create Extension Button ===
# Path: Extensions > UI Components > Buttons > Custom Buttons
# File: 34877.htm
# Relevance Score: 0.8932

# Interpret relevance score:
# 0.90+: Highly relevant, use as-is
# 0.75–0.89: Relevant, may need context from other results
# <0.75: Tangentially related, consider combining with other top results
```

### Indexing Workflow: Incremental Updates
```bash
# Day 1: First full index
python server.py

# Day 2: API docs updated, re-index (incremental)
python server.py  # Auto-detects new/changed files by filename

# Day 3: Need fresh index
refresh_knowledge(target_category="all", reset_db=True)
```

### Multi-Category vs. Single-Category Trade-offs
```python
# Broad search (all categories):
search_plesk_unified(query="button", category=None)
# Returns: Mixes results from guide, cli, api—good for discovery

# Targeted search (API only):
search_plesk_unified(query="button", category="api")
# Returns: Higher precision, narrower scope—good for API devs

# Tip: Use category filter when user context is known
```

## Critical Implementation Details

1. **Model Auto-Download**: First run downloads ~2GB ML models silently with `TQDM_DISABLE` and `TRANSFORMERS_VERBOSITY=error` to suppress noise
2. **Chunk Overlap**: 500-char overlap between chunks prevents semantic breaks
3. **LanceDB Table**: `UnifiedSchema` with embedding metadata ensures consistency; reuse with `get_table(create_new=False)` for incremental updates
4. **Category Filtering**: Applied at query time via LanceDB `.where()` clause, not at index time—supports broad indexing with flexible filtering
5. **Filename Deduplication**: Prevents re-indexing same file when `reset_db=False` by storing processed filenames

## Performance & Resource Considerations

### Indexing Resource Profile
| Source | Files | Est. Chunks | Memory Peak | Time (M1) |
|--------|-------|-------------|-------------|-----------|
| HTML (guide, cli, api) | ~300 | ~300 | 500MB | 5–8 min |
| PHP stubs | ~50 | ~40 | 200MB | 1–2 min |
| JS SDK | ~100 | ~100 | 300MB | 2–3 min |
| **Full Reindex** | ~500 | ~450 | ~1GB | 10–15 min |

### Storage Requirements
- ML Models: ~2GB (cached to `~/.cache/huggingface/`)
- Vector DB (`storage/lancedb/`): ~500MB–1GB
- Total KB folder: ~100MB (HTML + PHP + JS sources)

### Search Latency Breakdown
- Vector similarity (top-25): ~20–50ms
- Reranking (top-25 → top-5): ~50–150ms
- Network roundtrip: ~10–50ms
- **Total per-query**: ~100–300ms

### Optimization Tips
- Category filtering reduces rerank workload
- Batch indexing with `reset_db=False` is much faster than full reindex
- Use `BATCH_SIZE_FILES = 10` as baseline; increase to 25 if memory permits

## Common Pitfalls & Troubleshooting

### Memory Issues During Indexing
- **Problem**: Indexing large sources causes OOM or slowdown
- **Root Cause**: `BATCH_SIZE_FILES = 10` batches files, not chunks. Large files → many chunks per batch
- **Fix**: Reduce batch size in [server.py](server.py#L233) or increase chunk overlap to consolidate chunks

### First-Run Model Download Fails
- **Problem**: Server hangs or crashes during first initialization
- **Root Cause**: ~2GB model downloads over slow network
- **Fix**: Run with increased timeout; check internet connectivity. Models cache to `~/.cache/huggingface/`

### Database Corruption After Interrupt
- **Problem**: LanceDB table is corrupted or inaccessible after interrupting `python server.py`
- **Root Cause**: LanceDB batch writes aren't atomic; partial data committed
- **Fix**: Delete `storage/lancedb/` and rerun with `reset_db=True`

### TOC Mismatch & Missing Breadcrumbs
- **Problem**: Some HTML files index without breadcrumb context
- **Root Cause**: `toc.json` filenames don't match actual files (e.g., `28709.htm` vs `28709.html`)
- **Fix**: Verify filename matching in `load_toc_map()` output; fallback uses `<title>` tag only

### Duplicate Chunks in Results
- **Problem**: Same document appears multiple times in search results
- **Root Cause**: Multiple chunks from same source file returned by reranker
- **Fix**: Expected behavior. To deduplicate: group results by `filename` client-side

## Database Schema & Query Patterns

### UnifiedSchema Fields
LanceDB table `plesk_knowledge` has this structure:
- `vector`: 1024-dim embedding (BAAI/bge-m3) — auto-indexed for similarity search
- `text`: Full document chunk with category prefix and breadcrumb context
- `title`: Page/file title from TOC or `<title>` tag
- `filename`: Source filename (e.g., `28709.htm`, `Button.php`)
- `category`: Source category (`"guide"`, `"cli"`, `"api"`, `"php-stubs"`, `"js-sdk"`)
- `breadcrumb`: Hierarchical path from TOC (e.g., `"Extensions > UI > Buttons"`)

### Direct LanceDB Queries (for debugging)
```python
import lancedb
from pathlib import Path

db = lancedb.connect(str(Path(__file__).parent / "storage" / "lancedb"))
table = db.open_table("plesk_knowledge")

# Get all docs from category
docs = table.search().where("category = 'api'").limit(10).to_list()

# Count chunks per source
results = table.search().where("category = 'php-stubs'").limit(10000).to_list()
files = {r["filename"] for r in results}
print(f"Indexed {len(files)} files in php-stubs")
```

### Query Performance Notes
- `.where()` filtering scans all vectors — use specific categories if possible
- Reranking (top-25 → top-5) takes ~50–200ms; initial search is <50ms
- No pagination: retrieval limit capped at 25 pre-rerank, 5 post-rerank

## Testing & Validation

- No automated tests in repo; verify via:
  - Manual indexing: `python server.py` completes without errors
  - Search query returns results with valid relevance scores
  - LanceDB persisted in `storage/lancedb/` survives restarts

## Dependencies & Environment

- **Python**: 3.12+
- **Key Packages**: fastmcp (≥2.14.5), lancedb (≥0.29.1), sentence-transformers (≥5.2.2)
- **Storage**: ~2GB models + ~1GB embeddings + variable KB size
- **Virtual Environment**: `.venv/` folder, activated before running
