# /// script
# dependencies = [
#   "fastmcp",
#   "lancedb",
#   "beautifulsoup4",
#   "GitPython",
#   "torch",
#   "transformers",
#   "sentence-transformers",
#   "numpy",
#   "pydantic"
# ]
# ///

import logging

# ruff: noqa: E402
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# --- LOGGING SETUP ---
# Must be done before importing heavy libraries.
# This ensures we capture their initialization warnings if needed.
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "storage" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = os.environ.get("LOG_FILE", str(LOG_DIR / "plesk_unified.log"))
# Convert string level (e.g. "INFO") to integer
LOG_LEVEL_NAME = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_NAME, logging.INFO)

# Create logger
logger = logging.getLogger("plesk_unified")
logger.setLevel(LOG_LEVEL)

# Formatter
formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# 1. File Handler (Rotating)
file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=10_485_760,
    backupCount=5,
    encoding="utf-8",  # 10MB
)
file_handler.setFormatter(formatter)
file_handler.setLevel(LOG_LEVEL)

# 2. Stream Handler (stderr) - CRITICAL for MCP protocol
stream_handler = logging.StreamHandler(sys.stderr)
stream_handler.setFormatter(formatter)
stream_handler.setLevel(LOG_LEVEL)

# Avoid adding duplicate handlers if reloaded
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

# Silence noisy third-party libraries unless they error
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("git").setLevel(logging.WARNING)

logger.info(f"Logging initialized. Level: {LOG_LEVEL_NAME}, File: {LOG_FILE}")


# --- SILENCE THE NOISE ---
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from typing import Any, Dict, List, Optional, Tuple

import lancedb  # type: ignore
from bs4 import BeautifulSoup
from fastmcp import FastMCP
from lancedb.embeddings import get_registry  # type: ignore
from lancedb.pydantic import LanceModel, Vector  # type: ignore
from pydantic import Field
from plesk_unified import html_utils, chunking, io_utils

# Detect best device
# Priority: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
device = "cpu"
try:
    import torch

    # Check for CUDA first (NVIDIA GPUs on Windows/Linux)
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("NVIDIA GPU (CUDA) detected. Using for acceleration.")
    # Check for MPS (Apple Silicon on macOS)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Apple Silicon GPU (MPS) detected. Using for acceleration.")
    else:
        logger.info("No GPU acceleration available. Using CPU.")
except ImportError:
    logger.warning("Torch not available; using CPU.", exc_info=True)
except Exception:
    logger.warning("Error detecting device; using CPU.", exc_info=True)


# Initialize MCP
mcp = FastMCP("mcp-plesk-unified")

# --- Configuration ---
KB_DIR = BASE_DIR / "knowledge_base"
DB_PATH = BASE_DIR / "storage" / "lancedb"

KB_DIR.mkdir(exist_ok=True)
(BASE_DIR / "storage").mkdir(exist_ok=True)

SOURCES = [
    {
        "path": KB_DIR / "stubs",
        "cat": "php-stubs",
        "type": "php",
        "repo_url": "https://github.com/plesk/pm-api-stubs.git",
    },
    {
        "path": KB_DIR / "sdk",
        "cat": "js-sdk",
        "type": "js",
        "repo_url": "https://github.com/plesk/plesk-ext-sdk.git",
    },
    {
        "path": KB_DIR / "api-rpc",
        "cat": "api",
        "type": "md",
        "repo_url": None,
    },
    {
        "path": KB_DIR / "cli-linux",
        "cat": "cli",
        "type": "md",
        "repo_url": None,
    },
    {
        "path": KB_DIR / "extensions-guide",
        "cat": "guide",
        "type": "md",
        "repo_url": None,
    },
]

# --- Database Setup ---
logger.info("Initializing embedding model BAAI/bge-m3 on %s...", device)
try:
    try:
        embedding_model = (
            get_registry().get("huggingface").create(name="BAAI/bge-m3", device=device)
        )
    except TypeError:
        # Provider may not accept `device` kwarg; retry without it
        logger.debug("Device argument failed, retrying without device kwarg.")
        embedding_model = get_registry().get("huggingface").create(name="BAAI/bge-m3")
    logger.info("Embedding model initialized successfully.")
except Exception:
    logger.warning("Embedding model init failed. Retrying CPU-only.", exc_info=True)
    try:
        embedding_model = get_registry().get("huggingface").create(name="BAAI/bge-m3")
        logger.info("Embedding model initialized (CPU fallback).")
    except Exception:
        logger.critical("Embedding model could not be initialized.", exc_info=True)
        raise


class UnifiedSchema(LanceModel):
    # Use LanceDB's Vector type to ensure a FixedSizeList Arrow column is created.
    # BAAI/bge-m3 uses 1024-dimensional embeddings.
    vector: Vector(1024) = embedding_model.VectorField()  # type: ignore
    text: str = embedding_model.SourceField()
    title: str
    filename: str
    category: str
    breadcrumb: str


def get_table(create_new: bool = False) -> Any:
    """Connect to or create the LanceDB table."""
    logger.debug("Connecting to LanceDB at %s", DB_PATH)
    db = lancedb.connect(str(DB_PATH))
    try:
        if create_new:
            logger.info("Creating/overwriting table 'plesk_knowledge'")
            return db.create_table(
                "plesk_knowledge", schema=UnifiedSchema, mode="overwrite"
            )
        return db.open_table("plesk_knowledge")
    except Exception:
        logger.info(
            "Table not found or error opening. Creating new 'plesk_knowledge' table."
        )
        return db.create_table("plesk_knowledge", schema=UnifiedSchema, mode="create")


# --- Source Handling are now in plesk_unified.io_utils ---


# --- Content Parsers ---
def parse_html(
    file_path: Path,
    toc_metadata: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse an HTML file and extract title, breadcrumb, and text content."""
    try:
        html = file_path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        title = "Untitled"
        breadcrumb = ""
        if toc_metadata:
            title = toc_metadata.get("title", title)
            breadcrumb = toc_metadata.get("breadcrumb", "")
        elif soup.title and soup.title.string:
            title = soup.title.string.replace(
                " — Plesk Obsidian documentation", ""
            ).strip()

        content = soup.find("div", attrs={"itemprop": "articleBody"})
        if not content:
            content = soup.body

        if content:
            for tag in content(["script", "style", "nav", "footer", "iframe"]):
                tag.decompose()
            clean_text = content.get_text(separator="\n", strip=True)
        else:
            clean_text = ""
        if breadcrumb:
            clean_text = f"Context: {breadcrumb}\n\n{clean_text}"
        return title, breadcrumb, clean_text
    except Exception:
        logger.warning("Error parsing HTML file: %s", file_path.name, exc_info=True)
        return None, None, None


def parse_code(file_path: Path) -> Tuple[Optional[str], str, Optional[str]]:
    """Parse a code file and return filename, empty string, and content."""
    try:
        return (
            file_path.name,
            "",
            file_path.read_text(encoding="utf-8", errors="ignore"),
        )
    except Exception:
        logger.warning("Error parsing code file: %s", file_path.name, exc_info=True)
        return None, "", None


# --- Chunking Strategies ---
def chunk_by_lines(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """Chunks text by lines with overlap."""
    lines = text.splitlines()
    if not lines:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(lines), step):
        chunk = "\n".join(lines[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def chunk_by_chars(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """Chunks text by characters with overlap."""
    if not text:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(text), step):
        chunk = text[i : i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# --- Tools ---
def get_file_parser_and_toc(source):
    """Returns the correct parser and load TOC map for the given source type."""
    if source["type"] == "html":
        return html_utils.parse_html_file, io_utils.load_toc_map(source["path"])
    return parse_code, {}


def build_and_chunk_docs(source, file_path, title, breadcrumb, text):
    """Chunks text and builds document records."""
    if not text or len(text) <= 10:
        return []

    if source["type"] == "html":
        chunks = chunking.chunk_by_chars(text, 1500, 200)
    elif source["type"] == "php":
        chunks = chunking.chunk_by_lines(text, 150, 20)
    else:
        chunks = chunking.chunk_by_lines(text, 60, 10)

    if not chunks:
        return []

    records = chunking.build_doc_records(
        file_path.name,
        chunks,
        {"title": title, "category": source["cat"], "breadcrumb": breadcrumb},
    )

    for r in records:
        r["text"] = f"[{source['cat'].upper()}] {title}\n---\n{r['text']}"

    return records


def process_source_files(source, table, existing_files):
    """Processes all files for a given source and indexes them."""
    parser, toc_map = get_file_parser_and_toc(source)
    files = io_utils.collect_files_for_source(source)
    logger.info("Found %d files for source %s", len(files), source["cat"])

    cat_docs = []
    files_processed_count = 0
    BATCH_SIZE_FILES = 50

    for f in files:
        if f.name.startswith("_") or f.name == "toc.json" or f.name in existing_files:
            continue

        meta = toc_map.get(f.name) if toc_map else None

        if source["type"] == "html":
            title, breadcrumb, text = parser(f, meta)
        else:
            title, breadcrumb, text = parser(f)

        records = build_and_chunk_docs(source, f, title, breadcrumb, text)
        cat_docs.extend(records)

        files_processed_count += 1
        if files_processed_count >= BATCH_SIZE_FILES:
            if cat_docs:
                logger.info(
                    "Saving batch of %d chunks for %s...", len(cat_docs), source["cat"]
                )
                chunking.persist_batch(table, cat_docs)
                cat_docs = []
            files_processed_count = 0

    if cat_docs:
        logger.info(
            "Saving final batch of %d chunks for %s...", len(cat_docs), source["cat"]
        )
        chunking.persist_batch(table, cat_docs)


@mcp.tool
def refresh_knowledge(
    target_category: str = Field(
        "all",
        description=(
            "Category to index. Choose one: 'guide', 'cli', 'api', 'php-stubs', "
            "'js-sdk' or 'all'."
        ),
    ),
    reset_db: bool = Field(
        False,
        description=(
            "Set to True ONLY for the first run to wipe the database. "
            "Default is False (resume)."
        ),
    ),
):
    """
    Indexes Plesk documentation into LanceDB.
    """
    logger.info(
        "Starting refresh_knowledge: target=%s, reset_db=%s", target_category, reset_db
    )

    if reset_db:
        table = get_table(create_new=True)
        logger.warning("Database wiped by request.")
        existing_files = set()
    else:
        table = get_table(create_new=False)
        existing_files = set()
        if target_category != "all":
            try:
                results = (
                    table.search()
                    .where(f"category='{target_category}'")
                    .limit(10000)
                    .to_list()
                )
                existing_files.update(r["filename"] for r in results)
                logger.info(
                    "Found %d existing files/chunks in DB.", len(existing_files)
                )
            except Exception as e:
                logger.warning("Could not fetch existing files: %s", e)

    report = []

    for source in SOURCES:
        if target_category != "all" and source["cat"] != target_category:
            continue

        logger.info("Processing source: %s", source["cat"])
        if not io_utils.ensure_source_exists(source):
            msg = f"SKIPPED {source['cat']} (Source check failed)"
            logger.error(msg)
            report.append(msg)
            continue

        try:
            process_source_files(source, table, existing_files)
        except Exception:
            logger.exception("Error processing source %s", source["cat"])

        msg = f"Finished pass for {source['cat']}."
        logger.info(msg)
        report.append(msg)

    return "\n".join(report)


@mcp.tool
def search_plesk_unified(
    query: str = Field(..., description="The search query (e.g. 'how to add button')"),
    category: str | None = Field(
        None,
        description="Filter by category: 'guide', 'cli', 'api', 'php-stubs', 'js-sdk'",
    ),
):
    """
    Search the Unified Knowledge Base.
    """
    # Truncate query for logging to avoid leaking sensitive data
    safe_query = (query[:100] + "...") if len(query) > 100 else query
    logger.info("Search request: q='%s' category='%s'", safe_query, category)

    table = get_table()

    # Perform the search (no hard-coded pre-limit to allow provider to optimize)
    search_op = table.search(query)
    if category:
        search_op = search_op.where(f"category = '{category}'")

    results = search_op.limit(5).to_list()

    logger.info("Search returned %d results.", len(results))

    formatted_results = []
    for r in results:
        # LanceDB returns `_distance` for vector search and `_score` for FTS.
        # Use .get() to avoid KeyError if the backend changes.
        score = (
            r.get("_distance")
            if r.get("_distance") is not None
            else r.get("_score", 0.0)
        )

        formatted_results.append(
            (
                f"=== {r['category'].upper()} | {r['title']} ===\n"
                f"Path: {r.get('breadcrumb', '')}\n"
                f"File: {r.get('filename', '')}\n"
                f"Score/Distance: {score:.4f}\n\n"
                f"{r.get('text', '')}\n"
            )
        )

    return "\n".join(formatted_results)


if __name__ == "__main__":
    logger.info("Starting Plesk Unified MCP Server...")
    try:
        mcp.run()
    except Exception:
        logger.critical("Server crashed", exc_info=True)
        raise


def main() -> None:
    """Console entrypoint for the MCP server (used by package scripts/devtools)."""
    logger.info("Starting Plesk Unified MCP Server (entrypoint)...")
    try:
        mcp.run()
    except Exception:
        logger.critical("Server crashed (entrypoint)", exc_info=True)
        raise
