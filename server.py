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

# ruff: noqa: E402
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import json

# --- LOGGING SETUP ---
# Must be done before importing heavy libraries to capture their init warnings if needed.
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

from fastmcp import FastMCP
from bs4 import BeautifulSoup
import lancedb
from git import Repo
from lancedb.pydantic import LanceModel
from lancedb.embeddings import get_registry
from lancedb.rerankers import CrossEncoderReranker
from pydantic import Field
from typing import Any, Dict, List, Optional, Tuple

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

# Reranking Model
logger.info("Loading Reranker (BAAI/bge-reranker-base) on %s...", device)
reranker = None
try:
    try:
        reranker = CrossEncoderReranker(
            model_name="BAAI/bge-reranker-base", device=device
        )
    except TypeError:
        reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-base")
    logger.info("Reranker initialized successfully.")
except Exception as e:
    logger.warning(
        "Reranker init failed: %s. Reranking will be skipped.", e, exc_info=True
    )
    reranker = None


# Prepare vector field
_vector_field = embedding_model.VectorField()
try:
    extra = getattr(_vector_field, "json_schema_extra", None)
    if extra is None:
        _vector_field.json_schema_extra = {"dim": 1024}
    else:
        extra["dim"] = 1024
except Exception:
    pass


class UnifiedSchema(LanceModel):
    vector: List[float] = _vector_field
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


# --- Helper: Git Auto-Loader ---
def ensure_source_exists(source: Dict[str, Any]) -> bool:
    """Ensure the source repository exists and is not empty."""
    if source["path"].exists() and any(source["path"].iterdir()):
        logger.debug("Source %s already exists at %s", source["cat"], source["path"])
        return True

    if source["repo_url"]:
        logger.info("Downloading %s from %s...", source["cat"], source["repo_url"])
        try:
            Repo.clone_from(source["repo_url"], source["path"])
            logger.info("Clone succeeded for %s", source["cat"])
            return True
        except Exception:
            logger.error("Clone failed for %s", source["cat"], exc_info=True)
            return False
    return False


# --- TOC Parsing Logic ---
def parse_toc_recursive(
    nodes: List[Dict[str, Any]],
    parent_path: str = "",
    file_map: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, Dict[str, str]]:
    """Recursively parse TOC nodes to build a file map."""
    if file_map is None:
        file_map = {}
    for node in nodes:
        title = node.get("text", "Untitled")
        url_raw = node.get("url", "")
        current_path = f"{parent_path} > {title}" if parent_path else title
        filename = url_raw.split("#")[0]
        if filename and filename not in file_map:
            file_map[filename] = {"title": title, "breadcrumb": current_path}
        if "children" in node:
            parse_toc_recursive(node["children"], current_path, file_map)
    return file_map


def load_toc_map(folder_path: Path) -> Dict[str, Dict[str, str]]:
    """Load and parse a TOC map from a folder's toc.json file."""
    toc_path = folder_path / "toc.json"
    if not toc_path.exists():
        return {}
    try:
        data = json.loads(toc_path.read_text(encoding="utf-8"))
        return parse_toc_recursive(data)
    except Exception:
        logger.warning("Failed to parse TOC at %s", toc_path, exc_info=True)
        return {}


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


# --- Tools ---
@mcp.tool
def refresh_knowledge(
    target_category: str = Field(
        "all",
        description="Category to index. Choose one: 'guide', 'cli', 'api', 'php-stubs', 'js-sdk' or 'all'.",
    ),
    reset_db: bool = Field(
        False,
        description="Set to True ONLY for the first run to wipe the database. Default is False (resume).",
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
                logger.info(
                    "Checking existing files for category '%s'...", target_category
                )
                results = (
                    table.search()
                    .where(f"category='{target_category}'")
                    .limit(10000)
                    .to_list()
                )
                for r in results:
                    existing_files.add(r["filename"])
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
        if not ensure_source_exists(source):
            msg = f"SKIPPED {source['cat']} (Source check failed)"
            logger.error(msg)
            report.append(msg)
            continue

        toc_map = {}
        if source["type"] == "html":
            toc_map = load_toc_map(source["path"])
            files = list(source["path"].rglob("*.htm")) + list(
                source["path"].rglob("*.html")
            )
            parser: Any = parse_html
            chunk_size = 3000
        elif source["type"] == "php":
            files = list(source["path"].rglob("*.php"))
            parser: Any = parse_code
            chunk_size = 6000
        else:
            files = list(source["path"].rglob("*.js")) + list(
                source["path"].rglob("*.md")
            )
            parser: Any = parse_code
            chunk_size = 5000

        logger.info("Found %d files for source %s", len(files), source["cat"])

        cat_docs = []
        files_processed_count = 0
        BATCH_SIZE_FILES = 50

        for f in files:
            if f.name.startswith("_") or f.name == "toc.json":
                continue

            if f.name in existing_files:
                continue

            meta = toc_map.get(f.name) if toc_map else None
            if source["type"] == "html":
                title, breadcrumb, text = parser(f, meta)
            else:
                title, breadcrumb, text = parser(f)

            if text and len(text) > 50:
                chunks = [
                    text[i : i + chunk_size]
                    for i in range(0, len(text), chunk_size - 500)
                ]
                for chunk in chunks:
                    cat_docs.append(
                        {
                            "text": f"[{source['cat'].upper()}] {title}\n---\n{chunk}",
                            "title": title,
                            "filename": f.name,
                            "category": source["cat"],
                            "breadcrumb": breadcrumb,
                        }
                    )

            files_processed_count += 1

            if files_processed_count >= BATCH_SIZE_FILES:
                if cat_docs:
                    logger.info(
                        "Saving batch of %d chunks for %s...",
                        len(cat_docs),
                        source["cat"],
                    )
                    try:
                        table.add(cat_docs)
                        cat_docs = []
                    except Exception:
                        logger.exception("Failed to add batch to LanceDB")
                files_processed_count = 0

        if cat_docs:
            logger.info(
                "Saving final batch of %d chunks for %s...",
                len(cat_docs),
                source["cat"],
            )
            try:
                table.add(cat_docs)
            except Exception:
                logger.exception("Failed to add final batch to LanceDB")

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
    Search the Unified Knowledge Base with Reranking.
    """
    # Truncate query for logging to avoid leaking sensitive data
    safe_query = (query[:100] + "...") if len(query) > 100 else query
    logger.info("Search request: q='%s' category='%s'", safe_query, category)

    table = get_table()

    # 1. Busca Rápida
    search_op = table.search(query).limit(25)
    if category:
        search_op = search_op.where(f"category = '{category}'")

    logger.debug("Executing vector search + reranking...")

    # 2. Reranking
    if reranker is not None:
        try:
            results = search_op.rerank(reranker=reranker).limit(5).to_list()
        except Exception:
            logger.error("Reranking failed during search", exc_info=True)
            results = search_op.limit(5).to_list()
    else:
        logger.warning("Reranker unavailable; using vector search only.")
        results = search_op.limit(5).to_list()

    logger.info("Search returned %d results.", len(results))
    if results:
        logger.info("Top result score: %.4f", results[0].get("_relevance_score", 0.0))

    return "\n".join(
        [
            f"=== {r['category'].upper()} | {r['title']} ===\n"
            f"Path: {r['breadcrumb']}\n"
            f"File: {r['filename']}\n"
            f"Relevance Score: {r['_relevance_score']:.4f}\n\n"
            f"{r['text']}\n"
            for r in results
        ]
    )


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
