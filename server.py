import os
import sys
import json
import time
import threading
import requests
import re
from pathlib import Path
from pydantic import Field
from typing import Any, cast

# Lightweight import needed for MCP registration
from fastmcp import FastMCP

# --- SILENCE THE NOISE ---
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Initialize MCP
mcp = FastMCP("plesk-unified-master")

# --- Configuration ---
BASE_DIR = Path(__file__).parent
KB_DIR = BASE_DIR / "knowledge_base"
DB_PATH = BASE_DIR / "storage" / "lancedb"
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY")

KB_DIR.mkdir(exist_ok=True, parents=True)
(BASE_DIR / "storage").mkdir(exist_ok=True, parents=True)

SOURCES = [
    {"path": KB_DIR / "guide", "cat": "guide", "type": "html", "repo_url": None},
    {"path": KB_DIR / "cli", "cat": "cli", "type": "html", "repo_url": None},
    {"path": KB_DIR / "api", "cat": "api", "type": "html", "repo_url": None},
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

# --- Global Cache for Lazy Loading ---
_GLOBALS: dict[str, Any] = {
    "embedding_model": None,
    "reranker": None,
    "Schema": None,
    "lancedb": None,
}

# Global Status Tracker for background ingestion jobs
_JOB_STATUS = {
    "is_running": False,
    "last_message": "Idle",
    "history": [],
}


def log_job(msg: str):
    """Helper to update global status and print to stderr"""
    _JOB_STATUS["last_message"] = msg
    _JOB_STATUS["history"].append(msg)
    print(f"[LOG] {msg}", file=sys.stderr)


def get_resources() -> tuple[Any, Any, type]:
    """Lazy loads heavy AI models to prevent MCP startup timeouts."""
    if _GLOBALS["Schema"] is not None:
        return _GLOBALS["embedding_model"], _GLOBALS["reranker"], _GLOBALS["Schema"]

    print("[LOG] Lazy loading AI models...", file=sys.stderr)
    import lancedb
    # [FIX 1] Import Vector here
    from lancedb.pydantic import LanceModel, Vector
    from lancedb.embeddings import get_registry
    from lancedb.rerankers import CrossEncoderReranker

    registry = get_registry()
    embedding_model = registry.get("huggingface").create(name="BAAI/bge-m3")
    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-base")

    class UnifiedSchema(LanceModel):
        # BGE-M3 uses 1024 dimensions. This tells Arrow exactly how to store the data.
        vector: Vector(1024) = embedding_model.VectorField() # type: ignore
        text: str = embedding_model.SourceField()
        title: str = ""
        filename: str = ""
        category: str = ""
        breadcrumb: str = ""

    _GLOBALS["embedding_model"] = embedding_model
    _GLOBALS["reranker"] = reranker
    _GLOBALS["Schema"] = UnifiedSchema
    _GLOBALS["lancedb"] = lancedb

    return embedding_model, reranker, UnifiedSchema


def get_table(create_new=False):
    _, _, Schema = get_resources()
    import lancedb

    db = lancedb.connect(str(DB_PATH))
    try:
        if create_new:
            return db.create_table("plesk_knowledge", schema=Schema, mode="overwrite")
        return db.open_table("plesk_knowledge")
    except Exception:
        return db.create_table("plesk_knowledge", schema=Schema, mode="create")


# --- AI Enrichment ---
def get_ai_description(file_path, file_name):
    """Summarizes file with a 3-tier fallback chain."""
    models = [
        "arcee-ai/trinity-large-preview:free",
        "stepfun/step-3-5-flash:free",
        "liquid/lfm-2.5-1.2b-thinking:free",
    ]

    if not OPENROUTER_KEY:
        return "API Key missing."

    content = ""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(2500)
    except Exception:
        return "File unreadable."

    for model in models:
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "Content-Type": "application/json",
                },
                data=json.dumps(
                    {
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": f"Summarize the technical purpose of '{file_name}' in 1 concise sentence.\n\n{content}",
                            }
                        ],
                        "max_tokens": 100,
                    }
                ),
                timeout=15,
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            continue
    return "No description available (All AI fallbacks failed)."


# --- Helper: Git & TOC Logic ---
def ensure_source_exists(source):
    if source["path"].exists() and any(source["path"].iterdir()):
        return True
    if source["repo_url"]:
        print(f"[LOG] Downloading {source['cat']}...", file=sys.stderr)
        try:
            from git import Repo
            Repo.clone_from(source["repo_url"], source["path"])
            return True
        except Exception as e:
            print(f"[ERR] Failed to clone {source['cat']}: {e}", file=sys.stderr)
            return False
    return False


def generate_and_enrich_toc(source_path, category_name, enrich_ai=True):
    ignore_list = {".git", ".github", "tests", "vendor", "node_modules", "bin"}
    files_found = []

    toc_file = Path(source_path) / "virtual_toc.json"
    existing_data = {}
    if toc_file.exists():
        try:
            with open(toc_file, "r") as f:
                old_json = json.load(f)
                existing_data = {
                    f["path"]: f.get("description") for f in old_json.get("files", [])
                }
        except Exception:
            pass

    print(f"[LOG] Processing TOC for {category_name}...", file=sys.stderr)
    for root, _, files in os.walk(source_path):
        if any(ignore in root for ignore in ignore_list):
            continue
        for file in files:
            if file.endswith((".php", ".js", ".ts", ".md")):
                rel_path = str((Path(root) / file).relative_to(source_path))
                desc = existing_data.get(rel_path)
                if not desc and enrich_ai:
                    desc = get_ai_description(Path(root) / file, file)
                    time.sleep(0.5)

                files_found.append(
                    {
                        "name": file,
                        "path": rel_path,
                        "description": desc or "No description available.",
                    }
                )

    with open(toc_file, "w", encoding="utf-8") as f:
        json.dump({"category": category_name, "files": files_found}, f, indent=2)


def chunk_php_stubs(file_path):
    """
    Intelligently splits PHP Stubs/Classes into semantic chunks.
    """
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    lines = content.split('\n')
    
    chunks = []
    current_class = "Global"
    docblock_buffer = []
    in_docblock = False
    
    class_pattern = re.compile(r'^\s*(?:abstract\s+|final\s+)*(class|interface|trait)\s+(\w+)', re.IGNORECASE)
    method_pattern = re.compile(r'^\s*(?:public|protected|private|static|\s)*function\s+(\w+)', re.IGNORECASE)

    for line in lines:
        stripped = line.strip()
        
        if stripped.startswith('/**'):
            in_docblock = True
            docblock_buffer = [line]
            continue
        if in_docblock:
            docblock_buffer.append(line)
            if stripped.endswith('*/'):
                in_docblock = False
            continue

        class_match = class_pattern.search(line)
        if class_match:
            current_class = class_match.group(2)
            full_text = "\n".join(docblock_buffer) + "\n" + line
            chunks.append({
                "title": f"Class {current_class}",
                "text": full_text,
                "breadcrumb": f"PHP > {current_class}"
            })
            docblock_buffer = [] 
            continue

        method_match = method_pattern.search(line)
        if method_match:
            method_name = method_match.group(1)
            full_text = f"// Class: {current_class}\n" 
            full_text += "\n".join(docblock_buffer) + "\n" + line
            
            chunks.append({
                "title": f"{current_class}::{method_name}",
                "text": full_text,
                "breadcrumb": f"PHP > {current_class} > {method_name}"
            })
            docblock_buffer = [] 
            continue
            
    return chunks


def _worker_ingest(target_category: str, reset_db: bool, enrich_ai: bool):
    """Background worker that performs the heavy ingestion work.

    Updates `_JOB_STATUS` so callers can poll progress.
    """
    global _JOB_STATUS
    _JOB_STATUS["is_running"] = True
    _JOB_STATUS["history"] = []

    try:
        log_job("Starting resource loading...")
        get_resources()

        log_job("Opening database...")
        table = get_table(create_new=reset_db)

        total_chunks = 0

        for source in SOURCES:
            if target_category != "all" and source["cat"] != target_category:
                continue

            log_job(f"Processing source: {source['cat']}...")
            if not ensure_source_exists(source):
                log_job(f"FAILED: {source['cat']} (Source missing)")
                continue

            if source["cat"] in ["php-stubs", "js-sdk"]:
                generate_and_enrich_toc(source["path"], source["cat"], enrich_ai=enrich_ai)

            # 1. IDENTIFY FILES
            files = []
            if source["type"] == "html":
                files = list(source["path"].rglob("*.htm*"))
                chunk_size = 3000
            elif source["type"] == "php":
                files = list(source["path"].rglob("*.php"))
            else:
                files = list(source["path"].rglob("*.js")) + list(source["path"].rglob("*.md"))
                chunk_size = 5000

            # 2. PROCESS FILES
            cat_docs = []
            for f in files:
                if f.name.startswith("_") or f.name.endswith(".json"):
                    continue

                try:
                    # === BRANCH A: CUSTOM PHP CHUNKING ===
                    if source["type"] == "php":
                        php_chunks = chunk_php_stubs(f)
                        for c in php_chunks:
                            cat_docs.append({
                                "text": c["text"],
                                "title": c["title"],
                                "filename": f.name,
                                "category": source["cat"],
                                "breadcrumb": c["breadcrumb"],
                            })
                        continue

                    # === BRANCH B: STANDARD TEXT CHUNKING ===
                    if source["type"] == "html":
                        title, breadcrumb, text = parse_html(f)
                    else:
                        title, breadcrumb, text = parse_code(f)

                    if text and len(text) > 50:
                        chunks = [
                            text[i : i + chunk_size] for i in range(0, len(text), chunk_size - 500)
                        ]
                        for chunk in chunks:
                            cat_docs.append({
                                "text": f"[{source['cat'].upper()}] {title}\n---\n{chunk}",
                                "title": title or "Untitled",
                                "filename": f.name,
                                "category": source["cat"],
                                "breadcrumb": breadcrumb or "",
                            })
                except Exception as e:
                    print(f"[WARN] Failed to parse {f.name}: {e}", file=sys.stderr)

            if cat_docs:
                log_job(f"Inserting {len(cat_docs)} chunks for {source['cat']}...")
                table.add(cat_docs)
                total_chunks += len(cat_docs)
                log_job(f"SUCCESS: {source['cat']} complete.")

        log_job(f"Ingestion finished. Added {total_chunks} total chunks.")

    except Exception as e:
        log_job(f"CRITICAL ERROR: {str(e)}")
    finally:
        _JOB_STATUS["is_running"] = False

# --- MCP Resources ---

@mcp.resource("plesk://docs/{category}/toc")
def get_category_toc(category: str) -> str:
    mapping = {
        "guide": "guide", "cli": "cli", "api": "api",
        "php-stubs": "stubs", "js-sdk": "sdk",
    }
    target = mapping.get(category)
    if not target:
        return f"Error: Category '{category}' is invalid."

    base_path = KB_DIR / target
    for toc_name in ["virtual_toc.json", "toc.json"]:
        path = base_path / toc_name
        if path.exists():
            return path.read_text(encoding="utf-8")

    return f"No TOC found for {category}."


@mcp.resource("plesk://docs/list")
def list_categories() -> str:
    category_descriptions = {
        "guide": "General Plesk Obsidian administration and user guides (HTML).",
        "cli": "Plesk Command Line Interface (CLI) reference and utilities.",
        "api": "XML-RPC and REST API documentation for remote integration.",
        "php-stubs": "Internal Plesk PHP classes (pm_ namespace) for extension development.",
        "js-sdk": "Frontend SDK components and documentation for the Plesk GUI.",
    }
    output = ["Available Plesk Documentation Categories:"]
    for source in SOURCES:
        cat = source["cat"]
        output.append(f"- {cat}: {category_descriptions.get(cat, 'No desc.')}")
    return "\n".join(output)


# --- Content Parsers ---
def parse_html(file_path, toc_metadata=None):
    try:
        from bs4 import BeautifulSoup
        html = file_path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        title, breadcrumb = "Untitled", ""
        if toc_metadata:
            title = toc_metadata.get("title", title)
            breadcrumb = toc_metadata.get("breadcrumb", "")
        elif soup.title and soup.title.string:
            title = soup.title.string.replace(" — Plesk Obsidian documentation", "").strip()

        content = soup.find("div", attrs={"itemprop": "articleBody"}) or soup.body
        if content:
            for tag in content(["script", "style", "nav", "footer", "iframe"]):
                tag.decompose()
            clean_text = content.get_text(separator="\n", strip=True)
            if breadcrumb:
                clean_text = f"Context: {breadcrumb}\n\n{clean_text}"
            return title, breadcrumb, clean_text
    except Exception:
        pass
    return None, None, None


def parse_code(file_path):
    try:
        return (file_path.name, "", file_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None, None, None


# --- Tools ---

@mcp.tool
def refresh_knowledge(
    target_category: str = Field("all", description="Target category."),
    reset_db: bool = Field(False, description="Reset index."),
    enrich_ai: bool = Field(True, description="Use AI descriptions."),
) -> str:
    """Starts the knowledge ingestion process in the background."""
    if _JOB_STATUS["is_running"]:
        return "ERROR: An ingestion job is already running. Please wait for it to finish."

    # Start the worker in a separate thread
    t = threading.Thread(
        target=_worker_ingest, args=(target_category, reset_db, enrich_ai), daemon=True
    )
    t.start()

    return "Ingestion started in background. Use 'check_ingestion_status' to monitor progress."


@mcp.tool
def check_ingestion_status() -> str:
    """Checks the status of the background ingestion job."""
    status_str = "RUNNING" if _JOB_STATUS["is_running"] else "IDLE"
    history = "\n".join([f"- {h}" for h in _JOB_STATUS["history"][-5:]])
    return f"Status: {status_str}\nLast Message: {_JOB_STATUS['last_message']}\n\nRecent Log:\n{history}"


@mcp.tool
def search_plesk_unified(
    query: str = Field(..., description="Search query."),
    category: str | None = Field(None, description="Optional category filter."),
):
    """Deep semantic search across Plesk documentation and code stubs."""
    _, reranker, _ = get_resources()
    table = get_table()
    search_op = table.search(query).limit(25)
    if category:
        search_op = search_op.where(f"category = '{category}'")

    print(f"[LOG] Reranking results for: '{query}'...", file=sys.stderr)
    results = search_op.rerank(reranker=cast(Any, reranker)).limit(5).to_list()

    return "\n".join(
        [
            f"=== {r['category'].upper()} | {r['title']} ===\nFile: {r['filename']}\nScore: {r['_relevance_score']:.4f}\n\n{r['text']}\n"
            for r in results
        ]
    )

if __name__ == "__main__":
    mcp.run()