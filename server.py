import os
import sys

# --- SILENCE THE NOISE ---
# This prevents the 14,000 "Loading weights" notifications
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from fastmcp import FastMCP
from pathlib import Path
from bs4 import BeautifulSoup
import json
import lancedb
from git import Repo
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from lancedb.rerankers import CrossEncoderReranker
from pydantic import Field  # <--- IMPORT NECESSÁRIO PARA AS DESCRIÇÕES

# Initialize
mcp = FastMCP("plesk-unified-master")

# --- Configuration ---
BASE_DIR = Path(__file__).parent
KB_DIR = BASE_DIR / "knowledge_base"
DB_PATH = BASE_DIR / "storage" / "lancedb"

KB_DIR.mkdir(exist_ok=True)
(BASE_DIR / "storage").mkdir(exist_ok=True)

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

# --- Database Setup ---
# Using BAAI/bge-m3 (No custom code required)
embedding_model = get_registry().get("huggingface").create(name="BAAI/bge-m3")

# NOVO: Modelo de Reranking
# Isso vai baixar o modelo automaticamente na primeira execução (~1GB)
print("Loading Reranker (BAAI/bge-reranker-base)...", file=sys.stderr)
reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-base")


class UnifiedSchema(LanceModel):
    vector: Vector(1024) = embedding_model.VectorField()  # pyright: ignore[reportInvalidTypeForm]
    text: str = embedding_model.SourceField()
    title: str = ""
    filename: str = ""
    category: str = ""
    breadcrumb: str = ""


def get_table(create_new=False):
    db = lancedb.connect(str(DB_PATH))
    try:
        if create_new:
            return db.create_table(
                "plesk_knowledge", schema=UnifiedSchema, mode="overwrite"
            )
        return db.open_table("plesk_knowledge")
    except Exception:
        return db.create_table("plesk_knowledge", schema=UnifiedSchema, mode="create")


# --- Helper: Git Auto-Loader ---
def ensure_source_exists(source):
    if source["path"].exists() and any(source["path"].iterdir()):
        return True
    if source["repo_url"]:
        print(f"[LOG] Downloading {source['cat']}...", file=sys.stderr)
        try:
            Repo.clone_from(source["repo_url"], source["path"])
            return True
        except Exception:
            return False
    return False


# --- TOC Parsing Logic ---
def parse_toc_recursive(nodes, parent_path="", file_map=None):
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


def load_toc_map(folder_path):
    toc_path = folder_path / "toc.json"
    if not toc_path.exists():
        return {}
    try:
        data = json.loads(toc_path.read_text(encoding="utf-8"))
        return parse_toc_recursive(data)
    except Exception:
        return {}


# --- Content Parsers ---
def parse_html(file_path, toc_metadata=None):
    try:
        html = file_path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        title = "Untitled"
        breadcrumb = ""
        if toc_metadata:
            title = toc_metadata.get("title", title)
            breadcrumb = toc_metadata.get("breadcrumb", "")
        elif soup.title:
            if soup.title and soup.title.string:
                title = soup.title.string.replace(
                    " — Plesk Obsidian documentation", ""
                ).strip()

        content = soup.find("div", attrs={"itemprop": "articleBody"})
        if not content:
            content = soup.body
        if content:
            for tag in content(["script", "style", "nav", "footer", "iframe"]):
                tag.decompose()

        clean_text = content.get_text(separator="\n", strip=True) if content else ""
        if breadcrumb:
            clean_text = f"Context: {breadcrumb}\n\n{clean_text}"
        return title, breadcrumb, clean_text
    except Exception:
        return None, None, None


def parse_code(file_path):
    try:
        return (
            file_path.name,
            "",
            file_path.read_text(encoding="utf-8", errors="ignore"),
        )
    except Exception:
        return None, None, None


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
    if reset_db:
        table = get_table(create_new=True)
        print("[LOG] Database wiped.", file=sys.stderr)
        existing_files = set()
    else:
        table = get_table(create_new=False)
        existing_files = set()
        if target_category != "all":
            try:
                print("[LOG] Checking existing files...", file=sys.stderr)
                results = (
                    table.search()
                    .where(f"category='{target_category}'")
                    .limit(10000)
                    .to_list()
                )
                for r in results:
                    existing_files.add(r["filename"])
                print(
                    f"[LOG] Found {len(existing_files)} existing chunks.",
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"[WARN] Could not fetch existing files: {e}", file=sys.stderr)

    report = []

    for source in SOURCES:
        if target_category != "all" and source["cat"] != target_category:
            continue

        print(f"[LOG] Processing {source['cat']}...", file=sys.stderr)
        if not ensure_source_exists(source):
            report.append(f"SKIPPED {source['cat']}")
            continue

        toc_map = {}
        if source["type"] == "html":
            toc_map = load_toc_map(source["path"])
            files = list(source["path"].rglob("*.htm")) + list(
                source["path"].rglob("*.html")
            )
            chunk_size = 3000
        elif source["type"] == "php":
            files = list(source["path"].rglob("*.php"))
            chunk_size = 6000
        else:
            files = list(source["path"].rglob("*.js")) + list(
                source["path"].rglob("*.md")
            )
            chunk_size = 5000

        cat_docs = []
        files_processed_count = 0
        BATCH_SIZE_FILES = 10

        for f in files:
            if f.name.startswith("_") or f.name == "toc.json":
                continue

            if f.name in existing_files:
                continue

            meta = toc_map.get(f.name) if toc_map else None
            if source["type"] == "html":
                title, breadcrumb, text = parse_html(f, meta)
            else:
                title, breadcrumb, text = parse_code(f)

            if text and len(text) > 50:
                chunks = [
                    text[i : i + chunk_size]
                    for i in range(0, len(text), chunk_size - 500)
                ]
                # Ensure we have valid string values
                safe_title = title or "Untitled"
                safe_breadcrumb = breadcrumb or ""
                for chunk in chunks:
                    cat_docs.append(
                        {
                            "text": f"[{source['cat'].upper()}] {safe_title}\n---\n{chunk}",
                            "title": safe_title,
                            "filename": f.name,
                            "category": source["cat"],
                            "breadcrumb": safe_breadcrumb,
                        }
                    )

            files_processed_count += 1

            if files_processed_count >= BATCH_SIZE_FILES:
                if cat_docs:
                    print(
                        f"[LOG] Saving batch of {len(cat_docs)} chunks...",
                        file=sys.stderr,
                    )
                    table.add(cat_docs)
                    cat_docs = []
                files_processed_count = 0

        if cat_docs:
            table.add(cat_docs)

        report.append(f"Finished pass for {source['cat']}.")

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
    table = get_table()

    # 1. Busca Rápida
    search_op = table.search(query).limit(25)
    if category:
        search_op = search_op.where(f"category = '{category}'")

    # LOG VISUAL
    print(f"[LOG] Reranking results for: '{query}'...", file=sys.stderr)

    # 2. Reranking (A parte "lenta")
    results = search_op.rerank(reranker=reranker).limit(5).to_list()

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
    mcp.run()
