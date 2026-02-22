import json
import logging
import os
import shutil
import stat
from pathlib import Path
from typing import Any, Dict, List, Optional
from git import Repo

logger = logging.getLogger(__name__)


def ensure_source_exists(source: Dict[str, Any]) -> bool:
    """Ensure the source repository exists and is not empty."""
    source_path = source.get("path")

    if (
        source_path
        and isinstance(source_path, Path)
        and source_path.exists()
        and any(source_path.iterdir())
    ):
        logger.debug("Source %s already exists at %s", source.get("cat"), source_path)
        return True

    repo_url = source.get("repo_url")
    if repo_url and source_path:
        logger.info("Downloading %s from %s...", source.get("cat"), repo_url)
        try:
            Repo.clone_from(repo_url, source_path)
            logger.info("Clone succeeded for %s", source.get("cat"))

            def on_rm_error(func, path, exc_info):
                # path contains the path of the file that couldn't be removed
                # let's just assume that it's read-only and unlink it.
                os.chmod(path, stat.S_IWRITE)
                func(path)

            # Cleanup unnecessary artifacts
            for folder in [".git", ".github", "tests"]:
                target = source_path / folder
                if target.exists() and target.is_dir():
                    shutil.rmtree(target, onerror=on_rm_error)

            return True
        except Exception:
            logger.error("Clone failed for %s", source.get("cat"), exc_info=True)
            return False
    return False


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
        # handle case where toc structure might have a top level "files" array
        # vs direct nested nodes
        # The TOC JSON may be either a list of nodes or a dict with a
        # top-level "files" list. Normalize to a `List[Dict[str, Any]]`
        # before calling `parse_toc_recursive` so the type matches what
        # the parser expects (and to satisfy type checkers like Pylance).
        if (
            isinstance(data, dict)
            and "files" in data
            and isinstance(data["files"], list)
        ):
            nodes = data["files"]
        elif isinstance(data, list):
            nodes = data
        else:
            return {}

        return parse_toc_recursive(nodes)
    except Exception:
        logger.warning("Failed to parse TOC at %s", toc_path, exc_info=True)
        return {}


def collect_files_for_source(source: Dict[str, Any]) -> List[Path]:
    """Collects files for a source depending on its type."""
    source_path = source.get("path")
    if not source_path or not isinstance(source_path, Path):
        return []

    stype = source.get("type")

    if stype == "html":
        return list(source_path.rglob("*.htm")) + list(source_path.rglob("*.html"))
    elif stype == "php":
        return list(source_path.rglob("*.php"))
    else:
        return list(source_path.rglob("*.js")) + list(source_path.rglob("*.md"))
