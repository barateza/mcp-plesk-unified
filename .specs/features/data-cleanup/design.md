# Design: Data Cleanup and Path Updates

## Architecture & Logic Changes

### 1. GitHub Clone Cleanup
- **Target:** `plesk_unified/io_utils.py:ensure_source_exists`
- **Method:** After `Repo.clone_from` completes successfully, recursively use `shutil.rmtree` to remove `.git`, `.github`, and `.pytest_cache` folders within the destination path. This minimizes storage requirements.

### 2. Zip Extraction Cleanup
- **Target:** `scripts/manage_plesk_docs.py:GuideManager.download_and_extract`
- **Method:** Add `self.paths["zip"].unlink(missing_ok=True)` at the end of the method (after the successful extraction and JSON dump process).

### 3. Path Migration
- **Target:** `scripts/manage_plesk_docs.py`
- **Method:** Change `BASE_STORAGE_DIR` to `Path("knowledge_base")` instead of `Path("storage")`. This aligns with the new destination folder for manual documents.

### 4. Indexing Awareness
- **Target:** `plesk_unified/server.py`
- **Method:** Expand the `SOURCES` list to include:
  ```python
  {
      "path": KB_DIR / "extensions-guide" / "md",
      "cat": "guide",
      "type": "md",
      "repo_url": None,
  },
  {
      "path": KB_DIR / "cli-linux" / "md",
      "cat": "cli",
      "type": "md",
      "repo_url": None,
  },
  {
      "path": KB_DIR / "api-rpc" / "md",
      "cat": "api",
      "type": "md",
      "repo_url": None,
  }
  ```
  Note: `manage_plesk_docs.py` converts the `.htm` files to Markdown (`.md`) inside the `md` subdirectory. So the MCP server should parse those markdown documents instead of raw HTML if that was the intent. Wait, if `.md` was not indexed before, the MCP server might need adjusting, but `collect_files_for_source` for `type != "html" and type != "php"` checks for `*.js` and `*.md` files. Thus, type `"md"` is natively supported in `server.py` and `io_utils.py`.

## Edge Cases
- **Missing Paths:** The `rmtree` call for cleanup must ignore errors or gracefully check if the directory exists before deletion.
- **Extraction Failures:** Ensure broken zip files don't result in partially created folders without a deletion fallback. Ensure `shutil.rmtree` handles `.git` read-only files properly on Windows.
