# Tasks: Data Cleanup and Path Updates

- [ ] Update `scripts/manage_plesk_docs.py`: Update `BASE_STORAGE_DIR` to point to `knowledge_base` and append zip deletion logic.
- [ ] Update `plesk_unified/io_utils.py`: Enhance `ensure_source_exists` to wipe `.git` directories post-clone. Note: `shutil.rmtree` requires `onerror` handler on Windows to delete `.git` properly.
- [ ] Update `plesk_unified/server.py`: Register `api-rpc`, `cli-linux`, and `extensions-guide` in `SOURCES` for indexing.
- [ ] Run automated tests via `uv run pytest`.
- [ ] Format and lint with pre-commit via `uv run pre-commit run --all-files`.
- [ ] Update knowledge base folder names if not aligned.
