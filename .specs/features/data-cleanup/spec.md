# Specification: Data Cleanup and Path Updates

## Background
The user noticed that the codebase was leaving unnecessary files behind after downloading from its sources. `sdk` and `stubs` retain unnecessary files like `.git`. The zip files containing `api-rpc`, `cli-linux`, and `extensions-guide` are not deleted after extraction. Additionally, these folders were relocated from `storage` to `knowledge_base` manually by the user, and the codebase logic does not currently handle this change.

## Objectives
1. Automatically clean up unused project artifacts (e.g. `.git`) from GitHub repositories cloned to `knowledge_base`.
2. Automatically delete the `.zip` artifacts downloaded by `scripts/manage_plesk_docs.py` after extraction.
3. Fix path hardcoding in `scripts/manage_plesk_docs.py` by transitioning `storage` usage to `knowledge_base`.
4. Ensure `api-rpc`, `cli-linux`, and `extensions-guide` are discovered and indexed by `server.py` in the LanceDB generation loop.

## In Scope
- Applying cleanup to `plesk_unified.io_utils.ensure_source_exists`.
- Fixing paths in `scripts/manage_plesk_docs.py`.
- Adding sources to `plesk_unified.server.SOURCES`.

## Out of Scope
- Complete architectural rewrite of the indexing pipeline.
- Modifying how AI summaries are generated.
