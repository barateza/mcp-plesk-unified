# TLC: Refactor large functions to remove `# noqa: C901`

**Project:** mcp-plesk-unified
**Date:** 2026-02-21
**Author:** (automated plan)

## 1. Specify

- Goal: Lower cognitive complexity of large functions so `C901` suppressions can be removed while preserving behavior and tests.
- Target functions:
  - `refresh_knowledge` in `server.py`
  - `GuideManager.enrich_toc()` in `manage-plesk-docs.py`
  - `GuideManager.convert_to_markdown()` in `manage-plesk-docs.py`
  - Also: `enrich_all_tocs` in `enrich_toc.py` (AI callers)

### Success criteria

- All extracted helpers have unit tests covering core logic.
- Top-level functions become short coordinators (≤ 20–30 lines), no `# noqa: C901` needed.
- Integration smoke runs produce identical outputs for sample fixtures.
- Linting (`ruff`) shows no complexity suppressions for these functions.

### Constraints & assumptions

- Preserve incremental-save semantics and batching sizes.
- Preserve existing external behaviors: Git clone semantics, LLM fallback behavior, LanceDB writes.
- Tests must avoid live network/DB calls by using injected stubs/mocks.

## 2. Design

High-level approach:

- Extract pure/predictable logic into small helper modules under `plesk_unified/`:
  - `html_utils.py`: parsing and cleaning HTML.
  - `chunking.py`: chunking and doc-record building and `persist_batch` wrapper.
  - `ai_client.py`: thin wrapper for LLM calls with retry policy (injectable).
  - `io_utils.py` (optional): repo/TOC/file collection helpers.
- Convert big functions into coordinators that compose helpers and manage side-effects.
- Use dependency injection for external systems (DB factory, git_checker, llm_client) to enable tests.
- Add tests-first for helpers; then swap implementations in top-level functions and run integration smoke tests.

Example coordinator pseudocode (for `refresh_knowledge`):

- def refresh_knowledge(..., table_factory=get_table, git_checker=ensure_repo_cloned, llm_client=None):
  - table = table_factory()
  - for source in SOURCES:
    - if not git_checker(source): continue
    - toc_map = load_toc_map(source)
    - files = collect_files_for_source(source)
    - for file in files:
      - title, breadcrumb, text = parse_html_file(file, toc_map.get(filename))
      - chunks = chunk_by_chars(text)
      - docs = build_doc_records(filename, chunks, meta)
      - persist_batch(table, docs)
  - return report

## 3. Tasks (atomic, with verification)

1. Scaffold helper package `plesk_unified/` (completed)
   - Files: `html_utils.py`, `chunking.py`, `ai_client.py` (stub), `io_utils.py` (optional)
   - Verification: importable; unit tests run for `html_utils` and `chunking`.

2. Add unit tests for helpers (completed for initial helpers)
   - Tests: `tests/test_html_utils.py`, `tests/test_chunking.py`.
   - Verification: `pytest` passes locally for these tests.

3. Implement `ai_client.py` abstraction
   - API: `generate_description(text, model_list=None, retry_policy=None)`
   - Should be easy to monkeypatch in tests.
   - Verification: unit test uses stub to validate call and response handling.

4. Implement `io_utils.py`
   - Helpers: `ensure_repo_cloned(source)`, `collect_files_for_source(source)`, `load_toc_map(path)`.
   - Verification: unit tests use temporary directories and sample `toc.json`.

5. Refactor `GuideManager.enrich_toc()` → use `process_toc_nodes`, `node_needs_description`, `generate_node_description`
   - Steps:
     - Add extracted helpers.
     - Replace in-place closure with `process_toc_nodes(toc, action)`.
     - Keep `save_toc_incremental(toc_path, toc, n)` semantics.
   - Verification:
     - Unit tests for `node_needs_description` on fixtures.
     - Integration run against `tests/fixtures/sample_guide` produces same `toc.json` changes.

6. Refactor `GuideManager.convert_to_markdown()` → `convert_html_to_markdown`, `write_markdown`
   - Verification: run `manage-plesk-docs.py` in dry-run mode or on sample fixture and assert `.md` outputs identical to baseline.

7. Refactor `refresh_knowledge()` in `server.py`
   - Replace large inline loop with coordinator and calls to helpers.
   - Inject `table_factory` and `git_checker` for tests.
   - Preserve batching sizes and incremental filename checks.
   - Verification:
     - Unit-level test where `fake_table` records `add()` calls and `git_stub` simulates repo presence.
     - Integration dry-run against a small source with `--dry-run` to validate no writes or to a temp LanceDB path.

8. Replace live LLM calls in `enrich_toc.py` with `ai_client` (or inject client via param) and add tests with `llm_stub`.
   - Verification: tests validate fallback model behavior and rate-limit handling.

9. Run full test suite, linter, and formatters
   - Commands:
     - `pytest -q`
     - `ruff check . --fix`
     - `black .`
   - Verification: tests pass; ruff reports no `C901` suppressions on refactored functions.

10. Remove `# noqa: C901` lines and finalize PR
    - Verification: run `ruff` to confirm no complexity warnings.

## 4. Implement + Validate (detailed steps per task)

- For each helper extracted:
  - Write tests first (cover edge cases: empty HTML, missing main tag, huge text chunking).
  - Implement helper and run tests.
  - Add typing and short docstring.

- For top-level function swap:
  - Keep original function body in version control; perform replacement in a single commit with tests passing.
  - Use feature branch and small commits per file.

- Rollback plan:
  - If regressions detected, revert the refactor commit and open a follow-up PR with smaller changes.

## 5. Estimates & ownership

- `refresh_knowledge`: 90–120 minutes (extraction, tests, patching)
- `enrich_toc`: 45–90 minutes
- `convert_to_markdown`: 30–60 minutes
- `ai_client` + `io_utils` + tests: 60–90 minutes
- CI updates & lint clean: 30–60 minutes

Assigned owner: you (repo maintainer) / me for scaffolding and initial PRs if you want.

## 6. Risks & mitigations

- Risk: Tight coupling to module-level objects (global `get_table()` or `embedding_model`).
  - Mitigation: Add wrapper factories and allow injection; document breaking-change points.
- Risk: LLM/network flakiness during tests.
  - Mitigation: Use `llm_stub` and mock `requests.post`.
- Risk: LanceDB writes in tests corrupt local DB.
  - Mitigation: Use temp dirs or fake table.

## 7. Test plan & CI

- Add `pytest` job that runs unit tests for helpers and an integration smoke on fixtures.
- Add `ruff`/`pre-commit` gating to ensure no `noqa` re-introduced.

## 8. Acceptance criteria

- All targeted functions no longer require `# noqa: C901`.
- Tests cover helper logic with >80% branch coverage for those helpers.
- Integration smoke verifies no behavioral regressions on sample data.
