# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-21

### Changed

- Evaluated and overhauled core documentation (`README.md`, `CONTRIBUTING.md`, `SECURITY.md`) to comply with industry standard `docs-writer` guidelines.
- Standardized documentation to use active voice, imperative mood instructions, and consistent 80-character line wrapping.

### Removed

- Deleted obsolete `TLC-Refactor-C901.md` log file.

## [0.1.0] - 2025-02-08

### Added

- Initial release of Plesk Unified MCP Server
- Unified knowledge base aggregating multiple Plesk documentation sources:
  - API Documentation
  - CLI Reference
  - Admin Guide
  - PHP API Stubs
  - JavaScript SDK
- Semantic search using BAAI/bge-m3 embeddings
- Intelligent reranking with BAAI/bge-reranker-base cross-encoder
- LanceDB vector database for efficient storage and retrieval
- Auto-Git integration for automatic PHP stubs and JS SDK updates
- FastMCP server implementation for Model Context Protocol compatibility
- HTML/PHP/JavaScript documentation parsing
- Comprehensive README and documentation
- MIT License
- Contributing guidelines

### Features

- 🧠 Multilingual semantic search beyond keyword matching
- 🎯 Cross-encoder reranking for improved result relevance
- ⚡ Efficient vector database with Apache Arrow backend
- 🔄 Automatic repository cloning and updates
- 🔌 Full MCP protocol support for integration with Claude and other AI tools

---

## Unreleased

### Planned

- [ ] Batch API for multiple queries
- [ ] Caching layer for frequently accessed documents
- [ ] Web UI for documentation browsing
- [ ] REST API endpoint option
- [ ] Support for additional Plesk locales
- [ ] Performance optimization for large-scale deployments
- [ ] Custom embedding model support
- [ ] Integration tests
- [ ] Docker support

---

[0.2.0]: https://github.com/barateza/mcp-plesk-unified/releases/tag/v0.2.0
[0.1.0]: https://github.com/barateza/mcp-plesk-unified/releases/tag/v0.1.0
