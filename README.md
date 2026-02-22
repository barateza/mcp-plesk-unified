# Plesk Unified MCP Server

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue?style=flat-square)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Model Context Protocol](https://img.shields.io/badge/MCP-Compatible-green?style=flat-square)](https://modelcontextprotocol.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

A powerful Model Context Protocol (MCP) server that provides unified access to
comprehensive Plesk documentation through intelligent semantic search and
retrieval-augmented generation (RAG).

## Overview

Plesk Unified seamlessly integrates Plesk's scattered documentation resources
into a single, intelligent knowledge base. Using advanced embeddings and
reranking, it enables natural language queries to surface the most relevant
documentation, making it perfect for developers, system administrators, and
AI-powered tools like Claude.

## ✨ Features

- **Unified knowledge base**: Aggregates documentation from multiple Plesk
  sources:
  - 📚 API documentation
  - 💻 CLI reference
  - 📖 Admin guide
  - 🔧 PHP API stubs
  - 🚀 JavaScript SDK

- **🧠 Semantic search**: Uses BAAI/bge-m3 multilingual embeddings to
  understand queries semantically beyond simple keyword matching.

- **🎯 Intelligent reranking**: Cross-encoder reranking ensures the most
  relevant results surface first to improve accuracy.

- **⚡ Vector database**: Leverages LanceDB for efficient, scalable vector
  storage and retrieval with Apache Arrow.

- **🔄 Auto-Git integration**: Automatically downloads and indexes PHP stubs
  and JS SDK from GitHub repositories.

- **🔌 MCP compatible**: Works seamlessly with Claude, LLaMA, and other
  MCP-compatible AI tools for context-aware assistance.

## 📋 Requirements

- Python 3.12 or higher
- ~2GB storage for embeddings and vector database
- ~1GB for ML models (auto-downloaded on first run)
- Internet connection for initial setup

## 🚀 Quick start

### Installation

```bash
# Clone the repository
git clone https://github.com/barateza/mcp-plesk-unified.git
cd mcp-plesk-unified

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

### GPU acceleration (optional)

The server automatically detects and uses GPU acceleration when available:

| Platform | GPU type | Acceleration |
| -------- | -------- | ------------ |
| macOS M1/M2/M3 | Apple Silicon MPS | ✅ Automatic |
| Windows + NVIDIA | CUDA | ✅ Automatic |
| Linux + NVIDIA | CUDA | ✅ Automatic |
| Other | CPU | ✅ Fallback |

#### Install with GPU support

The default `sentence-transformers` includes CPU-only PyTorch. For GPU
acceleration, install PyTorch with CUDA from the official source:

**macOS (Apple Silicon):**

```bash
# Install PyTorch with Apple Silicon support
pip install torch
# Or use uv
uv pip install torch
```

**Windows/Linux (NVIDIA CUDA):**

```bash
# Install PyTorch with CUDA 12.4 (latest stable)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Or use uv
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

> **Important:** If you already have a CUDA-enabled PyTorch installed, it will
> not be uninstalled. The server automatically detects and uses your GPU.

#### Force a specific device

Override automatic detection by setting the `FORCE_DEVICE` environment variable:

```bash
# Force CPU even if GPU is available
export FORCE_DEVICE=cpu  # Linux/macOS
set FORCE_DEVICE=cpu     # Windows

# Or run
FORCE_DEVICE=cpu python server.py
```

### ⚠️ First-run warm-up (required)

> **Do this before registering the server with any MCP client.**

MCP clients (Claude Desktop, Cursor, etc.) enforce strict request timeouts
(~60 seconds). On first use, the server must download two AI models totalling
~1.8 GB:

| Model | Size | Purpose |
| ----- | ---- | ------- |
| `BAAI/bge-m3` | ~1.5 GB | Semantic embeddings |
| `BAAI/bge-reranker-base` | ~300 MB | Cross-encoder reranking |

If you register the server before the models are cached, the first tool call
triggers the download mid-request and **times out silently**. Run the warm-up
script first:

```bash
python main.py
```

Sample output:

```text
  mcp-plesk-unified — Model Warm-Up

Downloading and caching AI models...
This may take several minutes on the first run (~1.8 GB total).

✅  Warm-up complete in 142.3s.

Models are now cached. You can safely register the MCP server
in your client configuration without risk of timeout errors.
```

Once the warm-up completes, all subsequent server starts load from the local
HuggingFace cache and are nearly instantaneous.

### Initialize the knowledge base

```bash
python server.py
```

The server will:

1. ✅ Download Plesk documentation sources
2. ✅ Parse HTML, PHP, and JavaScript files
3. ✅ Generate semantic embeddings (using cached models)
4. ✅ Index everything into LanceDB vector database
5. ✅ Start the MCP server

### Use with Claude/MCP

Configure your Claude client to use this MCP server:

```json
{
  "mcpServers": {
    "mcp-plesk-unified": {
      "command": "python",
      "args": ["/path/to/mcp-plesk-unified/server.py"]
    }
  }
}
```

Query: "How do I add a button to the Plesk admin panel?" and receive
contextual documentation excerpts.

## 🏗️ Architecture

| Component | Technology | Purpose |
| --------- | ---------- | ------- |
| **Embeddings** | BAAI/bge-m3 | Multilingual, bidirectional semantic embeddings |
| **Reranker** | BAAI/bge-reranker-base | Cross-encoder for ranking retrieval results |
| **Vector DB** | LanceDB | Apache Arrow-based vector storage |
| **Server** | FastMCP | Model Context Protocol implementation |
| **Parser** | BeautifulSoup4 | HTML parsing for documentation |

## 📦 Dependencies

- **fastmcp** (≥2.14.5): MCP server framework
- **lancedb** (≥0.29.1): Vector database
- **sentence-transformers** (≥5.2.2): Embedding and reranking models
- **beautifulsoup4** (≥4.14.3): HTML parsing
- **gitpython** (≥3.1.46): Git repository management

## 💡 Usage examples

### Python API

```python
from server import mcp

# Search the knowledge base
results = mcp.search_plesk_unified(query="API authentication methods")

for result in results:
    print(f"Title: {result['title']}")
    print(f"Category: {result['category']}")
    print(f"Content: {result['text'][:200]}...")
```

### Command line

```bash
# Initialize and start server
python server.py

# In another terminal, query the server
# (Instructions for client-side querying)
```

## 🔧 Configuration

### TOC enrichment (`enrich_toc.py`)

`enrich_toc.py` uses the [OpenRouter](https://openrouter.ai/) API to
auto-generate one-sentence descriptions for every file in the knowledge-base
Table of Contents. Before running it, export your API key:

```bash
# macOS / Linux
export OPENROUTER_API_KEY="sk-or-v1-..."

# Windows (PowerShell)
$env:OPENROUTER_API_KEY = "sk-or-v1-..."

# Windows (Command Prompt)
set OPENROUTER_API_KEY=sk-or-v1-...
```

Optionally, override the knowledge-base root (defaults to `./knowledge_base`):

```bash
export KB_ROOT="/path/to/your/knowledge_base"
```

Run:

```bash
python enrich_toc.py
```

> **Tip:** Add `OPENROUTER_API_KEY` to a `.env` file and load it with `dotenv`
> or your shell's `source` command — never hard-code it in source files.

### Server (`server.py`)

Edit `server.py` to customize:

```python
# Knowledge base sources
SOURCES = [
    {"path": KB_DIR / "guide", "cat": "guide", ...},
    {"path": KB_DIR / "api", "cat": "api", ...},
    # ... more sources
]

# Embedding model
embedding_model = get_registry().get("huggingface").create(
    name="BAAI/bge-m3"
)

# Reranker model
reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-base")
```

## 🗂️ Project structure

```text
plesk-unified/
├── server.py              # Main MCP server implementation
├── main.py               # Warm-up script — run once before registering with MCP clients
├── pyproject.toml        # Project metadata and dependencies
├── README.md             # This file
├── LICENSE               # MIT License
├── knowledge_base/       # Documentation sources
│   ├── api/             # API documentation
│   ├── cli/             # CLI reference
│   ├── guide/           # Admin guide
│   ├── php-stubs/       # PHP API stubs
│   └── sdk/             # JavaScript SDK
└── storage/             # Generated data
    └── lancedb/         # Vector database
```

## Naming conventions

This repository follows a naming policy for user-facing files and documentation:

- Use kebab-case for documentation filenames and public assets (e.g.,
  `getting-started.md`, `extension-sdk-features.md`).
- Keep code/module names in their language-native form (Python: snake_case,
  JS: camelCase or as-is). Do not rename Python modules or package directories
  to kebab-case.
- Excluded paths (do not rename): `storage/lancedb`, `knowledge_base/stubs`,
  and `plesk_unified.egg-info`.

When renaming files, update any internal links (TOC, README, and other
markdown files) to point to the new kebab-case names. Run formatters and
linters after applying changes:

```bash
# Fix Python formatting and linting
ruff check . --fix
black .
```

## 🚧 Development

### Set up development environment

```bash
# Install with dev dependencies (if added)
pip install -e ".[dev]"

# Run tests (when added)
pytest

# Format code
black .

# Lint
pylint **/*.py
```

### Regenerate the vector database

```bash
# Delete the old database
rm -rf storage/lancedb

# Reinitialize
python server.py
```

## 📝 License

This project is licensed under the MIT License - see the
[LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Consider the following workflow:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Guidelines

- Follow PEP 8 style guidelines
- Update documentation for new features
- Test changes thoroughly
- Regenerate the vector database after content changes

## 🐛 Troubleshooting

### Models not downloading

Ensure you have internet connectivity and sufficient disk space (~2GB for
models).

### LanceDB connection issues

Delete the storage directory and reinitialize:

```bash
rm -rf storage/
python server.py
```

### Out of memory errors

Reduce batch size in `server.py` or run on a machine with more RAM.

## 📖 Resources

- [Model Context Protocol documentation](https://modelcontextprotocol.io/)
- [Plesk official documentation](https://docs.plesk.com/)
- [LanceDB documentation](https://lancedb.com/)
- [Sentence Transformers](https://www.sbert.net/)

## 🙏 Acknowledgments

- Built with [FastMCP](https://github.com/jlouis/fastmcp)
- Embeddings powered by [BAAI](https://www.baai.ac.cn/)
- Vector database by [LanceDB](https://lancedb.com/)
- Documentation from [Plesk](https://www.plesk.com/)

## 📞 Support

For issues or questions:

- 🐛 [Report a bug](https://github.com/barateza/mcp-plesk-unified/issues)
- 💡 [Request a feature](https://github.com/barateza/mcp-plesk-unified/issues)
- 📧 Open an issue on GitHub

---

## **Built with ❤️ for the Plesk community**

