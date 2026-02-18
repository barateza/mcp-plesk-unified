"""
Warm-up script for mcp-plesk-unified.

Run this ONCE before registering the server with your MCP client (Claude Desktop,
Cursor, etc.).  It downloads and caches the two AI models used by the server:

  • BAAI/bge-m3          — embedding model  (~1.5 GB)
  • BAAI/bge-reranker-base — reranker model  (~300 MB)

MCP clients enforce strict timeouts (~60 s).  Without this warm-up the first
tool call will trigger the download mid-request and almost certainly time out.
After running this script the models are cached locally by HuggingFace and
subsequent server starts are instant.
"""

import sys
import time


def main():
    print("=" * 60)
    print("  mcp-plesk-unified — Model Warm-Up")
    print("=" * 60)
    print()
    print("Downloading and caching AI models...")
    print("This may take several minutes on the first run (~1.8 GB total).")
    print()

    start = time.time()
    try:
        # Import get_resources from the server module to trigger the exact same
        # loading path that tools will use at runtime.
        from server import get_resources

        get_resources()

        elapsed = time.time() - start
        print()
        print(f"✅  Warm-up complete in {elapsed:.1f}s.")
        print()
        print("Models are now cached. You can safely register the MCP server")
        print("in your client configuration without risk of timeout errors.")
    except Exception as exc:
        elapsed = time.time() - start
        print(f"\n❌  Warm-up failed after {elapsed:.1f}s: {exc}", file=sys.stderr)
        print(
            "\nEnsure you have an active internet connection and ~2 GB of free disk space.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
