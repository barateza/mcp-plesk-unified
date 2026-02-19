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
import json
import os
from pathlib import Path
import argparse

def generate_mcp_config(mask_sensitive: bool = True):
    """Generates the JSON config for Claude Desktop/Cursor.

    If `mask_sensitive` is True (default) the API key value is replaced with
    a redaction token to avoid printing secrets to stdout/logs.
    """
    # Get the absolute path to the python interpreter in the venv
    python_path = sys.executable
    
    # Get the absolute path to server.py
    # Assuming server.py is in the same directory as main.py
    script_dir = Path(__file__).parent.resolve()
    server_script = script_dir / "server.py"

    # Try to get the key from the current environment, else use placeholder
    api_key = os.environ.get("OPENROUTER_API_KEY", "YOUR_KEY_HERE")

    config = {
        "mcpServers": {
            "plesk-unified": {
                "command": python_path,
                "args": [
                    str(server_script)
                ],
                "env": {
                    "OPENROUTER_API_KEY": api_key
                }
            }
        }
    }
    cfg_json = json.dumps(config, indent=4)
    if mask_sensitive and api_key and api_key != "YOUR_KEY_HERE":
        cfg_json = cfg_json.replace(api_key, "<REDACTED>")
    return cfg_json

def main():
    parser = argparse.ArgumentParser(description="Warm-up models and generate MCP config")
    parser.add_argument("--show-secret", action="store_true",
                        help="Print full config including any API key (unsafe)")
    args = parser.parse_args()

    print("=" * 60)
    print("  mcp-plesk-unified — Model Warm-Up & Config Generator")
    print("=" * 60)
    print()
    print("Step 1: Checking AI Models...")

    start = time.time()
    try:
        # Import get_resources to trigger the download/cache
        # This assumes server.py is in the same folder and valid
        from server import get_resources
        get_resources()

        elapsed = time.time() - start
        print(f"✅  Models ready ({elapsed:.1f}s).")
        print("-" * 60)
        print()
        print("Step 2: Configuration")
        print("Copy the JSON below into your MCP Client configuration file:")
        print("  • Claude Desktop: %APPDATA%\\Claude\\claude_desktop_config.json")
        print("  • Cursor: Features > MCP > Add New MCP Server")
        print()
        print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
        print(generate_mcp_config(mask_sensitive=not args.show_secret))
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print()
        
    except ImportError:
        print("\n❌ Error: Could not import 'server.py'. Make sure you run this script")
        print("   from the same directory as your server code.")
        sys.exit(1)
    except Exception as exc:
        print(f"\n❌ Warm-up failed: {exc}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()