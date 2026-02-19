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

Platform Detection:
This script automatically detects your platform and configures optimal settings:
  • Windows with NVIDIA GPU -> CUDA acceleration
  • macOS with Apple Silicon (M1/M2/M3) -> MPS acceleration
  • Linux with NVIDIA GPU -> CUDA acceleration
  • Otherwise -> CPU fallback
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

    # Try to get the key from the current environment. Do NOT include the
    # real secret in the returned JSON unless `mask_sensitive` is False.
    api_key = os.environ.get("OPENROUTER_API_KEY")

    # Use an explicit redaction token by default. Only include the real value
    # when the caller explicitly requests unmasked output.
    env_value = api_key if (not mask_sensitive and api_key) else "<REDACTED>"

    config = {
        "mcpServers": {
            "plesk-unified": {
                "command": python_path,
                "args": [str(server_script)],
                "env": {
                    "OPENROUTER_API_KEY": env_value
                }
            }
        }
    }

    return json.dumps(config, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Warm-up models and generate MCP config")
    parser.add_argument("--show-secret", action="store_true",
                        help="Print full config including any API key (unsafe)")
    parser.add_argument("--dangerous-yes-i-know", action="store_true",
                        help="Explicit confirmation required to print secrets."
                             " Must be used together with --show-secret.")
    args = parser.parse_args()

    if args.show_secret and not args.dangerous_yes_i_know:
        print("\nERROR: --show-secret is dangerous. To proceed, pass:\n"
              "  python main.py --show-secret --dangerous-yes-i-know\n",
              file=sys.stderr)
        sys.exit(2)

    print("=" * 60)
    print("  mcp-plesk-unified — Model Warm-Up & Config Generator")
    print("=" * 60)
    print()

    # Import platform utilities to show device info
    try:
        from platform_utils import (
            get_optimal_device,
            print_platform_info,
            get_platform_info,
        )

        info = get_platform_info()
        print(f"Detected Platform: {info.get('system')} {info.get('machine')}")

        device = get_optimal_device()
        print(f"Using Device: {device.upper()}")
        print()

        # Print detailed platform info to stderr
        print_platform_info()
    except ImportError:
        print("Platform detection not available, using default settings.")
        print()

    print("Downloading and caching AI models...")
    print("This may take several minutes on the first run (~1.8 GB total).")
    print()

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
        print("Models are now cached. You can safely register the MCP server")
        print("in your client configuration without risk of timeout errors.")

        # Show cache location
        try:
            from platform_utils import get_model_cache_dir

            cache_dir = get_model_cache_dir()
            print(f"\nModel cache location: {cache_dir}")
        except ImportError:
            pass
    except Exception as exc:
        print(f"\n❌ Warm-up failed: {exc}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()