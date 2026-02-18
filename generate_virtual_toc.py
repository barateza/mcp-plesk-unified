import json
import os
from pathlib import Path


def generate_plesk_toc(repo_path, category_name):
    base = Path(repo_path)
    if not base.exists():
        print(f"Error: Path {repo_path} does not exist.")
        return

    # Define what we want to ignore to keep the TOC clean
    ignore_list = {".git", ".github", "tests", "vendor", "node_modules", "bin"}

    virtual_toc = {"category": category_name, "files": []}

    # Walk through the directory
    for root, dirs, files in os.walk(base):
        # Remove ignored directories in-place
        dirs[:] = [d for d in dirs if d not in ignore_list]

        for file in files:
            # Focus on the meat: PHP for stubs, JS/TS/MD for SDK
            if file.endswith((".php", ".js", ".ts", ".md")):
                full_path = Path(root) / file
                rel_path = full_path.relative_to(base)

                virtual_toc["files"].append(
                    {
                        "name": file,
                        "path": str(rel_path),
                        "depth": len(rel_path.parts) - 1,
                        "folder": (
                            str(rel_path.parent)
                            if str(rel_path.parent) != "."
                            else "Root"
                        ),
                    }
                )

    output_path = base / "virtual_toc.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(virtual_toc, f, indent=2)

    print(f"✅ Created Virtual TOC for {category_name} at: {output_path}")


if __name__ == "__main__":
    # Update these paths to match your local machine
    KB_ROOT = Path("/Users/gilsonsiqueira/mcpServers/mcp-plesk-unified/knowledge_base")

    generate_plesk_toc(KB_ROOT / "stubs", "PHP API Stubs")
    generate_plesk_toc(KB_ROOT / "sdk", "Extension SDK")
