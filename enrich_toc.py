import json
import os
import time
from pathlib import Path

import requests

# --- Configuration ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
KB_ROOT = Path(os.environ.get("KB_ROOT", Path(__file__).parent / "knowledge_base"))
MODELS = [
    "arcee-ai/trinity-large-preview:free",
    "stepfun/step-3-5-flash:free",
    "liquid/lfm-2.5-1.2b-thinking:free",
]


def get_ai_description(file_path, file_name):
    """Attempts to get a description using a tiered fallback system."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(2500)
    except Exception:
        return "File unreadable."

    for model in MODELS:
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                data=json.dumps(
                    {
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": f"Summarize the technical purpose of the Plesk file '{file_name}' in exactly one concise sentence.\n\n{content}",
                            }
                        ],
                        "max_tokens": 100,
                    }
                ),
                timeout=15,
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            continue
    return "Description unavailable."


def enrich_all_tocs():
    # Process both SDK and Stubs
    categories = {"sdk": "Extension SDK", "stubs": "PHP API Stubs"}

    for folder, label in categories.items():
        toc_path = KB_ROOT / folder / "virtual_toc.json"
        if not toc_path.exists():
            print(f"Skipping {label}: TOC not found.")
            continue

        with open(toc_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"\n--- Enriching {label} ({len(data['files'])} files) ---")

        for i, item in enumerate(data["files"]):
            # Skip if description already exists
            if (
                item.get("description")
                and item["description"] != "Description unavailable."
            ):
                continue

            file_path = KB_ROOT / folder / item["path"]
            print(f"[{i + 1}/{len(data['files'])}] Describing: {item['name']}")

            item["description"] = get_ai_description(file_path, item["name"])

            # Save every 5 files to prevent data loss if script is interrupted
            if i % 5 == 0:
                with open(toc_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

            time.sleep(0.5)  # Rate limit safety

        # Final save for the category
        with open(toc_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"✅ {label} updated.")


if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY environment variable is not set.")
    enrich_all_tocs()
