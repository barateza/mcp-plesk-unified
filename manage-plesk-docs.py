import json
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# --- CONFIGURATION ---

# 1. AI Settings
USE_LOCAL_LLM = True

# LM Studio / Ollama Configuration
LOCAL_API_URL = "http://localhost:1234/v1"
LOCAL_MODEL_ID = "llama-3.1-8b-instruct"

# Fallback to OpenRouter
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = "liquid/lfm-2.5-1.2b-thinking:free"

# 2. Storage Paths (Staging Area)
# Files will be processed here. You can move 'md' folders to 'knowledge_base' later.
BASE_STORAGE_DIR = Path("storage")

# 3. Documentation Sources
GUIDES = {
    "extensions-guide": "https://docs.plesk.com/en-US/obsidian/zip/extensions-guide.zip",
    "cli-linux": "https://docs.plesk.com/en-US/obsidian/zip/cli-linux.zip",
    "api-rpc": "https://docs.plesk.com/en-US/obsidian/zip/api-rpc.zip",
    # Add more guides here as needed
}

# --- AI HELPERS ---


def clean_llm_response(text):
    """Trim conversational filler from LLM output and return a concise summary."""
    text = text.strip().strip('"').strip("'")

    # Common conversational prefixes to strip
    garbage_prefixes = [
        "Here is a concise sentence summarizing",
        "Here is a concise sentence",
        "Here is a summary",
        "The technical purpose of the file is to",
        "The technical purpose of the file",
        "The technical purpose of this file",
        "In this section, the documentation",
        "This section",
        "concise sentence:",
        "Summary:",
        "Description:",
    ]

    # 1. Strip known prefixes (case-insensitive check)
    for prefix in garbage_prefixes:
        if text.lower().startswith(prefix.lower()):
            # If there's a colon, split on it (e.g. "Summary: Explains...")
            if ":" in text:
                parts = text.split(":", 1)
                if len(parts) > 1:
                    text = parts[1]
            else:
                # Otherwise just slice off the prefix
                text = text[len(prefix) :]

    # 2. Secondary cleanup
    text = text.strip()

    # 3. Handle "titled 'Name'" artifacts
    # e.g. "titled 'Meta XML': Defines the structure..."
    if text.lower().startswith("titled"):
        if ":" in text:
            text = text.split(":", 1)[1]

    return text.strip()


def generate_description(filename, section_title, content_snippet):
    """Generates a specific description for a section."""

    # Strict prompt to force directness
    prompt = (
        "You are a technical indexer.\n"
        f"Context: The file '{filename}' contains documentation for Plesk Extensions.\n"
        "Task: Write exactly one concise sentence summarizing the specific section "
        f"titled '{section_title}'.\n"
        "Rules:\n"
        "1. Start directly with a verb (e.g., 'Explains', 'Defines', 'Configures').\n"
        "2. Do NOT say 'Here is a summary'.\n"
        "3. Do NOT mention the filename.\n"
        "4. Focus ONLY on the section.\n\n"
        "Content Snippet:\n" + content_snippet[:3500]
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise technical documenter. Output only the summary."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        if USE_LOCAL_LLM:
            resp = requests.post(
                f"{LOCAL_API_URL}/chat/completions",
                json={
                    "model": LOCAL_MODEL_ID,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 80,
                    "stream": False,
                },
                timeout=60,
            )
            if resp.status_code != 200:
                print(f"[!] Local LLM Error: {resp.text}")
                return ""
            raw_text = resp.json()["choices"][0]["message"]["content"]
            return clean_llm_response(raw_text)
        else:
            # OpenRouter
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": messages,
                    "max_tokens": 100,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                raw_text = resp.json()["choices"][0]["message"]["content"]
                return clean_llm_response(raw_text)
            return ""
    except Exception as e:
        print(f"[!] AI Generation Exception: {e}")
        return ""


# --- CORE LOGIC ---


def process_toc_nodes(nodes, action):
    """
    Recursively processes a list of TOC nodes.
    Calls `action(node)` on each, propagating True if action returned True.
    """
    updated_count = 0
    for node in nodes:
        if action(node):
            updated_count += 1
        if "children" in node:
            updated_count += process_toc_nodes(node["children"], action)
    return updated_count


def node_needs_description(node):
    """Checks if a node lacks a description or has a chatty one."""
    desc = node.get("description", "")
    if not desc:
        return True
    if "concise sentence" in desc or "Here is a summary" in desc:
        return True
    return False


def generate_node_description(node, html_dir):
    """Read HTML and generate an AI summary for a single node.
    Returns True if updated.
    """
    if not node_needs_description(node):
        return False
    if "url" not in node or ".htm" not in node["url"]:
        return False

    filename = node["url"].split("#")[0]
    section_title = node.get("text", "this section")
    file_path = html_dir / filename

    if not file_path.exists():
        return False

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            text_clean = BeautifulSoup(text, "html.parser").get_text()

        print(f"    > Generating summary for: '{section_title}'")
        new_desc = generate_description(filename, section_title, text_clean)

        if new_desc:
            node["description"] = new_desc
            return True
    except Exception as e:
        print(f"    ! Error processing {filename}: {e}")

    return False


class GuideManager:
    def __init__(self, name, url):
        self.name = name
        self.url = url
        self.root = BASE_STORAGE_DIR / name
        self.dirs = {
            "html": self.root / "html",
            "md": self.root / "md",
        }
        self.paths = {
            "zip": self.root / f"{name}.zip",
            "toc": self.root / "toc.json",
            "meta": self.root / "metadata.json",
        }

    def needs_update(self):
        try:
            head = requests.head(self.url, timeout=10)
            remote_mod = head.headers.get("Last-Modified")

            if not self.paths["zip"].exists() or not self.paths["meta"].exists():
                return True, remote_mod

            with open(self.paths["meta"], "r") as f:
                local_meta = json.load(f)
                if local_meta.get("last_modified") != remote_mod:
                    return True, remote_mod

            return False, remote_mod
        except Exception as e:
            print(f"[!] Network warning for {self.name}: {e}")
            return not self.paths["zip"].exists(), None

    def flatten_html_directory(self):
        """Moves files up if they were extracted into a nested subfolder."""
        found_toc = list(self.dirs["html"].rglob("toc.json"))
        if not found_toc:
            return

        toc_location = found_toc[0]
        # If TOC is not in the root html dir, we need to flatten
        if toc_location.parent != self.dirs["html"]:
            print("[-] Flattening nested directory structure...")
            nested_root = toc_location.parent

            for item in nested_root.iterdir():
                destination = self.dirs["html"] / item.name
                if destination.exists():
                    if destination.is_dir():
                        shutil.rmtree(destination)
                    else:
                        destination.unlink()
                shutil.move(str(item), str(self.dirs["html"]))

            shutil.rmtree(nested_root)

    def download_and_extract(self, remote_mod):
        print(f"[-] Downloading {self.name}...")
        self.root.mkdir(parents=True, exist_ok=True)
        self.dirs["html"].mkdir(parents=True, exist_ok=True)

        try:
            with requests.get(self.url, stream=True) as r:
                r.raise_for_status()
                with open(self.paths["zip"], "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception as e:
            print(f"[!] Download failed: {e}")
            return False

        print(f"[-] Extracting {self.name}...")
        try:
            if self.dirs["html"].exists():
                shutil.rmtree(self.dirs["html"])
            self.dirs["html"].mkdir()

            with zipfile.ZipFile(self.paths["zip"], "r") as z:
                z.extractall(self.dirs["html"])
        except Exception as e:
            print(f"[!] Extraction failed: {e}")
            return False

        # Fix nested folders
        self.flatten_html_directory()

        # Move TOC to root
        extracted_toc = self.dirs["html"] / "toc.json"
        if extracted_toc.exists():
            shutil.move(str(extracted_toc), str(self.paths["toc"]))
        else:
            print("[!] WARNING: toc.json still not found after extraction.")

        with open(self.paths["meta"], "w") as f:
            json.dump(
                {"last_modified": remote_mod, "updated": datetime.now().isoformat()}, f
            )

        return True

    def enrich_toc(self):
        if not self.paths["toc"].exists():
            print(f"[!] No toc.json found for {self.name}. Skipping enrichment.")
            return

        print(f"[-] Enriching TOC for {self.name}...")
        with open(self.paths["toc"], "r", encoding="utf-8") as f:
            toc = json.load(f)

        # We need to maintain state for incremental saves, so wrap action
        class EncrichAction:
            def __init__(self, html_dir, toc_path, full_toc):
                self.html_dir = html_dir
                self.toc_path = toc_path
                self.full_toc = full_toc
                self.internal_count = 0

            def __call__(self, node):
                updated = generate_node_description(node, self.html_dir)
                if updated:
                    self.internal_count += 1
                    if self.internal_count % 5 == 0:
                        with open(self.toc_path, "w", encoding="utf-8") as f_save:
                            json.dump(self.full_toc, f_save, indent=2)
                return updated

        action = EncrichAction(self.dirs["html"], self.paths["toc"], toc)
        updated_count = process_toc_nodes(toc, action)

        if updated_count > 0:
            with open(self.paths["toc"], "w", encoding="utf-8") as f:
                json.dump(toc, f, indent=2)
            print(f"[+] Updated {updated_count} descriptions.")
        else:
            print("[-] No new descriptions needed.")

    def convert_to_markdown(self):
        if not self.paths["toc"].exists():
            return

        print(f"[-] Converting {self.name} to Markdown...")
        self.dirs["md"].mkdir(parents=True, exist_ok=True)

        with open(self.paths["toc"], "r", encoding="utf-8") as f:
            toc = json.load(f)

        processed_files = set()

        class ConvertAction:
            def __init__(self, html_dir, md_dir, processed):
                self.html_dir = html_dir
                self.md_dir = md_dir
                self.processed = processed

            def __call__(self, node):
                if "url" not in node or ".htm" not in node["url"]:
                    return False

                filename = node["url"].split("#")[0]
                html_path = self.html_dir / filename

                if filename in self.processed or not html_path.exists():
                    return False

                out_path = self.md_dir / filename.replace(".htm", ".md")
                success = convert_html_to_markdown(filename, html_path, out_path, node)
                if success:
                    self.processed.add(filename)

                return success

        action = ConvertAction(self.dirs["html"], self.dirs["md"], processed_files)
        process_toc_nodes(toc, action)

        print(f"[+] Converted {len(processed_files)} files to {self.dirs['md']}")


def convert_html_to_markdown(filename, html_path, out_path, node):
    """Parses HTML, removes junk bars, and converts to markdown."""
    try:
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        main = soup.find("div", {"class": "document"}) or soup.find("body")
        if main:
            for junk in main.find_all(
                ["div", "nav"],
                {"class": ["sphinxsidebar", "related", "footer"]},
            ):
                junk.decompose()

            md_text = md(str(main), heading_style="ATX", code_language="php")

            header = f"# {node.get('text', 'Untitled')}\n\n"
            if node.get("description"):
                header += f"> **AI Summary:** {node['description']}\n\n"

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(header + md_text)

            return True
    except Exception as e:
        print(f"    ! Error converting {filename}: {e}")
    return False


# --- MAIN EXECUTION ---

if __name__ == "__main__":
    if USE_LOCAL_LLM:
        try:
            requests.get(f"{LOCAL_API_URL}/models", timeout=2)
            print("[+] Connected to Local LLM (LM Studio)")
        except Exception:
            print("[!] Warning: Could not connect to LM Studio.")

    for name, url in GUIDES.items():
        print(f"\n=== Processing Guide: {name} ===")
        mgr = GuideManager(name, url)

        # 1. Sync
        should_update, remote_mod = mgr.needs_update()

        # Force re-download if TOC is missing (Fixes previous nested folder issues)
        if not should_update and not mgr.paths["toc"].exists():
            print("[-] TOC missing. Forcing re-download to fix structure...")
            mgr.download_and_extract(remote_mod)
        elif should_update:
            if remote_mod:
                print(f"[-] Update available (Remote: {remote_mod})")
            mgr.download_and_extract(remote_mod)
        else:
            print("[-] Local cache is up to date.")

        # 2. Enrich (Regenerates descriptions if they look "chatty")
        mgr.enrich_toc()

        # 3. Convert
        mgr.convert_to_markdown()
