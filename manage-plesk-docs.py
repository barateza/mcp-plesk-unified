import os
import requests
import zipfile
import json
import shutil
from pathlib import Path
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from datetime import datetime

# --- CONFIGURATION ---

# 1. AI Provider Settings
USE_LOCAL_LLM = True

# LM Studio Configuration
# Note: LM Studio usually processes requests regardless of the 'model' string
# as long as a model is loaded in the GUI.
LOCAL_API_URL = "http://localhost:1234/v1"
LOCAL_MODEL_ID = "llama-3.1-8b-instruct"

# Fallback to OpenRouter if local fails or is disabled
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = "liquid/lfm-2.5-1.2b-thinking:free"

# 2. Storage Paths
BASE_STORAGE_DIR = Path("storage")

# 3. Documentation Sources
# Add any other Plesk zip URLs here. The script handles them all.
GUIDES = {
    "extensions-guide": "https://docs.plesk.com/en-US/obsidian/zip/extensions-guide.zip",
    "cli-linux": "https://docs.plesk.com/en-US/obsidian/zip/cli-linux.zip",
    "api-rpc": "https://docs.plesk.com/en-US/obsidian/zip/api-rpc.zip",
    "rest-api": "https://docs.plesk.com/en-US/obsidian/zip/rest-api.zip",
}

# --- AI HELPERS ---


def generate_description(filename, content_snippet):
    """Generates a one-liner description using the selected provider."""
    # Context prompt designed for Llama 3.1
    prompt = (
        f"You are a technical documentation assistant. "
        f"Summarize the technical purpose of the file '{filename}' in exactly one concise sentence. "
        f"Do not use introductory phrases like 'This file contains'. Just state the purpose directly.\n\n"
        f"File Content Snippet:\n{content_snippet[:3000]}"
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful technical assistant. You provide concise, one-sentence summaries.",
        },
        {"role": "user", "content": prompt},
    ]

    try:
        if USE_LOCAL_LLM:
            # LM Studio / Ollama Request
            resp = requests.post(
                f"{LOCAL_API_URL}/chat/completions",
                json={
                    "model": LOCAL_MODEL_ID,
                    "messages": messages,
                    "temperature": 0.3,  # Low temp for factual accuracy
                    "max_tokens": 100,
                    "stream": False,
                },
                timeout=60,  # Llama 3.1 8B might take a few seconds on first load
            )
            if resp.status_code != 200:
                print(f"[!] Local LLM Error: {resp.text}")
                return ""
            return resp.json()["choices"][0]["message"]["content"].strip()
        else:
            # OpenRouter Request
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
                return resp.json()["choices"][0]["message"]["content"].strip()
            else:
                print(f"[!] OpenRouter Error: {resp.text}")
                return ""
    except Exception as e:
        print(f"[!] AI Generation Exception: {e}")
        return ""


# --- CORE LOGIC ---


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
        """Checks Last-Modified header against local metadata."""
        try:
            head = requests.head(self.url, timeout=10)
            remote_mod = head.headers.get("Last-Modified")

            # If we've never downloaded it, we need it.
            if not self.paths["zip"].exists() or not self.paths["meta"].exists():
                return True, remote_mod

            # If we have it, check if the remote date changed
            with open(self.paths["meta"], "r") as f:
                local_meta = json.load(f)
                if local_meta.get("last_modified") != remote_mod:
                    return True, remote_mod

            return False, remote_mod
        except Exception as e:
            print(f"[!] Network warning for {self.name}: {e}")
            # If network fails but we have files, assume we are good for now
            return not self.paths["zip"].exists(), None

    def download_and_extract(self, remote_mod):
        """Downloads and unzips the guide."""
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
            # Clean old HTML to prevent orphans
            if self.dirs["html"].exists():
                shutil.rmtree(self.dirs["html"])
            self.dirs["html"].mkdir()

            with zipfile.ZipFile(self.paths["zip"], "r") as z:
                z.extractall(self.dirs["html"])
        except Exception as e:
            print(f"[!] Extraction failed: {e}")
            return False

        # Move TOC to root if it exists inside html
        extracted_toc = self.dirs["html"] / "toc.json"
        if extracted_toc.exists():
            shutil.move(str(extracted_toc), str(self.paths["toc"]))

        with open(self.paths["meta"], "w") as f:
            json.dump(
                {"last_modified": remote_mod, "updated": datetime.now().isoformat()}, f
            )

        return True

    def enrich_toc(self):
        """Iterates TOC and adds AI descriptions if missing."""
        if not self.paths["toc"].exists():
            print(f"[!] No toc.json found for {self.name}. Skipping enrichment.")
            return

        print(f"[-] Enriching TOC for {self.name}...")
        with open(self.paths["toc"], "r", encoding="utf-8") as f:
            toc = json.load(f)

        updated_count = 0

        # We use a recursive function to traverse the TOC tree
        def process_nodes_recursively(nodes):
            nonlocal updated_count
            for node in nodes:
                # Logic: Is it a file? (.htm) AND Does it lack a description?
                if (
                    "url" in node
                    and ".htm" in node["url"]
                    and not node.get("description")
                ):
                    filename = node["url"].split("#")[0]
                    file_path = self.dirs["html"] / filename

                    if file_path.exists():
                        try:
                            with open(
                                file_path, "r", encoding="utf-8", errors="ignore"
                            ) as f:
                                text = f.read()
                                # Lightweight strip to save tokens
                                text_clean = BeautifulSoup(
                                    text, "html.parser"
                                ).get_text()

                            print(f"    > Generating summary for: {filename}")
                            desc = generate_description(filename, text_clean)
                            if desc:
                                node["description"] = desc
                                updated_count += 1
                                # Save incrementally in case of crash
                                if updated_count % 5 == 0:
                                    with open(
                                        self.paths["toc"], "w", encoding="utf-8"
                                    ) as f_save:
                                        json.dump(toc, f_save, indent=2)
                        except Exception as e:
                            print(f"    ! Error processing {filename}: {e}")

                if "children" in node:
                    process_nodes_recursively(node["children"])

        process_nodes_recursively(toc)

        if updated_count > 0:
            with open(self.paths["toc"], "w", encoding="utf-8") as f:
                json.dump(toc, f, indent=2)
            print(f"[+] Added {updated_count} new descriptions.")
        else:
            print("[-] No new descriptions needed.")

    def convert_to_markdown(self):
        """Converts HTML to Markdown using TOC structure."""
        if not self.paths["toc"].exists():
            return

        print(f"[-] Converting {self.name} to Markdown...")
        self.dirs["md"].mkdir(parents=True, exist_ok=True)

        with open(self.paths["toc"], "r", encoding="utf-8") as f:
            toc = json.load(f)

        processed_files = set()

        def process_nodes_recursively(nodes):
            for node in nodes:
                if "url" in node and ".htm" in node["url"]:
                    filename = node["url"].split("#")[0]
                    html_path = self.dirs["html"] / filename

                    if filename not in processed_files and html_path.exists():
                        try:
                            with open(html_path, "r", encoding="utf-8") as f:
                                soup = BeautifulSoup(f.read(), "html.parser")

                            # Clean HTML (Remove sidebar, footer, search)
                            main = soup.find("div", {"class": "document"}) or soup.find(
                                "body"
                            )
                            if main:
                                for junk in main.find_all(
                                    ["div", "nav"],
                                    {"class": ["sphinxsidebar", "related", "footer"]},
                                ):
                                    junk.decompose()

                                # Convert to Markdown
                                md_text = md(
                                    str(main), heading_style="ATX", code_language="php"
                                )

                                # Add Header & AI Metadata
                                header = f"# {node.get('text', 'Untitled')}\n\n"
                                if node.get("description"):
                                    header += (
                                        f"> **AI Summary:** {node['description']}\n\n"
                                    )

                                # Save
                                out_path = self.dirs["md"] / filename.replace(
                                    ".htm", ".md"
                                )
                                out_path.parent.mkdir(parents=True, exist_ok=True)
                                with open(out_path, "w", encoding="utf-8") as f:
                                    f.write(header + md_text)

                                processed_files.add(filename)
                        except Exception as e:
                            print(f"    ! Error converting {filename}: {e}")

                if "children" in node:
                    process_nodes_recursively(node["children"])

        process_nodes_recursively(toc)
        print(f"[+] Converted {len(processed_files)} files to {self.dirs['md']}")


# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # Check if Local LLM or OpenRouter is ready
    if USE_LOCAL_LLM:
        try:
            requests.get(f"{LOCAL_API_URL}/models", timeout=2)
            print("[+] Connected to Local LLM (LM Studio)")
        except Exception:
            print(
                "[!] Warning: Could not connect to LM Studio. Make sure 'Start Server' is clicked."
            )
            if not OPENROUTER_KEY:
                print("    Aborting AI enrichment (No OpenRouter key found).")

    for name, url in GUIDES.items():
        print(f"\n=== Processing Guide: {name} ===")
        mgr = GuideManager(name, url)

        # 1. Sync
        should_update, remote_mod = mgr.needs_update()
        if should_update:
            if remote_mod:
                print(f"[-] Update available (Remote: {remote_mod})")
            mgr.download_and_extract(remote_mod)
        else:
            print("[-] Local cache is up to date.")

        # 2. Enrich (Runs on every execution to catch missing summaries)
        mgr.enrich_toc()

        # 3. Convert (Runs on every execution to bake in new summaries)
        mgr.convert_to_markdown()
