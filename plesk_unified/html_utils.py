from pathlib import Path
from typing import Optional, Tuple

from bs4 import BeautifulSoup


def parse_html_file(
    path: Path, toc_meta: Optional[dict] = None
) -> Tuple[str, Optional[str], str]:
    """Parse an HTML file and return (title, breadcrumb, text).

    - Removes nav/footer/script/style/aside elements before extracting text.
    - Prefers <main> or <article> when available.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        html = fh.read()

    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    title = (
        title_tag.get_text(strip=True)
        if title_tag
        else (toc_meta or {}).get("title", "")
    )

    # Remove common noisy elements
    for sel in soup.select(
        "nav, footer, script, style, aside, .sidebar, .toc, noscript"
    ):
        sel.decompose()

    main = soup.find("main") or soup.find("article") or soup.body
    text = (
        main.get_text(separator="\n", strip=True)
        if main
        else soup.get_text(separator="\n", strip=True)
    )

    breadcrumb = (toc_meta or {}).get("breadcrumb")

    return title, breadcrumb, text


def clean_html_for_markdown(html: str) -> str:
    """Return cleaned HTML string suitable for markdown conversion.

    This removes nav/footer/script/style/aside nodes and returns the inner
    HTML of main/article/body.
    """
    soup = BeautifulSoup(html, "html.parser")
    for sel in soup.select(
        "nav, footer, script, style, aside, .sidebar, .toc, noscript"
    ):
        sel.decompose()
    main = soup.find("main") or soup.find("article") or soup.body
    return str(main) if main else str(soup)
