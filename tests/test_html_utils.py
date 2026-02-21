from pathlib import Path

from plesk_unified.html_utils import parse_html_file, clean_html_for_markdown


def test_parse_html_file(tmp_path):
    src = Path("tests/fixtures/sample.html")
    title, breadcrumb, text = parse_html_file(src)
    assert "Sample Page" in title
    assert "Heading" in text
    assert "Navigation" not in text


def test_clean_html_for_markdown():
    src = Path("tests/fixtures/sample.html")
    html = src.read_text(encoding="utf-8")
    cleaned = clean_html_for_markdown(html)
    assert "Navigation" not in cleaned
    assert "Heading" in cleaned
