from plesk_unified.chunking import chunk_by_chars, build_doc_records


def test_chunk_by_chars_empty():
    assert chunk_by_chars("") == []


def test_chunk_by_chars_overlap():
    text = "a" * 5000
    chunks = chunk_by_chars(text, size=1000, overlap=200)
    assert len(chunks) >= 5
    # ensure overlap by checking consecutive chunks share content
    assert chunks[0][-1] == "a"


def test_build_doc_records():
    chunks = ["one", "two"]
    meta = {"title": "T", "category": "cat", "breadcrumb": "A > B"}
    recs = build_doc_records("file.html", chunks, meta)
    assert len(recs) == 2
    assert recs[0]["title"] == "T"
    assert recs[1]["filename"] == "file.html"
