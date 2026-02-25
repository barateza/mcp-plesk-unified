import json
from unittest.mock import patch

from plesk_unified.io_utils import (
    collect_files_for_source,
    ensure_source_exists,
    load_toc_map,
    parse_toc_recursive,
)


def test_ensure_source_exists_already_there(tmp_path):
    # Setup mock dir with a file
    (tmp_path / "test.txt").write_text("hello")
    source = {"path": tmp_path, "cat": "test"}
    assert ensure_source_exists(source)


@patch("plesk_unified.io_utils.Repo.clone_from")
def test_ensure_source_exists_clones(mock_clone, tmp_path):
    source = {
        "path": tmp_path / "new_repo",
        "repo_url": "http://example.com/repo.git",
        "cat": "test",
    }
    assert ensure_source_exists(source)
    mock_clone.assert_called_once_with(
        "http://example.com/repo.git", tmp_path / "new_repo"
    )


@patch("plesk_unified.io_utils.Repo.clone_from")
def test_ensure_source_exists_clone_fails(mock_clone, tmp_path):
    mock_clone.side_effect = Exception("Failed")
    source = {
        "path": tmp_path / "new_repo",
        "repo_url": "http://example.com/repo.git",
        "cat": "test",
    }
    assert not ensure_source_exists(source)


def test_parse_toc_recursive():
    nodes = [
        {"text": "A", "url": "a.htm"},
        {
            "text": "B",
            "url": "b.htm#anchor",
            "children": [{"text": "C", "url": "c.htm"}],
        },
    ]
    res = parse_toc_recursive(nodes)
    assert res["a.htm"]["title"] == "A"
    assert res["a.htm"]["breadcrumb"] == "A"

    assert res["b.htm"]["title"] == "B"
    assert res["b.htm"]["breadcrumb"] == "B"

    assert res["c.htm"]["title"] == "C"
    assert res["c.htm"]["breadcrumb"] == "B > C"


def test_load_toc_map(tmp_path):
    nodes = [
        {"text": "A", "url": "a.htm"},
    ]
    (tmp_path / "toc.json").write_text(json.dumps(nodes))
    res = load_toc_map(tmp_path)
    assert "a.htm" in res


def test_load_toc_map_not_found(tmp_path):
    assert load_toc_map(tmp_path) == {}


def test_collect_files_for_source(tmp_path):
    (tmp_path / "test.htm").touch()
    (tmp_path / "test.php").touch()

    source = {"path": tmp_path, "type": "html"}
    files = collect_files_for_source(source)
    assert len(files) == 1
    assert files[0].name == "test.htm"

    source = {"path": tmp_path, "type": "php"}
    files = collect_files_for_source(source)
    assert len(files) == 1
    assert files[0].name == "test.php"
