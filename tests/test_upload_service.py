"""上传服务校验逻辑（不依赖 ES / embedding）。"""
from __future__ import annotations

from pathlib import Path

from resume_search.services.upload import (
    list_existing_doc_filenames,
    upload_resume_files,
)


def test_list_existing_doc_filenames(tmp_path: Path) -> None:
    (tmp_path / "a.doc").write_bytes(b"x")
    (tmp_path / "b.DOC").write_bytes(b"x")
    (tmp_path / "c.pdf").write_bytes(b"x")
    (tmp_path / ".hidden.doc").write_bytes(b"x")
    names = list_existing_doc_filenames(tmp_path)
    assert names == ["a.doc", "b.DOC"]


def test_reject_invalid_format_and_empty(tmp_path: Path) -> None:
    result = upload_resume_files(
        [
            ("resume.pdf", b"%PDF"),
            ("empty.doc", b""),
            ("../escape.doc", b"content"),
        ],
        data_dir=tmp_path,
    )
    assert result["summary"]["succeeded"] == 0
    assert result["summary"]["failed"] == 3
    codes = {item["code"] for item in result["results"]}
    assert "invalid_format" in codes
    assert "empty_file" in codes


def test_reject_duplicate_against_disk_and_batch(tmp_path: Path) -> None:
    (tmp_path / "exists.doc").write_bytes(b"already")
    result = upload_resume_files(
        [
            ("exists.doc", b"new"),
            ("fresh.doc", b"one"),
            ("fresh.doc", b"two"),
        ],
        data_dir=tmp_path,
    )
    # fresh.doc will attempt index and fail without ES; exists + batch dup fail earlier
    by_name = {item["filename"]: item for item in result["results"]}
    assert by_name["exists.doc"]["code"] == "duplicate_filename"
    # 第二个同名 batch 内重复
    dups = [item for item in result["results"] if item["filename"] == "fresh.doc"]
    assert any(item.get("code") == "duplicate_filename" for item in dups)
