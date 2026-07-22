"""简历上传：校验 → 落盘 data/ → 增量解析与索引。

与 CLI 全量导入分离：只处理本次上传的文件，写入现有 ES 索引（不 recreate）。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from indexing.pipeline import import_resume_paths

from resume_search.config import DATA_DIR

ALLOWED_SUFFIX = ".doc"
# 与 data 目录现有命名一致：禁止路径分隔与控制字符，允许中文与常见文件名字符
_SAFE_FILENAME_RE = re.compile(r"^[^\\/:\*\?\"<>\|\x00-\x1f]+\.doc$", re.IGNORECASE)


def list_existing_doc_filenames(data_dir: Path | None = None) -> list[str]:
    root = Path(data_dir or DATA_DIR)
    if not root.is_dir():
        return []
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() == ALLOWED_SUFFIX and not p.name.startswith(".")
    )


def upload_resume_files(
    files: list[tuple[str, bytes]],
    *,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """处理一批上传文件。

    ``files``: ``(original_filename, content_bytes)`` 列表。
    返回 summary + 逐文件 results（status: success | error）。
    """
    root = Path(data_dir or DATA_DIR)
    root.mkdir(parents=True, exist_ok=True)

    existing = {name.lower(): name for name in list_existing_doc_filenames(root)}
    batch_names: set[str] = set()
    results: list[dict[str, Any]] = []

    # 第一遍：纯校验，不落盘；通过的进入 pending
    pending: list[tuple[str, bytes, Path]] = []
    for raw_name, content in files:
        filename = _normalize_filename(raw_name)
        if filename is None:
            results.append(
                _error(
                    raw_name or "(未命名)",
                    "invalid_format",
                    "仅支持 .doc 格式，且文件名不能包含路径或非法字符",
                )
            )
            continue
        if not content:
            results.append(_error(filename, "empty_file", "文件为空"))
            continue
        key = filename.lower()
        if key in existing:
            results.append(
                _error(
                    filename,
                    "duplicate_filename",
                    f"data 目录下已存在同名文件：{existing[key]}",
                )
            )
            continue
        if key in batch_names:
            results.append(
                _error(filename, "duplicate_filename", "本次上传中存在同名文件")
            )
            continue
        batch_names.add(key)
        pending.append((filename, content, root / filename))

    # 第二遍：落盘并增量索引（逐文件，互不影响）
    for filename, content, dest in pending:
        try:
            dest.write_bytes(content)
        except OSError as exc:
            results.append(
                _error(filename, "internal_error", f"保存文件失败：{exc}")
            )
            continue

        try:
            import_result = import_resume_paths([dest])
        except Exception as exc:
            results.append(
                _error(
                    filename,
                    "index_failed",
                    f"文件已保存，但索引失败：{exc}",
                )
            )
            continue

        failed = import_result.get("failed") or []
        if failed:
            err_msgs = []
            for item in failed:
                err_msgs.extend(item.get("errors") or [])
            message = "；".join(err_msgs) if err_msgs else "解析失败"
            results.append(_error(filename, "parse_failed", message))
            continue

        ok_items = import_result.get("results") or []
        resume_id = ok_items[0].get("resume_id") if ok_items else None
        if not resume_id:
            results.append(
                _error(filename, "index_failed", "文件已保存，但未得到 resume_id")
            )
            continue

        results.append(
            {
                "filename": filename,
                "status": "success",
                "resume_id": resume_id,
                "message": "已解析并索引",
            }
        )

    succeeded = sum(1 for r in results if r.get("status") == "success")
    failed_n = len(results) - succeeded
    return {
        "summary": {
            "total": len(results),
            "succeeded": succeeded,
            "failed": failed_n,
        },
        "results": results,
    }


def _normalize_filename(raw_name: str) -> str | None:
    name = Path(raw_name or "").name.strip()
    if not name or not _SAFE_FILENAME_RE.match(name):
        return None
    if not name.lower().endswith(ALLOWED_SUFFIX):
        return None
    return name


def _error(filename: str, code: str, message: str) -> dict[str, Any]:
    return {
        "filename": filename,
        "status": "error",
        "code": code,
        "message": message,
    }
