"""导入编排：加载简历文档 → 富化 → 构建证据切片 → 版本化写入 → 切换别名。

`import_resumes` 是导入链路的顶层入口，被 CLI 调用。支持从 JSONL/JSON 或 .doc 目录
加载，全量重建（versioned index + alias 原子切换）或增量更新（可选删除陈旧文档）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from indexing.resume_parser import discover_doc_files, parse_resume_batch

from indexing.enrichment import _enrich_doc
from indexing.es_admin import (
    _bulk_index,
    _delete_missing_docs,
    _delete_missing_evidence_docs,
    _request,
    _switch_alias,
    _target_exists,
    _versioned_index_name,
    _wait_for_index_ready,
    _write_target,
)
from indexing.evidence import _build_evidence_docs, add_evidence_embeddings
from indexing.mappings import (
    DEFAULT_ALIAS,
    DEFAULT_ES_URL,
    DEFAULT_EVIDENCE_ALIAS,
    DEFAULT_EVIDENCE_INDEX,
    DEFAULT_INDEX,
    EVIDENCE_INDEX_BODY,
    INDEX_BODY,
)


def import_resumes(
    data_path: str | Path,
    es_url: str = DEFAULT_ES_URL,
    index: str = DEFAULT_INDEX,
    alias: str = DEFAULT_ALIAS,
    evidence_index: str = DEFAULT_EVIDENCE_INDEX,
    evidence_alias: str = DEFAULT_EVIDENCE_ALIAS,
    recreate: bool = True,
    delete_missing: bool = False,
) -> dict[str, Any]:
    docs = _load_resume_docs(data_path)
    docs = [_enrich_doc(doc) for doc in docs if doc.get("parse_status") == "ok"]
    if recreate and not docs:
        raise RuntimeError("no parsed documents; aborting index rebuild")
    evidence_docs = _build_evidence_docs(docs)
    add_evidence_embeddings(evidence_docs)

    target_index = _versioned_index_name(index) if recreate else _write_target(es_url, index, alias)
    target_evidence_index = (
        _versioned_index_name(evidence_index)
        if recreate
        else _write_target(es_url, evidence_index, evidence_alias)
    )
    if recreate:
        _request("PUT", f"{es_url}/{target_index}", json_body=INDEX_BODY, ok_statuses={200})
        _request(
            "PUT",
            f"{es_url}/{target_evidence_index}",
            json_body=EVIDENCE_INDEX_BODY,
            ok_statuses={200},
        )
        _wait_for_index_ready(es_url, target_index)
        _wait_for_index_ready(es_url, target_evidence_index)
    elif not _target_exists(es_url, target_index):
        _request("PUT", f"{es_url}/{target_index}", json_body=INDEX_BODY, ok_statuses={200})
        _wait_for_index_ready(es_url, target_index)
    if not recreate and not _target_exists(es_url, target_evidence_index):
        _request(
            "PUT",
            f"{es_url}/{target_evidence_index}",
            json_body=EVIDENCE_INDEX_BODY,
            ok_statuses={200},
        )
        _wait_for_index_ready(es_url, target_evidence_index)

    if docs:
        _bulk_index(es_url, target_index, docs, id_field="resume_id")
        _request("POST", f"{es_url}/{target_index}/_refresh", ok_statuses={200})
    if evidence_docs:
        _bulk_index(es_url, target_evidence_index, evidence_docs, id_field="evidence_id")
        _request("POST", f"{es_url}/{target_evidence_index}/_refresh", ok_statuses={200})

    if delete_missing and not recreate:
        live_ids = {doc["resume_id"] for doc in docs}
        _delete_missing_docs(es_url, target_index, live_ids)
        _delete_missing_evidence_docs(es_url, target_evidence_index, live_ids)

    count = _request("GET", f"{es_url}/{target_index}/_count", ok_statuses={200})["count"]
    if recreate and count != len(docs):
        raise RuntimeError(f"indexed count mismatch: expected {len(docs)}, got {count}")
    evidence_count = _request(
        "GET",
        f"{es_url}/{target_evidence_index}/_count",
        ok_statuses={200},
    )["count"]
    if recreate and evidence_count != len(evidence_docs):
        raise RuntimeError(
            f"indexed evidence count mismatch: expected {len(evidence_docs)}, got {evidence_count}"
        )

    if recreate or not _target_exists(es_url, alias):
        _switch_alias(es_url, target_index, alias)
    if recreate or not _target_exists(es_url, evidence_alias):
        _switch_alias(es_url, target_evidence_index, evidence_alias)

    alias_count = _request("GET", f"{es_url}/{alias}/_count", ok_statuses={200})["count"]
    evidence_alias_count = _request(
        "GET",
        f"{es_url}/{evidence_alias}/_count",
        ok_statuses={200},
    )["count"]
    return {
        "index": target_index,
        "alias": alias,
        "evidence_index": target_evidence_index,
        "evidence_alias": evidence_alias,
        "parsed": len(docs),
        "indexed": count,
        "evidence_indexed": evidence_count,
        "alias_count": alias_count,
        "evidence_alias_count": evidence_alias_count,
    }


def _load_resume_docs(data_path: str | Path) -> list[dict[str, Any]]:
    path = Path(data_path)
    if path.is_file() and path.suffix.lower() == ".jsonl":
        return _load_jsonl_docs(path)
    if path.is_file() and path.suffix.lower() == ".json":
        return _load_json_docs(path)
    return parse_resume_batch(discover_doc_files(path))


def _load_jsonl_docs(path: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_no} must contain a JSON object per line")
        docs.append(item)
    return docs


def _load_json_docs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError(f"{path} must contain a list of JSON objects")
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"{path} must contain a JSON object or a list of JSON objects")


