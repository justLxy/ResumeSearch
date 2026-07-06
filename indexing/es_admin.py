"""Elasticsearch 索引管理：版本化索引创建、别名切换、批量写入、增量删除。

封装导入流程需要的所有 ES 运维操作。所有 HTTP 经 `_request`，统一超时与错误抛出。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import requests

from indexing.mappings import BULK_BATCH_SIZE, REQUEST_TIMEOUT_SECONDS


def _request(
    method: str,
    url: str,
    *,
    json_body: dict[str, Any] | None = None,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    ok_statuses: set[int],
) -> dict[str, Any]:
    response = requests.request(
        method,
        url,
        json=json_body,
        data=data,
        headers=headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code not in ok_statuses:
        raise RuntimeError(f"{method} {url} -> {response.status_code}: {response.text[:800]}")
    if response.text:
        return response.json()
    return {}


def _versioned_index_name(base_index: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    return f"{base_index}_{stamp}"


def _write_target(es_url: str, index: str, alias: str) -> str:
    if _target_exists(es_url, alias):
        return alias
    return index


def _target_exists(es_url: str, target: str) -> bool:
    response = requests.head(f"{es_url}/{target}", timeout=10)
    return response.status_code == 200


def _wait_for_index_ready(es_url: str, index: str) -> None:
    _request(
        "GET",
        f"{es_url}/_cluster/health/{index}?wait_for_status=yellow&timeout=90s",
        ok_statuses={200},
    )


def _bulk_index(
    es_url: str,
    index: str,
    docs: list[dict[str, Any]],
    *,
    id_field: str,
) -> None:
    for start in range(0, len(docs), BULK_BATCH_SIZE):
        batch = docs[start : start + BULK_BATCH_SIZE]
        lines = []
        for doc in batch:
            doc_id = doc.get(id_field)
            if not doc_id:
                raise ValueError(f"document is missing required id field: {id_field}")
            lines.append(
                json.dumps(
                    {"index": {"_index": index, "_id": str(doc_id)}},
                    ensure_ascii=False,
                )
            )
            lines.append(json.dumps(doc, ensure_ascii=False))
        response = _request(
            "POST",
            f"{es_url}/_bulk",
            data=("\n".join(lines) + "\n").encode("utf-8"),
            headers={"Content-Type": "application/x-ndjson"},
            ok_statuses={200},
        )
        if response.get("errors"):
            failures = [
                item
                for item in response.get("items", [])
                if item.get("index", {}).get("error")
            ]
            raise RuntimeError(f"bulk import failed: {failures[:3]}")


def _delete_missing_docs(es_url: str, target: str, live_ids: set[str]) -> None:
    if not live_ids:
        return
    _request(
        "POST",
        f"{es_url}/{target}/_delete_by_query?refresh=true",
        json_body={
            "query": {
                "bool": {
                    "must_not": {
                        "ids": {"values": sorted(live_ids)}
                    }
                }
            }
        },
        ok_statuses={200},
    )


def _delete_missing_evidence_docs(es_url: str, target: str, live_resume_ids: set[str]) -> None:
    if not live_resume_ids:
        return
    _request(
        "POST",
        f"{es_url}/{target}/_delete_by_query?refresh=true",
        json_body={
            "query": {
                "bool": {
                    "must_not": {
                        "terms": {"resume_id": sorted(live_resume_ids)}
                    }
                }
            }
        },
        ok_statuses={200},
    )


def _switch_alias(es_url: str, index: str, alias: str) -> None:
    _request(
        "POST",
        f"{es_url}/_aliases",
        json_body={
            "actions": [
                {"remove": {"index": "*", "alias": alias}},
                {"add": {"index": index, "alias": alias, "is_write_index": True}},
            ]
        },
        ok_statuses={200},
    )


