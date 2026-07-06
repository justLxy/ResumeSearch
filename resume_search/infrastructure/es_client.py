"""Elasticsearch HTTP 访问的唯一入口。

所有对 ES 的读写都经过 `es_request`。业务层不直接持有 requests / URL 拼接，
便于统一超时、错误处理，也便于测试整体打桩。
"""
from __future__ import annotations

from typing import Any

import requests

from resume_search.config import ES_REQUEST_TIMEOUT_SECONDS, ES_URL


def es_request(method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.request(
        method,
        f"{ES_URL}{path}",
        json=body,
        timeout=ES_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()
