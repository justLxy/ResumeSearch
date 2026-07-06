"""LLM query-parser 的 HTTP 边界。

只负责把已构造好的请求体 POST 给 OpenAI-兼容 chat/completions 接口并返回原始 JSON。
prompt 构造、响应解析、意图判定等逻辑属于 query_planning service，不放这里。
"""
from __future__ import annotations

from typing import Any

import requests

from resume_search.config import (
    QUERY_PARSER_API_KEY,
    QUERY_PARSER_API_URL,
    QUERY_PARSER_TIMEOUT_SECONDS,
)


def post_query_parser(body: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        QUERY_PARSER_API_URL,
        headers={
            "Authorization": f"Bearer {QUERY_PARSER_API_KEY}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=QUERY_PARSER_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()
