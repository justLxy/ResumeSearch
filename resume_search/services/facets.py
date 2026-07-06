"""facet 聚合与筛选词表：为前端提供各维度可选项与计数。

聚合请求经 es_client 发往 ES；院校档位计数复用 filters 的档位子句。带进程内 TTL
缓存，降低对 ES 的重复聚合压力。
"""
from __future__ import annotations

import time
from typing import Any

from resume_search.config import (
    FACETS_CACHE_TTL_SECONDS,
    FILTER_VOCAB_CACHE_TTL_SECONDS,
    INDEX_ALIAS,
    SKILL_FACET_AGG_SIZE,
    SKILL_FACET_DISPLAY_SIZE,
)
from resume_search.domain.constants import (
    CANONICAL_SKILL_LABELS,
    SCHOOL_TIER_LABELS,
    SCHOOL_TIER_OTHER,
)
from resume_search.infrastructure import es_client
from resume_search.services.normalization import (
    _casefold_key,
    _school_tier_filter,
    _skill_label_sort_key,
)

# facet / 筛选词表的进程内 TTL 缓存。
_facets_cache: tuple[float, dict[str, Any]] | None = None
_filter_vocab_cache: tuple[float, dict[str, set[str]]] | None = None


def _facet_keys(facets: dict[str, Any], name: str) -> set[str]:
    return {
        str(item.get("key")).strip()
        for item in facets.get(name, [])
        if item.get("key")
    }


def _load_facets() -> dict[str, Any]:
    global _facets_cache
    now = time.monotonic()
    if _facets_cache and _facets_cache[0] > now:
        return _facets_cache[1]

    body = {
        "size": 0,
        "aggs": {
            "degrees": {"terms": {"field": "candidate.highest_degree", "size": 20}},
            "cities": {"terms": {"field": "application.expected_work_cities", "size": 20}},
            "skills": {"terms": {"field": "skills", "size": SKILL_FACET_AGG_SIZE}},
            "positions": {"terms": {"field": "application.position_name.keyword", "size": 20}},
            "max_years": {"max": {"field": "candidate.years_experience"}},
            **_school_tier_aggs(),
        },
    }
    result = es_client.es_request("POST", f"/{INDEX_ALIAS}/_search", body)
    aggs = result.get("aggregations", {})
    facets = {
        name: [
            {"key": bucket["key"], "count": bucket["doc_count"]}
            for bucket in aggs.get(name, {}).get("buckets", [])
            if bucket.get("key")
        ]
        for name in ("degrees", "cities", "skills", "positions")
    }
    facets["skills"] = _merge_case_insensitive_skill_buckets(
        aggs.get("skills", {}).get("buckets", []),
        SKILL_FACET_DISPLAY_SIZE,
    )
    facets["school_tiers"] = _school_tier_facet_counts(aggs)
    max_years_val = aggs.get("max_years", {}).get("value")
    facets["max_years"] = round(max_years_val, 1) if max_years_val else 5
    _facets_cache = (now + FACETS_CACHE_TTL_SECONDS, facets)
    return facets


def _school_tier_aggs() -> dict[str, Any]:
    """为每个院校档位构建一个 filter 聚合，供前端按钮显示命中人数。"""
    aggs: dict[str, Any] = {}
    for key in SCHOOL_TIER_LABELS:
        tier_filter = _school_tier_filter(key)
        if tier_filter:
            aggs[f"school_tier_{key}"] = {"filter": tier_filter}
    other_filter = _school_tier_filter(SCHOOL_TIER_OTHER)
    if other_filter:
        aggs["school_tier_other"] = {"filter": other_filter}
    return aggs


def _school_tier_facet_counts(aggs: dict[str, Any]) -> list[dict[str, Any]]:
    """从聚合结果提取各档位计数，按 UI 顺序返回 [{key, label, count}, ...]。"""
    counts: list[dict[str, Any]] = []
    for key, label in SCHOOL_TIER_LABELS.items():
        bucket = aggs.get(f"school_tier_{key}")
        if bucket is not None:
            counts.append({"key": key, "label": label, "count": bucket.get("doc_count", 0)})
    other = aggs.get("school_tier_other")
    if other is not None:
        counts.append(
            {"key": SCHOOL_TIER_OTHER, "label": SCHOOL_TIER_OTHER, "count": other.get("doc_count", 0)}
        )
    return counts


def _merge_case_insensitive_skill_buckets(
    buckets: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for bucket in buckets:
        raw_key = str(bucket.get("key", "")).strip()
        if not raw_key:
            continue
        key = _casefold_key(raw_key)
        item = merged.setdefault(key, {"count": 0, "variants": []})
        item["count"] += int(bucket.get("doc_count", bucket.get("count", 0)) or 0)
        item["variants"].append(raw_key)

    items = [
        {"key": _display_skill_label(item["variants"]), "count": item["count"]}
        for item in merged.values()
    ]
    return sorted(items, key=lambda item: (-item["count"], _casefold_key(item["key"])))[:limit]


def _display_skill_label(variants: list[str]) -> str:
    cleaned = [str(value).strip() for value in variants if str(value).strip()]
    if not cleaned:
        return ""
    canonical = CANONICAL_SKILL_LABELS.get(_casefold_key(cleaned[0]))
    if canonical:
        return canonical
    return sorted(set(cleaned), key=lambda value: (_skill_label_sort_key(value), value))[0]


def _load_filter_vocab() -> dict[str, set[str]]:
    global _filter_vocab_cache
    now = time.monotonic()
    if _filter_vocab_cache and _filter_vocab_cache[0] > now:
        return _filter_vocab_cache[1]

    body = {
        "size": 0,
        "aggs": {
            "degrees": {"terms": {"field": "candidate.highest_degree", "size": 50}},
            "cities": {"terms": {"field": "application.expected_work_cities", "size": 200}},
            "skills": {"terms": {"field": "skills", "size": 1000}},
        },
    }
    result = es_client.es_request("POST", f"/{INDEX_ALIAS}/_search", body)
    aggs = result.get("aggregations", {})
    vocab = {
        name: {
            str(bucket.get("key")).strip()
            for bucket in aggs.get(name, {}).get("buckets", [])
            if bucket.get("key")
        }
        for name in ("degrees", "cities", "skills")
    }
    _filter_vocab_cache = (now + FILTER_VOCAB_CACHE_TTL_SECONDS, vocab)
    return vocab


