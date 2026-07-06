"""规范化与去重工具：学历、院校档位、技能标签的清洗与归一。

纯函数（除院校名单的进程内缓存外无副作用），被 filters / query_planning / facets
等多处复用。依赖 domain.constants 的词表和 config 的路径，不依赖上层 service。
"""
from __future__ import annotations

import json
import logging
from typing import Any

from resume_search.config import SCHOOL_TIERS_PATH
from resume_search.domain.constants import (
    DEGREE_ALIASES,
    DEGREE_ORDER,
    SCHOOL_TIER_ALIASES,
    SCHOOL_TIER_LABELS,
    SCHOOL_TIER_OTHER,
)

logger = logging.getLogger(__name__)

# 院校档位名单的进程内缓存（(各档位 key→校名列表, 全部校名并集)）。
_school_tiers_cache: tuple[dict[str, list[str]], set[str]] | None = None


def _normalize_highest_degree(degree: str) -> str:
    return DEGREE_ALIASES.get(degree, degree)


def _normalize_degree_value(degree: str) -> str:
    normalized = _normalize_highest_degree(str(degree).strip())
    return normalized if normalized in DEGREE_ORDER else ""


def _normalize_degree_list(value: Any) -> list[str]:
    """规范化"可接受学历集合"：去重、过滤非法值、按 本科<硕士<博士 排序。

    LLM 负责把任何学历表达（精确"硕士"、下限"本科及以上"、枚举"本科或硕士"）
    统一展开成它能接受的具体学历列表，后端只需一条 terms 过滤，不再区分
    精确/下限/枚举三种情况。
    """
    seen: set[str] = set()
    result: list[str] = []
    for item in value if isinstance(value, list) else []:
        normalized = _normalize_degree_value(str(item).strip())
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return sorted(result, key=DEGREE_ORDER.index)


def _load_school_tiers() -> tuple[dict[str, list[str]], set[str]]:
    """加载院校档位静态名单，返回 (各档位 key→校名列表, 所有名单校名并集)。

    名单是纯参考数据，进程内缓存一次。文件缺失/损坏时降级为空名单——
    院校筛选不可用，但不影响其余检索。
    """
    global _school_tiers_cache
    if _school_tiers_cache is not None:
        return _school_tiers_cache
    tiers: dict[str, list[str]] = {}
    known: set[str] = set()
    try:
        raw = json.loads(SCHOOL_TIERS_PATH.read_text(encoding="utf-8"))
        for key in SCHOOL_TIER_LABELS:
            names = [str(n).strip() for n in raw.get(key, []) if str(n).strip()]
            tiers[key] = sorted(set(names))
            known.update(names)
    except Exception:
        logger.exception("loading school_tiers.json failed; school-tier filter disabled")
        tiers, known = {}, set()
    _school_tiers_cache = (tiers, known)
    return _school_tiers_cache


def _normalize_school_tier(tier: str) -> str:
    cleaned = str(tier or "").strip()
    if not cleaned:
        return ""
    if cleaned in SCHOOL_TIER_ALIASES:
        return SCHOOL_TIER_ALIASES[cleaned]
    return cleaned if cleaned in SCHOOL_TIER_LABELS else ""


def _normalize_school_tier_list(value: Any) -> list[str]:
    """规范化"可接受院校档位集合"：去重、过滤非法值、保持稳定顺序。

    支持多选（OR）：用户在前端勾选多个档位，或 LLM 抽出 "985 或 留学生"
    这类并列需求时，后端按并集过滤。
    """
    if value is None:
        return []
    raw = value if isinstance(value, list) else [value]
    order = list(SCHOOL_TIER_LABELS.keys()) + [SCHOOL_TIER_OTHER]
    seen: set[str] = set()
    result: list[str] = []
    for item in raw:
        normalized = _normalize_school_tier(str(item).strip())
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return sorted(result, key=lambda t: order.index(t) if t in order else len(order))


def _school_tier_clause(tier: str) -> dict[str, Any] | None:
    """单个档位 → 一个 ES 查询子句。封闭档用 terms；"其他"用 must_not 补集。"""
    tiers, known = _load_school_tiers()
    if tier == SCHOOL_TIER_OTHER:
        if not known:
            return None  # 名单为空时不过滤，安全降级
        return {"bool": {"must_not": {"terms": {"candidate.all_schools.keyword": sorted(known)}}}}
    names = tiers.get(tier)
    if not names:
        return None
    return {"terms": {"candidate.all_schools.keyword": names}}


def _school_tier_filter(tiers: Any) -> dict[str, Any] | None:
    """档位（单个或多个）→ 一行 ES 过滤。多档位取并集（OR）。

    查 candidate.all_schools.keyword（候选人全部就读学校）→ 任一学历命中即算。
    封闭档（985/211/双一流/C9/海外QS50）合并成一条 terms；含"其他"时用
    bool.should（minimum_should_match=1）把补集与封闭档并起来。
    """
    normalized = _normalize_school_tier_list(tiers)
    if not normalized:
        return None
    tier_names, known = _load_school_tiers()
    closed_names: list[str] = []
    include_other = False
    for tier in normalized:
        if tier == SCHOOL_TIER_OTHER:
            include_other = True
        else:
            closed_names.extend(tier_names.get(tier, []))
    closed_names = sorted(set(closed_names))

    # 只有封闭档：一条 terms 即可。
    if not include_other:
        return {"terms": {"candidate.all_schools.keyword": closed_names}} if closed_names else None

    other_clause = _school_tier_clause(SCHOOL_TIER_OTHER)
    # 只有"其他"：直接返回补集。
    if not closed_names:
        return other_clause
    # 封闭档 OR 其他：should 取并集。
    shoulds = [{"terms": {"candidate.all_schools.keyword": closed_names}}]
    if other_clause:
        shoulds.append(other_clause)
    return {"bool": {"should": shoulds, "minimum_should_match": 1}}


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _dedupe_casefold(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        key = _casefold_key(cleaned)
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _casefold_key(value: str) -> str:
    return str(value).strip().casefold()


def _skill_filter(skill: str, skill_vocab: set[str] | None = None) -> dict[str, Any]:
    variants = _skill_variants(skill, skill_vocab)
    if len(variants) == 1:
        return {"term": {"skills": variants[0]}}
    return {"terms": {"skills": variants}}


def _skill_variants(skill: str, skill_vocab: set[str] | None = None) -> list[str]:
    cleaned = str(skill).strip()
    if not cleaned or not skill_vocab:
        return [cleaned]

    target_key = _casefold_key(cleaned)
    variants = [
        str(value).strip()
        for value in skill_vocab
        if str(value).strip() and _casefold_key(str(value)) == target_key
    ]
    if not variants:
        return [cleaned]

    return sorted(set(variants), key=lambda value: (_skill_label_sort_key(value), value))


def _skill_label_sort_key(value: str) -> tuple[int, int, str]:
    return (-_skill_label_score(value), len(value), value.casefold())


def _skill_label_score(value: str) -> int:
    stripped = value.strip()
    letters = [char for char in stripped if char.isalpha()]
    has_upper = any(char.isupper() for char in letters)
    has_lower = any(char.islower() for char in letters)
    if has_upper and has_lower:
        return 4 if any(char.isupper() for char in stripped[1:]) else 3
    if has_upper:
        return 2 if len(stripped) <= 3 else 1
    return 0


def _clean_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    raw_items = value if isinstance(value, list) else [value]
    result: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = str(item).strip()
        if not text:
            continue
        key = _casefold_key(text)
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _clean_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


