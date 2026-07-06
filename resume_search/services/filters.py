"""结构化过滤条件构建：学历/城市/技能/年限/院校档位 → ES filter 子句。

把用户显式筛选（前端参数）和 LLM 抽取的约束都翻译成 ES bool.filter 子句。
规范化细节委托给 normalization。这里只负责"约束 → 查询结构"的映射。
"""
from __future__ import annotations

from typing import Any

from resume_search.config import (
    MIN_YEARS_TOLERANCE_FLOOR,
    MIN_YEARS_TOLERANCE_RATIO,
)
from resume_search.services.normalization import (
    _clean_float,
    _clean_string_list,
    _dedupe,
    _dedupe_casefold,
    _normalize_degree_list,
    _normalize_highest_degree,
    _school_tier_filter,
    _skill_filter,
)


def _build_filters(
    degree: str,
    cities: list[str],
    skills: list[str],
    min_years: float,
    skill_vocab: set[str] | None = None,
    school_tiers: Any = None,
) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []
    if degree:
        highest_degree = _normalize_highest_degree(degree)
        filters.append({"term": {"candidate.highest_degree": highest_degree}})
    if cities:
        filters.append({"terms": {"application.expected_work_cities": _dedupe(cities)}})
    if skills:
        for skill in _dedupe_casefold(skills):
            filters.append(_skill_filter(skill, skill_vocab))
    if min_years > 0:
        filters.append(_min_years_filter(min_years))
    school_filter = _school_tier_filter(school_tiers)
    if school_filter:
        filters.append(school_filter)
    return filters


def _min_years_filter(min_years: float) -> dict[str, Any]:
    """Range filter on years_experience with a soft tolerance band.

    "N 年以上" is a fuzzy ask, not a hard boundary. We lower the gte by the
    larger of a ratio (10%) or a flat floor (0.5y) so near-misses like a 3.9y
    candidate still surface for a "4 年以上" query; true seniority is preserved
    by ranking, not by clipping the candidate set.

    Candidates whose years_experience is *unknown* (field absent) are NOT
    excluded: a plain range filter silently drops docs missing the field, and
    in this corpus a large share of resumes have no parsed experience value.
    Unknown != disqualified, so we OR in a "field missing" branch and let
    ranking sort out the rest. Otherwise a single "N 年以上" line in a JD
    collapses recall to the handful of resumes that happen to have the field.
    """
    tolerance = max(min_years * MIN_YEARS_TOLERANCE_RATIO, MIN_YEARS_TOLERANCE_FLOOR)
    effective_floor = max(0.0, round(min_years - tolerance, 3))
    return {
        "bool": {
            "should": [
                {"range": {"candidate.years_experience": {"gte": effective_floor}}},
                {"bool": {"must_not": {"exists": {"field": "candidate.years_experience"}}}},
            ],
            "minimum_should_match": 1,
        }
    }


def _filters_from_llm_constraints(constraints: dict[str, Any]) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []
    degrees = _normalize_degree_list(constraints.get("degrees"))
    if degrees:
        # 可接受学历集合 → 一条 terms 过滤（本科/硕士/博士 任意子集）
        filters.append({"terms": {"candidate.highest_degree": degrees}})
    cities = _clean_string_list(constraints.get("cities"))
    if cities:
        filters.append({"terms": {"application.expected_work_cities": _dedupe(cities)}})
    min_years = _clean_float(constraints.get("min_years"))
    if min_years is not None and min_years > 0:
        filters.append(_min_years_filter(min_years))
    school_filter = _school_tier_filter(
        constraints.get("school_tiers") or constraints.get("school_tier")
    )
    if school_filter:
        filters.append(school_filter)
    return filters


