"""候选人文档富化：注入派生字段、清洗调试字段、估算工作年限、汇总就读院校。

在写入 ES 前对每份简历文档做规范化与派生计算。纯函数，无 IO。
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any

from indexing.mappings import OBSOLETE_VECTOR_FIELDS


def _enrich_doc(doc: dict[str, Any]) -> dict[str, Any]:
    for field in OBSOLETE_VECTOR_FIELDS:
        doc.pop(field, None)
    doc.pop("search_text", None)
    doc.pop("embedding", None)
    candidate = doc.setdefault("candidate", {})
    years_experience = _coerce_years_experience(candidate.get("years_experience"))
    if years_experience is None:
        years_experience = _estimate_years_experience(doc)
    if years_experience is None:
        candidate.pop("years_experience", None)
    else:
        candidate["years_experience"] = years_experience
    candidate["all_schools"] = _collect_all_schools(doc)
    doc["skills_text"] = " ".join(doc.get("skills") or [])
    # 清理无效的奖项记录（has_award 为否或无名称的）
    if "awards" in doc:
        doc["awards"] = [
            award for award in doc["awards"]
            if award.get("has_award") not in (None, "否", False) and award.get("name")
        ] or []
    # 清理无效的 offer_internship（所有关键字段都为空）
    offer = doc.get("offer_internship")
    if offer and not any(offer.get(k) for k in ("post_graduation_intention", "can_intern", "available_start_date", "weekly_workdays", "internship_period")):
        doc.pop("offer_internship", None)
    _drop_index_debug_fields(doc)
    return doc


def _collect_all_schools(doc: dict[str, Any]) -> list[str]:
    """候选人就读过的全部学校（最高学历校 + 各段教育的学校），去重保序。

    供院校档位筛选用：在 candidate.all_schools 上做 terms 过滤即可实现
    "任一学历命中即算"。
    """
    candidate = doc.get("candidate") or {}
    all_schools: list[str] = []
    if candidate.get("school"):
        name = str(candidate.get("school")).strip()
        if name:
            all_schools.append(name)
    for edu in doc.get("education") or []:
        if isinstance(edu, dict) and edu.get("school"):
            name = str(edu.get("school")).strip()
            if name and name not in all_schools:
                all_schools.append(name)
    return all_schools


def _coerce_years_experience(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        years = float(value)
    elif isinstance(value, str):
        match = re.search(r"\d+(?:\.\d+)?", value.strip())
        if not match:
            return None
        years = float(match.group(0))
    else:
        return None
    if years < 0:
        return None
    return round(years, 1)


def _estimate_years_experience(doc: dict[str, Any]) -> float | None:
    reference_end = _resume_reference_date(doc)
    spans = []
    for item in doc.get("internships", []):
        start = _parse_date(item.get("start_date"))
        end = _parse_date(item.get("end_date")) or reference_end
        if reference_end and end and end > reference_end:
            end = reference_end
        if start and end and end >= start:
            spans.append((start, end))

    if not spans:
        return None
    days = sum((end - start).days for start, end in _merge_spans(spans))
    return round(days / 365, 1)


def _resume_reference_date(doc: dict[str, Any]) -> date:
    application = doc.get("application") or {}
    file_meta = doc.get("file") or {}
    return (
        _parse_date(application.get("apply_time"))
        or _parse_date(file_meta.get("mtime"))
        or date.today()
    )


def _merge_spans(spans: list[tuple[date, date]]) -> list[tuple[date, date]]:
    merged: list[list[date]] = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
            continue
        if end > merged[-1][1]:
            merged[-1][1] = end
    return [(start, end) for start, end in merged]


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _drop_index_debug_fields(value: Any) -> Any:
    if isinstance(value, dict):
        value.pop("raw_fields", None)
        value.pop("raw_sections", None)
        for item in list(value.values()):
            _drop_index_debug_fields(item)
    elif isinstance(value, list):
        for item in value:
            _drop_index_debug_fields(item)
    return value


