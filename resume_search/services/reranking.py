"""Cross-encoder 精排：对 RRF 头部窗口做 query-doc 相关性重排。

调用 rerank_service 打分，低于相关性地板时不弃权、只打低相关度标记（判断权交给 HR）。
文档拼装复用 formatting 的清洗工具。
"""
from __future__ import annotations

import logging
from typing import Any

from resume_search.config import (
    ENABLE_RERANK,
    RERANK_RELEVANCE_FLOOR,
    RERANK_TOP_N,
)
from resume_search.services.formatting import _append_doc_line, _clean_doc_text
from resume_search.services.quality import score_bucket

logger = logging.getLogger(__name__)


def _rerank_results(
    query_text: str,
    results: list[dict[str, Any]],
    *,
    top_n: int = RERANK_TOP_N,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    if not ENABLE_RERANK or not query_text.strip() or len(results) < 2 or top_n <= 0:
        return results, []

    window_size = min(top_n, len(results))
    window = results[:window_size]
    documents = [_rerank_document(result) for result in window]
    try:
        scores = _score_rerank_documents(query_text, documents)
    except Exception as exc:
        logger.exception("reranker failed")
        return results, [f"reranker failed: {exc}"]

    if len(scores) != len(window):
        return results, [
            f"reranker returned {len(scores)} scores for {len(window)} candidates"
        ]

    # 低相关度提示（不弃权）：即便最相关的候选分数也很低，通常意味着库中没有
    # 与此 query 精确匹配的人。我们【不】清空结果——HR 只看排名靠前的，把判断权
    # 交还给他，由 UI 据此 warning 给出"低相关度，库中可能无精确匹配"的提示，
    # 而不是由系统替他做"全有或全无"的决定。曾经的硬地板会把分数仅 0.489 的真·
    # 相关 query 误判成空，弊大于利。
    top_rerank_score = max(float(score) for score in scores)
    if top_rerank_score < RERANK_RELEVANCE_FLOOR:
        warnings.append(
            f"low relevance: top relevance {top_rerank_score:.3f} "
            f"below floor {RERANK_RELEVANCE_FLOOR}; results may not precisely match"
        )

    scored: list[tuple[float, float, int, dict[str, Any]]] = []
    for pre_rank, (result, score) in enumerate(zip(window, scores), start=1):
        rerank_score = float(score)
        item = dict(result)
        quality_score = float(item.get("quality_score") or 0.0)
        debug = dict(item.get("retrieval_debug") or {})
        debug.update(
            {
                "rerank_model": "qwen3-rerank-api",
                "rerank_applied": True,
                "rerank_window_size": window_size,
                "rerank_score": round(rerank_score, 6),
                "pre_rerank_rank": pre_rank,
                "pre_rerank_score": item.get("score"),
            }
        )
        item["retrieval_debug"] = debug
        item["score"] = round(rerank_score, 4)
        # 逐候选人低相关度标记：rerank 分数低于门槛 → 前端在姓名旁打 tag，
        # 提示"该候选人与查询的相关度较低"。判断权交给 HR，不再清空。
        item["low_relevance"] = rerank_score < RERANK_RELEVANCE_FLOOR
        scored.append((rerank_score, quality_score, pre_rank, item))

    # 排序键：rerank 分分桶降序 → 同桶（相关性实质相同）内质量降序 → 原 RRF 名次。
    # 质量只在 cross-encoder 分量化相等时才起作用，不覆盖相关性判断。
    scored.sort(key=lambda row: (-score_bucket(row[0]), -row[1], row[2]))
    reranked_window: list[dict[str, Any]] = []
    for rerank_rank, (_score, _quality, _pre_rank, item) in enumerate(scored, start=1):
        debug = dict(item.get("retrieval_debug") or {})
        debug["rerank_rank"] = rerank_rank
        item["retrieval_debug"] = debug
        reranked_window.append(item)

    tail: list[dict[str, Any]] = []
    for pre_rank, result in enumerate(results[window_size:], start=window_size + 1):
        item = dict(result)
        debug = dict(item.get("retrieval_debug") or {})
        debug.update(
            {
                "rerank_model": "qwen3-rerank-api",
                "rerank_applied": False,
                "rerank_skip_reason": "outside_top_n",
                "rerank_window_size": window_size,
                "pre_rerank_rank": pre_rank,
            }
        )
        item["retrieval_debug"] = debug
        tail.append(item)
    return [*reranked_window, *tail], warnings


def _score_rerank_documents(query_text: str, documents: list[str]) -> list[float]:
    from resume_search.infrastructure.rerank_service import score_pairs

    return score_pairs(query_text, documents)


def _rerank_document(result: dict[str, Any]) -> str:
    source = result.get("source") or {}
    candidate = source.get("candidate") or result.get("candidate") or {}
    application = source.get("application") or result.get("application") or {}
    lines: list[str] = []

    _append_doc_line(lines, "应聘岗位", application.get("position_name"))
    background = " ".join(
        _clean_doc_text(candidate.get(field))
        for field in ("highest_degree", "school", "major")
        if candidate.get(field)
    )
    _append_doc_line(lines, "候选人背景", background)
    _append_doc_line(lines, "技能", "、".join(source.get("skills") or result.get("skills") or []))

    lang = source.get("languages") or {}
    lang_parts = []
    if lang.get("english_exam_score"):
        lang_parts.append(f"英语等级考试成绩 {lang["english_exam_score"]}")
    if lang.get("english_spoken_level"):
        lang_parts.append(f"英语口语水平 {lang['english_spoken_level']}")
    _append_doc_line(lines, "语言能力", "，".join(lang_parts))

    for award in source.get("awards") or []:
        if award.get("has_award") not in (None, "否", False) and award.get("name"):
            text = " ".join(
                _clean_doc_text(award.get(field))
                for field in ("name", "level", "description")
                if award.get(field)
            )
            _append_doc_line(lines, "获奖经历", text)

    offer = source.get("offer_internship") or {}
    offer_parts = []
    if offer.get("can_intern"):
        offer_parts.append(f"可实习")
    if offer.get("available_start_date"):
        offer_parts.append(f"到岗{offer['available_start_date']}")
    if offer.get("weekly_workdays"):
        offer_parts.append(f"每周{offer['weekly_workdays']}天")
    if offer.get("internship_period"):
        offer_parts.append(f"周期{offer['internship_period']}")
    if offer.get("post_graduation_intention"):
        offer_parts.append(offer["post_graduation_intention"])
    _append_doc_line(lines, "实习与入职意向", "，".join(offer_parts))

    for edu in source.get("education") or []:
        text = " ".join(
            _clean_doc_text(edu.get(field))
            for field in ("school", "college", "degree", "major", "research_direction", "lab_name")
            if edu.get(field)
        )
        _append_doc_line(lines, "教育经历", text)

    for project in source.get("projects") or []:
        text = " ".join(
            _clean_doc_text(project.get(field))
            for field in ("name", "description", "responsibility")
            if project.get(field)
        )
        _append_doc_line(lines, "项目", text)
    for internship in source.get("internships") or []:
        text = " ".join(
            _clean_doc_text(internship.get(field))
            for field in ("company", "department", "title", "description")
            if internship.get(field)
        )
        _append_doc_line(lines, "经历", text)

    return "\n".join(line for line in lines if line.strip())


