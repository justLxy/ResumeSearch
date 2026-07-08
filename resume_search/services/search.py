"""检索编排：组合规划 → 过滤 → 混合检索 → RRF 融合 → 精排 → 格式化的顶层入口。

`search()` 是整个检索链路的编排者，被 api 层直接调用。它自身不含检索算法细节，
只负责按 QueryPlan 把各 service 串起来，并处理分页、浏览/筛选降级、计数与告警。
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import Query

from resume_search.config import (
    DEFAULT_SEARCH_LIMIT,
    INDEX_ALIAS,
    MAX_BROWSE_RESULT_SIZE,
    RRF_RANK_WINDOW_SIZE,
    SOURCE_EXCLUDES,
)
from resume_search.infrastructure.es_client import es_request as _es
from resume_search.infrastructure.embedding_service import encode_single
from resume_search.services.facets import _load_facets, _load_filter_vocab
from resume_search.services.filters import _build_filters
from resume_search.services.formatting import _format_hit
from resume_search.services.query_builder import _filter_browse_body
from resume_search.services.query_planning import _plan_query
from resume_search.services.reranking import _rerank_results
from resume_search.services.retrieval import (
    _hybrid_total,
    _lexical_total,
    _rrf_merge,
    _rrf_route_weights,
    _run_hybrid_search,
)

logger = logging.getLogger(__name__)


def search(
    q: str = "",
    degree: str = "",
    cities: list[str] = Query(default=[]),
    skills: list[str] = Query(default=[]),
    min_years: float = 0,
    school_tiers: list[str] = Query(default=[]),
    limit: int = 0,
    offset: int = 0,
) -> dict[str, Any]:
    raw_query_text = q.strip()
    page_size = _normalize_limit(limit)
    page_offset = _normalize_offset(offset)
    result_window_size = RRF_RANK_WINDOW_SIZE
    facets = _load_facets()
    skill_vocab = _load_filter_vocab()["skills"] if skills else None
    explicit_filters = _build_filters(
        degree, cities, skills, min_years, skill_vocab=skill_vocab, school_tiers=school_tiers
    )
    plan = _plan_query(raw_query_text, explicit_filters, size=result_window_size, facets=facets)
    query_text = plan.lexical_query
    filters = plan.filters
    # 意图感知的 RRF 路由权重：semantic（JD/长文本/多技能）让 dense 主导，
    # 其余意图保持 BM25 主导。
    evidence_weight, dense_weight = _rrf_route_weights(plan.intent)
    retrieval_warnings: list[str] = []

    if query_text:
        # 证据优先检索：先检索证据切片，再按 resume_id 聚合回候选人维度。
        use_dense = plan.enable_dense
        query_vector: list[float] = []
        if use_dense:
            try:
                query_vector = encode_single(plan.semantic_query)
            except Exception as exc:
                logger.exception("query embedding failed")
                retrieval_warnings.append(f"dense embedding failed: {exc}")
                use_dense = False
        responses, retriever_warnings = _run_hybrid_search(
            query_text,
            query_vector,
            filters,
            plan.rank_window_size,
            use_dense=use_dense,
            query_intent=plan.intent,
        )
        retrieval_warnings.extend(retriever_warnings)
        matched_total = _lexical_total(responses)
        candidate_total = _hybrid_total(responses)
        results = _rrf_merge(
            responses,
            result_window_size,
            query_text=query_text,
            evidence_weight=evidence_weight,
            dense_weight=dense_weight,
        )
        if plan.enable_rerank:
            results, rerank_warnings = _rerank_results(plan.semantic_query, results)
            retrieval_warnings.extend(rerank_warnings)
            # 结果为空时（重排不再弃权，但 RRF 本身可能无命中），让上报的计数
            # 与空结果集保持一致。
            if not results:
                matched_total = 0
                candidate_total = 0
    elif filters:
        browse_size = MAX_BROWSE_RESULT_SIZE
        body = _filter_browse_body(filters, browse_size)
        # ES 侧按投递时间降序拉取，Python 侧再按质量分重排——浏览模式无相关性可言
        # （score 全 0），用户是在"逛"候选人库，优质候选人应排在前。投递时间序被
        # 保留为同质量者的 stable tie-break（越新越靠前）。
        body["sort"] = [
            {"application.apply_time": {"order": "desc", "unmapped_type": "date"}},
            {"resume_id": {"order": "asc"}},
        ]
        es_result = _es("POST", f"/{INDEX_ALIAS}/_search", body)
        matched_total = es_result.get("hits", {}).get("total", {}).get("value", 0)
        candidate_total = matched_total
        results = _sort_browse_by_quality(
            [_format_hit(hit) for hit in es_result.get("hits", {}).get("hits", [])]
        )
    else:
        browse_size = MAX_BROWSE_RESULT_SIZE
        body = {
            "size": browse_size,
            "query": {"match_all": {}},
            "sort": [
                {"application.apply_time": {"order": "desc", "unmapped_type": "date"}},
                {"resume_id": {"order": "asc"}},
            ],
            "_source": {"excludes": SOURCE_EXCLUDES},
        }
        es_result = _es("POST", f"/{INDEX_ALIAS}/_search", body)
        matched_total = es_result.get("hits", {}).get("total", {}).get("value", 0)
        candidate_total = matched_total
        results = _sort_browse_by_quality(
            [_format_hit(hit) for hit in es_result.get("hits", {}).get("hits", [])]
        )

    available_count = len(results)
    paged_results = results[page_offset : page_offset + page_size]
    next_offset = page_offset + len(paged_results)
    has_more = next_offset < available_count

    return {
        "query": q,
        "effective_query": query_text,
        "parsed_constraints": plan.constraints,
        "query_plan": plan.to_debug_dict(),
        "total": available_count,
        "returned_count": len(paged_results),
        "offset": page_offset,
        "limit": page_size,
        "result_window_size": result_window_size,
        "has_more": has_more,
        "next_offset": next_offset if has_more else None,
        "matched_total": matched_total,
        "candidate_total": candidate_total,
        "retrieval_warnings": retrieval_warnings,
        "results": paged_results,
        "facets": facets,
    }


def _sort_browse_by_quality(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """浏览模式按候选人质量分降序【稳定】重排。

    ES 已按 apply_time 降序返回，稳定排序使同质量分的候选人保持投递时间序
    （越新越靠前）。质量分由 _format_hit 挂在结果上，此处直接读，不重复计算。
    """
    results.sort(key=lambda r: -float(r.get("quality_score") or 0.0))
    return results


def _normalize_limit(limit: int | None) -> int:
    if limit is None or limit <= 0:
        return DEFAULT_SEARCH_LIMIT
    return max(1, min(limit, MAX_BROWSE_RESULT_SIZE))


def _normalize_offset(offset: int | None) -> int:
    if offset is None or offset <= 0:
        return 0
    return min(offset, MAX_BROWSE_RESULT_SIZE)


