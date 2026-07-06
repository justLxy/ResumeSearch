"""混合检索与 RRF 融合：并行跑证据 BM25 + kNN 向量两路，聚合到候选人维度融合排序。

自实现 RRF（ES Basic 许可证不含内置 RRF）。把切片级命中按 resume_id 聚合成候选人，
每个候选人只做一次 RRF；查询词覆盖（coverage）跨切片取并集，据此做相关性加权。
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from resume_search.config import (
    COVERAGE_QUERY_PREFIX,
    DENSE_RANK_WINDOW_SIZE,
    DENSE_RETRIEVER,
    DENSE_RRF_WEIGHT,
    EVIDENCE_DENSE_RETRIEVER,
    EVIDENCE_DENSE_RRF_WEIGHT,
    EVIDENCE_DENSE_RRF_WEIGHT_SEMANTIC,
    EVIDENCE_EXACT_QUERY_PREFIX,
    EVIDENCE_INDEX_ALIAS,
    EVIDENCE_PHRASE_QUERY_PREFIX,
    EVIDENCE_POOL_EXTRA_WEIGHTS,
    EVIDENCE_RETRIEVER,
    EVIDENCE_RRF_WEIGHT,
    EVIDENCE_RRF_WEIGHT_SEMANTIC,
    EVIDENCE_VECTOR_FIELD,
    INDEX_ALIAS,
    INTENT_SEMANTIC,
    RRF_RANK_CONSTANT,
    SOURCE_EXCLUDES,
)
from resume_search.infrastructure.es_client import es_request as _es
from resume_search.services.formatting import (
    _evidence_match_debug,
    _format_hit,
)
from resume_search.services.query_builder import (
    _coverage_tokens,
    _evidence_body,
    _evidence_knn_body,
    _is_dense_retriever,
)

logger = logging.getLogger(__name__)


def _rrf_route_weights(intent: str | None) -> tuple[float, float]:
    """按意图返回 (evidence_bm25_weight, dense_weight)。

    semantic（JD / 长自然语言 / 多技能组合）让 dense 语义召回主导——这类查询
    的有效信号在原文语义里，压缩成关键词的 BM25 价值低且可能引入噪声。其余意图
    （keyword 实体、lookup 精确）保持 BM25 主导，尊重精确匹配。
    """
    if intent == INTENT_SEMANTIC:
        return EVIDENCE_RRF_WEIGHT_SEMANTIC, EVIDENCE_DENSE_RRF_WEIGHT_SEMANTIC
    return EVIDENCE_RRF_WEIGHT, DENSE_RRF_WEIGHT


def _run_hybrid_search(
    query_text: str,
    query_vector: list[float],
    filters: list[dict[str, Any]],
    rank_window_size: int,
    *,
    use_dense: bool,
    query_intent: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    requests_to_run = [
        (
            EVIDENCE_RETRIEVER,
            EVIDENCE_RRF_WEIGHT,
            _evidence_body(query_text, filters, rank_window_size, query_intent=query_intent),
            None,
        ),
    ]
    dense_size = min(rank_window_size, DENSE_RANK_WINDOW_SIZE)
    if use_dense and query_vector:
        requests_to_run.append(
            (
                EVIDENCE_DENSE_RETRIEVER,
                EVIDENCE_DENSE_RRF_WEIGHT,
                _evidence_knn_body(query_vector, filters, dense_size),
                EVIDENCE_VECTOR_FIELD,
            )
        )

    responses: list[dict[str, Any]] = []
    warnings: list[str] = []
    with ThreadPoolExecutor(max_workers=len(requests_to_run)) as executor:
        futures = {
            executor.submit(
                _es,
                "POST",
                f"/{_retriever_index_alias(name)}/_search",
                body,
            ): (name, weight, field)
            for name, weight, body, field in requests_to_run
        }
        for future in as_completed(futures):
            name, weight, field = futures[future]
            try:
                response = future.result()
            except Exception as exc:
                # 单路检索失败不应让整个搜索崩溃；用成功的那一路（或几路）降级返回。
                logger.exception("%s retriever failed", name)
                warnings.append(f"{name} retriever failed: {exc}")
                continue
            response["_retriever_name"] = name
            response["_rrf_weight"] = weight
            if field:
                response["_vector_field"] = field
            responses.append(response)
    return responses, warnings


def _retriever_index_alias(retriever_name: str) -> str:
    if retriever_name in {EVIDENCE_RETRIEVER, EVIDENCE_DENSE_RETRIEVER}:
        return EVIDENCE_INDEX_ALIAS
    return INDEX_ALIAS


def _is_evidence_retriever(name: Any) -> bool:
    return name in {EVIDENCE_RETRIEVER, EVIDENCE_DENSE_RETRIEVER}


def _hit_resume_id(hit: dict[str, Any], response: dict[str, Any] | None = None) -> str:
    retriever_name = response.get("_retriever_name") if response else None
    if _is_evidence_retriever(retriever_name):
        source = hit.get("_source") or {}
        return str(source.get("resume_id") or hit.get("_id"))
    return str(hit.get("_id"))


def _is_evidence_hit(hit: dict[str, Any] | None) -> bool:
    if not hit:
        return False
    source = hit.get("_source") or {}
    return bool(source.get("evidence_id") or source.get("section_type"))


def _hybrid_total(
    responses: list[dict[str, Any]],
) -> int:
    ids: set[str] = set()
    for response in responses:
        for _rank, hit, _debug in _accepted_hits(response):
            ids.add(_hit_resume_id(hit, response))
    return len(ids)


def _lexical_total(responses: list[dict[str, Any]]) -> int:
    lexical_responses = [
        response
        for response in responses
        if response.get("_retriever_name") == EVIDENCE_RETRIEVER
    ]
    if not lexical_responses:
        return 0
    ids: set[str] = set()
    for response in lexical_responses:
        for hit in response.get("hits", {}).get("hits", []):
            ids.add(_hit_resume_id(hit, response))
    return len(ids)


def _rrf_merge(
    responses: list[dict[str, Any]],
    limit: int,
    query_text: str = "",
    *,
    evidence_weight: float = EVIDENCE_RRF_WEIGHT,
    dense_weight: float = DENSE_RRF_WEIGHT,
) -> list[dict[str, Any]]:
    rrf_scores: dict[str, float] = {}
    dense_route_ranks: dict[str, list[int]] = {}
    dense_best_route_rank: dict[str, int] = {}
    evidence_route_ranks: dict[str, list[int]] = {}
    hit_map: dict[str, dict[str, Any]] = {}
    best_rank: dict[str, int] = {}
    # 跨切片并集：候选人命中的不同 coverage token 名集合。用 max(单切片命中数)
    # 会把"腾讯经历"和"阿里经历"分属两切片、各命中 1 词的全命中候选人低估成
    # coverage=1，与只有一段经历者无法区分。取并集后全命中者 coverage=2。
    term_coverage_names: dict[str, set[str]] = {}
    lexical_tier: dict[str, int] = {}
    retrieval_debug: dict[str, dict[str, Any]] = {}
    coverage_enabled = len(_coverage_tokens(query_text)) >= 2
    evidence_best_route_rank: dict[str, int] = {}

    for response in responses:
        retriever_name = response.get("_retriever_name")
        weight = float(response.get("_rrf_weight", 1.0))
        is_dense = _is_dense_retriever(retriever_name)
        is_evidence = _is_evidence_retriever(retriever_name)
        for rank, hit, dense_debug in _accepted_hits(response):
            doc_id = _hit_resume_id(hit, response)
            route_contribution = weight / (RRF_RANK_CONSTANT + rank)
            if is_dense:
                dense_route_ranks.setdefault(doc_id, []).append(rank)
                dense_best_route_rank[doc_id] = min(dense_best_route_rank.get(doc_id, rank), rank)
            elif is_evidence:
                evidence_route_ranks.setdefault(doc_id, []).append(rank)
                evidence_best_route_rank[doc_id] = min(evidence_best_route_rank.get(doc_id, rank), rank)
            if coverage_enabled and retriever_name == EVIDENCE_RETRIEVER:
                term_coverage_names.setdefault(doc_id, set()).update(
                    _matched_coverage_names(hit)
                )
            if is_evidence:
                hit_map.setdefault(doc_id, hit)
            elif doc_id not in hit_map or hit.get("highlight") or _is_evidence_hit(hit_map.get(doc_id)):
                hit_map[doc_id] = hit
            debug = retrieval_debug.setdefault(
                doc_id,
                {
                    "retrieval_sources": [],
                    "dense_rank": None,
                },
            )
            if not is_dense and retriever_name not in debug["retrieval_sources"]:
                debug["retrieval_sources"].append(retriever_name)
            if is_evidence:
                if not is_dense:
                    if EVIDENCE_RETRIEVER not in debug["retrieval_sources"]:
                        debug["retrieval_sources"].append(EVIDENCE_RETRIEVER)
                    debug["evidence_weight"] = round(weight, 3)
                    debug["matched_queries"] = _merge_matched_queries(
                        debug.get("matched_queries") or [],
                        hit.get("matched_queries") or [],
                    )
                    lexical_tier[doc_id] = max(
                        lexical_tier.get(doc_id, 0),
                        _matched_lexical_tier(hit),
                    )
                    debug["evidence_rank"] = min(debug.get("evidence_rank") or rank, rank)
                    debug["evidence_score"] = max(
                        float(debug.get("evidence_score") or 0),
                        float(hit.get("_score") or 0),
                    )
                    matches = debug.setdefault("evidence_matches", [])
                    if len(matches) < 3:
                        matches.append(_evidence_match_debug(hit, retriever_name, rank))
            if is_dense:
                dense_match = {
                    "retriever": retriever_name,
                    "field": dense_debug.get("dense_field"),
                    "rank": rank,
                    "score": dense_debug.get("dense_score"),
                    "weight": dense_debug.get("dense_weight"),
                    "contribution": round(route_contribution, 6),
                }
                if is_evidence:
                    dense_match.update(_evidence_match_debug(hit, retriever_name, rank))
                matches = debug.setdefault("dense_matches", [])
                if len(matches) < 3:
                    matches.append(dense_match)

    # --- 将证据 BM25 聚合到候选人维度，每个候选人只做一次 RRF。 ---
    evidence_pools = {
        doc_id: _evidence_pool_score(ranks)
        for doc_id, ranks in evidence_route_ranks.items()
    }
    evidence_group_ids = sorted(
        evidence_pools.keys(),
        key=lambda doc_id: (
            -float(evidence_pools[doc_id]["score"]),
            evidence_best_route_rank[doc_id],
            doc_id,
        ),
    )
    evidence_group_rank = {
        doc_id: rank
        for rank, doc_id in enumerate(evidence_group_ids, start=1)
    }
    for doc_id in evidence_group_ids:
        ev_rank = evidence_group_rank[doc_id]
        evidence_contribution = evidence_weight / (RRF_RANK_CONSTANT + ev_rank)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + evidence_contribution
        best_rank[doc_id] = min(best_rank.get(doc_id, ev_rank), ev_rank)
        debug = retrieval_debug.get(doc_id)
        if debug:
            debug["evidence_group_rank"] = ev_rank
            debug["evidence_inner_score"] = round(float(evidence_pools[doc_id]["score"]), 6)
            debug["evidence_support_count"] = evidence_pools[doc_id]["support_count"]
            debug["evidence_rrf_contribution"] = round(evidence_contribution, 6)
            # 用实际参与计分的意图感知权重覆盖逐响应的 _rrf_weight，
            # 否则 debug 面板会显示陈旧的 1.2 而与贡献数学不符。
            debug["evidence_weight"] = round(evidence_weight, 3)

    # --- 将 dense 聚合到候选人维度，每个候选人只做一次 RRF。 ---
    dense_pools = {
        doc_id: _evidence_pool_score(ranks)
        for doc_id, ranks in dense_route_ranks.items()
    }
    dense_group_ids = sorted(
        dense_pools.keys(),
        key=lambda doc_id: (
            -float(dense_pools[doc_id]["score"]),
            dense_best_route_rank[doc_id],
            doc_id,
        ),
    )
    all_dense_group_rank = {
        doc_id: rank
        for rank, doc_id in enumerate(dense_group_ids, start=1)
    }
    for doc_id in dense_group_ids:
        dense_rank = all_dense_group_rank.get(doc_id, dense_best_route_rank.get(doc_id, 10**9))
        dense_pool = dense_pools[doc_id]
        dense_contribution = dense_weight / (RRF_RANK_CONSTANT + dense_rank)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + dense_contribution
        best_rank[doc_id] = min(best_rank.get(doc_id, dense_rank), dense_rank)
        debug = retrieval_debug.setdefault(
            doc_id,
            {
                "retrieval_sources": [],
                "dense_rank": None,
            },
        )
        if DENSE_RETRIEVER not in debug["retrieval_sources"]:
            debug["retrieval_sources"].append(DENSE_RETRIEVER)
        dense_matches = sorted(
            debug.get("dense_matches") or [],
            key=lambda match: (
                match.get("rank") or 10**9,
                -float(match.get("score") or 0),
            ),
        )
        debug["dense_matches"] = dense_matches
        best_dense_match = min(
            dense_matches,
            key=lambda match: (
                match.get("rank") or 10**9,
                -float(match.get("score") or 0),
            ),
        ) if dense_matches else {}
        debug["dense_rank"] = dense_rank
        debug["dense_group_rank"] = dense_rank
        debug["dense_inner_score"] = round(float(dense_pool["score"]), 6)
        debug["dense_outer_weight"] = round(dense_weight, 3)
        debug["dense_support_count"] = dense_pool["support_count"]
        debug["dense_pooling"] = "top_k_route_rerank"
        debug["dense_rrf_contribution"] = round(dense_contribution, 6)
        debug["dense_route_rank"] = best_dense_match.get("rank")
        debug["dense_score"] = best_dense_match.get("score")
        debug["dense_field"] = best_dense_match.get("field")
        debug["dense_retriever"] = best_dense_match.get("retriever")

    # 命中的不同 token 名集合大小 = 该候选人跨全部切片命中的关键词数。
    term_coverage = {
        doc_id: len(names) for doc_id, names in term_coverage_names.items()
    }

    final_scores: dict[str, float] = {}
    score_multipliers: dict[str, float] = {}
    for doc_id in rrf_scores:
        tier = lexical_tier.get(doc_id, 0)
        coverage = term_coverage.get(doc_id, 0)
        multiplier = 1.0 + (0.15 * tier) + (0.05 * coverage)
        score_multipliers[doc_id] = round(multiplier, 2)
        final_scores[doc_id] = rrf_scores[doc_id] * multiplier

    sorted_ids = sorted(
        final_scores.keys(),
        key=lambda k: (
            -final_scores[k],
            best_rank.get(k, 10**9),
            k,
        ),
    )[:limit]
    fetched_resume_hits = _fetch_resume_hits_for_evidence(sorted_ids, hit_map)

    results = []
    for doc_id in sorted_ids:
        hit = dict(fetched_resume_hits.get(doc_id) or hit_map[doc_id])
        hit["_retrieval_debug"] = {
            **retrieval_debug.get(doc_id, {}),
            "raw_rrf_score": round(rrf_scores[doc_id], 6),
            "score_multiplier": score_multipliers[doc_id],
            "rrf_score": round(final_scores[doc_id], 6),
            "lexical_tier": lexical_tier.get(doc_id, 0),
        }
        if coverage_enabled:
            hit["_retrieval_debug"]["term_coverage"] = term_coverage.get(doc_id, 0)
        results.append(_format_hit(hit, final_scores[doc_id]))
    return results


def _evidence_pool_score(route_ranks: list[int]) -> dict[str, float | int]:
    sorted_route_ranks = sorted(rank for rank in route_ranks if rank > 0)
    weights = (1.0, *EVIDENCE_POOL_EXTRA_WEIGHTS)
    contributions = [
        weight / (RRF_RANK_CONSTANT + route_rank)
        for weight, route_rank in zip(weights, sorted_route_ranks)
    ]
    return {
        "score": sum(contributions),
        "support_count": len(contributions),
    }


def _fetch_resume_hits_for_evidence(
    doc_ids: list[str],
    hit_map: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    missing_ids = [
        doc_id
        for doc_id in doc_ids
        if _is_evidence_hit(hit_map.get(doc_id))
    ]
    if not missing_ids:
        return {}

    body = {
        "size": len(missing_ids),
        "query": {"ids": {"values": missing_ids}},
        "_source": {"excludes": SOURCE_EXCLUDES},
    }
    try:
        result = _es("POST", f"/{INDEX_ALIAS}/_search", body)
    except Exception:
        logger.exception("fetching parent resumes for evidence hits failed")
        return {}
    return {
        hit["_id"]: hit
        for hit in result.get("hits", {}).get("hits", [])
    }


def _matched_coverage_names(hit: dict[str, Any]) -> set[str]:
    """本切片命中的 coverage query 名集合（形如 "query_term:0"）。

    每个 coverage 子句对应查询里的一个 token（见 _evidence_term_coverage_queries）。
    在候选人维度对多个切片的名集合取并集，即得该候选人跨全部经历命中的关键词数——
    这是"腾讯 阿里巴巴"两段经历分属两切片时，仍能算出 coverage=2 的关键。
    """
    matched_queries = hit.get("matched_queries") or []
    return {
        query_name
        for query_name in matched_queries
        if isinstance(query_name, str) and query_name.startswith(COVERAGE_QUERY_PREFIX)
    }


def _matched_term_coverage(hit: dict[str, Any]) -> int:
    return len(_matched_coverage_names(hit))


def _merge_matched_queries(existing: list[Any], incoming: list[Any]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for query_name in [*existing, *incoming]:
        if not isinstance(query_name, str) or query_name in seen:
            continue
        merged.append(query_name)
        seen.add(query_name)
    return merged


def _matched_lexical_tier(hit: dict[str, Any]) -> int:
    matched_queries = hit.get("matched_queries") or []
    if any(
        isinstance(query_name, str) and query_name.startswith(EVIDENCE_EXACT_QUERY_PREFIX)
        for query_name in matched_queries
    ):
        return 3
    if any(
        isinstance(query_name, str) and query_name.startswith(EVIDENCE_PHRASE_QUERY_PREFIX)
        for query_name in matched_queries
    ):
        return 2
    return 1 if matched_queries else 0


def _accepted_hits(response: dict[str, Any]) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    hits = response.get("hits", {}).get("hits", [])
    if not _is_dense_retriever(response.get("_retriever_name")):
        return [(rank, hit, {}) for rank, hit in enumerate(hits, start=1)]

    accepted: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for rank, hit in enumerate(hits, start=1):
        raw_score = float(hit.get("_score") or 0)
        dense_debug = {
            "dense_score": round(raw_score, 6),
            "dense_field": response.get("_vector_field"),
            "dense_weight": round(float(response.get("_rrf_weight", 1.0)), 3),
        }
        accepted.append((rank, hit, dense_debug))
    return accepted


