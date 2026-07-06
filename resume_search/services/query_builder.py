"""Elasticsearch 查询 DSL 构建：把查询文本/意图翻译成证据检索的 bool/dis_max/knn 结构。

这是检索的"词面与向量召回"核心。包含证据切片检索体、实体字段匹配、查询词覆盖
（coverage）子句、多关键词 OR 召回、kNN 向量检索体等。纯函数，不触碰 IO。
"""
from __future__ import annotations

import re
from typing import Any

from resume_search.config import (
    COVERAGE_QUERY_PREFIX,
    EVIDENCE_DENSE_RETRIEVER,
    EVIDENCE_SOURCE_EXCLUDES,
    EVIDENCE_VECTOR_FIELD,
    INTENT_KEYWORD,
    INTENT_LOOKUP,
    KNN_NUM_CANDIDATES,
    MAX_QUERY_COVERAGE_TERMS,
    PARTIAL_TERMS_MINIMUM_SHOULD_MATCH,
    QUERY_TERM_COVERAGE_BOOST,
    SOURCE_EXCLUDES,
)
from resume_search.services.normalization import _normalize_highest_degree


def _query_tokens(query_text: str) -> list[str]:
    return [
        token
        for token in re.split(r"[\s,，、;/；]+", query_text.strip())
        if token
    ]


def _coverage_tokens(query_text: str) -> list[str]:
    seen: set[str] = set()
    tokens: list[str] = []
    for token in _query_tokens(query_text):
        key = token.casefold()
        if key in seen:
            continue
        seen.add(key)
        tokens.append(token)
        if len(tokens) >= MAX_QUERY_COVERAGE_TERMS:
            break
    return tokens


def _filter_browse_body(filters: list[dict[str, Any]], size: int) -> dict[str, Any]:
    return {
        "size": size,
        "query": {"bool": {"must": [{"match_all": {}}], "filter": filters}},
        "_source": {
            "excludes": SOURCE_EXCLUDES,
        },
    }


def _evidence_body(
    query_text: str,
    filters: list[dict[str, Any]],
    size: int,
    *,
    query_intent: str | None = None,
) -> dict[str, Any]:
    return {
        "size": size,
        "query": {
            "bool": {
                "must": [_evidence_lexical_query(query_text, query_intent=query_intent)],
                "filter": filters,
            }
        },
        "highlight": {
            "pre_tags": ["<mark>"],
            "post_tags": ["</mark>"],
            "fields": {
                "title": {"fragment_size": 80, "number_of_fragments": 1},
                "text": {"fragment_size": 80, "number_of_fragments": 1},
                "skills_text": {"fragment_size": 80, "number_of_fragments": 1},
                "candidate.name": {"fragment_size": 80, "number_of_fragments": 1},
                "candidate.school": {"fragment_size": 80, "number_of_fragments": 1},
                "candidate.major": {"fragment_size": 80, "number_of_fragments": 1},
                "application.position_name": {"fragment_size": 120, "number_of_fragments": 1},
            },
        },
        "_source": {"excludes": EVIDENCE_SOURCE_EXCLUDES},
    }


def _evidence_lexical_query(
    query_text: str,
    *,
    query_intent: str | None = None,
) -> dict[str, Any]:
    if query_intent == INTENT_LOOKUP:
        return _lookup_lexical_query(query_text)

    normalized_degree = _normalize_highest_degree(query_text)
    queries = [
        _profile_query(
            _term_query("application.candidate_no", query_text.upper(), 60, "evidence_exact:candidate_no")
        ),
        _profile_query(
            _term_query("application.position_code", query_text.upper(), 55, "evidence_exact:position_code")
        ),
        _profile_query(
            _term_query("candidate.name.keyword", query_text, 45, "evidence_exact:candidate_name")
        ),
        _profile_query(
            _term_query("candidate.phone", query_text, 45, "evidence_exact:candidate_phone")
        ),
        _profile_query(
            _term_query("candidate.email", query_text, 45, "evidence_exact:candidate_email")
        ),
        _section_query(
            "skills",
            _term_query("skills", query_text, 40, "evidence_exact:skills"),
        ),
        # 实体字段: term (精确 token 匹配) + match (分词后受控召回)
        # term 保证 "阿里巴巴" 精准命中，match 保证 "阿里巴巴实习" 也能通过分词命中。
        # match 使用 minimum_should_match，避免 "Columbia University 哥伦比亚大学"
        # 只凭 "大学" 这类泛词把候选池扩到全库。
        _profile_query(_entity_field_query(
            "candidate.all_schools.keyword", "candidate.all_schools",
            query_text, 36, "evidence_exact:candidate_school", "evidence_match:candidate_school",
        )),
        _profile_query(_entity_field_query(
            "candidate.major.keyword", "candidate.major",
            query_text, 34, "evidence_exact:candidate_major", "evidence_match:candidate_major",
        )),
        _profile_query(_entity_field_query(
            "application.company", "application.company",
            query_text, 30, "evidence_exact:application_company", "evidence_match:company",
        )),
        _profile_query(_entity_field_query(
            "application.position_name.keyword", "application.position_name",
            query_text, 30, "evidence_exact:position_name", "evidence_match:position_name",
        )),
        _profile_query(
            _term_query("candidate.highest_degree", normalized_degree, 15, "evidence_exact:highest_degree")
        ),
        _term_query("title.keyword", query_text, 18, "evidence_exact:title"),
        _profile_query(
            _match_phrase_query("candidate.major.phrase", query_text, 24, "evidence_phrase:candidate_major")
        ),
        _profile_query(
            _match_phrase_query("candidate.all_schools.phrase", query_text, 18, "evidence_phrase:candidate_school")
        ),
        _profile_query(
            _match_phrase_query(
                "application.position_name.phrase",
                query_text,
                18,
                "evidence_phrase:position_name",
            ),
        ),
        _match_phrase_query("title.phrase", query_text, 12, "evidence_phrase:title"),
        _match_phrase_query("text.phrase", query_text, 10, "evidence_phrase:text"),
        {
            "multi_match": {
                "_name": "evidence_term:all_terms:W4",
                "query": query_text,
                "fields": [
                    "title^5",
                    "text^4",
                    "skills_text^5",
                ],
                "type": "best_fields",
                "operator": "and",
                "boost": 4,
            }
        },
    ]
    # partial_terms 是 70% OR-token 的"部分命中"召回，对多技能语义查询有用
    # （"Python NLP SQL" 命中 2/3 也算相关）；但对 keyword 实体查询有害——
    # "哥伦比亚大学" 被切成 [哥伦比亚, 大学]，泛词"大学"命中全库每份简历，
    # 70% 门槛形同直通车。keyword 意图下实体已由 term + 短语路精确处理，
    # 不需要也不应走 partial_terms，故仅在非 keyword 意图下启用。
    if query_intent != INTENT_KEYWORD:
        queries.append(_partial_terms_query(query_text))
    elif len(_coverage_tokens(query_text)) >= 2:
        # 多关键词 keyword 查询（用户用空格/顿号分隔，如 "腾讯 阿里巴巴"）：
        # 命中任一 token 即召回。否则上面那条 operator:"and" 要求所有词出现在
        # 同一证据切片里，而"腾讯经历"和"阿里经历"分属不同切片，永远 0 召回。
        # 单实体查询（"哥伦比亚大学"，无空格 → 单 token）不进此路，避免 IK 切出
        # 的泛词"大学"把候选池扩到全库。命中词数多寡由 coverage should + 候选人
        # 维度 multiplier 主导排序，全命中者自然靠前。
        queries.append(_keyword_or_recall_query(query_text))
    scoring_query = {
        "dis_max": {
            "tie_breaker": 0.0,
            "queries": queries,
        }
    }
    coverage_should = _evidence_term_coverage_queries(query_text)
    if not coverage_should:
        return scoring_query
    return {
        "bool": {
            "must": [scoring_query],
            "should": coverage_should,
        }
    }


def _lookup_lexical_query(query_text: str) -> dict[str, Any]:
    return {
        "dis_max": {
            "tie_breaker": 0.0,
            "queries": [
                _profile_query(
                    _term_query("application.candidate_no", query_text.upper(), 60, "evidence_exact:candidate_no")
                ),
                _profile_query(
                    _term_query("application.position_code", query_text.upper(), 55, "evidence_exact:position_code")
                ),
                _profile_query(
                    _term_query("candidate.name.keyword", query_text, 45, "evidence_exact:candidate_name")
                ),
                _profile_query(
                    _term_query("candidate.phone", query_text, 45, "evidence_exact:candidate_phone")
                ),
                _profile_query(
                    _term_query("candidate.email", query_text, 45, "evidence_exact:candidate_email")
                ),
            ],
        }
    }


def _partial_terms_query(query_text: str) -> dict[str, Any]:
    return {
        "multi_match": {
            "_name": "evidence_term:partial_terms:W1",
            "query": query_text,
            "fields": [
                "title^5",
                "text^4",
                "skills_text^5",
            ],
            "type": "best_fields",
            "operator": "or",
            "minimum_should_match": PARTIAL_TERMS_MINIMUM_SHOULD_MATCH,
            "boost": 1,
        }
    }


def _keyword_or_recall_query(query_text: str) -> dict[str, Any]:
    """多关键词 keyword 查询的 OR 召回：命中任一 token 即召回。

    与 _partial_terms_query 的区别：
    - 这里 minimum_should_match=1（命中一个就召回），不是 70%——"腾讯 阿里巴巴"
      只匹配其中一个的候选人也要被召回（诉求 2），70% 在 2 词时会退化成 AND。
    - 按 token 逐个组 should 子句，而非把整串交给 multi_match 的 operator:"or"，
      这样一个 token 内部若被 IK 切成多词（如 "阿里巴巴"→[阿里,巴巴]）仍按短语
      连续匹配，不会因单个泛词碎片命中全库。
    - boost 低（1），只负责"开召回门"；命中词数对排序的影响由候选人维度的
      coverage multiplier 承担，全命中者靠前。
    """
    return {
        "bool": {
            "_name": "evidence_term:keyword_or_recall:W1",
            "should": [
                {
                    "multi_match": {
                        "query": token,
                        "fields": [
                            "title^5",
                            "text^4",
                            "skills_text^5",
                        ],
                        "type": "phrase",
                    }
                }
                for token in _coverage_tokens(query_text)
            ],
            "minimum_should_match": 1,
            "boost": 1,
        }
    }


def _entity_field_query(
    keyword_field: str,
    text_field: str,
    query_text: str,
    boost: float,
    term_name: str,
    match_name: str,
) -> dict[str, Any]:
    """对实体字段组合 term + 短语匹配，兼顾精确匹配和受控分词召回。

    term 查询：当 query 本身就是一个完整 token 时精准命中（如 "阿里巴巴"）。
    match_phrase 查询：当 query 含修饰词或被 IK 切成多 token 时（如 "北京邮电大学"
    → [北京邮电大学, 北京邮电, 大学]），要求这些 token 在字段里**按序连续**出现才算
    命中。这从根本上挡住了"哥伦比亚大学"被切成 [哥伦比亚, 大学] 后、仅凭泛词"大学"
    把整库学校全召回的问题——早先用 `match` + `minimum_should_match=70%` 时，2 个
    token 只需命中 1 个，泛词"大学/学院"形同直通车。短语匹配是内容无关的结构约束，
    不需要维护停用词表。boost 略低于精确 term，避免排在精准匹配前面。
    """
    return {
        "dis_max": {
            "tie_breaker": 0.0,
            "queries": [
                {"term": {keyword_field: {"value": query_text, "boost": boost, "_name": f"{term_name}:W{boost}"}}},
                {
                    "match_phrase": {
                        text_field: {
                            "query": query_text,
                            "slop": 1,
                            "boost": boost * 0.55,
                            "_name": f"{match_name}:W{boost * 0.55}",
                        }
                    }
                },
            ],
        }
    }


def _term_query(field: str, value: str | int, boost: float, name: str | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {"value": value, "boost": boost}
    if name:
        params["_name"] = f"{name}:W{boost}"
    return {"term": {field: params}}


def _match_phrase_query(
    field: str,
    query_text: str,
    boost: float,
    name: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"query": query_text, "slop": 0, "boost": boost}
    if name:
        params["_name"] = f"{name}:W{boost}"
    return {"match_phrase": {field: params}}


def _profile_query(query: dict[str, Any]) -> dict[str, Any]:
    return _section_query("profile", query)


def _section_query(section_type: str, query: dict[str, Any]) -> dict[str, Any]:
    return {
        "bool": {
            "filter": {"term": {"section_type": section_type}},
            "must": [query],
        }
    }


def _evidence_term_coverage_queries(query_text: str) -> list[dict[str, Any]]:
    tokens = _coverage_tokens(query_text)
    if len(tokens) < 2:
        return []
    return [
        {
            "constant_score": {
                "_name": f"{COVERAGE_QUERY_PREFIX}{index}",
                "filter": _evidence_term_coverage_filter(token),
                "boost": QUERY_TERM_COVERAGE_BOOST,
            }
        }
        for index, token in enumerate(tokens)
    ]


def _evidence_term_coverage_filter(token: str) -> dict[str, Any]:
    return {
        "bool": {
            "should": [
                {"term": {"application.candidate_no": token.upper()}},
                {"term": {"application.position_code": token.upper()}},
                {"term": {"candidate.name.keyword": token}},
                {"term": {"candidate.phone": token}},
                {"term": {"candidate.email": token}},
                {"term": {"candidate.school.keyword": token}},
                {"term": {"candidate.major.keyword": token}},
                {"term": {"application.company": token}},
                {"term": {"application.position_name.keyword": token}},
                {"term": {"skills": token}},
                {"term": {"candidate.highest_degree": _normalize_highest_degree(token)}},
                {
                    "multi_match": {
                        "query": token,
                        "fields": [
                            "title",
                            "text",
                            "skills_text",
                        ],
                        "type": "best_fields",
                    }
                },
            ],
            "minimum_should_match": 1,
        }
    }


def _is_dense_retriever(name: Any) -> bool:
    return name == EVIDENCE_DENSE_RETRIEVER


def _evidence_knn_body(
    query_vector: list[float],
    filters: list[dict[str, Any]],
    size: int,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "size": size,
        "knn": {
            "field": EVIDENCE_VECTOR_FIELD,
            "query_vector": query_vector,
            "k": size,
            "num_candidates": max(size, min(KNN_NUM_CANDIDATES, max(size * 3, 50))),
        },
        "_source": {"excludes": EVIDENCE_SOURCE_EXCLUDES},
    }
    if filters:
        body["knn"]["filter"] = {"bool": {"filter": filters}}
    return body


