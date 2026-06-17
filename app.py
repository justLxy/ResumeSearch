from __future__ import annotations

import html
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from embedding_service import encode_single


ES_URL = "http://localhost:9200"
INDEX_ALIAS = "resumes_current"
BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

RRF_RANK_CONSTANT = 60
RRF_RANK_WINDOW_SIZE = 100
MAX_RESULT_SIZE = 50
KNN_NUM_CANDIDATES = 300
FACETS_CACHE_TTL_SECONDS = 60
VECTOR_FIELDS_CACHE_TTL_SECONDS = 300
BM25_RETRIEVER = "bm25"
DENSE_RETRIEVER = "dense"
BM25_RRF_WEIGHT = 1.0
DENSE_RRF_WEIGHT = 1.0
DENSE_ONLY_MIN_SCORE = 0.84
DENSE_ONLY_SCORE_BAND = 0.02
DENSE_ONLY_MAX_RESULTS = 8
VECTOR_FIELDS = ("semantic_profile_vector",)
LEGACY_VECTOR_FIELDS = (
    "search_text_vector",
    "skills_vector",
    "projects_vector",
    "internships_vector",
    "education_vector",
)
SOURCE_EXCLUDES = [
    "raw_text",
    "raw_sections",
    "search_text",
    "skills_text",
    *VECTOR_FIELDS,
    *LEGACY_VECTOR_FIELDS,
]

_facets_cache: tuple[float, dict[str, Any]] | None = None
_vector_fields_cache: tuple[float, tuple[str, ...]] | None = None

app = FastAPI(title="Resume Search Prototype")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/search")
def search(
    q: str = "",
    degree: str = "",
    cities: list[str] = Query(default=[]),
    skills: list[str] = Query(default=[]),
    min_years: float = 0,
    limit: int = 20,
) -> dict[str, Any]:
    query_text = q.strip()
    size = _normalize_limit(limit)
    filters = _build_filters(degree, cities, skills, min_years)

    if query_text:
        # Hybrid: BM25 + semantic profile kNN merged with manual RRF.
        use_dense = _use_dense(query_text)
        query_vector = encode_single(query_text) if use_dense else []
        rank_window_size = max(size, RRF_RANK_WINDOW_SIZE)
        responses = _run_hybrid_search(query_text, query_vector, filters, rank_window_size, use_dense)
        allow_dense_only = use_dense
        candidate_total = _hybrid_total(responses, allow_dense_only)
        matched_total = candidate_total
        results = _rrf_merge(responses, size, allow_dense_only)
    elif filters:
        browse_size = MAX_RESULT_SIZE
        body = _bm25_body(query_text, filters, browse_size)
        body.pop("highlight", None)
        body["sort"] = [
            {"application.apply_time": {"order": "desc", "unmapped_type": "date"}},
            {"resume_id": {"order": "asc"}},
        ]
        es_result = _es("POST", f"/{INDEX_ALIAS}/_search", body)
        matched_total = es_result.get("hits", {}).get("total", {}).get("value", 0)
        candidate_total = matched_total
        results = [_format_hit(hit) for hit in es_result.get("hits", {}).get("hits", [])]
    else:
        browse_size = MAX_RESULT_SIZE
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
        results = [_format_hit(hit) for hit in es_result.get("hits", {}).get("hits", [])]

    return {
        "query": q,
        "total": len(results),
        "returned_count": len(results),
        "matched_total": matched_total,
        "candidate_total": candidate_total,
        "results": results,
        "facets": _load_facets(),
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    try:
        result = _es("GET", "/_cluster/health")
        return {
            "es_online": True,
            "status": result.get("status", "unknown"),
            "indices": result.get("number_of_indices", 0),
        }
    except Exception:
        return {"es_online": False, "status": "offline", "indices": 0}


@app.get("/api/resumes/{resume_id}")
def get_resume(resume_id: str) -> dict[str, Any]:
    result = _es("GET", f"/{INDEX_ALIAS}/_doc/{resume_id}")
    return result.get("_source", {})


# ---------------------------------------------------------------------------
# query builders
# ---------------------------------------------------------------------------

def _normalize_limit(limit: int) -> int:
    return max(1, min(limit, MAX_RESULT_SIZE))

def _build_filters(
    degree: str,
    cities: list[str],
    skills: list[str],
    min_years: float,
) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []
    if degree:
        highest_degree = _normalize_highest_degree(degree)
        filters.append({"term": {"candidate.highest_degree": highest_degree}})
    if cities:
        filters.append({"terms": {"application.expected_work_cities": cities}})
    if skills:
        filters.append({"terms": {"skills": skills}})
    if min_years > 0:
        filters.append({"range": {"candidate.years_experience": {"gte": min_years}}})
    return filters


def _normalize_highest_degree(degree: str) -> str:
    aliases = {
        "学士": "本科",
        "本科": "本科",
        "硕士研究生": "硕士",
    }
    return aliases.get(degree, degree)


def _bm25_body(query_text: str, filters: list[dict[str, Any]], size: int) -> dict[str, Any]:
    must: list[dict[str, Any]]
    if query_text:
        must = [_lexical_query(query_text)]
    else:
        must = [{"match_all": {}}]

    body: dict[str, Any] = {
        "size": size,
        "query": {"bool": {"must": must, "filter": filters}},
        "highlight": {
            "pre_tags": ["<mark>"],
            "post_tags": ["</mark>"],
            "fields": {
                "application.position_name": {"fragment_size": 120, "number_of_fragments": 1},
                "candidate.major": {"fragment_size": 80, "number_of_fragments": 1},
                "section_text.projects": {"fragment_size": 160, "number_of_fragments": 2},
                "section_text.internships": {"fragment_size": 160, "number_of_fragments": 1},
                "section_text.education": {"fragment_size": 160, "number_of_fragments": 1},
                "candidate.school": {"fragment_size": 80, "number_of_fragments": 1},
                "skills_text": {"fragment_size": 200, "number_of_fragments": 1},
            },
        },
        "_source": {
            "excludes": SOURCE_EXCLUDES,
        },
    }
    return body


def _lexical_query(query_text: str) -> dict[str, Any]:
    should: list[dict[str, Any]] = [
        {"term": {"candidate.name.keyword": {"value": query_text, "boost": 30}}},
        {"term": {"candidate.school.keyword": {"value": query_text, "boost": 24}}},
        {"term": {"application.position_name.keyword": {"value": query_text, "boost": 16}}},
        {"term": {"skills": {"value": query_text, "boost": 14}}},
        {"match_phrase": {"candidate.school": {"query": query_text, "boost": 12}}},
        {"match_phrase": {"section_text.education": {"query": query_text, "boost": 8}}},
        {"match_phrase": {"application.position_name": {"query": query_text, "boost": 8}}},
        {"match_phrase": {"section_text.projects": {"query": query_text, "boost": 6}}},
        {"match_phrase": {"section_text.internships": {"query": query_text, "boost": 6}}},
        {
            "nested": {
                "path": "education",
                "score_mode": "max",
                "query": {
                    "bool": {
                        "should": [
                            {
                                "term": {
                                    "education.school.keyword": {
                                        "value": query_text,
                                        "boost": 24,
                                    }
                                }
                            },
                            {
                                "match_phrase": {
                                    "education.school": {
                                        "query": query_text,
                                        "boost": 12,
                                    }
                                }
                            },
                            {
                                "match_phrase": {
                                    "education.major": {
                                        "query": query_text,
                                        "boost": 5,
                                    }
                                }
                            },
                            {
                                "match_phrase": {
                                    "education.research_direction": {
                                        "query": query_text,
                                        "boost": 5,
                                    }
                                }
                            },
                        ],
                        "minimum_should_match": 1,
                    }
                },
            }
        },
        {
            "nested": {
                "path": "internships",
                "score_mode": "max",
                "query": {
                    "bool": {
                        "should": [
                            {
                                "term": {
                                    "internships.company.keyword": {
                                        "value": query_text,
                                        "boost": 14,
                                    }
                                }
                            },
                            {
                                "match_phrase": {
                                    "internships.company": {
                                        "query": query_text,
                                        "boost": 7,
                                    }
                                }
                            },
                            {
                                "match_phrase": {
                                    "internships.title": {
                                        "query": query_text,
                                        "boost": 5,
                                    }
                                }
                            },
                        ],
                        "minimum_should_match": 1,
                    }
                },
            }
        },
        {
            "nested": {
                "path": "projects",
                "score_mode": "max",
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match_phrase": {
                                    "projects.name": {
                                        "query": query_text,
                                        "boost": 7,
                                    }
                                }
                            },
                            {
                                "match_phrase": {
                                    "projects.description": {
                                        "query": query_text,
                                        "boost": 5,
                                    }
                                }
                            },
                            {
                                "match_phrase": {
                                    "projects.responsibility": {
                                        "query": query_text,
                                        "boost": 5,
                                    }
                                }
                            },
                        ],
                        "minimum_should_match": 1,
                    }
                },
            }
        },
        {
            "multi_match": {
                "query": query_text,
                "fields": [
                    "application.position_name^4",
                    "candidate.name^4",
                    "candidate.major^2",
                    "section_text.projects^3",
                    "section_text.internships^3",
                    "section_text.education^2",
                    "skills_text^6",
                ],
                "type": "best_fields",
                "operator": "or",
                "minimum_should_match": "2<70%",
            }
        },
    ]
    return {"bool": {"should": should, "minimum_should_match": 1}}


def _knn_body(
    field: str,
    query_vector: list[float],
    filters: list[dict[str, Any]],
    size: int,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "size": size,
        "knn": {
            "field": field,
            "query_vector": query_vector,
            "k": size,
            "num_candidates": max(size, min(KNN_NUM_CANDIDATES, max(size * 3, 50))),
        },
        "_source": {
            "excludes": SOURCE_EXCLUDES,
        },
    }
    if filters:
        body["knn"]["filter"] = {"bool": {"filter": filters}}
    return body


def _run_hybrid_search(
    query_text: str,
    query_vector: list[float],
    filters: list[dict[str, Any]],
    rank_window_size: int,
    use_dense: bool = True,
) -> list[dict[str, Any]]:
    requests_to_run = [
        (BM25_RETRIEVER, BM25_RRF_WEIGHT, _bm25_body(query_text, filters, rank_window_size)),
    ]
    vector_fields = _available_vector_fields() if use_dense else ()
    if vector_fields:
        requests_to_run.append(
            (
                DENSE_RETRIEVER,
                DENSE_RRF_WEIGHT,
                _knn_body(vector_fields[0], query_vector, filters, rank_window_size),
            )
        )

    responses: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(requests_to_run)) as executor:
        futures = {
            executor.submit(_es, "POST", f"/{INDEX_ALIAS}/_search", body): (name, weight)
            for name, weight, body in requests_to_run
        }
        for future in as_completed(futures):
            name, weight = futures[future]
            try:
                response = future.result()
            except Exception:
                # One retriever failing should not crash the entire search;
                # degrade gracefully with whatever retriever(s) succeeded.
                continue
            response["_retriever_name"] = name
            response["_rrf_weight"] = weight
            responses.append(response)
    return responses


def _available_vector_fields() -> tuple[str, ...]:
    global _vector_fields_cache
    now = time.monotonic()
    if _vector_fields_cache and _vector_fields_cache[0] > now:
        return _vector_fields_cache[1]

    mapping = _es("GET", f"/{INDEX_ALIAS}/_mapping")
    properties: dict[str, Any] = {}
    for index_mapping in mapping.values():
        properties.update(index_mapping.get("mappings", {}).get("properties", {}))

    fields = tuple(
        field
        for field in VECTOR_FIELDS
        if properties.get(field, {}).get("type") == "dense_vector"
    )
    _vector_fields_cache = (now + VECTOR_FIELDS_CACHE_TTL_SECONDS, fields)
    return fields


# ---------------------------------------------------------------------------
# manual RRF merge  (ES Basic license does not include built-in RRF)
# ---------------------------------------------------------------------------

def _hybrid_total(
    responses: list[dict[str, Any]],
    allow_dense_only: bool = True,
) -> int:
    ids: set[str] = set()
    lexical_ids = _lexical_doc_ids(responses)
    for response in responses:
        for _rank, hit, _debug in _accepted_hits(response, lexical_ids, allow_dense_only):
            ids.add(hit["_id"])
    return len(ids)

def _rrf_merge(
    responses: list[dict[str, Any]],
    limit: int,
    allow_dense_only: bool = True,
) -> list[dict[str, Any]]:
    rrf_scores: dict[str, float] = {}
    hit_map: dict[str, dict[str, Any]] = {}
    best_rank: dict[str, int] = {}
    retrieval_debug: dict[str, dict[str, Any]] = {}
    lexical_ids = _lexical_doc_ids(responses)

    for response in responses:
        retriever_name = response.get("_retriever_name")
        weight = float(response.get("_rrf_weight", 1.0))
        for rank, hit, dense_debug in _accepted_hits(response, lexical_ids, allow_dense_only):
            doc_id = hit["_id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + weight / (RRF_RANK_CONSTANT + rank)
            best_rank[doc_id] = min(best_rank.get(doc_id, rank), rank)
            if doc_id not in hit_map or hit.get("highlight"):
                hit_map[doc_id] = hit
            debug = retrieval_debug.setdefault(
                doc_id,
                {
                    "retrieval_sources": [],
                    "bm25_rank": None,
                    "dense_rank": None,
                },
            )
            if retriever_name not in debug["retrieval_sources"]:
                debug["retrieval_sources"].append(retriever_name)
            if retriever_name == BM25_RETRIEVER:
                debug["bm25_rank"] = rank
            elif retriever_name == DENSE_RETRIEVER:
                debug["dense_rank"] = rank
                debug.update(dense_debug)

    sorted_ids = sorted(
        rrf_scores.keys(),
        key=lambda k: (-rrf_scores[k], best_rank.get(k, 10**9), k),
    )[:limit]

    results = []
    for doc_id in sorted_ids:
        hit = dict(hit_map[doc_id])
        hit["_retrieval_debug"] = {
            **retrieval_debug.get(doc_id, {}),
            "rrf_score": round(rrf_scores[doc_id], 6),
        }
        results.append(_format_hit(hit, rrf_scores[doc_id]))
    return results


def _accepted_hits(
    response: dict[str, Any],
    lexical_ids: set[str],
    allow_dense_only: bool,
) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    hits = response.get("hits", {}).get("hits", [])
    if response.get("_retriever_name") != DENSE_RETRIEVER:
        return [(rank, hit, {}) for rank, hit in enumerate(hits, start=1)]

    threshold = _dense_only_threshold(response)
    accepted: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    dense_only_count = 0
    for rank, hit in enumerate(hits, start=1):
        doc_id = hit["_id"]
        raw_score = float(hit.get("_score") or 0)
        dense_debug = {
            "dense_score": round(raw_score, 6),
            "dense_only_threshold": round(threshold, 6) if threshold is not None else None,
        }
        if doc_id in lexical_ids:
            dense_debug["dense_only_accepted"] = False
            accepted.append((rank, hit, dense_debug))
            continue
        if not allow_dense_only or threshold is None or raw_score < threshold:
            continue
        if dense_only_count >= DENSE_ONLY_MAX_RESULTS:
            continue
        dense_only_count += 1
        dense_debug["dense_only_accepted"] = True
        accepted.append((rank, hit, dense_debug))
    return accepted


def _dense_only_threshold(response: dict[str, Any]) -> float | None:
    hits = response.get("hits", {}).get("hits", [])
    if not hits:
        return None
    top_score = float(hits[0].get("_score") or 0)
    if top_score < DENSE_ONLY_MIN_SCORE:
        return None
    return max(DENSE_ONLY_MIN_SCORE, top_score - DENSE_ONLY_SCORE_BAND)


def _lexical_doc_ids(responses: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for response in responses:
        if response.get("_retriever_name") != BM25_RETRIEVER:
            continue
        for hit in response.get("hits", {}).get("hits", []):
            ids.add(hit["_id"])
    return ids


def _use_dense(query_text: str) -> bool:
    compact = "".join(query_text.split())
    if len(compact) >= 8:
        return True
    return len(query_text.split()) >= 2


# ---------------------------------------------------------------------------
# hit formatting
# ---------------------------------------------------------------------------

def _format_hit(hit: dict[str, Any], rrf_score: float | None = None) -> dict[str, Any]:
    source = hit.get("_source", {})
    score = round(rrf_score, 4) if rrf_score is not None else round(hit.get("_score") or 0, 3)
    candidate = source.get("candidate", {})
    education = source.get("education") or []
    projects = source.get("projects") or []
    internships = source.get("internships") or []
    highlight = hit.get("highlight", {})

    snippets = (
        highlight.get("application.position_name")
        or highlight.get("candidate.major")
        or highlight.get("section_text.projects")
        or highlight.get("section_text.internships")
        or highlight.get("section_text.education")
        or highlight.get("candidate.school")
        or highlight.get("skills_text")
        or [_default_snippet(projects, internships)]
    )
    years_experience = candidate.get("years_experience")

    return {
        "id": hit.get("_id"),
        "score": score,
        "candidate": candidate,
        "application": source.get("application", {}),
        "education_summary": _education_summary(candidate, education),
        "project_snippet": _safe_snippet(" ... ".join(snippets)),
        "skills": source.get("skills", []),
        "years_experience": years_experience,
        "experience_display": _experience_display(years_experience),
        "retrieval_debug": hit.get("_retrieval_debug", {}),
        "source": source,
    }


def _education_summary(candidate: dict[str, Any], education: list[dict[str, Any]]) -> str:
    if education:
        parts: list[str] = []
        for edu in education:
            school = edu.get("school")
            degree = edu.get("degree") or edu.get("education_level")
            major = edu.get("major")
            if school or degree or major:
                parts.append(
                    " / ".join(item for item in [school, degree, major] if item)
                )
        if parts:
            return " · ".join(parts)

    if candidate.get("school") or candidate.get("highest_degree") or candidate.get("major"):
        return " / ".join(
            item
            for item in [
                candidate.get("school"),
                candidate.get("highest_degree"),
                candidate.get("major"),
            ]
            if item
        )
    return "教育信息待补充"


def _default_snippet(projects: list[dict[str, Any]], internships: list[dict[str, Any]]) -> str:
    lines: list[str] = []

    proj_names = [p.get("name") for p in projects if p.get("name")]
    if proj_names:
        lines.append(f"项目经历：{' | '.join(proj_names)}")

    intern_names: list[str] = []
    for intern in internships:
        company = intern.get("company")
        title = intern.get("title")
        if company and title:
            intern_names.append(f"{company} / {title}")
        elif title:
            intern_names.append(title)
        elif company:
            intern_names.append(company)
    if intern_names:
        lines.append(f"实习经历：{' | '.join(intern_names)}")

    if lines:
        text = "\n".join(lines)
        return html.escape(text)
    return "暂无项目摘要"


def _experience_display(years_experience: float | None) -> str:
    if years_experience is None:
        return "无工作经验"
    return f"{years_experience:.1f} 年工作经验"


def _safe_snippet(snippet: str) -> str:
    allowed = (
        snippet.replace("&lt;mark&gt;", "<mark>")
        .replace("&lt;/mark&gt;", "</mark>")
    )
    while "</mark><mark>" in allowed:
        allowed = allowed.replace("</mark><mark>", "")
    return allowed


# ---------------------------------------------------------------------------
# facets / health
# ---------------------------------------------------------------------------

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
            "skills": {"terms": {"field": "skills", "size": 30}},
            "positions": {"terms": {"field": "application.position_name.keyword", "size": 20}},
        },
    }
    result = _es("POST", f"/{INDEX_ALIAS}/_search", body)
    aggs = result.get("aggregations", {})
    facets = {
        name: [
            {"key": bucket["key"], "count": bucket["doc_count"]}
            for bucket in aggs.get(name, {}).get("buckets", [])
            if bucket.get("key")
        ]
        for name in ("degrees", "cities", "skills", "positions")
    }
    _facets_cache = (now + FACETS_CACHE_TTL_SECONDS, facets)
    return facets


def _es(method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.request(
        method,
        f"{ES_URL}{path}",
        json=body,
        timeout=20,
    )
    response.raise_for_status()
    return response.json()
