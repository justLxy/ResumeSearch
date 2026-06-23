from __future__ import annotations

import html
import logging
import re
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
logger = logging.getLogger(__name__)

RRF_RANK_CONSTANT = 60
RRF_RANK_WINDOW_SIZE = 100
MAX_BROWSE_RESULT_SIZE = 10_000
KNN_NUM_CANDIDATES = 300
FACETS_CACHE_TTL_SECONDS = 60
FILTER_VOCAB_CACHE_TTL_SECONDS = 300
SKILL_FACET_AGG_SIZE = 200
SKILL_FACET_DISPLAY_SIZE = 30
BM25_RETRIEVER = "bm25"
DENSE_RETRIEVER = "dense"
BM25_RRF_WEIGHT = 1.0
DENSE_RRF_WEIGHT = 1.0
DENSE_ONLY_MIN_SCORE = 0.855
DENSE_ONLY_SCORE_BAND = 0.02
DENSE_ONLY_MAX_RESULTS = 8
QUERY_TERM_COVERAGE_BOOST = 0.001
MAX_QUERY_COVERAGE_TERMS = 8
COVERAGE_QUERY_PREFIX = "query_term:"
LEXICAL_EXACT_QUERY_PREFIX = "lexical_exact:"
LEXICAL_PHRASE_QUERY_PREFIX = "lexical_phrase:"
VECTOR_FIELDS = ("semantic_profile_vector",)
SOURCE_EXCLUDES = [
    "raw_text",
    "raw_sections",
    "search_text",
    "skills_text",
    *VECTOR_FIELDS,
]
EXACT_LOOKUP_RE = re.compile(
    r"^(?:[A-Za-z]\d{3,}|M\d{6,}|\d{6,}|1[3-9]\d{9}|[^@\s]+@[^@\s]+\.[^@\s]+)$",
    re.I,
)
YEAR_FILTER_RE = re.compile(r"^(?P<years>\d+(?:\.\d+)?)\s*年(?:以上|及以上|\+)?$")
EXACT_ENTITY_SUFFIXES = ("大学", "学院", "公司", "集团")
DEGREE_ALIASES = {
    "博士研究生": "博士",
    "博士": "博士",
    "硕士研究生": "硕士",
    "硕士": "硕士",
    "学士": "本科",
    "本科": "本科",
}
CANONICAL_SKILL_LABELS = {
    "c": "C",
    "c++": "C++",
    "c#": "C#",
    "css": "CSS",
    "docker": "Docker",
    "html": "HTML",
    "java": "Java",
    "javascript": "JavaScript",
    "jvm": "JVM",
    "linux": "Linux",
    "mysql": "MySQL",
    "nlp": "NLP",
    "pytorch": "PyTorch",
    "redis": "Redis",
    "spark": "Spark",
    "spring boot": "Spring Boot",
    "spring cloud": "Spring Cloud",
    "sql": "SQL",
    "tensorflow": "TensorFlow",
    "typescript": "TypeScript",
    "vue": "Vue",
}

_facets_cache: tuple[float, dict[str, Any]] | None = None
_filter_vocab_cache: tuple[float, dict[str, set[str]]] | None = None
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
    limit: int = 0,
) -> dict[str, Any]:
    raw_query_text = q.strip()
    parsed_query = _parse_query_constraints(raw_query_text) if raw_query_text else _empty_parsed_query()
    query_text = parsed_query["query_text"]
    size = _normalize_limit(limit)
    skill_vocab = _load_filter_vocab()["skills"] if skills else None
    filters = [
        *_build_filters(degree, cities, skills, min_years, skill_vocab=skill_vocab),
        *parsed_query["filters"],
    ]
    retrieval_warnings: list[str] = []

    if query_text:
        # Hybrid: BM25 + semantic profile kNN merged with manual RRF.
        use_dense = _use_dense(query_text)
        query_vector: list[float] = []
        if use_dense:
            try:
                query_vector = encode_single(query_text)
            except Exception as exc:
                logger.exception("query embedding failed")
                retrieval_warnings.append(f"dense embedding failed: {exc}")
                use_dense = False
        rank_window_size = max(size, RRF_RANK_WINDOW_SIZE)
        responses, retriever_warnings = _run_hybrid_search(
            query_text,
            query_vector,
            filters,
            rank_window_size,
            use_dense,
        )
        retrieval_warnings.extend(retriever_warnings)
        matched_total = _lexical_total(responses)
        allow_dense_only = use_dense and _allow_dense_only(query_text)
        candidate_total = _hybrid_total(responses, allow_dense_only)
        results = _rrf_merge(responses, size, allow_dense_only, query_text=query_text)
    elif filters:
        browse_size = MAX_BROWSE_RESULT_SIZE
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
        results = [_format_hit(hit) for hit in es_result.get("hits", {}).get("hits", [])]

    return {
        "query": q,
        "effective_query": query_text,
        "parsed_constraints": parsed_query["constraints"],
        "total": len(results),
        "returned_count": len(results),
        "matched_total": matched_total,
        "candidate_total": candidate_total,
        "retrieval_warnings": retrieval_warnings,
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

def _normalize_limit(limit: int | None) -> int:
    if limit is None or limit <= 0:
        return MAX_BROWSE_RESULT_SIZE
    return max(1, min(limit, MAX_BROWSE_RESULT_SIZE))

def _build_filters(
    degree: str,
    cities: list[str],
    skills: list[str],
    min_years: float,
    skill_vocab: set[str] | None = None,
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
        filters.append({"range": {"candidate.years_experience": {"gte": min_years}}})
    return filters


def _normalize_highest_degree(degree: str) -> str:
    return DEGREE_ALIASES.get(degree, degree)


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


def _empty_parsed_query() -> dict[str, Any]:
    return {"query_text": "", "filters": [], "constraints": {}}


def _parse_query_constraints(
    query_text: str,
    facets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text = query_text.strip()
    if not text:
        return _empty_parsed_query()

    tokens = _query_tokens(text)
    year_tokens = [token for token in tokens if _parse_year_filter_token(token) is not None]
    should_parse = len(tokens) >= 2 or bool(year_tokens)
    if not should_parse:
        return {"query_text": text, "filters": [], "constraints": {}}

    if facets is None:
        vocab = _load_filter_vocab()
        known_cities = vocab["cities"]
        known_skills = vocab["skills"]
        known_degrees = vocab["degrees"] | set(DEGREE_ALIASES)
    else:
        known_cities = _facet_keys(facets, "cities")
        known_skills = _facet_keys(facets, "skills")
        known_degrees = _facet_keys(facets, "degrees") | set(DEGREE_ALIASES)

    remove_tokens: set[str] = set()
    constraints: dict[str, Any] = {}
    filters: list[dict[str, Any]] = []

    parsed_years = [_parse_year_filter_token(token) for token in tokens]
    parsed_years = [years for years in parsed_years if years is not None]
    if parsed_years:
        min_years = max(parsed_years)
        filters.append({"range": {"candidate.years_experience": {"gte": min_years}}})
        constraints["min_years"] = min_years
        remove_tokens.update(year_tokens)

    degree_tokens = [token for token in tokens if token in known_degrees]
    if degree_tokens:
        degree = _normalize_highest_degree(degree_tokens[0])
        filters.append({"term": {"candidate.highest_degree": degree}})
        constraints["degree"] = degree
        remove_tokens.update(degree_tokens)

    cities = _dedupe([token for token in tokens if token in known_cities])
    if cities:
        filters.append({"terms": {"application.expected_work_cities": cities}})
        constraints["cities"] = cities
        remove_tokens.update(cities)

    # Only promote skill tokens to hard filters when the same free-text input
    # already contains an explicit structured constraint. Plain skill queries
    # remain broad BM25+dense searches to preserve recall.
    if filters:
        skills = _known_skill_tokens(tokens, known_skills)
        if skills:
            for skill in skills:
                filters.append(_skill_filter(skill, known_skills))
            constraints["skills"] = skills

    remaining_tokens = [token for token in tokens if token not in remove_tokens]
    remaining_query = " ".join(remaining_tokens).strip()
    return {
        "query_text": remaining_query,
        "filters": filters,
        "constraints": constraints,
    }


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


def _facet_keys(facets: dict[str, Any], name: str) -> set[str]:
    return {
        str(item.get("key")).strip()
        for item in facets.get(name, [])
        if item.get("key")
    }


def _known_skill_tokens(tokens: list[str], known_skills: set[str]) -> list[str]:
    preferred_by_key: dict[str, str] = {}
    for skill in known_skills:
        key = _casefold_key(skill)
        if not key:
            continue
        current = preferred_by_key.get(key)
        if current is None or _skill_label_sort_key(skill) < _skill_label_sort_key(current):
            preferred_by_key[key] = skill

    result: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        key = _casefold_key(token)
        if key in preferred_by_key and key not in seen:
            seen.add(key)
            result.append(_display_skill_label([preferred_by_key[key]]))
    return result


def _parse_year_filter_token(token: str) -> float | None:
    match = YEAR_FILTER_RE.match(token)
    if not match:
        return None
    return float(match.group("years"))


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
    exact_should = _lexical_exact_queries(query_text)
    phrase_should = _lexical_phrase_queries(query_text)
    term_should = _lexical_term_queries(query_text)
    scoring_should = [*exact_should, *phrase_should, *term_should]
    coverage_should = _term_coverage_queries(query_text)
    if not coverage_should:
        return {"bool": {"should": scoring_should, "minimum_should_match": 1}}
    return {
        "bool": {
            "must": [
                {
                    "bool": {
                        "should": scoring_should,
                        "minimum_should_match": 1,
                    }
                }
            ],
            "should": coverage_should,
        }
    }


def _lexical_exact_queries(query_text: str) -> list[dict[str, Any]]:
    normalized_degree = _normalize_highest_degree(query_text)
    return [
        _term_query("application.candidate_no", query_text.upper(), 60, _exact_name("candidate_no")),
        _term_query("application.position_code", query_text.upper(), 55, _exact_name("position_code")),
        _term_query("candidate.name.keyword", query_text, 45, _exact_name("candidate_name")),
        _term_query("candidate.phone", query_text, 45, _exact_name("candidate_phone")),
        _term_query("candidate.email", query_text, 45, _exact_name("candidate_email")),
        _term_query("candidate.school.keyword", query_text, 36, _exact_name("candidate_school")),
        _term_query("candidate.major.keyword", query_text, 34, _exact_name("candidate_major")),
        _term_query("application.company", query_text, 30, _exact_name("application_company")),
        _term_query("application.position_name.keyword", query_text, 30, _exact_name("position_name")),
        _term_query("skills", query_text, 28, _exact_name("skills")),
        _term_query("candidate.highest_degree", normalized_degree, 10, _exact_name("highest_degree")),
        _nested_query(
            "application.wishes",
            [
                _term_query(
                    "application.wishes.company",
                    query_text,
                    30,
                    _exact_name("wish_company"),
                ),
            ],
            _exact_name("wishes"),
        ),
        _nested_query(
            "education",
            [
                _term_query(
                    "education.school.keyword",
                    query_text,
                    36,
                    _exact_name("education_school"),
                ),
                _term_query(
                    "education.major.keyword",
                    query_text,
                    34,
                    _exact_name("education_major"),
                ),
                _term_query(
                    "education.education_level",
                    normalized_degree,
                    10,
                    _exact_name("education_level"),
                ),
                _term_query("education.degree", query_text, 8, _exact_name("education_degree")),
            ],
            _exact_name("education"),
        ),
        _nested_query(
            "internships",
            [
                _term_query(
                    "internships.company.keyword",
                    query_text,
                    24,
                    _exact_name("internship_company"),
                ),
                _term_query(
                    "internships.work_type",
                    query_text,
                    8,
                    _exact_name("internship_work_type"),
                ),
            ],
            _exact_name("internships"),
        ),
        _nested_query(
            "projects",
            [
                _term_query(
                    "projects.name.keyword",
                    query_text,
                    26,
                    _exact_name("project_name"),
                ),
            ],
            _exact_name("projects"),
        ),
    ]


def _lexical_phrase_queries(query_text: str) -> list[dict[str, Any]]:
    return [
        _match_phrase_query("candidate.major.phrase", query_text, 24, _phrase_name("candidate_major")),
        _match_phrase_query("candidate.school.phrase", query_text, 18, _phrase_name("candidate_school")),
        _match_phrase_query(
            "application.position_name.phrase",
            query_text,
            18,
            _phrase_name("position_name"),
        ),
        _match_phrase_query(
            "section_text.education.phrase",
            query_text,
            12,
            _phrase_name("section_education"),
        ),
        _match_phrase_query("section_text.projects.phrase", query_text, 9, _phrase_name("section_projects")),
        _match_phrase_query(
            "section_text.internships.phrase",
            query_text,
            9,
            _phrase_name("section_internships"),
        ),
        _nested_query(
            "application.wishes",
            [
                _match_phrase_query(
                    "application.wishes.position_name.phrase",
                    query_text,
                    18,
                    _phrase_name("wish_position"),
                ),
            ],
            _phrase_name("wishes"),
        ),
        _nested_query(
            "education",
            [
                _match_phrase_query(
                    "education.major.phrase",
                    query_text,
                    24,
                    _phrase_name("education_major"),
                ),
                _match_phrase_query(
                    "education.school.phrase",
                    query_text,
                    18,
                    _phrase_name("education_school"),
                ),
                _match_phrase_query(
                    "education.college.phrase",
                    query_text,
                    14,
                    _phrase_name("education_college"),
                ),
                _match_phrase_query(
                    "education.research_direction.phrase",
                    query_text,
                    10,
                    _phrase_name("education_research"),
                ),
                _match_phrase_query(
                    "education.lab_name.phrase",
                    query_text,
                    8,
                    _phrase_name("education_lab"),
                ),
            ],
            _phrase_name("education"),
        ),
        _nested_query(
            "internships",
            [
                _match_phrase_query(
                    "internships.company.phrase",
                    query_text,
                    14,
                    _phrase_name("internship_company"),
                ),
                _match_phrase_query(
                    "internships.title.phrase",
                    query_text,
                    12,
                    _phrase_name("internship_title"),
                ),
                _match_phrase_query(
                    "internships.department.phrase",
                    query_text,
                    8,
                    _phrase_name("internship_department"),
                ),
                _match_phrase_query(
                    "internships.description.phrase",
                    query_text,
                    6,
                    _phrase_name("internship_description"),
                ),
            ],
            _phrase_name("internships"),
        ),
        _nested_query(
            "projects",
            [
                _match_phrase_query(
                    "projects.name.phrase",
                    query_text,
                    18,
                    _phrase_name("project_name"),
                ),
                _match_phrase_query(
                    "projects.description.phrase",
                    query_text,
                    7,
                    _phrase_name("project_description"),
                ),
                _match_phrase_query(
                    "projects.responsibility.phrase",
                    query_text,
                    7,
                    _phrase_name("project_responsibility"),
                ),
            ],
            _phrase_name("projects"),
        ),
    ]


def _lexical_term_queries(query_text: str) -> list[dict[str, Any]]:
    fields = [
        "application.position_name^4",
        "candidate.name^4",
        "candidate.school^3",
        "candidate.major^4",
        "section_text.projects^3",
        "section_text.internships^3",
        "section_text.education^3",
        "skills_text^6",
    ]
    return [
        {
            "multi_match": {
                "query": query_text,
                "fields": fields,
                "type": "best_fields",
                "operator": "and",
                "boost": 4,
            }
        },
        {
            "multi_match": {
                "query": query_text,
                "fields": fields,
                "type": "best_fields",
                "operator": "or",
                "minimum_should_match": "2<70%",
                "boost": 1,
            }
        },
        _nested_query(
            "education",
            [
                {"match": {"education.school": {"query": query_text, "operator": "and", "boost": 4}}},
                {"match": {"education.college": {"query": query_text, "operator": "and", "boost": 4}}},
                {"match": {"education.major": {"query": query_text, "operator": "and", "boost": 5}}},
                {
                    "match": {
                        "education.research_direction": {
                            "query": query_text,
                            "operator": "and",
                            "boost": 3,
                        }
                    }
                },
                {"match": {"education.lab_name": {"query": query_text, "operator": "and", "boost": 2}}},
            ],
        ),
        _nested_query(
            "projects",
            [
                {"match": {"projects.name": {"query": query_text, "operator": "and", "boost": 4}}},
                {"match": {"projects.description": {"query": query_text, "operator": "and", "boost": 2}}},
                {"match": {"projects.responsibility": {"query": query_text, "operator": "and", "boost": 2}}},
            ],
        ),
        _nested_query(
            "internships",
            [
                {"match": {"internships.title": {"query": query_text, "operator": "and", "boost": 3}}},
                {"match": {"internships.department": {"query": query_text, "operator": "and", "boost": 2}}},
                {"match": {"internships.description": {"query": query_text, "operator": "and", "boost": 2}}},
            ],
        ),
    ]


def _term_query(field: str, value: str, boost: float, name: str | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {"value": value, "boost": boost}
    if name:
        params["_name"] = name
    return {"term": {field: params}}


def _match_phrase_query(
    field: str,
    query_text: str,
    boost: float,
    name: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"query": query_text, "slop": 0, "boost": boost}
    if name:
        params["_name"] = name
    return {"match_phrase": {field: params}}


def _nested_query(
    path: str,
    should: list[dict[str, Any]],
    name: str | None = None,
) -> dict[str, Any]:
    nested: dict[str, Any] = {
        "path": path,
        "score_mode": "max",
        "query": {
            "bool": {
                "should": should,
                "minimum_should_match": 1,
            }
        },
    }
    if name:
        nested["_name"] = name
    return {"nested": nested}


def _exact_name(label: str) -> str:
    return f"{LEXICAL_EXACT_QUERY_PREFIX}{label}"


def _phrase_name(label: str) -> str:
    return f"{LEXICAL_PHRASE_QUERY_PREFIX}{label}"


def _term_coverage_queries(query_text: str) -> list[dict[str, Any]]:
    tokens = _coverage_tokens(query_text)
    if len(tokens) < 2:
        return []
    return [
        {
            "constant_score": {
                "_name": f"{COVERAGE_QUERY_PREFIX}{index}",
                "filter": _term_coverage_filter(token),
                "boost": QUERY_TERM_COVERAGE_BOOST,
            }
        }
        for index, token in enumerate(tokens)
    ]


def _term_coverage_filter(token: str) -> dict[str, Any]:
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
                            "application.position_name",
                            "candidate.name",
                            "candidate.school",
                            "candidate.major",
                            "section_text.projects",
                            "section_text.internships",
                            "section_text.education",
                            "skills_text",
                        ],
                        "type": "best_fields",
                    }
                },
                {
                    "nested": {
                        "path": "application.wishes",
                        "score_mode": "none",
                        "query": {
                            "bool": {
                                "should": [
                                    {"term": {"application.wishes.company": token}},
                                    {
                                        "match": {
                                            "application.wishes.position_name": token
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
                        "path": "education",
                        "score_mode": "none",
                        "query": {
                            "bool": {
                                "should": [
                                    {"term": {"education.school.keyword": token}},
                                    {"term": {"education.major.keyword": token}},
                                    {
                                        "term": {
                                            "education.education_level": (
                                                _normalize_highest_degree(token)
                                            )
                                        }
                                    },
                                    {"term": {"education.degree": token}},
                                    {"match": {"education.school": token}},
                                    {"match": {"education.college": token}},
                                    {"match": {"education.major": token}},
                                    {"match": {"education.research_direction": token}},
                                    {"match": {"education.lab_name": token}},
                                ],
                                "minimum_should_match": 1,
                            }
                        },
                    }
                },
                {
                    "nested": {
                        "path": "internships",
                        "score_mode": "none",
                        "query": {
                            "bool": {
                                "should": [
                                    {"term": {"internships.company.keyword": token}},
                                    {"term": {"internships.work_type": token}},
                                    {"match": {"internships.company": token}},
                                    {"match": {"internships.department": token}},
                                    {"match": {"internships.title": token}},
                                    {"match": {"internships.description": token}},
                                ],
                                "minimum_should_match": 1,
                            }
                        },
                    }
                },
                {
                    "nested": {
                        "path": "projects",
                        "score_mode": "none",
                        "query": {
                            "bool": {
                                "should": [
                                    {"term": {"projects.name.keyword": token}},
                                    {"match": {"projects.name": token}},
                                    {"match": {"projects.description": token}},
                                    {"match": {"projects.responsibility": token}},
                                ],
                                "minimum_should_match": 1,
                            }
                        },
                    }
                },
            ],
            "minimum_should_match": 1,
        }
    }


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
) -> tuple[list[dict[str, Any]], list[str]]:
    requests_to_run = [
        (BM25_RETRIEVER, BM25_RRF_WEIGHT, _bm25_body(query_text, filters, rank_window_size)),
    ]
    vector_fields = VECTOR_FIELDS if use_dense else ()
    if vector_fields:
        requests_to_run.append(
            (
                DENSE_RETRIEVER,
                DENSE_RRF_WEIGHT,
                _knn_body(vector_fields[0], query_vector, filters, rank_window_size),
            )
        )

    responses: list[dict[str, Any]] = []
    warnings: list[str] = []
    with ThreadPoolExecutor(max_workers=len(requests_to_run)) as executor:
        futures = {
            executor.submit(_es, "POST", f"/{INDEX_ALIAS}/_search", body): (name, weight)
            for name, weight, body in requests_to_run
        }
        for future in as_completed(futures):
            name, weight = futures[future]
            try:
                response = future.result()
            except Exception as exc:
                # One retriever failing should not crash the entire search;
                # degrade gracefully with whatever retriever(s) succeeded.
                logger.exception("%s retriever failed", name)
                warnings.append(f"{name} retriever failed: {exc}")
                continue
            response["_retriever_name"] = name
            response["_rrf_weight"] = weight
            responses.append(response)
    return responses, warnings

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


def _lexical_total(responses: list[dict[str, Any]]) -> int:
    for response in responses:
        if response.get("_retriever_name") != BM25_RETRIEVER:
            continue
        total = response.get("hits", {}).get("total", {})
        if isinstance(total, dict):
            return int(total.get("value") or 0)
        return int(total or 0)
    return 0

def _rrf_merge(
    responses: list[dict[str, Any]],
    limit: int,
    allow_dense_only: bool = True,
    query_text: str = "",
) -> list[dict[str, Any]]:
    rrf_scores: dict[str, float] = {}
    hit_map: dict[str, dict[str, Any]] = {}
    best_rank: dict[str, int] = {}
    bm25_rank: dict[str, int] = {}
    term_coverage: dict[str, int] = {}
    lexical_tier: dict[str, int] = {}
    retrieval_debug: dict[str, dict[str, Any]] = {}
    lexical_ids = _lexical_doc_ids(responses)
    coverage_enabled = len(_coverage_tokens(query_text)) >= 2

    for response in responses:
        retriever_name = response.get("_retriever_name")
        weight = float(response.get("_rrf_weight", 1.0))
        for rank, hit, dense_debug in _accepted_hits(response, lexical_ids, allow_dense_only):
            doc_id = hit["_id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + weight / (RRF_RANK_CONSTANT + rank)
            best_rank[doc_id] = min(best_rank.get(doc_id, rank), rank)
            if coverage_enabled and retriever_name == BM25_RETRIEVER:
                term_coverage[doc_id] = max(
                    term_coverage.get(doc_id, 0),
                    _matched_term_coverage(hit),
                )
            if retriever_name == BM25_RETRIEVER:
                bm25_rank[doc_id] = rank
                lexical_tier[doc_id] = max(
                    lexical_tier.get(doc_id, 0),
                    _matched_lexical_tier(hit),
                )
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
        key=lambda k: (
            -lexical_tier.get(k, 0),
            -term_coverage.get(k, 0),
            bm25_rank.get(k, 10**9) if lexical_tier.get(k, 0) >= 2 else 10**9,
            -rrf_scores[k],
            best_rank.get(k, 10**9),
            k,
        ),
    )[:limit]

    results = []
    for doc_id in sorted_ids:
        hit = dict(hit_map[doc_id])
        hit["_retrieval_debug"] = {
            **retrieval_debug.get(doc_id, {}),
            "rrf_score": round(rrf_scores[doc_id], 6),
            "lexical_tier": lexical_tier.get(doc_id, 0),
        }
        if coverage_enabled:
            hit["_retrieval_debug"]["term_coverage"] = term_coverage.get(doc_id, 0)
        results.append(_format_hit(hit, rrf_scores[doc_id]))
    return results


def _matched_term_coverage(hit: dict[str, Any]) -> int:
    matched_queries = hit.get("matched_queries") or []
    return len(
        {
            query_name
            for query_name in matched_queries
            if isinstance(query_name, str) and query_name.startswith(COVERAGE_QUERY_PREFIX)
        }
    )


def _matched_lexical_tier(hit: dict[str, Any]) -> int:
    matched_queries = hit.get("matched_queries") or []
    if any(
        isinstance(query_name, str) and query_name.startswith(LEXICAL_EXACT_QUERY_PREFIX)
        for query_name in matched_queries
    ):
        return 3
    if any(
        isinstance(query_name, str) and query_name.startswith(LEXICAL_PHRASE_QUERY_PREFIX)
        for query_name in matched_queries
    ):
        return 2
    return 1 if matched_queries else 0


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
    if not compact or _looks_like_exact_lookup(compact):
        return False
    if len(query_text.split()) >= 2:
        return True
    return len(compact) >= 4


def _allow_dense_only(query_text: str) -> bool:
    compact = "".join(query_text.split())
    return bool(compact) and not _looks_like_exact_lookup(compact)


def _looks_like_exact_lookup(compact_query: str) -> bool:
    if EXACT_LOOKUP_RE.match(compact_query):
        return True
    return compact_query.endswith(EXACT_ENTITY_SUFFIXES)


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

    snippets = _highlight_snippets(highlight) or [_default_snippet(projects, internships)]
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


def _highlight_snippets(highlight: dict[str, list[str]]) -> list[str]:
    fields = (
        "application.position_name",
        "candidate.major",
        "candidate.school",
        "skills_text",
        "section_text.internships",
        "section_text.projects",
        "section_text.education",
    )
    snippets: list[str] = []
    seen: set[str] = set()
    for field in fields:
        for fragment in highlight.get(field, []):
            text = str(fragment).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            snippets.append(text)
    return snippets


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
            "skills": {"terms": {"field": "skills", "size": SKILL_FACET_AGG_SIZE}},
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
    facets["skills"] = _merge_case_insensitive_skill_buckets(
        aggs.get("skills", {}).get("buckets", []),
        SKILL_FACET_DISPLAY_SIZE,
    )
    _facets_cache = (now + FACETS_CACHE_TTL_SECONDS, facets)
    return facets


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
    result = _es("POST", f"/{INDEX_ALIAS}/_search", body)
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


def _es(method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.request(
        method,
        f"{ES_URL}{path}",
        json=body,
        timeout=20,
    )
    response.raise_for_status()
    return response.json()
