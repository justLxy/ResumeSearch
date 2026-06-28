from __future__ import annotations

import html
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from embedding_service import encode_single


ES_URL = "http://localhost:9200"
INDEX_ALIAS = "resumes_current"
EVIDENCE_INDEX_ALIAS = "resume_evidence_current"
BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
logger = logging.getLogger(__name__)

RRF_RANK_CONSTANT = 60
RRF_RANK_WINDOW_SIZE = 1000
DEFAULT_SEARCH_LIMIT = 100
MAX_BROWSE_RESULT_SIZE = 1000
KNN_NUM_CANDIDATES = 300
FACETS_CACHE_TTL_SECONDS = 60
FILTER_VOCAB_CACHE_TTL_SECONDS = 300
SKILL_FACET_AGG_SIZE = 200
SKILL_FACET_DISPLAY_SIZE = 30
DENSE_RETRIEVER = "dense"
EVIDENCE_RETRIEVER = "evidence"
EVIDENCE_DENSE_RETRIEVER = "evidence_dense"
DENSE_RRF_WEIGHT = 1.0
EVIDENCE_RRF_WEIGHT = 1.2
EVIDENCE_DENSE_RRF_WEIGHT = 1.0
DENSE_RANK_WINDOW_SIZE = 300
ENABLE_RERANK = True
RERANK_TOP_N = 20
DENSE_ABSTAIN_MIN_SAMPLE_SIZE = 20
DENSE_ABSTAIN_IQR_MULTIPLIER = 1.5
EVIDENCE_POOL_EXTRA_WEIGHTS = (0.30, 0.15)
QUERY_TERM_COVERAGE_BOOST = 0.001
MAX_QUERY_COVERAGE_TERMS = 8
COVERAGE_QUERY_PREFIX = "query_term:"
EVIDENCE_EXACT_QUERY_PREFIX = "evidence_exact:"
EVIDENCE_PHRASE_QUERY_PREFIX = "evidence_phrase:"
EVIDENCE_TERM_QUERY_PREFIX = "evidence_term:"
INTENT_BROWSE = "browse"
INTENT_EXACT_LOOKUP = "exact_lookup"
INTENT_ENTITY = "entity"
INTENT_STRUCTURED = "structured"
INTENT_SKILL_COMBO = "skill_combo"
INTENT_SEMANTIC = "semantic"
INTENT_JD_MATCH = "jd_match"
RERANK_INTENTS = {INTENT_SEMANTIC, INTENT_JD_MATCH}
EVIDENCE_VECTOR_FIELD = "evidence_vector"
SOURCE_EXCLUDES = [
    "raw_text",
    "raw_sections",
    "skills_text",
]
EVIDENCE_SOURCE_EXCLUDES = [EVIDENCE_VECTOR_FIELD]
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


@dataclass
class QueryPlan:
    raw_query: str
    intent: str
    filters: list[dict[str, Any]]
    constraints: dict[str, Any]
    query_text: str
    lexical_query: str
    semantic_query: str
    must_terms: list[str]
    should_terms: list[str]
    enable_dense: bool
    enable_rerank: bool
    rank_window_size: int

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "lexical_query": self.lexical_query,
            "semantic_query": self.semantic_query,
            "must_terms": self.must_terms,
            "should_terms": self.should_terms,
            "enable_dense": self.enable_dense,
            "enable_rerank": self.enable_rerank,
            "rank_window_size": self.rank_window_size,
            "filter_count": len(self.filters),
        }


_facets_cache: tuple[float, dict[str, Any]] | None = None
_filter_vocab_cache: tuple[float, dict[str, set[str]]] | None = None
app = FastAPI(title="Resume Search Prototype")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/favicon.ico")
def favicon():
    return FileResponse(WEB_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/api/search")
def search(
    q: str = "",
    degree: str = "",
    cities: list[str] = Query(default=[]),
    skills: list[str] = Query(default=[]),
    min_years: float = 0,
    limit: int = 0,
    offset: int = 0,
) -> dict[str, Any]:
    raw_query_text = q.strip()
    page_size = _normalize_limit(limit)
    page_offset = _normalize_offset(offset)
    result_window_size = RRF_RANK_WINDOW_SIZE
    skill_vocab = _load_filter_vocab()["skills"] if skills else None
    explicit_filters = _build_filters(degree, cities, skills, min_years, skill_vocab=skill_vocab)
    plan = _plan_query(raw_query_text, explicit_filters, size=result_window_size)
    query_text = plan.lexical_query
    filters = plan.filters
    retrieval_warnings: list[str] = []

    if query_text:
        # Evidence-first retrieval: search evidence chunks, then aggregate back to resumes.
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
        )
        retrieval_warnings.extend(retriever_warnings)
        matched_total = _lexical_total(responses)
        candidate_total = _hybrid_total(responses)
        results = _rrf_merge(responses, result_window_size, query_text=query_text)
        if plan.enable_rerank:
            results, rerank_warnings = _rerank_results(plan.semantic_query, results)
            retrieval_warnings.extend(rerank_warnings)
    elif filters:
        browse_size = MAX_BROWSE_RESULT_SIZE
        body = _filter_browse_body(filters, browse_size)
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
        return DEFAULT_SEARCH_LIMIT
    return max(1, min(limit, MAX_BROWSE_RESULT_SIZE))


def _normalize_offset(offset: int | None) -> int:
    if offset is None or offset <= 0:
        return 0
    return min(offset, MAX_BROWSE_RESULT_SIZE)

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


def _plan_query(
    raw_query_text: str,
    explicit_filters: list[dict[str, Any]],
    *,
    size: int,
    facets: dict[str, Any] | None = None,
) -> QueryPlan:
    raw_query = raw_query_text.strip()
    parsed_query = _parse_query_constraints(raw_query, facets=facets) if raw_query else _empty_parsed_query()
    query_text = parsed_query["query_text"]
    filters = [*explicit_filters, *parsed_query["filters"]]
    known_skills = _planner_known_skills(facets) if raw_query else set()
    intent = _classify_query_intent(
        raw_query=raw_query,
        query_text=query_text,
        filters=filters,
        constraints=parsed_query["constraints"],
        known_skills=known_skills,
    )
    semantic_query = query_text
    must_terms = _plan_must_terms(intent, query_text, parsed_query["constraints"], known_skills)
    must_term_keys = {_casefold_key(term) for term in must_terms}
    should_terms = [
        token
        for token in _coverage_tokens(query_text)
        if _casefold_key(token) not in must_term_keys
    ]
    enable_dense = _plan_enable_dense(intent, semantic_query)
    return QueryPlan(
        raw_query=raw_query,
        intent=intent,
        filters=filters,
        constraints=parsed_query["constraints"],
        query_text=query_text,
        lexical_query=query_text,
        semantic_query=semantic_query,
        must_terms=must_terms,
        should_terms=should_terms,
        enable_dense=enable_dense,
        enable_rerank=_plan_enable_rerank(intent, semantic_query),
        rank_window_size=max(size, RRF_RANK_WINDOW_SIZE),
    )


def _planner_known_skills(facets: dict[str, Any] | None = None) -> set[str]:
    if facets is not None:
        return _facet_keys(facets, "skills")
    return _load_filter_vocab()["skills"]


def _classify_query_intent(
    *,
    raw_query: str,
    query_text: str,
    filters: list[dict[str, Any]],
    constraints: dict[str, Any],
    known_skills: set[str],
) -> str:
    if not raw_query:
        return INTENT_STRUCTURED if filters else INTENT_BROWSE

    if not query_text:
        return INTENT_STRUCTURED if filters else INTENT_BROWSE

    compact_query = "".join(query_text.split())
    if EXACT_LOOKUP_RE.match(compact_query):
        return INTENT_EXACT_LOOKUP
    if compact_query.endswith(EXACT_ENTITY_SUFFIXES):
        return INTENT_ENTITY
    if constraints:
        return INTENT_STRUCTURED
    if _is_skill_combo_query(query_text, known_skills):
        return INTENT_SKILL_COMBO
    if _looks_like_jd_query(raw_query, query_text):
        return INTENT_JD_MATCH
    if _use_dense(query_text):
        return INTENT_SEMANTIC
    return INTENT_ENTITY


def _is_skill_combo_query(query_text: str, known_skills: set[str]) -> bool:
    tokens = _query_tokens(query_text)
    if len(tokens) < 2 or not known_skills:
        return False
    skill_tokens = _known_skill_tokens(tokens, known_skills)
    return len(skill_tokens) >= 2 and len(skill_tokens) >= max(2, len(tokens) - 1)


def _looks_like_jd_query(raw_query: str, query_text: str) -> bool:
    compact = "".join(query_text.split())
    return "\n" in raw_query or len(compact) >= 80 or len(_query_tokens(query_text)) >= 16


def _plan_must_terms(
    intent: str,
    query_text: str,
    constraints: dict[str, Any],
    known_skills: set[str],
) -> list[str]:
    if not query_text:
        return []
    if intent in {INTENT_EXACT_LOOKUP, INTENT_ENTITY}:
        return [query_text]
    if intent == INTENT_STRUCTURED:
        return list(constraints.get("skills") or [])
    if intent == INTENT_SKILL_COMBO:
        skill_terms = _known_skill_tokens(_query_tokens(query_text), known_skills)
        return skill_terms or _coverage_tokens(query_text)
    return []


def _plan_enable_dense(intent: str, semantic_query: str) -> bool:
    if intent in {INTENT_BROWSE, INTENT_EXACT_LOOKUP, INTENT_ENTITY}:
        return False
    return _use_dense(semantic_query)


def _plan_enable_rerank(intent: str, semantic_query: str) -> bool:
    return ENABLE_RERANK and intent in RERANK_INTENTS and _use_dense(semantic_query)


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


def _filter_browse_body(filters: list[dict[str, Any]], size: int) -> dict[str, Any]:
    return {
        "size": size,
        "query": {"bool": {"must": [{"match_all": {}}], "filter": filters}},
        "_source": {
            "excludes": SOURCE_EXCLUDES,
        },
    }


def _evidence_body(query_text: str, filters: list[dict[str, Any]], size: int) -> dict[str, Any]:
    return {
        "size": size,
        "query": {
            "bool": {
                "must": [_evidence_lexical_query(query_text)],
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


def _evidence_lexical_query(query_text: str) -> dict[str, Any]:
    normalized_degree = _normalize_highest_degree(query_text)
    scoring_query = {
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
                _section_query(
                    "skills",
                    _term_query("skills", query_text, 40, "evidence_exact:skills"),
                ),
                _profile_query(
                    _term_query("candidate.all_schools.keyword", query_text, 36, "evidence_exact:candidate_school")
                ),
                _profile_query(
                    _term_query("candidate.major.keyword", query_text, 34, "evidence_exact:candidate_major")
                ),
                _profile_query(
                    _term_query("application.company", query_text, 30, "evidence_exact:application_company")
                ),
                _profile_query(
                    _term_query("application.position_name.keyword", query_text, 30, "evidence_exact:position_name")
                ),
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
                {
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
                        "minimum_should_match": "2<70%",
                        "boost": 1,
                    }
                },
            ],
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


def _dense_confidence(response: dict[str, Any]) -> dict[str, Any]:
    scores = [
        float(hit.get("_score") or 0)
        for hit in response.get("hits", {}).get("hits", [])
    ]
    sample_size = len(scores)
    if sample_size < DENSE_ABSTAIN_MIN_SAMPLE_SIZE:
        return {
            "abstained": False,
            "reason": "insufficient_sample",
            "sample_size": sample_size,
        }

    top_score = max(scores)
    q1 = _percentile(scores, 0.25)
    q3 = _percentile(scores, 0.75)
    iqr = q3 - q1
    threshold = q3 + (DENSE_ABSTAIN_IQR_MULTIPLIER * iqr)
    abstained = not (iqr > 0 and top_score > threshold)
    return {
        "abstained": abstained,
        "reason": "flat_distribution" if abstained else "clear_head",
        "sample_size": sample_size,
        "top_score": round(top_score, 6),
        "q1": round(q1, 6),
        "q3": round(q3, 6),
        "iqr": round(iqr, 6),
        "threshold": round(threshold, 6),
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    ratio = position - lower
    return (sorted_values[lower] * (1 - ratio)) + (sorted_values[upper] * ratio)


def _run_hybrid_search(
    query_text: str,
    query_vector: list[float],
    filters: list[dict[str, Any]],
    rank_window_size: int,
    *,
    use_dense: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    requests_to_run = [
        (
            EVIDENCE_RETRIEVER,
            EVIDENCE_RRF_WEIGHT,
            _evidence_body(query_text, filters, rank_window_size),
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
                # One retriever failing should not crash the entire search;
                # degrade gracefully with whatever retriever(s) succeeded.
                logger.exception("%s retriever failed", name)
                warnings.append(f"{name} retriever failed: {exc}")
                continue
            response["_retriever_name"] = name
            response["_rrf_weight"] = weight
            if field:
                response["_vector_field"] = field
            if _is_dense_retriever(name):
                confidence = _dense_confidence(response)
                response["_dense_confidence"] = confidence
                if confidence["abstained"]:
                    response["hits"]["hits"] = []
                    warnings.append(
                        f"{name} abstained: dense score distribution has no clear head"
                    )
            responses.append(response)
    return responses, warnings


def _retriever_index_alias(retriever_name: str) -> str:
    if retriever_name in {EVIDENCE_RETRIEVER, EVIDENCE_DENSE_RETRIEVER}:
        return EVIDENCE_INDEX_ALIAS
    return INDEX_ALIAS

# ---------------------------------------------------------------------------
# manual RRF merge  (ES Basic license does not include built-in RRF)
# ---------------------------------------------------------------------------

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
) -> list[dict[str, Any]]:
    rrf_scores: dict[str, float] = {}
    dense_route_ranks: dict[str, list[int]] = {}
    dense_best_route_rank: dict[str, int] = {}
    evidence_route_ranks: dict[str, list[int]] = {}
    hit_map: dict[str, dict[str, Any]] = {}
    best_rank: dict[str, int] = {}
    term_coverage: dict[str, int] = {}
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
                term_coverage[doc_id] = max(
                    term_coverage.get(doc_id, 0),
                    _matched_term_coverage(hit),
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

    # --- Aggregate evidence BM25 to candidate level, then RRF once per candidate. ---
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
        evidence_contribution = EVIDENCE_RRF_WEIGHT / (RRF_RANK_CONSTANT + ev_rank)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + evidence_contribution
        best_rank[doc_id] = min(best_rank.get(doc_id, ev_rank), ev_rank)
        debug = retrieval_debug.get(doc_id)
        if debug:
            debug["evidence_group_rank"] = ev_rank
            debug["evidence_inner_score"] = round(float(evidence_pools[doc_id]["score"]), 6)
            debug["evidence_support_count"] = evidence_pools[doc_id]["support_count"]
            debug["evidence_rrf_contribution"] = round(evidence_contribution, 6)

    # --- Aggregate dense to candidate level, then RRF once per candidate. ---
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
        dense_contribution = DENSE_RRF_WEIGHT / (RRF_RANK_CONSTANT + dense_rank)
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
        debug["dense_outer_weight"] = round(DENSE_RRF_WEIGHT, 3)
        debug["dense_support_count"] = dense_pool["support_count"]
        debug["dense_pooling"] = "top_k_route_rerank"
        debug["dense_rrf_contribution"] = round(dense_contribution, 6)
        debug["dense_route_rank"] = best_dense_match.get("rank")
        debug["dense_score"] = best_dense_match.get("score")
        debug["dense_field"] = best_dense_match.get("field")
        debug["dense_retriever"] = best_dense_match.get("retriever")

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


def _rerank_results(
    query_text: str,
    results: list[dict[str, Any]],
    *,
    top_n: int = RERANK_TOP_N,
) -> tuple[list[dict[str, Any]], list[str]]:
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

    scored: list[tuple[float, int, dict[str, Any]]] = []
    for pre_rank, (result, score) in enumerate(zip(window, scores), start=1):
        rerank_score = float(score)
        item = dict(result)
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
        scored.append((rerank_score, pre_rank, item))

    scored.sort(key=lambda row: (-row[0], row[1]))
    reranked_window: list[dict[str, Any]] = []
    for rerank_rank, (_score, _pre_rank, item) in enumerate(scored, start=1):
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
    return [*reranked_window, *tail], []


def _score_rerank_documents(query_text: str, documents: list[str]) -> list[float]:
    from rerank_service import score_pairs

    return score_pairs(query_text, documents)


def _rerank_document(result: dict[str, Any]) -> str:
    source = result.get("source") or {}
    candidate = source.get("candidate") or result.get("candidate") or {}
    application = source.get("application") or result.get("application") or {}
    lines: list[str] = []

    _append_doc_line(lines, "技能", "、".join(source.get("skills") or result.get("skills") or []))

    lang = source.get("languages") or {}
    lang_parts = []
    if lang.get("english_exam_score"):
        lang_parts.append(lang["english_exam_score"])
    if lang.get("english_spoken_level"):
        lang_parts.append(f"口语{lang['english_spoken_level']}")
    _append_doc_line(lines, "语言", "，".join(lang_parts))

    for award in source.get("awards") or []:
        if award.get("has_award") not in (None, "否", False) and award.get("name"):
            text = " ".join(
                _clean_doc_text(award.get(field))
                for field in ("name", "level", "description")
                if award.get(field)
            )
            _append_doc_line(lines, "奖项", text)

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
    _append_doc_line(lines, "意向", "，".join(offer_parts))

    for edu in source.get("education") or []:
        text = " ".join(
            _clean_doc_text(edu.get(field))
            for field in ("school", "college", "degree", "major", "research_direction", "lab_name")
            if edu.get(field)
        )
        _append_doc_line(lines, "教育", text)

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


def _append_doc_line(lines: list[str], label: str, value: Any) -> None:
    text = _clean_doc_text(value)
    if text:
        lines.append(f"{label}: {text}")


def _clean_doc_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _strip_html(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value)).strip()


def _evidence_match_debug(
    hit: dict[str, Any],
    retriever_name: Any,
    rank: int,
) -> dict[str, Any]:
    source = hit.get("_source") or {}
    return {
        "retriever": retriever_name,
        "rank": rank,
        "score": round(float(hit.get("_score") or 0), 4),
        "evidence_id": source.get("evidence_id"),
        "section_type": source.get("section_type"),
        "title": source.get("title"),
        "snippet": _evidence_snippet(hit, retriever_name),
    }

def _evidence_snippet(hit: dict[str, Any], retriever_name: str | None = None) -> str:
    highlight = hit.get("highlight") or {}
    snippets: list[str] = []
    
    field_labels = {
        "title": "标题",
        "text": "正文",
        "skills_text": "技能词",
    }
    
    import re
    def _clean(s: str) -> str:
        return re.sub(r'\W+', '', re.sub(r'<[^>]+>', '', s))
        
    seen_fragments: set[str] = set()
    
    for field, label in field_labels.items():
        if field in highlight and highlight[field]:
            fragments = [str(item).strip() for item in highlight[field] if str(item).strip()]
            if fragments:
                joined_fragments = " ... ".join(fragments)
                clean_frag = _clean(joined_fragments)
                is_duplicate = False
                for seen in seen_fragments:
                    if clean_frag in seen or seen in clean_frag:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    snippets.append(f'<span class="snippet-label">{label}</span> ' + joined_fragments)
                    seen_fragments.add(clean_frag)
                
    if snippets:
        return _safe_snippet(" <span class=\"snippet-sep\">|</span> ".join(snippets))
    if retriever_name and _is_dense_retriever(retriever_name):
        source = hit.get("_source") or {}
        title = str(source.get("title") or "").strip()
        text = str(source.get("text") or "").strip()
        if title and text.startswith(title):
            raw = text
        else:
            raw = f"{title}：{text}" if title else text
        escaped_raw = html.escape(raw[:100] + ("..." if len(raw) > 100 else ""))
        return f'<span class="snippet-label dense-label">Dense 匹配</span> {escaped_raw}'
        
    # If it's a BM25 hit without highlights, it means it only matched global candidate attributes.
    # Return empty string to prevent irrelevant chunk text from cluttering the UI.
    return ""

def _matched_term_coverage(hit: dict[str, Any]) -> int:
    matched_queries = hit.get("matched_queries") or []
    return len(
        {
            query_name
            for query_name in matched_queries
            if isinstance(query_name, str) and query_name.startswith(COVERAGE_QUERY_PREFIX)
        }
    )


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


def _use_dense(query_text: str) -> bool:
    compact = "".join(query_text.split())
    if not compact or _looks_like_exact_lookup(compact):
        return False
    if len(query_text.split()) >= 2:
        return True
    return len(compact) >= 4


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
    retrieval_debug = hit.get("_retrieval_debug", {})

    snippets = (
        _highlight_snippets(highlight)
        or _debug_evidence_snippets(retrieval_debug)
        or [_default_snippet(projects, internships)]
    )
    years_experience = candidate.get("years_experience")

    return {
        "id": hit.get("_id"),
        "score": score,
        "candidate": candidate,
        "application": source.get("application", {}),
        "education_summary": _education_summary(candidate, education),
        "project_snippet": _safe_snippet("".join(f'<div style="margin-bottom: 6px;">{s}</div>' for s in snippets)) if snippets else "",
        "skills": source.get("skills", []),
        "years_experience": years_experience,
        "experience_display": _experience_display(years_experience),
        "retrieval_debug": retrieval_debug,
        "source": source,
    }


def _debug_evidence_snippets(debug: dict[str, Any]) -> list[str]:
    seen_evidence_ids: set[str] = set()
    seen_clean_texts: set[str] = set()
    snippets: list[str] = []
    
    import re
    def _clean(s: str) -> str:
        s = re.sub(r'<span class="snippet-label[^>]*>.*?</span>', '', s)
        s = re.sub(r'\[.*?\]', '', s)
        s = re.sub(r'<[^>]+>', '', s)
        return re.sub(r'\W+', '', s)
        
    def _add_matches(matches: list[dict[str, Any]]) -> None:
        for item in matches:
            ev_id = item.get("evidence_id")
            snippet = str(item.get("snippet") or "").strip()
            if snippet and ev_id not in seen_evidence_ids:
                sub_snippets = [s.strip() for s in snippet.split('<span class="snippet-sep">|</span>')]
                surviving_subs = []
                for sub in sub_snippets:
                    if not sub:
                        continue
                    clean_text = _clean(sub)
                    is_duplicate = False
                    if clean_text:
                        for seen in seen_clean_texts:
                            if clean_text in seen or seen in clean_text:
                                is_duplicate = True
                                break
                    if not is_duplicate:
                        surviving_subs.append(sub)
                        if clean_text:
                            seen_clean_texts.add(clean_text)
                if surviving_subs:
                    snippets.append(' <span class="snippet-sep">|</span> '.join(surviving_subs))
                if ev_id:
                    seen_evidence_ids.add(ev_id)
                    
    _add_matches(debug.get("evidence_matches") or [])
    _add_matches(debug.get("dense_matches") or [])
    return snippets


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
            "max_years": {"max": {"field": "candidate.years_experience"}},
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
    max_years_val = aggs.get("max_years", {}).get("value")
    facets["max_years"] = round(max_years_val, 1) if max_years_val else 5
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
