from __future__ import annotations

import html
import json
import logging
import re
import threading
import time
from collections import OrderedDict
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
DENSE_ABSTAIN_MIN_MARGIN = 0.02
EVIDENCE_POOL_EXTRA_WEIGHTS = (0.30, 0.15)
QUERY_TERM_COVERAGE_BOOST = 0.001
MAX_QUERY_COVERAGE_TERMS = 8
PARTIAL_TERMS_MINIMUM_SHOULD_MATCH = "70%"
ENTITY_MATCH_MINIMUM_SHOULD_MATCH = "70%"
COVERAGE_QUERY_PREFIX = "query_term:"
EVIDENCE_EXACT_QUERY_PREFIX = "evidence_exact:"
EVIDENCE_PHRASE_QUERY_PREFIX = "evidence_phrase:"
EVIDENCE_TERM_QUERY_PREFIX = "evidence_term:"
INTENT_BROWSE = "browse"
INTENT_LOOKUP = "lookup"
INTENT_KEYWORD = "keyword"
INTENT_SEMANTIC = "semantic"
EVIDENCE_VECTOR_FIELD = "evidence_vector"
QUERY_PARSER_PROVIDER = "deepseek"
QUERY_PARSER_MODEL_ID = "deepseek-v4-flash"
QUERY_PARSER_API_URL = "https://api.deepseek.com/chat/completions"
QUERY_PARSER_API_KEY = "sk-1eed8c88508842c2a023399a7ed6b5c0"
QUERY_PARSER_TIMEOUT_SECONDS = 30
QUERY_PARSER_MAX_VOCAB_ITEMS = 120
QUERY_PLAN_CACHE_TTL_SECONDS = 300
QUERY_PLAN_CACHE_MAX_ENTRIES = 512
# Strict json_schema for DeepSeek structured output. All keys are required and
# additionalProperties is closed so the model can't drift the shape; the
# sanitizer still normalizes values, but the schema removes whole-field omissions.
QUERY_PLAN_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intent", "lexical_query", "semantic_query", "constraints", "enable_dense"],
    "properties": {
        "intent": {"type": "string", "enum": ["browse", "lookup", "keyword", "semantic"]},
        "lexical_query": {"type": "string"},
        "semantic_query": {"type": "string"},
        "enable_dense": {"type": "boolean"},
        "constraints": {
            "type": "object",
            "additionalProperties": False,
            "required": ["degree", "min_degree", "cities", "skills", "min_years"],
            "properties": {
                "degree": {"type": ["string", "null"], "enum": ["博士", "硕士", "本科", None]},
                "min_degree": {"type": ["string", "null"], "enum": ["博士", "硕士", "本科", None]},
                "cities": {"type": "array", "items": {"type": "string"}},
                "skills": {"type": "array", "items": {"type": "string"}},
                "min_years": {"type": ["number", "null"]},
            },
        },
    },
}
SOURCE_EXCLUDES = [
    "raw_text",
    "raw_sections",
    "skills_text",
]
EVIDENCE_SOURCE_EXCLUDES = [EVIDENCE_VECTOR_FIELD]
DEGREE_ALIASES = {
    "博士研究生": "博士",
    "博士": "博士",
    "硕士研究生": "硕士",
    "硕士": "硕士",
    "学士": "本科",
    "本科": "本科",
}
DEGREE_ORDER = ("本科", "硕士", "博士")
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
    lexical_query: str
    semantic_query: str
    enable_dense: bool
    enable_rerank: bool
    rank_window_size: int

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "intent": self.intent,
            "lexical_query": self.lexical_query,
            "semantic_query": self.semantic_query,
            "constraints": self.constraints,
            "enable_dense": self.enable_dense,
            "enable_rerank": self.enable_rerank,
            "rank_window_size": self.rank_window_size,
            "filter_count": len(self.filters),
        }


_facets_cache: tuple[float, dict[str, Any]] | None = None
_filter_vocab_cache: tuple[float, dict[str, set[str]]] | None = None
_query_plan_cache: "OrderedDict[str, tuple[float, dict[str, Any]]]" = OrderedDict()
_query_plan_cache_lock = threading.Lock()
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
    facets = _load_facets()
    skill_vocab = _load_filter_vocab()["skills"] if skills else None
    explicit_filters = _build_filters(degree, cities, skills, min_years, skill_vocab=skill_vocab)
    plan = _plan_query(raw_query_text, explicit_filters, size=result_window_size, facets=facets)
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
            query_intent=plan.intent,
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
        "facets": facets,
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


def _normalize_degree_value(degree: str) -> str:
    normalized = _normalize_highest_degree(str(degree).strip())
    return normalized if normalized in DEGREE_ORDER else ""


def _degree_floor_from_query(raw_query: str) -> str:
    compact_query = re.sub(r"\s+", "", raw_query)
    for label, canonical in sorted(
        DEGREE_ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if canonical not in DEGREE_ORDER:
            continue
        if re.search(rf"{re.escape(label)}(?:及以上|及其以上|或以上|以上)", compact_query):
            return canonical
    return ""


def _degree_floor_filter(min_degree: str) -> dict[str, Any]:
    normalized = _normalize_degree_value(min_degree)
    if not normalized:
        return {"term": {"candidate.highest_degree": _normalize_highest_degree(str(min_degree))}}
    start = DEGREE_ORDER.index(normalized)
    return {"terms": {"candidate.highest_degree": list(DEGREE_ORDER[start:])}}


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
    return {
        "intent": INTENT_BROWSE,
        "lexical_query": "",
        "semantic_query": "",
        "filters": [],
        "constraints": {},
        "enable_dense": False,
        "enable_rerank": False,
        "parser": QUERY_PARSER_PROVIDER,
    }


def _plan_query(
    raw_query_text: str,
    explicit_filters: list[dict[str, Any]],
    *,
    size: int,
    facets: dict[str, Any] | None = None,
) -> QueryPlan:
    raw_query = raw_query_text.strip()
    parsed_query = _parse_query_with_llm(raw_query, facets=facets) if raw_query else _empty_parsed_query()
    lexical_query = str(parsed_query.get("lexical_query") or "").strip()
    semantic_query = str(parsed_query.get("semantic_query") or lexical_query).strip()
    intent = _normalize_plan_intent(parsed_query.get("intent"), raw_query, lexical_query, parsed_query.get("constraints") or {})

    # 防御：lookup / semantic 意图下 lexical_query 不应为空。
    # 极少数情况下 LLM 仍可能返回空字符串，导致搜索退化到浏览模式；
    # 此处用 raw_query 兜底。
    if not lexical_query and raw_query and intent in (INTENT_LOOKUP, INTENT_SEMANTIC):
        lexical_query = raw_query
        if not semantic_query:
            semantic_query = raw_query

    llm_filters = _filters_from_llm_constraints(parsed_query.get("constraints") or {})
    filters = [*explicit_filters, *llm_filters]
    enable_dense = bool(parsed_query.get("enable_dense")) and bool(semantic_query)
    enable_rerank = (
        ENABLE_RERANK
        and intent == INTENT_SEMANTIC
        and bool(raw_query)
        and bool(lexical_query)
        and bool(semantic_query)
    )
    return QueryPlan(
        raw_query=raw_query,
        intent=intent,
        filters=filters,
        constraints=parsed_query.get("constraints") or {},
        lexical_query=lexical_query,
        semantic_query=semantic_query,
        enable_dense=enable_dense,
        enable_rerank=enable_rerank,
        rank_window_size=max(size, RRF_RANK_WINDOW_SIZE),
    )


def _parse_query_with_llm(
    raw_query: str,
    facets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    query = raw_query.strip()
    if not query:
        return _empty_parsed_query()

    # Fast-path: unambiguous unique identifiers (email / phone / candidate_no /
    # position_code) are 100% resolvable by regex. Short-circuit the LLM so the
    # cheapest queries don't pay a network round-trip.
    fast_path = _lookup_fast_path(query)
    if fast_path is not None:
        return fast_path

    cached = _get_cached_query_plan(query)
    if cached is not None:
        return cached

    try:
        payload = _call_deepseek_query_parser(query, facets=facets)
    except Exception as exc:
        logger.exception("LLM query parser failed")
        fallback = _llm_parser_fallback(query)
        fallback["constraints"]["parser_warning"] = str(exc)
        # Don't cache fallbacks: a transient LLM outage shouldn't poison the
        # cache for the whole TTL.
        return fallback
    sanitized = _sanitize_llm_query_plan(payload, query)
    _set_cached_query_plan(query, sanitized)
    return sanitized


_LOOKUP_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_LOOKUP_PHONE_RE = re.compile(r"^1[3-9]\d{9}$")
_LOOKUP_CANDIDATE_NO_RE = re.compile(r"^[A-Za-z]?\d{6,}$")
_LOOKUP_POSITION_CODE_RE = re.compile(r"^[A-Za-z]\d{3,}$")


def _lookup_fast_path(query: str) -> dict[str, Any] | None:
    """Return a lookup plan for unambiguous single-token identifiers, else None.

    Only fires for a single whitespace-free token that matches a known
    identifier shape. Anything with spaces or mixed content falls through to
    the LLM, which is better at disambiguating intent.
    """
    if not query or any(ch.isspace() for ch in query):
        return None
    is_identifier = (
        _LOOKUP_EMAIL_RE.match(query)
        or _LOOKUP_PHONE_RE.match(query)
        or _LOOKUP_CANDIDATE_NO_RE.match(query)
        or _LOOKUP_POSITION_CODE_RE.match(query)
    )
    if not is_identifier:
        return None
    return {
        "intent": INTENT_LOOKUP,
        "lexical_query": query,
        "semantic_query": query,
        "filters": [],
        "constraints": {},
        "enable_dense": False,
        "enable_rerank": False,
        "parser": "regex_fast_path",
    }


def _query_plan_cache_key(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip().casefold()


def _get_cached_query_plan(query: str) -> dict[str, Any] | None:
    key = _query_plan_cache_key(query)
    now = time.monotonic()
    with _query_plan_cache_lock:
        entry = _query_plan_cache.get(key)
        if entry is None:
            return None
        expires_at, plan = entry
        if expires_at <= now:
            _query_plan_cache.pop(key, None)
            return None
        _query_plan_cache.move_to_end(key)
        return _deep_copy_plan(plan)


def _set_cached_query_plan(query: str, plan: dict[str, Any]) -> None:
    key = _query_plan_cache_key(query)
    expires_at = time.monotonic() + QUERY_PLAN_CACHE_TTL_SECONDS
    with _query_plan_cache_lock:
        _query_plan_cache[key] = (expires_at, _deep_copy_plan(plan))
        _query_plan_cache.move_to_end(key)
        while len(_query_plan_cache) > QUERY_PLAN_CACHE_MAX_ENTRIES:
            _query_plan_cache.popitem(last=False)


def _deep_copy_plan(plan: dict[str, Any]) -> dict[str, Any]:
    # Cached plans are mutated downstream (e.g. parser_warning injection), so
    # hand out independent copies to keep the cache entry pristine.
    return json.loads(json.dumps(plan, ensure_ascii=False))


def _call_deepseek_query_parser(
    raw_query: str,
    facets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt_context = _query_parser_prompt_context(facets)
    body = {
        "model": QUERY_PARSER_MODEL_ID,
        "messages": [
            {"role": "system", "content": _query_parser_system_prompt()},
            {
                "role": "user",
                "content": (
                    f"{prompt_context}\n\n"
                    f"用户原始 query:\n{raw_query.strip()}"
                ),
            },
        ],
        "response_format": _query_parser_response_format(),
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "max_tokens": 900,
        "stream": False,
    }
    data = _post_query_parser(body)
    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"query parser response has no content: {data}")
    return json.loads(_strip_json_fence(content))


# DeepSeek 较新接口支持 json_schema 严格结构化输出；旧接口只支持 json_object。
# 优先用 schema 约束（能根治 LLM 偶发返回空 lexical_query 等问题），
# 如果接口拒绝 schema 则自动回退到 json_object，并记住该结果避免重复试探。
_structured_output_supported = True
_structured_output_lock = threading.Lock()


def _query_parser_response_format() -> dict[str, Any]:
    with _structured_output_lock:
        use_schema = _structured_output_supported
    if not use_schema:
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "resume_query_plan",
            "strict": True,
            "schema": QUERY_PLAN_JSON_SCHEMA,
        },
    }


def _post_query_parser(body: dict[str, Any]) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {QUERY_PARSER_API_KEY}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        QUERY_PARSER_API_URL,
        headers=headers,
        json=body,
        timeout=QUERY_PARSER_TIMEOUT_SECONDS,
    )
    # If the endpoint can't honor a json_schema response_format, retry once with
    # the widely-supported json_object mode and stop attempting schema mode.
    status_code = getattr(response, "status_code", None)
    if (
        status_code in (400, 422)
        and body.get("response_format", {}).get("type") == "json_schema"
    ):
        global _structured_output_supported
        with _structured_output_lock:
            _structured_output_supported = False
        logger.warning(
            "query parser json_schema rejected (HTTP %s); falling back to json_object",
            response.status_code,
        )
        retry_body = {**body, "response_format": {"type": "json_object"}}
        response = requests.post(
            QUERY_PARSER_API_URL,
            headers=headers,
            json=retry_body,
            timeout=QUERY_PARSER_TIMEOUT_SECONDS,
        )
    response.raise_for_status()
    return response.json()


def _query_parser_system_prompt() -> str:
    return (
        "你是一个招聘/简历检索系统的 query planner。"
        "你的唯一任务是把用户自由文本解析成可执行的检索计划，必须只输出一个 JSON object，不能输出 Markdown。"
        "不要根据候选人库是否存在来臆测结果，只解析用户意图。\n\n"
        "输出 schema:\n"
        "{\n"
        '  "intent": "browse|lookup|keyword|semantic",\n'
        '  "lexical_query": "给 BM25/短语/精确检索使用的文本；不得包含已抽取到 constraints 的学历/城市/年限纯筛选词；纯筛选时为空字符串",\n'
        '  "semantic_query": "给 embedding/rerank 使用的语义需求文本；不得包含已抽取到 constraints 的学历/城市/年限纯筛选词；不启用语义时为空字符串",\n'
        '  "constraints": {\n'
        '    "degree": null|"博士"|"硕士"|"本科",\n'
        '    "min_degree": null|"博士"|"硕士"|"本科",\n'
        '    "cities": ["北京"],\n'
        '    "skills": ["Python"],\n'
        '    "min_years": null|0.5\n'
        "  },\n"
        '  "enable_dense": true\n'
        "}\n\n"
        "判断准则:\n"
        "- lookup: 候选人编号、岗位编号、手机号、邮箱等唯一标识符直接定位查询；dense=false。\n"
        "- keyword: 关键词检索——学校、公司、专业、姓名、岗位名等实体查询；dense=false。注意：只把实体核心名称放入 lexical_query，修饰词如\"实习\"\"大学\"\"硕士\"等不要混进去。例如\"阿里巴巴实习\" → lexical_query=\"阿里巴巴\"。\n"
        "- semantic: 语义检索——自然语言能力描述（如\"做过大规模分布式系统架构设计\"）、多技能组合（如\"Python PyTorch NLP 大模型\"）、长岗位描述或 JD 粘贴；dense=true。\n"
        "- 学历/城市/年限只是结构化 filter，不决定 intent。抽掉这些 filter 后，如果剩余需求是技能组合、工程能力、项目经验或 JD，intent 必须是 semantic；如果剩余需求只是学校/公司/专业/姓名/岗位实体，intent 才是 keyword。\n"
        "- 学历精确要求（如\"本科\"）放入 degree；学历下限要求（如\"本科及以上\"\"硕士及以上\"）放入 min_degree，不要放入 degree。\n"
        "- 负向约束如\"不要纯推荐排序\"不要变成硬过滤；把核心正向需求放入 semantic_query，负向信息可留在 semantic_query 里。\n"
        "- 即使某些技能也放入 constraints.skills，也不要从 lexical_query/semantic_query 中删除这些技能词；技能词仍然是 BM25 和语义召回的重要线索。\n"
        "- 已放入 constraints 的学历、城市、年限不要再出现在 lexical_query 或 semantic_query 中，避免污染词面检索和 embedding。\n"
        "- 长 JD 或长自然语言需求中，lexical_query 不要复读原句；必须压缩为核心技能、实体、系统类型、业务短语和高价值检索词，去掉\"岗位\"\"职责\"\"要求\"\"负责\"\"熟悉\"\"优先\"等低信息量叙述词。semantic_query 才保留完整原文和上下文。\n"
        "- keyword 意图下，如果用户只输入了学历、城市、年限这类纯筛选条件（没有任何检索关键词），则 lexical_query 和 semantic_query 都返回空字符串。\n"
        "- skills 可记录用户明确点名的技能，供解释与检索调试；泛化语义能力不要硬塞进 skills。\n"
        "\n示例:\n"
        "输入: zhangwei_mock@example.com\n"
        '输出: {"intent":"lookup","lexical_query":"zhangwei_mock@example.com","semantic_query":"zhangwei_mock@example.com","constraints":{"degree":null,"min_degree":null,"cities":[],"skills":[],"min_years":null},"enable_dense":false}\n'
        "输入: Columbia University 哥伦比亚大学\n"
        '输出: {"intent":"keyword","lexical_query":"Columbia University 哥伦比亚大学","semantic_query":"","constraints":{"degree":null,"min_degree":null,"cities":[],"skills":[],"min_years":null},"enable_dense":false}\n'
        "输入: 北京 硕士 4年以上 RAG LangChain\n"
        '输出: {"intent":"semantic","lexical_query":"RAG LangChain","semantic_query":"RAG LangChain","constraints":{"degree":null,"min_degree":"硕士","cities":["北京"],"skills":["RAG","LangChain"],"min_years":4},"enable_dense":true}\n'
        "输入: 深圳 本科 5年以上 Golang Kubernetes\n"
        '输出: {"intent":"semantic","lexical_query":"Golang Kubernetes","semantic_query":"Golang Kubernetes","constraints":{"degree":"本科","min_degree":null,"cities":["深圳"],"skills":["Golang","Kubernetes"],"min_years":5},"enable_dense":true}\n'
        "输入: 杭州 硕士 8年以上 Java 架构\n"
        '输出: {"intent":"semantic","lexical_query":"Java 架构","semantic_query":"Java 架构","constraints":{"degree":null,"min_degree":"硕士","cities":["杭州"],"skills":["Java"],"min_years":8},"enable_dense":true}\n'
        "输入: 北京交通大学\n"
        '输出: {"intent":"keyword","lexical_query":"北京交通大学","semantic_query":"","constraints":{"degree":null,"min_degree":null,"cities":[],"skills":[],"min_years":null},"enable_dense":false}\n'
        "输入: 岗位：LLM/RAG 应用工程师。职责：负责企业知识库问答、文档解析、向量检索、召回排序、Prompt 设计和模型微调，能用 Python、PyTorch、LangChain 或 LlamaIndex 做工程落地。要求：熟悉 RAG 评测、长文本处理和业务系统集成，有 ToB 知识库项目经验优先。\n"
        '输出: {"intent":"semantic","lexical_query":"LLM RAG 企业知识库问答 文档解析 向量检索 召回排序 Prompt 模型微调 Python PyTorch LangChain LlamaIndex RAG评测 长文本处理 ToB知识库","semantic_query":"岗位：LLM/RAG 应用工程师。职责：负责企业知识库问答、文档解析、向量检索、召回排序、Prompt 设计和模型微调，能用 Python、PyTorch、LangChain 或 LlamaIndex 做工程落地。要求：熟悉 RAG 评测、长文本处理和业务系统集成，有 ToB 知识库项目经验优先。","constraints":{"degree":null,"min_degree":null,"cities":[],"skills":["LLM","RAG","Prompt","Python","PyTorch","LangChain","LlamaIndex"],"min_years":null},"enable_dense":true}\n'
    )


def _query_parser_prompt_context(facets: dict[str, Any] | None = None) -> str:
    vocab = _filter_vocab_for_prompt(facets)
    return (
        "可用规范化参考词表如下。词表只是帮助规范输出，不要把不在词表里的真实用户约束丢弃。\n"
        f"学历: {', '.join(vocab['degrees']) or '博士, 硕士, 本科'}\n"
        f"城市: {', '.join(vocab['cities']) or '无'}\n"
        f"技能样例: {', '.join(vocab['skills']) or '无'}"
    )


def _filter_vocab_for_prompt(facets: dict[str, Any] | None = None) -> dict[str, list[str]]:
    if facets is not None:
        degrees = sorted(_facet_keys(facets, "degrees"))
        cities = sorted(_facet_keys(facets, "cities"))
        skills = sorted(_facet_keys(facets, "skills"), key=_skill_label_sort_key)
    else:
        try:
            vocab = _load_filter_vocab()
            degrees = sorted(vocab["degrees"])
            cities = sorted(vocab["cities"])
            skills = sorted(vocab["skills"], key=_skill_label_sort_key)
        except Exception:
            logger.exception("loading parser vocabulary failed")
            degrees, cities, skills = [], [], []
    return {
        "degrees": degrees[:QUERY_PARSER_MAX_VOCAB_ITEMS],
        "cities": cities[:QUERY_PARSER_MAX_VOCAB_ITEMS],
        "skills": skills[:QUERY_PARSER_MAX_VOCAB_ITEMS],
    }


def _strip_json_fence(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _sanitize_llm_query_plan(payload: dict[str, Any], raw_query: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return _llm_parser_fallback(raw_query)

    constraints = payload.get("constraints") if isinstance(payload.get("constraints"), dict) else {}
    cleaned_constraints: dict[str, Any] = {}
    min_degree = _normalize_degree_value(str(constraints.get("min_degree") or "").strip())
    min_degree = min_degree or _degree_floor_from_query(raw_query)
    degree = constraints.get("degree")
    if min_degree:
        cleaned_constraints["min_degree"] = min_degree
    elif degree:
        cleaned_constraints["degree"] = _normalize_highest_degree(str(degree).strip())
    cities = _clean_string_list(constraints.get("cities"))
    if cities:
        cleaned_constraints["cities"] = cities
    skills = _clean_string_list(constraints.get("skills"))
    if skills:
        cleaned_constraints["skills"] = skills
    min_years = _clean_float(constraints.get("min_years"))
    if min_years is not None and min_years > 0:
        cleaned_constraints["min_years"] = min_years

    lexical_query = str(payload.get("lexical_query") or "").strip()
    semantic_query = str(payload.get("semantic_query") or "").strip()
    intent = _normalize_plan_intent(payload.get("intent"), raw_query, lexical_query, cleaned_constraints)
    enable_dense = bool(payload.get("enable_dense")) and bool(semantic_query)

    return {
        "intent": intent,
        "lexical_query": lexical_query,
        "semantic_query": semantic_query,
        "filters": [],
        "constraints": cleaned_constraints,
        "enable_dense": enable_dense,
        "enable_rerank": False,
        "parser": QUERY_PARSER_PROVIDER,
    }


def _llm_parser_fallback(raw_query: str) -> dict[str, Any]:
    query = raw_query.strip()
    return {
        "intent": INTENT_SEMANTIC if query else INTENT_BROWSE,
        "lexical_query": query,
        "semantic_query": query,
        "filters": [],
        "constraints": {},
        "enable_dense": False,
        "enable_rerank": False,
        "parser": QUERY_PARSER_PROVIDER,
    }


def _normalize_plan_intent(
    raw_intent: Any,
    raw_query: str,
    lexical_query: str,
    constraints: dict[str, Any],
) -> str:
    allowed = {
        INTENT_BROWSE,
        INTENT_LOOKUP,
        INTENT_KEYWORD,
        INTENT_SEMANTIC,
    }
    intent = str(raw_intent or "").strip()
    if intent in allowed:
        return intent
    if not raw_query.strip():
        return INTENT_KEYWORD if constraints else INTENT_BROWSE
    if constraints:
        return INTENT_KEYWORD
    return INTENT_SEMANTIC if lexical_query else INTENT_BROWSE


def _filters_from_llm_constraints(constraints: dict[str, Any]) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []
    min_degree = constraints.get("min_degree")
    if min_degree:
        filters.append(_degree_floor_filter(str(min_degree)))
        degree = None
    else:
        degree = constraints.get("degree")
    if degree:
        filters.append({"term": {"candidate.highest_degree": _normalize_highest_degree(str(degree))}})
    cities = _clean_string_list(constraints.get("cities"))
    if cities:
        filters.append({"terms": {"application.expected_work_cities": _dedupe(cities)}})
    min_years = _clean_float(constraints.get("min_years"))
    if min_years is not None and min_years > 0:
        filters.append({"range": {"candidate.years_experience": {"gte": min_years}}})
    return filters


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
        _partial_terms_query(query_text),
    ]
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


def _entity_field_query(
    keyword_field: str,
    text_field: str,
    query_text: str,
    boost: float,
    term_name: str,
    match_name: str,
) -> dict[str, Any]:
    """对实体字段组合 term + match，兼顾精确匹配和分词召回。

    term 查询：当 query 本身就是一个完整 token 时精准命中（如 "阿里巴巴"）。
    match 查询：当 query 含修饰词时（如 "阿里巴巴实习"），经分词后在 text 字段
    上做受控 OR 匹配，降低 boost 避免排在精准匹配前面。
    """
    return {
        "dis_max": {
            "tie_breaker": 0.0,
            "queries": [
                {"term": {keyword_field: {"value": query_text, "boost": boost, "_name": f"{term_name}:W{boost}"}}},
                {
                    "match": {
                        text_field: {
                            "query": query_text,
                            "operator": "or",
                            "minimum_should_match": ENTITY_MATCH_MINIMUM_SHOULD_MATCH,
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
    accepted_threshold = threshold + DENSE_ABSTAIN_MIN_MARGIN
    abstained = not (iqr > 0 and top_score > accepted_threshold)
    return {
        "abstained": abstained,
        "reason": "flat_distribution" if abstained else "clear_head",
        "sample_size": sample_size,
        "top_score": round(top_score, 6),
        "q1": round(q1, 6),
        "q3": round(q3, 6),
        "iqr": round(iqr, 6),
        "threshold": round(threshold, 6),
        "min_margin": round(DENSE_ABSTAIN_MIN_MARGIN, 6),
        "accepted_threshold": round(accepted_threshold, 6),
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
