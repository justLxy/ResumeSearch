"""查询规划：把用户自由文本解析成检索计划（意图 / 词面 / 语义 / 约束）。

优先走正则快速通道（编号/手机号/邮箱等唯一标识符），否则调用 LLM query planner，
带 TTL 缓存与失败降级。产出 QueryPlan 供 search 编排层使用。
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import OrderedDict
from typing import Any

from resume_search.config import (
    ENABLE_RERANK,
    INTENT_BROWSE,
    INTENT_KEYWORD,
    INTENT_LOOKUP,
    INTENT_SEMANTIC,
    QUERY_PARSER_API_KEY,
    QUERY_PARSER_API_URL,
    QUERY_PARSER_MAX_VOCAB_ITEMS,
    QUERY_PARSER_MODEL_ID,
    QUERY_PARSER_PROVIDER,
    QUERY_PARSER_TIMEOUT_SECONDS,
    QUERY_PLAN_CACHE_MAX_ENTRIES,
    QUERY_PLAN_CACHE_TTL_SECONDS,
    RRF_RANK_WINDOW_SIZE,
)
from resume_search.domain.models import QueryPlan
from resume_search.infrastructure import llm_client
from resume_search.services.facets import _facet_keys, _load_filter_vocab
from resume_search.services.filters import _filters_from_llm_constraints
from resume_search.services.normalization import (
    _clean_float,
    _clean_string_list,
    _normalize_degree_list,
    _normalize_school_tier_list,
    _skill_label_sort_key,
)

logger = logging.getLogger(__name__)

# 查询计划的 LRU + TTL 缓存。
_query_plan_cache: "OrderedDict[str, tuple[float, dict[str, Any]]]" = OrderedDict()
_query_plan_cache_lock = threading.Lock()


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

    # 快速通道：邮箱 / 手机号 / 候选人编号 / 岗位编号这类唯一标识符可被正则 100%
    # 判定。直接短路 LLM，避免最廉价的查询白白付一次网络往返。
    fast_path = _lookup_fast_path(query)
    if fast_path is not None:
        return fast_path

    cached = _get_cached_query_plan(query)
    if cached is not None:
        return cached

    try:
        payload = _call_query_parser_llm(query, facets=facets)
    except Exception as exc:
        logger.exception("LLM query parser failed")
        fallback = _llm_parser_fallback(query)
        fallback["constraints"]["parser_warning"] = str(exc)
        # 不缓存 fallback：LLM 短暂故障不应污染整个 TTL 内的缓存。
        return fallback
    sanitized = _sanitize_llm_query_plan(payload, query)
    _set_cached_query_plan(query, sanitized)
    return sanitized


_LOOKUP_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


_LOOKUP_PHONE_RE = re.compile(r"^1[3-9]\d{9}$")


_LOOKUP_CANDIDATE_NO_RE = re.compile(r"^[A-Za-z]?\d{6,}$")


_LOOKUP_POSITION_CODE_RE = re.compile(r"^[A-Za-z]\d{3,}$")


def _lookup_fast_path(query: str) -> dict[str, Any] | None:
    """对无歧义的单 token 标识符返回 lookup 计划，否则返回 None。

    仅对不含空白、且匹配已知标识符形态的单一 token 生效。带空格或混合内容的
    查询一律交给 LLM——它更擅长消解意图歧义。
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
    # 缓存的计划在下游会被改写（如注入 parser_warning），因此返回独立副本，
    # 保持缓存条目不被污染。
    return json.loads(json.dumps(plan, ensure_ascii=False))


def _call_query_parser_llm(
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
        "response_format": {"type": "json_object"},
        # 关闭思考链：planner 是结构化抽取任务，思考只增延迟无收益。不同 provider
        # 的关闭参数名不同——qwen 用 enable_thinking=False（顶层），deepseek/豆包用
        # thinking={"type":"disabled"}。两者都发，各 provider 忽略自己不认的键。
        "enable_thinking": False,
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


def _post_query_parser(body: dict[str, Any]) -> dict[str, Any]:
    # HTTP 边界在 infrastructure.llm_client；这里只做业务编排。
    return llm_client.post_query_parser(body)


def _query_parser_system_prompt() -> str:
    return (
        "你是招聘/简历检索系统的 query planner，把用户自由文本解析成检索计划。"
        "只输出一个 JSON object，不要 Markdown，不要臆测候选人库是否存在。\n\n"
        "schema:\n"
        "{\n"
        '  "intent": "browse|lookup|keyword|semantic",\n'
        '  "lexical_query": "给 BM25/精确检索的文本；纯筛选时为空",\n'
        '  "semantic_query": "给 embedding/rerank 的语义需求文本；不启用语义时为空",\n'
        '  "constraints": {"degrees": ["本科","硕士"], "cities": ["北京"], "skills": ["Python"], "min_years": null|0.5, "school_tiers": ["985"]},\n'
        '  "enable_dense": true|false\n'
        "}\n\n"
        "intent 判定（学历/城市/年限只是 filter，不决定 intent；抽掉它们后看剩余需求）:\n"
        "- lookup: 编号/手机号/邮箱等唯一标识符定位，dense=false。\n"
        "- keyword: 学校/公司/专业/姓名/岗位名等实体查询，dense=false。\n"
        "- semantic: 能力描述、多技能组合、长 JD 等需要语义理解的需求，dense=true。\n\n"
        "字段规则:\n"
        "- degrees: 仅当 query 里**字面出现**学历词（本科/学士/硕士/研究生/博士/及以上等）时才填，由 LLM 展开——\"硕士\"→[硕士]；\"本科或硕士\"→[本科,硕士]；\"本科及以上\"→[本科,硕士,博士]；\"硕士及以上\"→[硕士,博士]。query 没提学历就必须是 []——绝不能因为学校/专业/岗位\"看起来高端\"就臆测学历，那会把合格候选人误过滤掉。\n"
        "- school_tiers: 院校档位列表（可多选，并列/或的关系取并集）。仅当 query 里**字面出现**院校档位词时才填，每项取值之一：985 / 211 / 双一流 / c9（C9联盟）/ qs50_overseas（海外名校、留学、留学生、海归、QS前100等）/ 其他（普通院校、双非）。\"985\"→[\"985\"]；\"海归\"或\"留学生\"→[\"qs50_overseas\"]；\"留学生或985\"→[\"qs50_overseas\",\"985\"]；\"双非\"→[\"其他\"]。query 没提院校档位就必须是 []——绝不能因为岗位/专业\"看起来高端\"就臆测院校档位。具体某所学校名（如\"北京大学\"\"东南大学\"）不是档位词，仍走 lexical_query，school_tiers 留 []。\n"
        "- cities: 期望工作/所在城市列表。仅当 query 里**字面出现**独立的城市意图词（如\"北京\"\"在上海\"\"深圳工作\"）时才填。**绝不能从校名/公司名/机构名的城市前缀里抠城市**——\"北京邮电大学\"\"上海交通大学\"\"西安交通大学\"里的\"北京/上海/西安\"是学校名的一部分，不是工作城市，cities 必须留 []，整串学校名走 lexical_query。城市一旦填了会变成硬过滤，把在外地工作的合格候选人整个排除，所以拿不准就留 []。\n"
        "- lexical_query: 只放实体核心名/技能/高价值检索词，去掉已抽到 constraints 的学历/城市/年限，也去掉\"实习\"\"岗位\"\"职责\"\"要求\"\"熟悉\"等修饰/低信息词。长 JD 必须压缩成关键词串，不要复读原句。\n"
        "- semantic_query: 保留完整语义需求与上下文（长 JD 放原文）；负向约束（如\"不要纯推荐\"）也留在这里，不要变成硬过滤。\n"
        "- skills: 用户明确点名的技能，仍要保留在 lexical/semantic_query 里（是召回线索）；泛化能力不要硬塞。\n"
        "- 纯筛选（只有学历/城市/年限、无检索词）时 lexical_query 和 semantic_query 都为空。\n"
        "\n示例:\n"
        "输入: zhangwei_mock@example.com\n"
        '输出: {"intent":"lookup","lexical_query":"zhangwei_mock@example.com","semantic_query":"zhangwei_mock@example.com","constraints":{"degrees":[],"cities":[],"skills":[],"min_years":null,"school_tiers":[]},"enable_dense":false}\n'
        "输入: 东南大学本科或者硕士\n"
        '输出: {"intent":"keyword","lexical_query":"东南大学","semantic_query":"","constraints":{"degrees":["本科","硕士"],"cities":[],"skills":[],"min_years":null,"school_tiers":[]},"enable_dense":false}\n'
        "输入: 南京大学\n"
        '输出: {"intent":"keyword","lexical_query":"南京大学","semantic_query":"","constraints":{"degrees":[],"cities":[],"skills":[],"min_years":null,"school_tiers":[]},"enable_dense":false}\n'
        "输入: 北京邮电大学 并发长连接\n"
        '输出: {"intent":"keyword","lexical_query":"北京邮电大学 并发长连接","semantic_query":"","constraints":{"degrees":[],"cities":[],"skills":[],"min_years":null,"school_tiers":[]},"enable_dense":false}\n'
        "输入: 985硕士 做过推荐系统\n"
        '输出: {"intent":"semantic","lexical_query":"推荐系统","semantic_query":"做过推荐系统","constraints":{"degrees":["硕士"],"cities":[],"skills":[],"min_years":null,"school_tiers":["985"]},"enable_dense":true}\n'
        "输入: 留学生 或者 985\n"
        '输出: {"intent":"keyword","lexical_query":"","semantic_query":"","constraints":{"degrees":[],"cities":[],"skills":[],"min_years":null,"school_tiers":["qs50_overseas","985"]},"enable_dense":false}\n'
        "输入: 海归 计算机视觉\n"
        '输出: {"intent":"semantic","lexical_query":"计算机视觉","semantic_query":"计算机视觉","constraints":{"degrees":[],"cities":[],"skills":[],"min_years":null,"school_tiers":["qs50_overseas"]},"enable_dense":true}\n'
        "输入: 北京 硕士及以上 4年以上 RAG LangChain\n"
        '输出: {"intent":"semantic","lexical_query":"RAG LangChain","semantic_query":"RAG LangChain","constraints":{"degrees":["硕士","博士"],"cities":["北京"],"skills":["RAG","LangChain"],"min_years":4,"school_tiers":[]},"enable_dense":true}\n'
        "输入: 岗位：LLM/RAG 应用工程师。职责：负责企业知识库问答、文档解析、向量检索、召回排序、Prompt 设计和模型微调，能用 Python、PyTorch、LangChain 或 LlamaIndex 做工程落地。要求：熟悉 RAG 评测、长文本处理和业务系统集成，有 ToB 知识库项目经验优先。\n"
        '输出: {"intent":"semantic","lexical_query":"LLM RAG 企业知识库问答 文档解析 向量检索 召回排序 Prompt 模型微调 Python PyTorch LangChain LlamaIndex RAG评测 长文本处理 ToB知识库","semantic_query":"岗位：LLM/RAG 应用工程师。职责：负责企业知识库问答、文档解析、向量检索、召回排序、Prompt 设计和模型微调，能用 Python、PyTorch、LangChain 或 LlamaIndex 做工程落地。要求：熟悉 RAG 评测、长文本处理和业务系统集成，有 ToB 知识库项目经验优先。","constraints":{"degrees":[],"cities":[],"skills":["LLM","RAG","Prompt","Python","PyTorch","LangChain","LlamaIndex"],"min_years":null},"enable_dense":true}\n'
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
    # 学历统一为"可接受学历集合"：LLM 已把精确/下限/枚举都展开成具体学历列表，
    # 后端只做规范化，不再区分三种情况。
    degrees = _normalize_degree_list(constraints.get("degrees"))
    if degrees:
        cleaned_constraints["degrees"] = degrees
    # 城市抽取的正确性由 planner prompt 的 cities 字段规则保证（不得从校名/公司名
    # 的城市前缀抠城市），后端只做规范化清洗，不再有 substring 兜底规则。
    cities = _clean_string_list(constraints.get("cities"))
    if cities:
        cleaned_constraints["cities"] = cities
    skills = _clean_string_list(constraints.get("skills"))
    if skills:
        cleaned_constraints["skills"] = skills
    min_years = _clean_float(constraints.get("min_years"))
    if min_years is not None and min_years > 0:
        cleaned_constraints["min_years"] = min_years
    school_tiers = _normalize_school_tier_list(
        constraints.get("school_tiers") or constraints.get("school_tier")
    )
    if school_tiers:
        cleaned_constraints["school_tiers"] = school_tiers

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


