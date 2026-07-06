"""集中配置：环境变量、路径、检索/排序常量、规范化别名表。

这里只放"值"，不放逻辑。任何模块需要常量都从这里导入，避免常量散落在业务代码里、
也避免循环依赖（config 不依赖任何业务模块）。
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- 路径 ---
# config.py 位于 resume_search/ 下，项目根是它的上两级。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT
WEB_DIR = PROJECT_ROOT / "web"
SCHOOL_TIERS_PATH = PROJECT_ROOT / "school_tiers.json"

# --- Elasticsearch ---
ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
INDEX_ALIAS = "resumes_current"
EVIDENCE_INDEX_ALIAS = "resume_evidence_current"
ES_REQUEST_TIMEOUT_SECONDS = 20

# --- RRF 融合 / 检索窗口 ---
RRF_RANK_CONSTANT = 60
RRF_RANK_WINDOW_SIZE = 1000
DEFAULT_SEARCH_LIMIT = 100
MAX_BROWSE_RESULT_SIZE = 1000
KNN_NUM_CANDIDATES = 300
DENSE_RANK_WINDOW_SIZE = 300

# --- 缓存 TTL ---
FACETS_CACHE_TTL_SECONDS = 60
FILTER_VOCAB_CACHE_TTL_SECONDS = 300
QUERY_PLAN_CACHE_TTL_SECONDS = 300
QUERY_PLAN_CACHE_MAX_ENTRIES = 512

# --- facet 聚合规模 ---
SKILL_FACET_AGG_SIZE = 200
SKILL_FACET_DISPLAY_SIZE = 30

# --- 检索路由名 / RRF 权重 ---
DENSE_RETRIEVER = "dense"
EVIDENCE_RETRIEVER = "evidence"
EVIDENCE_DENSE_RETRIEVER = "evidence_dense"
DENSE_RRF_WEIGHT = 1.0
EVIDENCE_RRF_WEIGHT = 1.2
EVIDENCE_DENSE_RRF_WEIGHT = 1.0
# 意图感知的路由权重：semantic 意图（JD / 长自然语言 / 多技能组合）下，BM25 词面
# 的价值低（长文本压缩成关键词本就有损），应让 dense 语义召回主导；keyword / lookup
# 等实体/精确查询则保持 BM25 主导。这样 JD 匹配走"原文 dense 召回 + cross-encoder
# 精排"的正解，而不是依赖压缩关键词的 BM25。
EVIDENCE_RRF_WEIGHT_SEMANTIC = 1.0
EVIDENCE_DENSE_RRF_WEIGHT_SEMANTIC = 1.5
EVIDENCE_POOL_EXTRA_WEIGHTS = (0.30, 0.15)

# --- Rerank ---
ENABLE_RERANK = True
RERANK_TOP_N = 20
# Cross-encoder 重排分带有绝对的 query-doc 相关性语义（不像余弦相似度，余弦只
# 在同一 query 的批次内可比）。实测分布：库中无相关候选的离域 query（如量子计算、
# SAP、iOS 内核——本库没有这类简历）最高分都 <=0.27，而真正在域内的需求（含偏门
# 但确实存在的方向，如某些 CV/ML 岗）最低也到 ~0.49，中间是一条 0.27~0.49 的空白带。
# 若重排窗口中最相关的候选都低于此地板，说明 reranker 在告诉我们这里没有真正相关的人，
# 于是弃权返回空，而不是让 RRF 用噪声填满整页。地板取在空白带内（偏离域一侧留足余量），
# 既挡住离域噪声，又不会把在域内的边界 JD 误杀成"零召回"。这是基于模型自身输出的绝对
# 相关性判定，不是挂在 query 文本模式上的规则——它从不检查 query 内容。
RERANK_RELEVANCE_FLOOR = 0.35

# --- 查询词覆盖 / 部分召回 ---
QUERY_TERM_COVERAGE_BOOST = 0.001
MAX_QUERY_COVERAGE_TERMS = 8
PARTIAL_TERMS_MINIMUM_SHOULD_MATCH = "70%"

# --- matched_queries 命名前缀（用于 coverage / tier 判定）---
COVERAGE_QUERY_PREFIX = "query_term:"
EVIDENCE_EXACT_QUERY_PREFIX = "evidence_exact:"
EVIDENCE_PHRASE_QUERY_PREFIX = "evidence_phrase:"
EVIDENCE_TERM_QUERY_PREFIX = "evidence_term:"

# --- 意图 ---
INTENT_BROWSE = "browse"
INTENT_LOOKUP = "lookup"
INTENT_KEYWORD = "keyword"
INTENT_SEMANTIC = "semantic"

# --- 向量字段 ---
EVIDENCE_VECTOR_FIELD = "evidence_vector"

# --- LLM Query Planner ---
QUERY_PARSER_PROVIDER = "qwen"
QUERY_PARSER_MODEL_ID = "qwen3.5-flash"
QUERY_PARSER_API_URL = os.environ.get(
    "QUERY_PARSER_API_URL",
    "https://ws-nl8tvztfpss60i8t.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions",
)
QUERY_PARSER_API_KEY = os.environ.get("QUERY_PARSER_API_KEY", "")
QUERY_PARSER_TIMEOUT_SECONDS = 30
QUERY_PARSER_MAX_VOCAB_ITEMS = 120

# --- min_years 软容差 ---
# "N 年以上" 是用户的模糊表达，不是精确边界：一个 3.9 年的候选人对 "4 年以上"
# 几乎必然算命中。对 min_years 过滤施加软容差带，召回边界候选，真实年限仍由排序体现。
MIN_YEARS_TOLERANCE_RATIO = 0.10
MIN_YEARS_TOLERANCE_FLOOR = 0.5

# --- _source 裁剪 ---
SOURCE_EXCLUDES = [
    "raw_text",
    "raw_sections",
    "skills_text",
]
EVIDENCE_SOURCE_EXCLUDES = [EVIDENCE_VECTOR_FIELD]
