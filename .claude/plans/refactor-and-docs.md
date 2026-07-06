# 计划：完善 Project Review + 更新 README + Clean Architecture 重构

## 决策（已与用户确认）
- **目标结构**：新建 `resume_search/` 包（domain / infrastructure / services / api 分层），根目录留薄 shim
- **范围**：全部模块（检索链路 app.py + 索引链路 import_to_es.py + 其余）
- **验证**：同步更新测试的 import / monkeypatch 路径，保证 103 个测试全绿

## 现状约束（探查结论）
- `uvicorn app:app` 启动 → 必须保留可用的 `app:app`
- `tests/test_search_logic.py` monkeypatch 了 ~30 个 `app._xxx` 内部符号（`app._es`、`app._call_query_parser_llm`、`app._fetch_resume_hits_for_evidence`、`app._score_rerank_documents`、`app._query_plan_cache`、`app.requests.post` 等），且 `from app import (...)` 了 20+ 符号
- `evaluate_search.py` 依赖 `search_app.app / ES_URL / INDEX_ALIAS / EVIDENCE_INDEX_ALIAS`
- `import_to_es.py` 依赖 `embedding_service` 和 `resume_parser`；测试 `from import_to_es import (...)` 了 10 个符号
- 跨模块 import 图：app→embedding_service；import_to_es→embedding_service,resume_parser；evaluate_search→app
- monkeypatch 的 Python 陷阱：`patch("app._es")` 只改 `app` 命名空间里的名字。如果内部调用变成跨模块（`services.retrieval` 调 `infra.es_client`），`patch("app._es")` 改不到真正的调用点 → **测试的 patch 路径必须指向函数实际被查找的模块**

## 目标目录结构
```
resume_search/
  __init__.py
  config.py              # 所有环境变量 + 常量（ES_URL, 权重, 意图常量, 别名表...）
  domain/
    __init__.py
    models.py            # QueryPlan (dataclass)
    constants.py         # DEGREE_ALIASES, SCHOOL_TIER_*, CANONICAL_SKILL_LABELS, INTENT_*
  infrastructure/
    __init__.py
    es_client.py         # _es() —— 唯一 ES HTTP 访问点
    llm_client.py        # _post_query_parser / _call_query_parser_llm（LLM HTTP）
  services/
    __init__.py
    normalization.py     # 学历/院校档位/技能 规范化 + dedupe/casefold 工具
    filters.py           # _build_filters, _min_years_filter, _skill_filter, _school_tier_filter, _filters_from_llm_constraints
    query_planning.py     # _plan_query, _parse_query_with_llm, _lookup_fast_path, 缓存, prompt, sanitize
    query_builder.py     # _evidence_body/_lexical_query/_entity_field/_coverage/_keyword_or_recall/knn...
    retrieval.py         # _run_hybrid_search, _rrf_merge, 聚合/coverage/tier 辅助
    reranking.py         # _rerank_results, _score_rerank_documents, _rerank_document
    formatting.py        # _format_hit, snippets, education_summary, experience_display
    facets.py            # _load_facets, _load_filter_vocab, school_tier_aggs, skill buckets
    search.py            # search() 顶层编排（组合以上 service）
  api.py                 # FastAPI app、路由（/、/api/search、/api/health、/api/resumes/{id}）、StaticFiles

indexing/                # import_to_es.py 拆分
  __init__.py
  mappings.py            # INDEX_BODY, EVIDENCE_INDEX_BODY, 分词器/向量 mapping
  enrichment.py          # _enrich_doc, _collect_all_schools, years_experience, spans
  evidence.py            # _resume_evidence_docs, _evidence_doc, semantic/lexical text 构建
  es_admin.py            # versioned index, alias 切换, bulk, 增量删除, _request
  pipeline.py            # import_resumes, add_evidence_embeddings, load docs
  cli.py                 # argparse main()

# 根目录 shim（保持向后兼容 + CLI 入口）
app.py                   # from resume_search.api import app; from resume_search.<...> import * （re-export 测试所需符号）
import_to_es.py          # from indexing.* import * （re-export CLI + 测试符号）；保留 __main__
config.py?               # 见下：常量放 resume_search/config.py，根 config 不建
```

**注**：`resume_parser.py`(679)、`embedding_service.py`(389)、`rerank_service.py`(118)、`evaluate_search.py`(818)、`generate_mock_resumes.py`(1140)、`build_eval_queries.py`(493) —— 这些已是单一职责的独立模块。「全部模块」的分层目标下：
- `embedding_service` / `rerank_service` → 移入 `resume_search/infrastructure/`（外部服务客户端），根留 shim
- `resume_parser` → 移入 `indexing/`（属于离线数据链路），根留 shim
- `evaluate_search` / `generate_mock_resumes` / `build_eval_queries` → 是独立 CLI 工具脚本，**保持根目录**（它们不是运行时库代码，分层价值低、改动风险高），仅更新内部 import 指向新模块

## 保证不破坏行为的核心策略
1. **纯移动，不改逻辑**：每个函数原样搬到新模块，body 一字不改（除了 import 引用）
2. **shim re-export**：`app.py` 变成
   ```python
   from resume_search.api import app
   from resume_search.config import *          # 常量
   from resume_search.services.search import search
   from resume_search.infrastructure.es_client import _es
   ...（把测试 import / patch 的所有符号 re-export 到 app 命名空间）
   ```
3. **monkeypatch 路径修正**：这是最关键的坑。测试里 `patch("app._es")` 在重构后改不到 `services.retrieval` 内部对 `_es` 的调用。两条路线择一：
   - **路线 A（推荐）**：把测试的 patch 路径改到真实模块，如 `patch("resume_search.infrastructure.es_client._es")`。语义正确、长期可维护。用户已选「同步更新测试的导入/patch路径」→ 走这条。
   - 同理 `app._call_query_parser_llm` → `resume_search.services.query_planning._call_query_parser_llm`，`app._fetch_resume_hits_for_evidence` → `resume_search.services.retrieval.*`，`app.requests.post` → `resume_search.infrastructure.llm_client.requests.post`，`app._query_plan_cache` → `resume_search.services.query_planning._query_plan_cache`
4. **分步提交式验证**：每搬完一个模块就 `python -c "import app"` + 跑一次全量 pytest，绝不一次性搬完再测

## 执行顺序（自底向上，每步后跑测试）
1. 建包骨架 + `config.py`（常量先行，无依赖）
2. `domain/`（models, constants）
3. `infrastructure/`（es_client, llm_client, 移入 embedding/rerank）
4. `services/` 自底向上：normalization → filters → query_builder → query_planning → retrieval → reranking → formatting → facets → search
5. `api.py`
6. `app.py` 改 shim；跑全量测试 + 修 patch 路径
7. `indexing/` 拆分 import_to_es.py；`import_to_es.py` 改 shim；修相关测试
8. 更新 `evaluate_search.py` 内部 import（若需要）
9. 全量 pytest（103）+ `uvicorn app:app` 冒烟 + 端到端 curl 复现「腾讯 阿里巴巴」「腾讯 Java」验证行为不变

## 文档（在重构完成、结构稳定后写，避免写完又变）
### PROJECT_REVIEW.md（追加新章节，不删旧内容——复盘是历史记录）
- 新增「§十四：AND/OR 检索缺陷排查」：现象（腾讯∩阿里=0）、根因（切片级 operator:and + keyword 意图跳过 OR 兜底 + coverage 按切片 max）、方案（keyword_or_recall + coverage 跨切片并集）、验证数据、经验（切片级索引下"同一实体多值"必须在候选人维度聚合）
- 新增「§十五：Clean Architecture 重构」：为什么拆（2420 行单文件）、分层设计、monkeypatch 跨模块失效的坑与解法、shim 兼容策略、"纯移动不改逻辑 + 每步跑测试"的安全重构方法论

### README.md（就地替换过时内容，不追加）
- §2 架构总览 + 模块职责一览：整块替换为新的 `resume_search/` + `indexing/` 分层结构与目录树
- §5 检索流程：更新为「腾讯 阿里巴巴」示例反映 AND/OR 机制；§5.1-5.5 的函数/文件引用改为新模块路径
- §9 配置项：常量位置从 `app.py` 改为 `resume_search/config.py`
- §4 数据处理：`import_to_es.py` 引用改为 `indexing/` 子模块
- 启动命令 `uvicorn app:app` 保持不变（shim 保证）

## 风险与回退
- 风险：monkeypatch 路径遗漏 → 测试静默失效（patch 没生效但测试仍过）。缓解：逐个核对 30 处 patch，确认每处指向函数定义所在的新模块
- 风险：循环 import（services 之间）。缓解：严格自底向上分层，上层依赖下层，search.py 在最顶层
- 回退：git 未提交，任何一步测试失败可 `git checkout` 单文件回退

## 验收标准
- `python -m pytest tests/` → 103 passed
- `uvicorn app:app` 正常启动，/api/health 返回 green
- curl「腾讯 阿里巴巴」=17、「腾讯 Java」top3 coverage=2、「南京大学」=18（与重构前一致）
- 无单文件 > ~400 行（除数据/mapping 常量文件）
- README 与新结构一致，PROJECT_REVIEW 含本次两个新章节
