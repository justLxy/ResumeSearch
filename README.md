# 简历检索系统 (ResumeSearch)

一个面向招聘场景的 **混合检索原型系统**，围绕"**证据切片**"（Evidence Chunks）这一核心设计——将每份简历按结构化段落拆分为可独立检索的语义片段，在证据粒度上同时进行 BM25 词面检索和 kNN 向量检索，再按候选人维度聚合后通过 RRF 融合排序。

系统以 **Elasticsearch 9.x** 为检索引擎，以 **DeepSeek V4 Flash** 作为在线 **LLM Query Planner** 实现用户 query 的意图分类（精确查找/关键词检索/语义检索）与结构化约束抽取，以 **Qwen text-embedding-v4**（可选豆包 API 或本地 Yuan 1.5B）提供 2048 维证据向量化，以 **Qwen3 Reranker** 对 RRF 融合后的 top-N 结果做精排。前端为无框架纯 HTML/CSS/JS 单页应用，包含动态筛选面板、Debug 排名可解释性面板和完整的检索质量评估框架（76 条评测用例，覆盖 9 种查询类型，支持 P@K / R@K / MRR / NDCG 等指标）。

> **Parser 概念边界**：本项目当前只有用户输入 query 会调用 LLM。离线简历解析不依赖 LLM，`resume_parser.py` 是针对 HTML `.doc` 简历的规则化解析器；`app.py` 中的 LLM Query Planner 只负责在线检索时解析用户 query。数据文件里的 `parser_version`（如 `html-doc-v1` 或实验数据中的 `llm-diverse-v2`）描述的是简历数据来源/生成版本，不代表检索服务运行时会用 LLM 解析简历。

---

## 目录

- [1. 业务背景与问题定义](#1-业务背景与问题定义)
- [2. 项目架构总览](#2-项目架构总览)
- [3. 技术栈与依赖](#3-技术栈与依赖)
- [4. 数据处理流程](#4-数据处理流程)
  - [4.1 离线简历解析 (resume_parser.py，非 LLM)](#41-离线简历解析-resume_parserpy非-llm)
  - [4.2 证据切片构建与向量化 (import_to_es.py)](#42-证据切片构建与向量化-import_to_espy)
  - [4.3 Embedding 服务 (embedding_service.py)](#43-embedding-服务-embedding_servicepy)
  - [4.4 Elasticsearch 索引设计](#44-elasticsearch-索引设计)
- [5. 检索流程详解](#5-检索流程详解)
  - [5.1 请求入口与 Query 解析](#51-请求入口与-query-解析)
    - [5.1.1 LLM Query Planner 的意图分类与检索策略路由](#511-llm-query-planner-的意图分类与检索策略路由)
  - [5.2 两路并行检索](#52-两路并行检索)
  - [5.3 RRF 融合排序](#53-rrf-融合排序)
  - [5.4 结果格式化与返回](#54-结果格式化与返回)
  - [5.5 Rerank 精排与相关性弃权](#55-rerank-精排与相关性弃权)
- [6. 前端交互](#6-前端交互)
- [7. 检索效果评估](#7-检索效果评估)
- [8. 本地部署与运行](#8-本地部署与运行)
  - [9. 配置项与常量](#9-配置项与常量)
- [10. 当前实现说明与后续优化方向](#10-当前实现说明与后续优化方向)

---

## 1. 业务背景与问题定义

### 1.1 简历检索为什么不能只用一种技术

招聘场景中，用户的搜索意图极度多样化：

| 查询类型       | 用户输入示例                         | 实际意图                             |
|----------------|--------------------------------------|--------------------------------------|
| 编号精确查找   | `A0009`、`M20260001`                 | 按候选人编号或岗位编号直接定位       |
| 实体精确查找   | `北京交通大学`、`百度`               | 找特定学校/公司背景的候选人          |
| 专业查询       | `计算机科学与技术`                   | 专业字段中连续短语优先匹配           |
| 多技能组合     | `Python NLP SQL`                     | 同时具备这些技能的候选人             |
| 结构化筛选     | `北京 本科 0.5年以上 推荐系统`       | 多维度约束 + 关键词搜索              |
| 语义能力查询   | `做过推荐系统召回和 NLP 模型落地`     | 找经历在语义上相近的候选人           |

**单一技术无法覆盖所有场景**：

- **只用 keyword/term**：无法理解"NLP 模型落地"这类语义描述。
- **只用 BM25 分词**：能做关键词相关性排序，但对同义表达（如"机器学习"vs"ML"）泛化能力有限。
- **只用向量检索**：语义泛化过强——搜"北京大学"可能错误召回所有带"北京"的候选人。
- **只用 filter**：过于严格，召回量极低，且不能做相关性排序。

因此本项目采用 **混合检索**：让不同技术各自负责最擅长的部分，再通过 RRF 融合排名。

### 1.2 "证据切片"架构——为什么不对整份简历做向量化

这是项目架构中最核心的设计决策，需要先理解其动机：

**问题**：一份简历可能有 2000+ 字，包含教育、实习、项目、技能等多个截然不同的信息维度。如果把全文拼接成一个字符串去做 embedding，得到的向量只能表达一个"模糊的平均语义"。当用户搜"做过推荐系统"时，这个平均向量可能被简历里大段的教育经历描述所"稀释"，导致匹配精度下降。

**解决方案——证据切片（Evidence Chunks）**：

1. 把每份简历按**结构化段落**拆分成多个"证据片段"——每个项目一片、每段实习一片、每段教育经历一片、技能列表一片。
2. 对每个证据片段**独立做向量化**，存入独立的 `resume_evidence` 索引。
3. 检索时，向量检索在证据片段级别进行，找到最相关的片段后，再按 `resume_id` **聚合回候选人维度**。

这样，用户搜"推荐系统"时，系统能精准匹配到某份简历中"项目：推荐系统召回层设计"这个具体片段，而不是被整份简历的平均语义干扰。

---

## 2. 项目架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户浏览器                              │
│   web/index.html + web/app.js + web/styles.css                 │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│   │ 搜索框   │  │ 筛选面板 │  │ 结果列表 │  │ 详情抽屉 │      │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
└───────────────────────────┬─────────────────────────────────────┘
                            │  HTTP (fetch → /api/search)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI 后端 (app.py)                         │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐       │
│  │ LLM Query    │→│ 并行检索调度  │→│ RRF 融合 + 排序 │       │
│  │ Planner      │  │ ThreadPool   │  │ _rrf_merge()    │       │
│  │ DeepSeek V4  │  │ Executor     │  │ Qwen3 rerank    │       │
│  └──────────────┘  └──────────────┘  └─────────────────┘       │
│         │                │                                      │
│         │    ┌───────────┼───────────┐                          │
│         │    ▼           ▼           ▼                          │
│         │  Evidence    Evidence    主索引                        │
│         │  BM25检索    kNN检索    详情回填                       │
│         │  (词面)      (向量)                                     │
│         ▼                                                       │
│  embedding_service.py  ←→  Qwen / 豆包 API / 本地 Yuan embedding │
└────────────────────────┬────────────────────────────────────────┘
                         │  HTTP (requests → ES REST API)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Elasticsearch 9.x                              │
│                                                                 │
│  ┌─────────────────────┐    ┌──────────────────────────────┐   │
│  │ resumes_current     │    │ resume_evidence_current      │   │
│  │ (候选人主索引)       │    │ (证据切片索引)                │   │
│  │                     │    │                              │   │
│  │ - 结构化字段        │    │ - evidence_id / resume_id    │   │
│  │ - 嵌套对象          │    │ - section_type / title / text│   │
│  │ - section_text      │    │ - evidence_vector             │   │
│  │ - skills keyword    │    │ - 冗余 candidate/application │   │
│  │                     │    │ - skills / skills_text       │   │
│  └─────────────────────┘    └──────────────────────────────┘   │
│                                                                 │
│  IK 分词器 (ik_max_word / ik_smart)                            │
│  HNSW 向量索引 (m=32, ef_construction=300)                     │
└─────────────────────────────────────────────────────────────────┘

离线数据导入流程：
┌──────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────┐
│ .doc 文件 │ ──→ │ resume_parser│ ──→ │ import_to_es │ ──→ │   ES    │
│ (HTML格式)│     │ 解析结构化   │     │ 证据切片     │     │ 双索引  │
│           │     │ JSON         │     │ 向量化       │     │         │
└──────────┘     └──────────────┘     │ bulk index   │     └─────────┘
                                      └──────────────┘
```

### 模块职责一览

| 模块文件                | 核心职责                                                         |
|-------------------------|------------------------------------------------------------------|
| `resume_parser.py`      | 规则化解析 HTML 格式 `.doc` 简历文件，提取结构化字段（候选人、教育、实习、项目、技能等），不调用 LLM |
| `import_to_es.py`       | 加载解析结果 → 构建证据切片 → 调用 embedding 服务向量化 → 批量写入 ES 双索引 |
| `embedding_service.py`  | 统一封装 Qwen、豆包 API、本地 Yuan embedding 后端，提供 `encode_single` / `encode_batch` |
| `app.py`                | FastAPI 后端：DeepSeek LLM Query Planner → 并行混合检索 → RRF 融合 → Qwen3 rerank → 结果格式化 |
| `web/index.html`        | 前端页面骨架：搜索框、筛选面板、结果区域、详情抽屉               |
| `web/app.js`            | 前端交互逻辑：搜索触发、facet 渲染、结果卡片、Debug 面板        |
| `web/styles.css`        | 前端样式                                                         |
| `evaluate_search.py`    | 检索质量评估脚本，基于 eval_queries.jsonl 计算 P@K / R@K / MRR / 分级 NDCG |
| `tests/test_search_logic.py` | 单元测试，覆盖查询解析、RRF 融合、过滤器构建等核心逻辑     |

---

## 3. 技术栈与依赖

| 类别           | 技术选型                                          | 说明                                           |
|----------------|---------------------------------------------------|------------------------------------------------|
| 后端框架       | FastAPI + Uvicorn                                 | 异步 Web 框架，提供 REST API                   |
| 搜索引擎       | Elasticsearch 9.x                                 | 承载 BM25 全文检索 + kNN 向量检索 + 结构化过滤 |
| 中文分词       | IK Analysis Plugin                                | `ik_max_word`（索引时细粒度分词）/ `ik_smart`（搜索时智能分词） |
| Query Planner  | `deepseek-v4-flash` API                            | 仅在用户搜索时调用，将自由文本 query 解析为 intent、结构化约束、lexical/semantic query |
| Embedding 模型 | Qwen `text-embedding-v4` / 豆包 API / 本地 Yuan     | 当前默认 Qwen 2048 维；豆包 2048 维；本地 Yuan 1792 维 |
| Reranker 模型  | `qwen3-rerank` API                                | 通过 `rerank_service.py` 调用文本重排 API      |
| 模型框架       | sentence-transformers + PyTorch / requests        | 本地 embedding 推理、DeepSeek Query Planner、Qwen/豆包 embedding API、Qwen3 rerank API |
| 模型下载       | ModelScope + HuggingFace Hub                      | 本地 embedding 通过 ModelScope 下载             |
| HTML 解析      | BeautifulSoup4                                    | 离线规则解析 HTML 格式的 .doc 简历文件，不调用 LLM |
| 前端           | 原生 HTML + CSS + JavaScript                      | 无框架依赖的轻量前端                           |

### Python 依赖 (`requirements.txt`)

```
beautifulsoup4>=4.12
fastapi>=0.100
requests>=2.31
uvicorn>=0.23
sentence-transformers==3.4.1
transformers>=4.51.0
torch>=2.0
modelscope>=1.14
huggingface-hub>=0.20
```

---

## 4. 数据处理流程

数据处理分为三个阶段：**简历解析** → **证据切片构建与向量化** → **写入 ES 索引**。

### 4.1 离线简历解析 (`resume_parser.py`，非 LLM)

这里的“简历解析”是离线数据处理阶段的 HTML 结构化抽取，不是 `app.py` 中的 LLM Query Planner。当前运行链路中，简历解析不调用 DeepSeek 或其他 LLM；只有用户输入搜索 query 时，后端才会调用 LLM 做意图分类和约束抽取。

#### 输入格式

本项目处理的简历不是常见的 PDF 或 Word `.docx`，而是 **HTML 内容封装的 `.doc` 文件**——这是一些招聘系统导出简历时采用的格式。文件扩展名是 `.doc`，但实际内容是 HTML 表格。

#### 解析流程

```
.doc 文件 (实际是 HTML)
    │
    ▼
1. 编码检测与解码
   - 先检查 HTML meta 标签中的 charset 声明
   - 按 utf-8-sig → utf-8 → gb18030 顺序尝试解码
    │
    ▼
2. HTML 清洗
   - 移除 <script>、<style> 等非内容节点
   - 提取纯文本 (raw_text)
    │
    ▼
3. 文件名解析
   - 从文件名中提取公司、岗位名称、岗位编号、候选人姓名、候选人编号
   - 文件名格式：{公司}-{岗位}({岗位编号})-{姓名}({候选人编号}).doc
    │
    ▼
4. 表格行扫描与段落分区
   - 遍历所有 <tr> 行
   - 遇到段落标题行(如"个人信息""教育经历")时创建新分区
   - 非标题行解析为 key-value 对
    │
    ▼
5. 结构化字段映射
   - 个人信息 → candidate 对象 (姓名、性别、学历、学校、手机、邮箱等)
   - 教育经历 → education[] (学校、专业、学历、研究方向、实验室等)
   - 实习经历 → internships[] (公司、部门、职位、描述等)
   - 项目经验 → projects[] (名称、描述、职责等)
   - IT技能   → skills[] (去重的技能标签列表)
   - 语言能力 → languages (英语成绩、口语水平)
   - 奖项活动 → awards[]
    │
    ▼
6. 日期规范化
   - "2019年3月" → "2019-03-01" (ISO 格式)
   - "至今"/"现在" → null + is_current=true
    │
    ▼
7. 生成 resume_id
   - 优先使用候选人编号 (candidate_no)
   - 回退为文件内容的 SHA-256 哈希值
```

#### 输出结构

解析结果是一个 JSON 对象，关键字段：

```json
{
  "resume_id": "20190016837",
  "parse_status": "ok",
  "parser_version": "html-doc-v1",
  "file": { "path": "...", "sha256": "...", "encoding": "utf-8" },
  "application": {
    "candidate_no": "20190016837",
    "company": "奇安信",
    "position_name": "NLP工程师",
    "position_code": "A0009",
    "expected_work_cities": ["北京", "上海"],
    "wishes": [{ "rank": 1, "position_name": "...", "company": "..." }]
  },
  "candidate": {
    "name": "张三", "gender": "男", "highest_degree": "硕士",
    "school": "北京交通大学", "major": "计算机科学与技术",
    "phone": "13800138000", "email": "xxx@xxx.com"
  },
  "education": [{ "school": "...", "major": "...", "degree": "...", ... }],
  "internships": [{ "company": "...", "title": "...", "description": "...", ... }],
  "projects": [{ "name": "...", "description": "...", "responsibility": "...", ... }],
  "skills": ["Python", "NLP", "PyTorch", "TensorFlow"],
  "section_text": { "education": "...", "internships": "...", "projects": "..." }
}
```

#### 命令行用法

```bash
# 解析单个文件
python resume_parser.py path/to/resume.doc --pretty

# 解析目录下所有 .doc 文件，输出为 JSONL
python resume_parser.py data/ --jsonl -o parsed_resumes.jsonl
```

### 4.2 证据切片构建与向量化 (`import_to_es.py`)

这是数据管道的核心环节，完成 **解析结果 → 证据切片 → 向量化 → 双索引写入** 的全流程。

#### 整体流程

```
1. 加载数据
   - 支持三种输入：
     ① .jsonl 文件 (一行一个 JSON 对象)
     ② .json 文件 (JSON 数组 或 单个 JSON 对象)
     ③ 目录路径 → 自动调用 resume_parser 解析 .doc 文件
   - 只保留 parse_status == "ok" 的文档

2. 文档富化 (_enrich_doc)
   - 优先保留解析结果中的 candidate.years_experience
   - 当显式年限缺失或无效时，才从实习经历中估算工作年限
     计算方式：合并所有实习经历的时间跨度，处理重叠区间，
     以申请日期（或文件修改日期/当前日期）为截止时间，
     总天数 / 365 = 年数
   - 清除遗留向量字段（旧版 whole-doc 向量）
   - 生成 skills_text (所有技能标签空格连接的文本)

3. 构建证据切片 (_build_evidence_docs)
   - 对每份简历生成以下类型的证据片段：
     ┌─────────────┬──────────────────────────────────┬─────────┐
     │ section_type │ 内容来源                         │ 有向量？│
     ├─────────────┼──────────────────────────────────┼─────────┤
     │ profile     │ 候选人档案（编号/姓名/学校/岗位等）│ ✗       │
     │ skills      │ 所有技能标签拼接                  │ ✗       │
     │ project     │ 每个项目：名称 + 描述 + 职责      │ ✓       │
     │ internship  │ 每段实习：部门 + 职位 + 描述      │ ✓       │
     │ education   │ 每段教育：专业 + 研究方向 + 实验室 │ ✗       │
     └─────────────┴──────────────────────────────────┴─────────┘
   - evidence_id 格式：{resume_id}:{section_type}:{ordinal}
   - 每个证据片段冗余存储了候选人基本信息和申请信息，
     便于在证据索引上直接做过滤和词面检索
   - 证据文本会排除公司名、学校名、姓名等实体，
     避免这些高频信息干扰词面证据和语义向量的表达

4. 向量化 (add_evidence_embeddings)
   - 只对 project/internship 两类长经历证据做向量化
   - profile/skills/education 类型不做向量化，保留给 BM25、短语匹配和结构化过滤
   - 调用 embedding_service.encode_batch() 批量生成向量；维度由当前 embedding provider 决定
   - 向量做 L2 归一化 (normalize=True)

5. 创建 ES 索引并写入
   - 带时间戳创建新版本索引 (如 resumes_v1_20260625080000000000)
   - Bulk API 批量写入（每批 100 条）
   - 主索引和证据索引分别写入
   - 写入完成后原子切换 alias（resumes_current → 新索引）
   - 旧索引保留但不再被 alias 指向
```

#### 命令行用法

```bash
# 从 JSONL 文件导入（最常用）
python import_to_es.py data/ai_generated.jsonl

# 从目录解析并导入
python import_to_es.py data/

# 指定 ES 地址
python import_to_es.py data/ --es-url http://localhost:9200

# 增量更新（不重建索引）
python import_to_es.py new_resumes.jsonl --no-recreate

# 增量更新并删除不在本批次中的旧文档
python import_to_es.py current_resumes.jsonl --no-recreate --delete-missing
```

### 4.3 Embedding 服务 (`embedding_service.py`)

#### 模型选型

当前默认使用 **Qwen `text-embedding-v4`** API，输出 2048 维向量。`embedding_service.py` 也保留了豆包 API embedding 和本地 `IEITYuan/Yuan-embedding-2.0-zh` 两个可选后端，便于做模型对比实验。

当前默认配置为：

```python
EMBEDDING_PROVIDER = QWEN_PROVIDER
QWEN_MODEL_ID = "text-embedding-v4"
QWEN_VECTOR_DIMS = 2048
```

豆包 API 模式同样输出 2048 维向量，默认接入点为 `ep-20260412051954-zl5fm`。本地 Yuan 模型输出 1792 维向量。

本地模型结构：

```
Transformer Encoder (1024维 hidden)
    ↓
1_Pooling (mean pooling → 1024维)
    ↓
2_Dense (线性投影 1024→1792维)
    ↓
L2 归一化
```

#### 模型加载流程

本地模型加载需要处理 ModelScope 和 HuggingFace 的兼容性问题：

1. **从 ModelScope 下载模型主体**（`snapshot_download`）
2. **重建子目录结构**——ModelScope 会把文件平铺到根目录，但 `sentence-transformers` 期望 `1_Pooling/` 和 `2_Dense/` 子目录
3. **从 HuggingFace 下载 Dense 层权重**——Dense 层的 `model.safetensors` 或 `pytorch_model.bin` 需要单独下载
4. **构建 `SentenceTransformer` 实例**——模型懒加载，首次调用时初始化

Qwen 和豆包 API 模式不会下载或加载本地模型。Qwen 通过 DashScope embedding 接口批量请求；豆包当前接入点对应多模态 embedding 模型，因此调用 OpenAI-compatible `/embeddings/multimodal` 接口，文本会包装为 `{"type":"text","text":"..."}`。三种 provider 都会在本地继续做 L2 归一化，保证与 ES `cosine` 检索契约一致。

#### Provider 切换

默认使用 Qwen API。切换 provider 时修改 `embedding_service.py` 顶部常量。

切换到本地 Yuan 模型：

```python
EMBEDDING_PROVIDER = LOCAL_PROVIDER
```

切换到豆包 API 时改为：

```python
EMBEDDING_PROVIDER = DOUBAO_PROVIDER
DOUBAO_MODEL_ID = "ep-20260412051954-zl5fm"
DOUBAO_VECTOR_DIMS = 2048
```

Qwen、豆包 API key 当前都硬编码在 `embedding_service.py` 顶部，符合开发阶段快速实验的使用方式。`QWEN_VECTOR_DIMS` / `DOUBAO_VECTOR_DIMS` / `LOCAL_VECTOR_DIMS` 必须与接口或模型实际返回的向量维度一致；如果接口返回维度变化，导入脚本会直接报错，避免把错误维度写入 ES。

> 切换 embedding provider 或向量维度后，需要重新运行 `import_to_es.py` 重建索引。旧索引里的向量维度和模型元数据不会自动迁移，不能直接混用。

#### 关键 API

```python
encode_single(text: str) -> list[float]     # 单条文本 → 向量
encode_batch(texts: list[str]) -> list[list[float]]  # 批量编码
```

### 4.4 Elasticsearch 索引设计

系统维护 **两个索引**，通过 **alias** 机制实现版本切换：

#### 主索引 (`resumes_current`)

存储完整的候选人资料，用于 **结果展示** 和 **无 query 时的浏览模式**。

关键字段设计：

| 字段路径                          | 类型              | 设计意图                                       |
|-----------------------------------|-------------------|------------------------------------------------|
| `resume_id`                       | keyword           | 文档唯一标识                                   |
| `candidate.name`                  | text + keyword    | text 支持分词搜索，keyword 支持精确匹配        |
| `candidate.school`                | text + keyword + phrase | phrase 子字段支持连续短语匹配             |
| `candidate.major`                 | text + keyword + phrase | 同上                                     |
| `candidate.highest_degree`        | keyword           | 精确匹配（博士/硕士/本科）                     |
| `candidate.years_experience`      | float             | 范围过滤 (`gte`)                               |
| `application.expected_work_cities`| keyword           | 精确匹配，支持 terms 多值                      |
| `skills`                          | keyword           | 精确匹配技能标签（大小写敏感）                 |
| `skills_text`                     | text              | 技能标签拼接的文本，支持分词搜索               |
| `education`                       | **nested**        | 嵌套对象，支持同一教育经历内的关联查询         |
| `internships`                     | **nested**        | 同上                                           |
| `projects`                        | **nested**        | 同上                                           |
| `section_text.{education/internships/projects}` | text + phrase | 段落级全文检索             |

> **为什么 education/internships/projects 使用 nested 类型？**
>
> Elasticsearch 默认会把对象数组"扁平化"——如果一个候选人有两段教育经历，"学校A + 专业X"和"学校B + 专业Y"，扁平化后搜索"学校A + 专业Y"也能匹配。nested 类型保证每个子对象的字段关联性，避免跨条目的错误匹配。

#### 中文分词配置

```
索引时分词器 (index analyzer)：   ik_max_word  → 尽可能细粒度切分
搜索时分词器 (search analyzer)：  ik_smart     → 智能合并，保持搜索语义完整
```

**为什么索引和搜索使用不同的分词器？** 索引时用细粒度分词可以建立更多倒排表项，提高召回率。搜索时用粗粒度分词可以减少无意义的短词匹配，提高精确率。例如"计算机科学与技术"：

- `ik_max_word` → `["计算机", "计算", "科学", "与", "技术", "计算机科学", ...]`
- `ik_smart` → `["计算机科学", "与", "技术"]`

#### 证据索引 (`resume_evidence_current`)

存储简历的 **语义切片**，用于 **检索命中**。

关键字段：

| 字段                    | 类型          | 说明                                           |
|-------------------------|---------------|------------------------------------------------|
| `evidence_id`           | keyword       | 格式 `{resume_id}:{section_type}:{ordinal}`    |
| `resume_id`             | keyword       | 关联回主索引                                   |
| `section_type`          | keyword       | profile / skills / project / internship / education |
| `title`                 | text + keyword + phrase | 切片标题（如项目名称、公司/职位）      |
| `text`                  | text + phrase | 切片正文（描述、职责等）                       |
| `evidence_vector`       | dense_vector  | 当前默认 2048 维，cosine 相似度，HNSW 索引（m=32, ef=300） |
| `candidate.*`           | 冗余字段      | 候选人基本信息，用于直接在证据索引上做过滤     |
| `application.*`         | 冗余字段      | 申请信息，同上                                 |
| `skills` / `skills_text`| keyword / text| 冗余的技能信息                                 |

> **为什么证据索引要冗余存储 candidate 和 application 信息？**
>
> 因为检索时需要在证据索引上同时做"筛选条件过滤"和"BM25/kNN 检索"。如果不冗余，就需要先在主索引查出符合条件的 resume_id 列表，再用这个列表去证据索引做过滤，这会增加一次额外的 ES 请求和延迟。冗余存储是"以空间换时间"的典型取舍。

---

## 5. 检索流程详解

这是整个系统最核心的部分。以用户输入 `q = "Python 自然语言处理"` 为例，完整跟踪一次搜索请求。

### 5.1 请求入口与 Query 解析

**入口**：`GET /api/search?q=Python+自然语言处理`

**Step 1：构建显式过滤器**

从 URL 参数中的 `degree`、`cities`、`skills`、`min_years` 构建 Elasticsearch filter 子句（LLM 从自由文本抽取的 `constraints` 也走同一套转换逻辑）。

> **`min_years` 软容差**：`_min_years_filter` 不把"N 年以上"当成硬边界，而是把 `gte` 下调 `max(N×10%, 0.5 年)`。原因是"4 年以上"是用户的模糊表达，一个 3.9 年的候选人几乎必然算命中；硬卡 `>=4` 会把这类边界候选直接灭掉（评测中曾出现某 query 因此 R@100=0）。真实资历仍由排序体现，过滤只负责"放进候选池"。

**Step 2：LLM Query Planner** (`_parse_query_with_llm`)

这是在线检索阶段的 query parser，与离线简历解析器是两套独立机制。解析分三层，按成本从低到高短路：

1. **正则 fast-path**：邮箱、手机号、候选人编号、岗位编号这类**单 token 唯一标识符**可被正则 100% 判定，直接判为 `lookup` 并短路 LLM，无需为最廉价的查询付一次网络往返。
2. **解析缓存命中**：解析结果按规范化后的 query 文本（去多余空白、casefold）做带 TTL 的 LRU 缓存（默认 5 分钟、512 条）。命中则跳过 LLM——对前端 300ms 防抖自动搜索和重复 query 能显著降低延迟与 API 成本（解析失败的 fallback 不入缓存，避免短暂故障污染整个 TTL）。
3. **LLM 解析**：以上都未命中时，调用 DeepSeek `deepseek-v4-flash`，请求体显式设置 `thinking: {"type": "disabled"}` 优先低延迟，把用户输入解析为结构化 QueryPlan。

> **结构化输出说明**：代码保留了 `response_format: json_schema` 严格模式的完整链路（schema 把所有字段设为 required、关闭 additionalProperties，能从源头消除"偶发不返回 lexical_query"这类整字段缺失）。但实测 `deepseek-v4-flash` 拒绝 json_schema（返回 HTTP 400），因此当前默认使用广泛支持的 `response_format: json_object`，靠 `_sanitize_llm_query_plan` 做字段兜底。若将来切换到支持 schema 的模型，把 `_structured_output_supported` 初值改回 `True` 即可自动启用严格模式（并保留遇 400 自动回退 json_object 的能力）。

LLM 解析输出示例：

```
输入: "0.5年以上 北京 本科 Python 自然语言处理"
      ↓ DeepSeek V4 Flash
输出:
{
  "intent": "semantic",
  "lexical_query": "Python 自然语言处理",
  "semantic_query": "Python 自然语言处理",
  "constraints": {
    "min_years": 0.5,
    "degree": "本科",
    "cities": ["北京"],
    "skills": ["Python", "自然语言处理"]
  },
  "enable_dense": true
}
```

> 注意：学历/城市/年限只是结构化 filter，不决定 intent。抽掉这些 filter 后，剩余需求是技能组合/能力描述（如上例的 `Python 自然语言处理`），intent 即为 `semantic`、`enable_dense=true`；若剩余需求只是学校/公司/姓名等实体，intent 才是 `keyword`。

QueryPlan 的字段分工：

- `intent`：浏览、精确查找（lookup）、关键词检索（keyword）或语义检索（semantic）。
- `constraints`：学历、城市、技能、最低年限等硬过滤条件。后端只负责把 LLM 输出的结构化约束转换为 ES filter。
- `lexical_query`：交给 Evidence BM25 的词面检索文本。
- `semantic_query`：交给 embedding 和 reranker 的语义检索文本。
- `enable_dense`：是否启用 evidence kNN 向量召回。

`enable_rerank` 不是 LLM Query Planner 的职责。系统根据 `intent` 自动判定：只有 `intent == semantic`（且 `ENABLE_RERANK=True`、lexical/semantic query 均非空）时，才对 RRF 后的 top-N 候选启用 Qwen3 reranker。`lookup`（编号/手机/邮箱精确定位）、`keyword`（实体精确匹配）、`browse`（空 query / 纯筛选）都不做 rerank——这些场景词面/精确匹配本身就是最强信号，cross-encoder 重排只会增加延迟并可能扰乱已经正确的头部排序。

如果 DeepSeek 调用失败，系统会退化为保守的词面检索：保留原始 query 作为 `lexical_query`，intent 兜底为 `semantic`，不启用 dense；此时仍满足 `intent == semantic` 的 rerank 条件，因此会在 BM25/RRF 候选上执行系统级 rerank，避免解析失败导致搜索接口不可用。

#### 5.1.1 LLM Query Planner 的意图分类与检索策略路由

LLM 完成的不只是"抽 filter"，而是一次**检索策略路由决策**。它输出一个意图标签，系统据此决定走纯 BM25 还是 BM25+Dense 混合检索。这是一种 **Self-Querying** 技术的扩展——经典 Self-Querying 只把 NL 查询拆成 `(语义文本, 元数据 filter)` 两项，本系统额外产出了意图分类和多路检索文本。

##### 三种意图

| 意图 | 前端中文标签 | 典型输入 | 检索策略 | Dense | 说明 |
|---|---|---|---|---|---|
| `lookup` | 精确查找 | `A0009`、`M20260001`、`13800138000` | 纯 BM25 | ❌ | 编号/手机/邮箱等唯一标识符直接定位，语义泛化反而会引入噪声 |
| `keyword` | 关键词检索 | `阿里巴巴`、`北京交通大学`、`硕士 北京 3年 Python` | 纯 BM25（+ ES filter） | ❌ | 实体名精确匹配、多维度筛选+关键词。filter 承担硬筛，BM25 承担关键词召回 |
| `semantic` | 语义检索 | `做过大规模分布式系统架构设计`、`Python PyTorch NLP 大模型`、长 JD 粘贴 | **混合** | ✅ | 自然语言能力描述、多技能组合、长文本匹配，核心依赖语义理解 |

**核心原则**：Dense 永远是 BM25 的**外挂**，不存在纯 Dense 检索路径。`_run_hybrid_search` 中 BM25 证据检索是必跑的，Dense KNN 只在 `enable_dense=True` 时额外追加。这样设计的原因是——BM25 对实体、编号、技能关键词的精确召回是 Dense 无法替代的。

##### enable_dense 的决策链

```
LLM 输出 enable_dense: true/false
        ↓
_plan_query: AND bool(semantic_query)  ← 语义查询为空则强制关
        ↓
search(): embedding API 调用失败 → 运行时降级关
        ↓
LLM API 整体挂掉 → 兜底关（_llm_parser_fallback）
```

`enable_rerank` 不受 LLM 控制——系统根据 `ENABLE_RERANK` 常量、`intent == semantic` 及三个 query 字段是否非空自动判定。

##### 与经典 Self-Querying 的对比

| | 经典 Self-Querying (LangChain 2023) | 本系统 |
|---|---|---|
| LLM 输出 | `{query, filter}` | `{intent, lexical_query, semantic_query, constraints, enable_dense}` |
| 检索方式 | 纯向量 | **BM25 + 向量 + RRF 融合** |
| 策略路由 | 无（始终向量检索） | **4 种意图 → 不同检索策略** |
| 排序管道 | 单一相似度 | RRF → tier/coverage 乘数 → **Rerank 重排** |
| Filter 处理 | 从 query 中移除后单独 filter | 抽取 constraints 但**保留原词在 query 中**（避免 BM25 召回损失） |
| 容错 | LLM 失败 = 搜索失败 | LLM 失败 → 降级为保守词面检索 |

##### 证据分块 BM25 查询的三层匹配

在`_evidence_lexical_query` 中，实体字段（公司/学校/专业/岗位名）同时使用 `term`（精确 token 匹配）和 `match`（分词后 OR 匹配）：

```
dis_max(
    term("application.company", query),     // 完整匹配 "阿里巴巴" → boost 30
    match("application.company", query)     // 分词匹配 ["阿里巴巴","实习"] → boost 16.5
)
```

这解决了"搜'阿里巴巴'能命中，搜'阿里巴巴实习'反而命中不了"的问题——`match` 让 IK 分词器自动拆出 "阿里巴巴" 去匹配，无需手工分词规则。

### 5.2 两路并行检索

当 query 不为空时，系统通过 `ThreadPoolExecutor` **并行**发出最多两个检索请求（如果启用了向量检索）：

```
                       ┌──────────────────────────┐
                       │    ThreadPoolExecutor     │
                       └─────┬──────────┬──────────┘
                             │          │
                   ┌─────────▼──┐  ┌────▼──────────┐
                   │ Evidence   │  │ Evidence       │
                   │ BM25 检索  │  │ kNN 检索       │
                   │ (词面)     │  │ (向量)         │
                   │ weight=1.2 │  │ weight=1.0     │
                   └────────────┘  └───────────────┘
                       ▲                  ▲
                       │                  │
               evidence_current    evidence_current
               索引               索引
```

#### 路线 1：Evidence BM25 检索

在 **证据索引** 上执行复杂的词面检索。查询结构如下：

```
dis_max (tie_breaker=0.0)
├── 精确匹配层 (Exact)
│   ├── term: candidate_no (boost=60)
│   ├── term: position_code (boost=55)
│   ├── term: candidate.name.keyword (boost=45)
│   ├── term: candidate.phone (boost=45)
│   ├── term: candidate.email (boost=45)
│   ├── term: skills (boost=40)
│   ├── term: candidate.school.keyword (boost=36)
│   ├── term: candidate.major.keyword (boost=34)
│   ├── term: application.company (boost=30)
│   └── ... (更多精确匹配字段)
│
├── 短语匹配层 (Phrase)
│   ├── match_phrase: candidate.major.phrase (boost=24)
│   ├── match_phrase: candidate.school.phrase (boost=18)
│   ├── match_phrase: title.phrase (boost=12)
│   ├── match_phrase: text.phrase (boost=10)
│   └── ... (更多短语匹配字段)
│
└── 分词匹配层 (Term)
    ├── multi_match(operator=and, boost=4)   ← 所有词都命中
    └── multi_match(operator=or, min_match=70%, boost=1) ← 部分词命中
```

**为什么用 `dis_max` 而不是 `bool/should`？**

`dis_max` 取各子查询中得分最高的那个作为最终分数（`tie_breaker=0.0`）。这避免了"一个候选人在多个低权重字段都匹配到"导致分数虚高的问题——我们希望的是"在最相关的那个字段上匹配得好"就够了。

**三层匹配的权重设计逻辑**：

- **精确匹配**给最高分（45-60）：因为如果 query 恰好是某个编号或姓名，这几乎一定是用户想找的
- **短语匹配**给中等分（10-24）：连续短语比散词匹配更精确，比如搜"计算机科学"应该优先匹配专业名完全包含这四个字的候选人
- **分词匹配**给基础分（1-4）：用于兜底，保证语义相关但没有精确命中的候选人也能被召回

#### 路线 2：Evidence kNN 检索

在证据索引的 `evidence_vector` 字段上执行 kNN 近邻搜索：

```json
{
  "knn": {
    "field": "evidence_vector",
    "query_vector": [0.123, -0.456, ...],  // 当前默认 2048 维
    "k": 300,
    "num_candidates": 300,
    "filter": { "bool": { "filter": [...] } }
  }
}
```

- `k` 和 `num_candidates` 控制候选池大小（默认 300）
- 如果有筛选条件，会附带 filter 子句在向量检索阶段就做预过滤

#### Term Coverage 机制

除了主要的 dis_max 评分查询外，系统还会添加 **term coverage** 辅助查询。它的作用不是改变排序，而是为后续的 RRF 融合提供"这个候选人覆盖了多少个查询词"的信息。

对于多词 query（如"Python 自然语言处理"），会为每个词生成一个 `constant_score` 查询，检查该词是否在候选人的任意字段中出现。覆盖率越高的候选人，在 RRF 阶段会获得额外加分。

### 5.3 RRF 融合排序

#### 为什么需要手动实现 RRF

Elasticsearch 的内置 RRF 功能需要高级许可证（Platinum/Enterprise）。项目使用的是 Basic 许可证，因此在应用层手动实现 RRF。

#### 什么是 RRF

**Reciprocal Rank Fusion（倒数排名融合）** 是一种不依赖分数绝对值、只依赖排名的融合方法。它解决的问题是：BM25 的分数和向量相似度的分数完全不可比——前者可能是 0-100 的范围，后者是 0-1。直接加权平均没有意义。

RRF 的核心公式：

$$\text{RRF}(d) = \sum_{r \in \text{retrievers}} \frac{w_r}{k + \text{rank}_r(d)}$$

其中：
- $d$ 是一个文档（候选人）
- $r$ 是某个检索器（evidence BM25 / evidence kNN）
- $w_r$ 是该检索器的权重
- $\text{rank}_r(d)$ 是文档 $d$ 在检索器 $r$ 的结果中的排名
- $k$ 是常数（本项目 $k=60$），用于控制高排名和低排名的区分度

例如：某候选人在 BM25 证据检索中排第 3，在 kNN 证据检索中排第 10：

$$\text{RRF} = \frac{1.2}{60 + 3} + \frac{1.0}{60 + 10} = 0.01905 + 0.01429 = 0.03333$$

#### 证据片段的聚合与标准 RRF

标准 RRF 的核心原则是：**每个检索器对每个候选人贡献且仅贡献一个排名**。

由于系统的两路检索（BM25 和 kNN）返回的都是**证据片段级别**的结果，同一个候选人可能有多个不同排名位置的片段命中。

为了避免"写的段落多"的候选人因为片段分数累加次数多而总分虚高，同时保留"多段证据一致相关"的排序信号，系统在进入 RRF 前，对 BM25 和 Dense 两路都进行了**对称的候选人级别聚合**：

1. **片段 top-k pooling**：遍历某一路返回的所有片段，对于同一个候选人，只取排名最高的前 3 个片段，以 `1/(60 + best_rank) + 0.30/(60 + second_rank) + 0.15/(60 + third_rank)` 计算本路内部聚合分。
2. **候选人全局重排**：根据本路内部聚合分，对命中的候选人重新排序，得到候选人级别的全局排名（`evidence_group_rank` 和 `dense_group_rank`）。
3. **单项 RRF 贡献**：用重排后的候选人聚合排名参与 RRF 融合。

这样确保了多段证据只影响本路内部排名；跨路融合时，每路检索仍然只向最终的 RRF 贡献**1 项**分数。

#### 内部聚合排名示例

假设 query 是"推荐系统召回 NLP"，系统先分别拿到 BM25 evidence 和 Dense evidence 的片段级结果。

BM25 evidence 路返回的片段排名如下：

| 片段排名 | 候选人 | 命中的证据片段 |
|----------|--------|----------------|
| 1        | A      | 项目：推荐系统召回 |
| 2        | B      | 项目：推荐系统 |
| 5        | A      | 实习：排序模型 |
| 8        | C      | 项目：NLP |
| 12       | A      | 教育：自然语言处理 |

此时不是直接把 A 的 3 个片段都送进最终 RRF，而是先在 BM25 路内部聚合：

```text
A = 1/(60+1) + 0.30/(60+5) + 0.15/(60+12)
  = 0.01639 + 0.00462 + 0.00208
  = 0.02309

B = 1/(60+2)
  = 0.01613

C = 1/(60+8)
  = 0.01471
```

所以 BM25 路内部候选人排名是：

```text
evidence_group_rank:
A = 1
B = 2
C = 3
```

进入跨路 RRF 时，BM25 路仍然只给每个候选人贡献一次：

```text
A 的 BM25 贡献 = 1.2 / (60 + 1)
B 的 BM25 贡献 = 1.2 / (60 + 2)
C 的 BM25 贡献 = 1.2 / (60 + 3)
```

Dense evidence 路也做同样的事，只是片段排名来自向量检索：

| 片段排名 | 候选人 | 命中的证据片段 |
|----------|--------|----------------|
| 1        | B      | 项目向量：推荐系统 |
| 3        | A      | 项目向量：召回链路 |
| 4        | A      | 实习向量：排序模型 |
| 9        | C      | 项目向量：NLP |
| 15       | A      | 实习向量：模型优化 |

Dense 路内部聚合分：

```text
A = 1/(60+3) + 0.30/(60+4) + 0.15/(60+15)
  = 0.01587 + 0.00469 + 0.00200
  = 0.02256

B = 1/(60+1)
  = 0.01639

C = 1/(60+9)
  = 0.01449
```

所以 Dense 路内部候选人排名是：

```text
dense_group_rank:
A = 1
B = 2
C = 3
```

进入跨路 RRF 时，Dense 路同样只贡献一次：

```text
A 的 Dense 贡献 = 1.0 / (60 + 1)
B 的 Dense 贡献 = 1.0 / (60 + 2)
C 的 Dense 贡献 = 1.0 / (60 + 3)
```

这个例子里，B 有 Dense 路的全局最佳单片段，但 A 有多个相关片段，所以 A 在 Dense 路内部被重排到第 1。关键点是：**多证据只改变本路内部排名，不会在最终 RRF 里额外多加一笔分数**。

#### 关于"无结果"判定：从 Dense IQR 弃权到 Reranker 相关性地板

Elasticsearch kNN 会强制返回最近的 K 个证据片段，即使 query 和简历经历没有真实语义关系，也会从库里找出"最接近"的一批结果。早期版本在 Dense 路进入候选人聚合前做一次**分布形状判定**（基于 IQR 上侧离群围栏：`top_score <= Q3 + 1.5×IQR` 则 abstain），用来拦截"没有清晰头部"的向量召回。

**该机制已移除**（`_dense_confidence` / IQR abstain）。消融实验显示：关闭它后评测集所有指标（含负例 `empty_acc`、`forbidden@10`）**逐项零变化**。原因是它与下游的 **Reranker 相关性地板**（见 5.5）触发条件几乎完全重合（都只在 semantic 意图上生效），而"库中无相关人"的最终判定已由 reranker 地板更准确地兜住——cross-encoder 的绝对相关性分比 IQR 的相对形状判定更适合做这件事。删除它是"用一个相关性权威替代两套重叠弃权机制"的简化，符合"补架构缺口而非堆规则"的原则。

> 完整推导见 `PROJECT_REVIEW.md` [第十节](./PROJECT_REVIEW.md)。当前 Dense 路命中**始终**参与 RRF（在 `enable_dense=True` 时），是否返回空完全交给 5.5 的 reranker 地板决定。

#### 完整的 RRF 融合流程

```
Step 1: 检索路内部聚合 (Chunk to Candidate Top-K Pooling)
  - Evidence BM25 路：按候选人名下 top-k 词面证据重排，得出 evidence_group_rank
  - Evidence kNN 路：按候选人名下 top-k 向量证据重排，得出 dense_group_rank

Step 2: 计算基础 RRF 分数 (Standard RRF)
  - evidence BM25 贡献：weight(1.2) / (60 + evidence_group_rank)
  - dense 贡献：weight(1.0) / (60 + dense_group_rank)
  - 基础 RRF = 两者之和

Step 3: 匹配层级 (Lexical Tier) 加分
  - tier=3 (精确匹配命中)：额外 ×1.45
  - tier=2 (短语匹配命中)：额外 ×1.30
  - tier=1 (分词匹配命中)：额外 ×1.15
  - tier=0 (仅 dense 命中)：×1.00

Step 4: Term Coverage 加分
  - 每覆盖一个查询词，额外 ×(1 + 0.05)

Step 5: 最终得分
  final_score = base_rrf × (1.0 + 0.15×tier + 0.05×coverage)

Step 6: 排序输出
  - 按 final_score 降序
  - 同分时按 best_rank 升序
  - 内部最多保留 1000 个候选人
  - 按 offset/limit 返回当前批次，默认首批返回 100 个
```

#### 为什么要做 tier 加分

纯 RRF 只看排名，不区分"匹配质量"。但在简历检索场景下，精确匹配（比如 `skills` 字段中恰好有 "Python"）的可信度远高于仅分词命中（"Python" 出现在项目描述中但可能只是顺带提到）。Tier 加分让精确匹配的候选人获得额外优势。

### 5.4 结果格式化与返回

RRF 排序完成后，对于通过证据索引命中的候选人，系统会 **回查主索引** 获取完整的候选人资料（因为证据索引只有冗余的基本信息，展示详情需要完整字段）。

返回的每条结果包含：

```json
{
  "id": "20190016837",
  "score": 0.0333,
  "candidate": { "name": "张三", ... },
  "application": { "position_name": "NLP工程师", ... },
  "education_summary": "北京交通大学 / 硕士 / 计算机科学",
  "project_snippet": "<mark>自然语言处理</mark>平台...",
  "skills": ["Python", "NLP", ...],
  "years_experience": 0.9,
  "experience_display": "0.9 年工作经验",
  "retrieval_debug": {
    "retrieval_sources": ["evidence", "dense"],
    "evidence_rank": 3,
    "evidence_score": 45.2,
    "dense_rank": 10,
    "dense_score": 0.85,
    "raw_rrf_score": 0.0333,
    "score_multiplier": 1.45,
    "rrf_score": 0.0483,
    "lexical_tier": 3,
    "term_coverage": 2,
    "matched_queries": ["evidence_exact:skills:W40", ...],
    "evidence_matches": [
      { "evidence_id": "20190016837:skills:0", "title": "能力标签", ... }
    ]
  }
}
```

其中 `retrieval_debug` 提供了完整的排序可解释性信息——每个检索器贡献了多少分、命中了哪些字段、最终乘数是多少。这些信息会在前端的 "Debug 排名" 面板中展示。

### 5.5 Rerank 精排与相关性弃权

`intent == semantic` 时，RRF 融合后的 top-N（`RERANK_TOP_N=20`）候选会送入 Qwen3 reranker（cross-encoder）做精排，按 query-doc 相关性重排顺序。

#### 相关性地板：用 reranker 的绝对分做"无结果"判定

RRF 是**纯排名融合**，本质上永远会把结果页填满——它没有"绝对相关性"的概念。当库里根本没有相关的人时，BM25 仍会靠"性能优化""SQL""系统"这类泛 token 部分命中捞回一堆候选，把整页灌满。（早期 Dense 路有 IQR 形状弃权能拦一部分，但已移除——见 5.3 的说明：它与本节的 reranker 地板职责重叠，且 reranker 地板更准。）

解决思路不是再加一道门控，而是**补上链路里缺失的"相关性权威"**：reranker 算的本就是 query-doc 的**绝对**相关性分数（不像余弦只在同 query 批次内可比），但此前只被用来排序、绝对量级被丢弃。`_rerank_results` 现在多做一步判定——

```
若 重排窗口内的最高分 < RERANK_RELEVANCE_FLOOR(0.5)：
  判定为"库中无真正相关候选" → 整体弃权，返回空
否则：
  正常按 rerank 分重排
```

这个地板**不是**调参试出来的脆阈值，也**不挂在 query 文本模式上**。它落在一条经实测验证的宽空白带正中：

| | 真实语义 query（库中有相关人） | 离域负例 query（库中无相关人） |
|---|---|---|
| 最高 rerank 分 | min **0.84**，mean 0.93 | max **0.33**，mean 0.30 |

两个分布隔着 0.33→0.84 的空白带，地板取 0.5 时既不会漏掉任何真实查询（它们最低也有 0.84），又能切掉所有负例（最高才 0.33）。判定只看 reranker 自己的输出，是内容无关的绝对相关性事实。

> **为什么不复用 Dense 那套 IQR 形状判定？** 最初设想是把 5.3 的"分布有没有清晰头部"直接搬到 rerank 分上，但测量数据否决了它：负例虽然整体分低，相对自己的背景仍可能有个小头部（IQR 误判为"有结果"），而个别真实查询的头部相对背景反而不够突出。根因是 IQR 适合余弦这类只有相对意义的分数，而 cross-encoder 的分数本身带绝对相关性语义——这里应该看绝对量级，不是形状。这是一个"先测量再决策"否决了直觉方案的例子。

#### 容错

如果 DeepSeek query parser 调用失败，intent 兜底为 `semantic`，仍会触发 rerank（此时 lexical/semantic query 退化为原始 query），保证解析失败不影响搜索可用性。reranker API 本身失败时，`_rerank_results` 返回未重排的 RRF 结果并附 warning，不阻断请求。

---

## 6. 前端交互

前端是纯 HTML + CSS + JavaScript 实现的单页应用，没有使用任何框架。

### 页面结构

```
┌─────────────────────────────────────────────────────────┐
│ Header：品牌标识 | 搜索框 + 快捷搜索 | ES 状态指示灯   │
├───────────┬─────────────────────────────────────────────┤
│ 左侧筛选   │  结果区域                                   │
│           │                                             │
│ 学历 (radio)│  显示 N 条结果  搜索：xxx                  │
│ 经验 (range)│  ┌──────────────────────────────────┐      │
│ 城市 (chips)│  │ 候选人卡片                        │      │
│ 技能 (chips)│  │  姓名 | 岗位·编号                  │      │
│           │  │  教育信息 | 工作经验                │      │
│           │  │  匹配摘要 (带高亮)                  │      │
│           │  │  技能标签                           │      │
│           │  │  [Debug 排名] [查看详情]             │      │
│           │  └──────────────────────────────────┘      │
│           │                                             │
│           │  (更多卡片...)                              │
├───────────┴─────────────────────────────────────────────┤
│ 详情抽屉 (右侧滑出)                                     │
│  基础信息 | 应聘信息 | 教育经历 | 实习经历 | 项目经验 ...│
└─────────────────────────────────────────────────────────┘
```

### 交互功能

- **搜索框**：输入时 300ms 防抖自动搜索，Enter 键立即搜索
- **快捷搜索**：预设 "Python 自然语言处理" / "机器学习 医疗" / "Java 服务端"
- **筛选条件**：学历(radio)、经验(range slider)、城市(chips)、技能(chips)，数据来自 ES 聚合
- **结果卡片**：展示候选人摘要，带 `<mark>` 高亮的匹配片段
- **加载更多**：首屏展示 100 个候选人，最多可继续查看 RRF 窗口内的 1000 个候选人
- **Debug 面板**：展开可查看 BM25/Dense 召回、RRF 融合、rerank 重排状态、排名变化和匹配详情
- **详情抽屉**：右侧滑出面板，展示候选人完整资料
- **ES 健康检查**：每 30 秒轮询 `/api/health`，显示在线/离线状态

### API 接口

| 端点                        | 方法 | 用途             |
|-----------------------------|------|------------------|
| `/`                         | GET  | 返回前端页面     |
| `/api/search`               | GET  | 核心搜索接口     |
| `/api/health`               | GET  | ES 健康检查      |
| `/api/resumes/{resume_id}`  | GET  | 获取单份简历详情 |
| `/static/*`                 | GET  | 静态资源         |

---

## 7. 检索效果评估

项目包含一套基于 JSONL 格式的评估框架 (`evaluate_search.py` + `eval_queries.jsonl`)。当前评测集定位为 **向量模型/语义检索/精排评测集**，用于比较 embedding 模型、观察 hard negative 误召回、评估 rerank 对头部排序的收益与回退。

当前 `eval_queries.jsonl` 包含 76 条查询：

| 类型 | 数量 | 设计目的 |
|------|------|----------|
| `semantic_capability` | 8 | 直接考察语义能力匹配，如 RAG、DevSecOps、低延迟 C++、蓝队应急等 |
| `cross_language` | 8 | 中英混合/英文表达，检查模型对技术英文和缩写的理解 |
| `negative_semantic` | 8 | 库中不存在的人才需求，检查 no-result 判定 |
| `skill_combo` | 8 | 多技能组合查询，检查 BM25、向量和 RRF 是否能稳定覆盖组合能力 |
| `structured_filter` | 8 | 结构化约束 + 检索文本，检查 Query Planner 抽取出的学历/城市/年限过滤是否正确 |
| `exact_lookup` | 8 | 编号、手机号、邮箱、岗位编号等精确查找，检查实体查询不会被向量泛化污染 |
| `entity_exact` | 8 | 学校、公司、候选人等实体查询，检查短语/精确匹配优先级 |
| `major_query` | 8 | 专业名查询，检查中英文专业、相近专业和跨字段泛化的边界 |
| `jd_match` | 12 | 长 JD 匹配，检查 LLM Query Planner 与 reranker 的协同效果 |

评测集使用 3/2/1 分级相关性：

| 分数 | 含义 |
|------|------|
| 3 | 高度匹配，应该排在最前 |
| 2 | 明显相关，可以进入首屏候选 |
| 1 | 弱相关或可作为补充召回 |
| 0 / 未标注 | 不相关 |

### 评估指标

| 指标         | 含义                                   | 主要看什么 |
|--------------|----------------------------------------|------------|
| P@5 / P@10   | 前 K 条里有多少比例是相关候选人。比如 P@5=0.6 表示前 5 条里有 3 条相关。 | 前排结果精度 |
| R@5 / R@10   | 前 K 条覆盖了多少比例的全部相关候选人。比如相关候选共 4 个，前 10 条找到 3 个，则 R@10=0.75。 | 前排召回 |
| R@50 / R@100 | 前 50/100 条覆盖了多少比例的全部相关候选人。这个指标主要用于判断第一阶段召回池是否足够，尤其适合评估后续 rerank 是否有空间。 | 候选池召回 |
| MRR@10       | 前 10 条里第一个相关候选人的倒数排名。第 1 条相关则为 1.0，第 2 条相关则为 0.5；前 10 条没有相关则为 0。 | 首个好结果是否靠前 |
| NDCG@5 / NDCG@10 | 基于 3/2/1 分级相关性衡量排序质量。3 分候选越靠前，分数越高；把弱相关排在强相关前面会被扣分。 | 分级排序质量 |
| forbidden@10 | 前 10 条中出现了多少个明确不应出现的候选人，数值越低越好。适合检查语义泛化导致的误召回。 | 严重误召回 |
| empty_acc    | 对于 `expect_empty=true` 的负例查询，系统是否返回空结果。比如"量子计算芯片设计经验"这类库里没有的人才需求，返回空才算正确。 | 无结果判断 |

这些指标的侧重点不同：

- **看排序是否好**：优先看 `NDCG@5/NDCG@10`、`MRR@10`、`P@5/P@10`。
- **看召回是否够**：优先看 `R@50/R@100`。如果 R@100 很低，说明第一阶段检索没有把正确候选召回来，后面加 rerank 也救不了。
- **看负例是否稳**：优先看 `empty_acc` 和 `forbidden@10`。这类指标能暴露 dense 检索"总要返回一些相似候选"的问题。
- **看分查询类型表现**：优先看 `by_type`，不要只看 overall。实体查询、技能组合、自然语言语义查询的难度不同，混在一起平均容易掩盖问题。

### 运行方式

```bash
# 需要先启动 ES 并导入数据
python evaluate_search.py

# 输出 JSON 实验报告（含 overall / by_type / details）
python evaluate_search.py --output reports/current.json

# 与上一份报告做指标 delta 对比
python evaluate_search.py --output reports/current.json --compare-to reports/baseline.json

# 打印每个查询的详细结果
python evaluate_search.py --details
```

评估脚本默认请求 100 条结果，同时输出整体指标和按查询类型分组的指标，便于观察 `semantic_capability`、`paraphrase_intent`、`hard_negative_boundary`、`cross_language`、`negative_semantic` 等查询族的收益与回退。当前 `eval_queries.jsonl` 使用静态分级 qrels，与 `data/ai_generated.jsonl` 的 100 条模拟简历对齐，避免动态相关集为空时大量跳过用例。

### 评估用例格式

```jsonl
{
  "id": "hard_rag_not_recsys",
  "type": "hard_negative_boundary",
  "query": "算法候选，要能做 RAG 工程落地和知识库问答，不要纯推荐排序",
  "relevance": {
    "M20260013": 3,
    "M20260046": 3,
    "M20260086": 3,
    "M20260032": 2
  },
  "forbidden_ids": ["M20260005", "M20260039", "M20260062"]
}
```

也可以继续使用旧的 `relevant_ids` 二值格式，评估脚本会把它们视为 1 分相关。但新增用例建议优先使用 `relevance` 分级标注，并显式维护 `forbidden_ids`。

---

## 8. 本地部署与运行

### 前置条件

- Python 3.10+
- Elasticsearch 9.x（需安装 IK 分词插件）
- 本地 embedding 模式约需 4-8GB 可用内存；豆包 API 模式不需要加载本地模型

### 步骤 1：安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 步骤 2：安装并启动 Elasticsearch

```bash
# 下载 ES 9.x（项目目录下已有 elasticsearch-9.3.0，可直接使用）
cd elasticsearch-9.3.0

# 安装 IK 分词插件（如果尚未安装）
bin/elasticsearch-plugin install https://get.infini.cloud/elasticsearch/analysis-ik/9.3.0

# 启动 ES（开发模式，单节点）
bin/elasticsearch -d  # -d 表示后台运行

# 验证 ES 是否启动
curl http://localhost:9200/_cluster/health
```

### 步骤 3：选择 embedding 后端并导入数据

默认使用 Qwen `text-embedding-v4` API，不需要下载本地 embedding 模型。如果要切换到豆包 API，先把 `embedding_service.py` 顶部改成：

```python
EMBEDDING_PROVIDER = DOUBAO_PROVIDER
```

如果要切换到本地 Yuan 模型，则改成：

```python
EMBEDDING_PROVIDER = LOCAL_PROVIDER
```

然后导入数据：

```bash
# 如果有 .doc 简历文件，放入 data/ 目录
python import_to_es.py data/

# 如果使用 JSONL 格式的模拟数据
python import_to_es.py data/ai_generated.jsonl
```

> **⚠️ 只有切换到 `LOCAL_PROVIDER` 时才会下载本地 Yuan embedding 模型**（约 2-4GB），需要网络连接。如果国内访问 HuggingFace 受限，可设置 `HF_ENDPOINT=https://hf-mirror.com`。Qwen 和豆包 API 模式都不会下载本地模型。

### 步骤 4：启动后端

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 步骤 5：访问前端

浏览器打开 `http://localhost:8000`

### 步骤 6：运行测试

```bash
# 单元测试（不需要 ES）
python -m pytest -q

# 检索质量评估（需要 ES 在线且已导入数据）
python evaluate_search.py --details --output reports/current.json
```

---

## 9. 配置项与常量

### 环境变量

| 变量            | 说明                                        | 默认值                    |
|-----------------|---------------------------------------------|---------------------------|
| `HF_ENDPOINT`   | HuggingFace Hub 镜像地址                    | `https://huggingface.co`  |

### `app.py` 中的关键常量

| 常量                         | 值     | 说明                                         |
|------------------------------|--------|----------------------------------------------|
| `ES_URL`                     | `http://localhost:9200` | Elasticsearch 地址              |
| `INDEX_ALIAS`                | `resumes_current` | 主索引别名                          |
| `EVIDENCE_INDEX_ALIAS`       | `resume_evidence_current` | 证据索引别名              |
| `RRF_RANK_CONSTANT`          | 60     | RRF 公式中的 k 值                            |
| `RRF_RANK_WINDOW_SIZE`       | 1000   | Evidence BM25 召回证据片段数，也是 RRF 后保留的最大候选人窗口 |
| `DEFAULT_SEARCH_LIMIT`       | 100    | `/api/search` 默认返回候选人数               |
| `KNN_NUM_CANDIDATES`         | 300    | kNN 检索的候选池大小                         |
| `DENSE_RRF_WEIGHT`           | 1.0    | Dense 路在 RRF 中的外层权重                  |
| `EVIDENCE_RRF_WEIGHT`        | 1.2    | Evidence BM25 路在 RRF 中的权重              |
| `EVIDENCE_DENSE_RRF_WEIGHT`  | 1.0    | Evidence kNN 路在 RRF 中的权重               |
| `DENSE_RANK_WINDOW_SIZE`     | 300    | Dense 检索的最大结果窗口                     |
| `ENABLE_RERANK`              | `True` | 是否对 `intent == semantic` 的 query 启用 Qwen3 reranker |
| `RERANK_TOP_N`               | 20     | RRF 后进入 reranker 的候选人数               |
| `RERANK_RELEVANCE_FLOOR`     | 0.5    | 重排窗口最高分低于此地板则判定"库中无相关候选"，弃权返回空（见 5.5） |
| `MIN_YEARS_TOLERANCE_RATIO`  | 0.10   | `min_years` 软容差比例（见 5.1 Step 1）      |
| `MIN_YEARS_TOLERANCE_FLOOR`  | 0.5    | `min_years` 软容差的最小绝对值（年）          |
| `QUERY_PARSER_PROVIDER`      | `deepseek` | 自由文本 query 的 LLM 解析 provider      |
| `QUERY_PARSER_MODEL_ID`      | `deepseek-v4-flash` | Query Planner 模型名              |
| `QUERY_PARSER_API_URL`       | `https://api.deepseek.com/chat/completions` | DeepSeek OpenAI-compatible Chat Completions 接口 |
| `QUERY_PARSER_API_KEY`       | 已硬编码 | DeepSeek API key，开发阶段直接写在代码中      |
| `QUERY_PARSER_TIMEOUT_SECONDS` | 30   | Query Planner 请求超时时间                    |
| `QUERY_PLAN_CACHE_TTL_SECONDS` | 300  | Query Planner 解析结果缓存时间（规范化 query 为 key） |
| `QUERY_PLAN_CACHE_MAX_ENTRIES` | 512  | Query Planner 解析结果 LRU 缓存最大条数        |
| `DeepSeek thinking`          | `disabled` | Query Planner 请求体设置 `thinking: {"type": "disabled"}`，优先低延迟 |
| `EVIDENCE_POOL_EXTRA_WEIGHTS`| `(0.30, 0.15)` | 同候选人第 2、3 个证据在本路内部排序中的衰减权重 |
| `QUERY_TERM_COVERAGE_BOOST`  | 0.001  | Term Coverage 的 constant_score boost        |
| `MAX_BROWSE_RESULT_SIZE`     | 1,000  | 搜索和浏览最多可查看的候选人窗口             |
| `FACETS_CACHE_TTL_SECONDS`   | 60     | Facet 聚合结果缓存时间                       |
| `FILTER_VOCAB_CACHE_TTL_SECONDS` | 300 | 过滤词表缓存时间                            |

### `import_to_es.py` 中的关键常量

| 常量                           | 值     | 说明                                       |
|--------------------------------|--------|--------------------------------------------|
| `BULK_BATCH_SIZE`              | 100    | Bulk API 每批文档数                        |
| `VECTOR_EVIDENCE_SECTION_TYPES`| `{project, internship}` | 需要向量化的证据类型；技能标签和教育经历只走词面检索 |

### `embedding_service.py` 中的关键常量

| 常量          | 默认值     | 说明                     |
|---------------|------------|--------------------------|
| `EMBEDDING_PROVIDER` | `QWEN_PROVIDER` | 当前默认 embedding 后端；可改为 `DOUBAO_PROVIDER` 或 `LOCAL_PROVIDER` |
| `MODEL_ID`    | `qwen:text-embedding-v4` | 当前 provider 的模型标识 |
| `VECTOR_DIMS` | 2048       | 当前 provider 的向量维度 |
| `QWEN_API_KEY` | 已硬编码 | Qwen embedding API key |
| `QWEN_API_URL` | DashScope embedding 接口 | Qwen embedding 请求地址 |
| `QWEN_MODEL_ID` | `text-embedding-v4` | Qwen embedding 模型名 |
| `QWEN_VECTOR_DIMS` | 2048 | Qwen embedding 向量维度 |
| `QWEN_BATCH_SIZE` | 10 | Qwen embedding 每批请求条数 |
| `DOUBAO_API_KEY` | 已硬编码 | 豆包 API key |
| `DOUBAO_API_BASE` | `https://ark.cn-beijing.volces.com/api/v3` | 豆包 OpenAI-compatible API base URL |
| `DOUBAO_MODEL_ID` | `ep-20260412051954-zl5fm` | 豆包 embedding 接入点 / model id |
| `DOUBAO_VECTOR_DIMS` | 2048 | 豆包 embedding 向量维度 |
| `DOUBAO_BATCH_SIZE` | 64 | 豆包 embedding 每批请求条数 |
| `DOUBAO_TIMEOUT_SECONDS` | 60 | 豆包 embedding 请求超时时间 |
| `DOUBAO_MULTIMODAL` | `True` | 使用 `/embeddings/multimodal` 接口 |
| `DOUBAO_SEND_DIMENSIONS` | `True` | 请求体中带 `dimensions` 参数 |
| `LOCAL_MODEL_ID` | `IEITYuan/Yuan-embedding-2.0-zh` | 本地 Yuan embedding 模型名 |
| `LOCAL_VECTOR_DIMS` | 1792 | 本地 Yuan embedding 向量维度 |

### `rerank_service.py` 中的关键常量

| 常量          | 默认值     | 说明                     |
|---------------|------------|--------------------------|
| `RERANK_PROVIDER` | `dashscope_api` | 当前 reranker 后端标识 |
| `RERANK_MODEL_ID` | `qwen3-rerank` | API 模型名 |
| `RERANK_API_URL` | 阿里云文本重排接口 | reranker HTTP 调用地址 |
| `RERANK_API_KEY` | 已硬编码 | reranker API key |
| `RERANK_BATCH_SIZE` | 20 | reranker 每批候选文档数 |
| `RERANK_TIMEOUT_SECONDS` | 60 | reranker API 请求超时时间 |

---

## 10. 当前实现说明与后续优化方向

### 10.1 检索质量优化决策（基于评测集驱动）

以下几项改动都由 `evaluate_search.py` 的指标驱动——先定位塌陷点，再针对性优化，最后用 delta 对比验证收益与零回退。

| 优化项 | 解决的问题 | 做法 | 设计哲学 |
|---|---|---|---|
| **`min_years` 软容差** | "4 年以上"硬卡 `>=4` 把 3.9 年的相关候选直接灭掉（某 query 因此 R@100=0） | `gte` 下调 `max(N×10%, 0.5)`，资历仍由排序体现 | 模糊的用户表达不该当成硬边界；过滤负责"放进池子"，排序负责"排好序" |
| **Reranker 相关性地板** | 负例 query（库中无相关人）被 BM25 泛 token 命中灌满整页，`empty_acc` 仅 0.556 | 重排窗口最高分 < 0.5 则整体弃权返回空 | 补上链路缺失的"相关性权威"，复用 reranker 已有的绝对相关性分；地板落在实测空白带（真实查询 ≥0.84 vs 负例 ≤0.33）正中，内容无关、非脆阈值（详见 5.5） |
| **正则 fast-path** | 邮箱/手机/编号等可正则判定的查询白白付一次 LLM 往返 | 单 token 标识符直接判 `lookup` 短路 LLM | 最廉价的查询不该走最贵的链路 |
| **Query→Plan 缓存** | 前端 300ms 防抖 + 重复 query 反复打 LLM | 规范化 query 为 key 的 TTL LRU 缓存 | 同一意图不重复解析；fallback 不入缓存避免污染 |

> **关于"算法优雅 vs 特征工程"**：这套优化刻意避免了"为每种 query 模式加门控/特判"的老路（参见 `PROJECT_REVIEW.md` 第三节"针对'大学'做特判是错误决策"的教训）。相关性地板看的是 reranker 输出的绝对分布，`min_years` 容差是一个统一的数值规则，都不挂在 query 文本内容上——是"补架构缺口"而非"堆规则"。

### 10.2 Query Planner 工程实现

- **三层短路解析**：正则 fast-path → 解析缓存 → LLM（详见 5.1）。
- **结构化输出**：默认 `json_object`（实测 `deepseek-v4-flash` 拒绝 `json_schema`），代码保留严格 schema 链路与自动回退，换模型即可启用。
- **多层容错降级**：LLM 失败 → 保守词面检索；JSON 损坏 → fallback；空 query → 不调 LLM。解析失败不影响搜索可用性。

### 10.3 当前指标基线

最新 `reports/current.json`（100 条模拟简历 + 76 条评测 query）：overall NDCG@10≈0.96、MRR@10≈1.0、R@100≈0.99、empty_acc=1.0。详见第 7 节。

### 10.4 后续可继续的方向

- **有结果查询内部的排序噪声**：`forbidden@10` 主要来自 `jd_match`、`semantic_capability`、`cross_language` 这些库中确有相关人、但前 10 混入了 forbidden 的场景。这是精排质量问题（不是"该不该返回"），是下一步主要抓手。
- **API key 与配置外置**：当前 key 硬编码在源码中（开发阶段取舍），生产应移至环境变量或密钥管理。
- **旧索引清理策略**：versioned index 多次重建会留下旧索引，需保留最近 N 个可回滚、定期清理（详见 `PROJECT_REVIEW.md` 第九节）。
- **生产环境磁盘阈值**：本地曾临时关闭 ES 磁盘 watermark，生产须通过扩容/清理解决，不能关闭保护。
