# 简历检索系统 (ResumeSearch)

一个面向校招/实习简历筛选场景的 **混合检索原型系统**。系统以 Elasticsearch 为核心引擎，实现了 **结构化过滤 + 中文 BM25 全文检索 + 短语匹配 + 证据切片向量检索 + RRF 融合排序**，并配套前端 Web 界面供交互使用。

> **适读对象**：项目新接手者、面试评审方、需要理解系统全貌的协作开发者。读完本文档后，你应当能回答以下问题：
>
> - 这个项目的完整数据链路是怎样的（从 `.doc` 文件到可搜索索引）？
> - 用户输入一个 query 后，系统经历了哪些步骤才返回结果？
> - 为什么需要"证据索引"？它比直接对整份简历做向量化好在哪里？
> - RRF 融合排序具体是怎么计算的？
> - 系统有哪些已知边界和可优化方向？

---

## 目录

- [1. 业务背景与问题定义](#1-业务背景与问题定义)
- [2. 项目架构总览](#2-项目架构总览)
- [3. 技术栈与依赖](#3-技术栈与依赖)
- [4. 数据处理流程](#4-数据处理流程)
  - [4.1 简历解析 (resume_parser.py)](#41-简历解析-resume_parserpy)
  - [4.2 证据切片构建与向量化 (import_to_es.py)](#42-证据切片构建与向量化-import_to_espy)
  - [4.3 Embedding 服务 (embedding_service.py)](#43-embedding-服务-embedding_servicepy)
  - [4.4 Elasticsearch 索引设计](#44-elasticsearch-索引设计)
- [5. 检索流程详解](#5-检索流程详解)
  - [5.1 请求入口与 Query 解析](#51-请求入口与-query-解析)
  - [5.2 两路并行检索](#52-两路并行检索)
  - [5.3 RRF 融合排序](#53-rrf-融合排序)
  - [5.4 结果格式化与返回](#54-结果格式化与返回)
- [6. 前端交互](#6-前端交互)
- [7. 检索效果评估](#7-检索效果评估)
- [8. 本地部署与运行](#8-本地部署与运行)
  - [9. 配置项与常量](#9-配置项与常量)
- [10. 与旧 README 的差异说明](#10-与旧-readme-的差异说明)
- [11. 当前实现说明与后续优化方向](#11-当前实现说明与后续优化方向)

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
│  │ Query 解析   │→│ 并行检索调度  │→│ RRF 融合 + 排序 │       │
│  │ _parse_query │  │ ThreadPool   │  │ _rrf_merge()    │       │
│  │ _constraints │  │ Executor     │  │                 │       │
│  └──────────────┘  └──────────────┘  └─────────────────┘       │
│         │                │                                      │
│         │    ┌───────────┼───────────┐                          │
│         │    ▼           ▼           ▼                          │
│         │  Evidence    Evidence    主索引                        │
│         │  BM25检索    kNN检索    详情回填                       │
│         │  (词面)      (向量)                                     │
│         ▼                                                       │
│  embedding_service.py  ←→  本地 Yuan / 豆包 API embedding        │
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
| `resume_parser.py`      | 解析 HTML 格式 `.doc` 简历文件，提取结构化字段（候选人、教育、实习、项目、技能等） |
| `import_to_es.py`       | 加载解析结果 → 构建证据切片 → 调用 embedding 服务向量化 → 批量写入 ES 双索引 |
| `embedding_service.py`  | 统一封装本地 `Yuan-embedding-2.0-zh` 与豆包 API embedding 后端，提供 `encode_single` / `encode_batch` |
| `app.py`                | FastAPI 后端：Query 解析 → 并行混合检索 → RRF 融合排序 → 结果格式化 |
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
| Embedding 模型 | 本地 `IEITYuan/Yuan-embedding-2.0-zh` / 豆包 API   | 本地 1792 维；豆包默认 2048 维，可通过代码常量切换 |
| 模型框架       | sentence-transformers + PyTorch / requests        | 本地模型推理或 API 调用                         |
| 模型下载       | ModelScope + HuggingFace Hub                      | 仅本地模型需要下载（ModelScope 主体 + HF Dense 层权重） |
| HTML 解析      | BeautifulSoup4                                    | 解析 HTML 格式的 .doc 简历文件                 |
| 前端           | 原生 HTML + CSS + JavaScript                      | 无框架依赖的轻量前端                           |

### Python 依赖 (`requirements.txt`)

```
beautifulsoup4>=4.12
fastapi>=0.100
requests>=2.31
uvicorn>=0.23
sentence-transformers==3.4.1
torch>=2.0
modelscope>=1.14
huggingface-hub>=0.20
```

---

## 4. 数据处理流程

数据处理分为三个阶段：**简历解析** → **证据切片构建与向量化** → **写入 ES 索引**。

### 4.1 简历解析 (`resume_parser.py`)

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
   - 从实习经历中估算工作年限 (years_experience)
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
   - 证据文本有字符预算控制（长经历 section 512 字符, skills 256 字符）
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
python import_to_es.py mock_resumes_llm_diverse.jsonl

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

默认使用 **`IEITYuan/Yuan-embedding-2.0-zh`**，一个 1792 维的中文 embedding 模型。也可以通过 `embedding_service.py` 顶部常量切换到豆包 API embedding 后端，默认接入点为 `ep-20260412051954-zl5fm`。

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

豆包 API 模式不会下载或加载本地模型。当前接入点对应豆包多模态 embedding 模型，因此调用 OpenAI-compatible `/embeddings/multimodal` 接口，文本会包装为 `{"type":"text","text":"..."}`，并在本地继续做 L2 归一化，保证与 ES `cosine` 检索契约一致。

#### Provider 切换

默认使用本地模型，`embedding_service.py` 顶部配置为：

```python
EMBEDDING_PROVIDER = LOCAL_PROVIDER
```

切换到豆包 API 时改为：

```python
EMBEDDING_PROVIDER = DOUBAO_PROVIDER
DOUBAO_MODEL_ID = "ep-20260412051954-zl5fm"
DOUBAO_VECTOR_DIMS = 2048
```

豆包 API key 已在同文件顶部的 `DOUBAO_API_KEY` 常量中配置。`DOUBAO_VECTOR_DIMS` 必须与接口实际返回的向量维度一致；如果接口返回维度变化，导入脚本会直接报错，避免把错误维度写入 ES。

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
| `evidence_vector`       | dense_vector  | 1792维，cosine相似度，HNSW索引（m=32, ef=300） |
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

从 URL 参数中的 `degree`、`cities`、`skills`、`min_years` 构建 Elasticsearch filter 子句。

**Step 2：Query 文本解析** (`_parse_query_constraints`)

系统会尝试从自由文本中识别出结构化约束，并将其从 query 中剥离：

```
输入: "0.5年以上 北京 本科 Python 自然语言处理"
      ↓ 解析
识别出:
  - "0.5年以上" → filter: years_experience >= 0.5
  - "北京"     → filter: expected_work_cities = "北京"
  - "本科"     → filter: highest_degree = "本科"
残留 query_text: "Python 自然语言处理"
```

解析规则：
- **年限识别**：正则匹配 `数字+年(以上|+)` 的模式
- **城市识别**：与 ES 中实际存在的城市词表做精确匹配
- **学历识别**：与已知学历词表匹配，支持别名映射（"博士研究生"→"博士"）
- **技能提升为 filter**：仅当 query 中已包含其他结构化约束时，才把匹配到的技能词从自由搜索提升为硬过滤。这是为了避免纯技能查询（如"Python"）被错误地限定为只搜 skills 字段

**Step 3：决定是否启用向量检索** (`_use_dense`)

```python
# 不启用向量检索的情况：
# 1. query 为空
# 2. query 看起来是精确查找（编号、手机号、邮箱、以"大学/学院/公司/集团"结尾）
# 3. query 太短（去空格后 < 4 个字符 且只有 1 个 token）
#
# 启用的条件：
# - query 有 >= 2 个 token（空格分隔），或
# - 去空格后 >= 4 个字符
```

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
    "query_vector": [0.123, -0.456, ...],  // 1792 维
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

#### Dense 路的自适应拒绝召回

Elasticsearch kNN 会强制返回最近的 K 个证据片段，即使 query 和简历经历没有真实语义关系，也会从库里找出"最接近"的一批结果。因此 Dense 路在进入候选人聚合前，会先判断本次向量检索是否有清晰的头部信号。

项目不使用固定相似度阈值（例如 `score > 0.80`），而是使用本次 dense 结果分布的上侧离群判断：

```text
Q1 = dense scores 的 25 分位数
Q3 = dense scores 的 75 分位数
IQR = Q3 - Q1
upper_fence = Q3 + 1.5 * IQR

如果 top_score <= upper_fence：
  Dense 路 abstain，不参与 RRF
否则：
  Dense 路进入候选人聚合和 RRF
```

这个判断只看本次检索结果的分布形状，不关心 query 是不是学校、公司、技能或自然语言需求。比如搜索"哥伦比亚"时，dense topK 可能整体都在 0.75-0.78 之间，top1 没有明显高出背景分布，Dense 路会拒绝召回；而搜索"做过推荐系统召回和 NLP 模型落地的人"时，头部向量证据会明显高于背景，Dense 路才参与融合。

#### 完整的 RRF 融合流程

```
Step 1: Dense 路自适应判断
  - 如果 Dense topK 没有清晰头部，Dense 路 abstain
  - 如果 Dense topK 有清晰头部，继续进入候选人聚合

Step 2: 检索路内部聚合 (Chunk to Candidate Top-K Pooling)
  - Evidence BM25 路：按候选人名下 top-k 词面证据重排，得出 evidence_group_rank
  - Evidence kNN 路：按候选人名下 top-k 向量证据重排，得出 dense_group_rank

Step 3: 计算基础 RRF 分数 (Standard RRF)
  - evidence BM25 贡献：weight(1.2) / (60 + evidence_group_rank)
  - dense 贡献：weight(1.0) / (60 + dense_group_rank)
  - 基础 RRF = 两者之和

Step 4: 匹配层级 (Lexical Tier) 加分
  - tier=3 (精确匹配命中)：额外 ×1.45
  - tier=2 (短语匹配命中)：额外 ×1.30
  - tier=1 (分词匹配命中)：额外 ×1.15
  - tier=0 (仅 dense 命中)：×1.00

Step 5: Term Coverage 加分
  - 每覆盖一个查询词，额外 ×(1 + 0.05)

Step 6: 最终得分
  final_score = base_rrf × (1.0 + 0.15×tier + 0.05×coverage)

Step 7: 排序输出
  - 按 final_score 降序
  - 同分时按 best_rank 升序
  - 取 limit 条返回
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
- **Debug 面板**：展开可查看 RRF 融合计算过程、各路检索的排名/分数/匹配详情
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

项目包含一套基于 JSONL 格式的评估框架 (`evaluate_search.py` + `eval_queries.jsonl`)。当前评测集定位为 **向量模型/语义检索评测集**，用于比较 embedding 模型、观察 hard negative 误召回、评估后续 rerank 是否有必要。

当前 `eval_queries.jsonl` 包含 75 条查询：

| 类型 | 数量 | 设计目的 |
|------|------|----------|
| `semantic_capability` | 40 | 直接考察语义能力匹配，如 RAG、DevSecOps、低延迟 C++、蓝队应急等 |
| `paraphrase_intent` | 8 | 用自然语言转述能力需求，减少对关键词字面命中的依赖 |
| `hard_negative_boundary` | 10 | 查询中显式排除近似但错误的候选人，检查语义过度泛化 |
| `cross_language` | 8 | 中英混合/英文表达，检查模型对技术英文和缩写的理解 |
| `negative_semantic` | 9 | 库中不存在的人才需求，检查 no-result 判定 |

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

默认使用本地模型。如果要使用豆包 API，先把 `embedding_service.py` 顶部改成：

```python
EMBEDDING_PROVIDER = DOUBAO_PROVIDER
```

然后导入数据：

```bash
# 如果有 .doc 简历文件，放入 data/ 目录
python import_to_es.py data/

# 如果使用 JSONL 格式的模拟数据
python import_to_es.py mock_resumes_llm_diverse.jsonl
```

> **⚠️ 本地模型首次运行会自动下载 embedding 模型**（约 2-4GB），需要网络连接。如果国内访问 HuggingFace 受限，可设置 `HF_ENDPOINT=https://hf-mirror.com`。如果将 `EMBEDDING_PROVIDER` 常量改为 `DOUBAO_PROVIDER`，则不下载本地模型。

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
| `RRF_RANK_WINDOW_SIZE`       | 100    | 每路检索参与 RRF 的最大文档数                |
| `KNN_NUM_CANDIDATES`         | 300    | kNN 检索的候选池大小                         |
| `DENSE_RRF_WEIGHT`           | 1.0    | Dense 路在 RRF 中的外层权重                  |
| `EVIDENCE_RRF_WEIGHT`        | 1.2    | Evidence BM25 路在 RRF 中的权重              |
| `EVIDENCE_DENSE_RRF_WEIGHT`  | 1.0    | Evidence kNN 路在 RRF 中的权重               |
| `DENSE_RANK_WINDOW_SIZE`     | 300    | Dense 检索的最大结果窗口                     |
| `DENSE_ABSTAIN_MIN_SAMPLE_SIZE` | 20  | Dense 自适应拒绝召回所需的最小样本数        |
| `DENSE_ABSTAIN_IQR_MULTIPLIER` | 1.5  | Dense 上侧离群围栏的 IQR 倍率               |
| `EVIDENCE_POOL_EXTRA_WEIGHTS`| `(0.30, 0.15)` | 同候选人第 2、3 个证据在本路内部排序中的衰减权重 |
| `QUERY_TERM_COVERAGE_BOOST`  | 0.001  | Term Coverage 的 constant_score boost        |
| `MAX_BROWSE_RESULT_SIZE`     | 10,000 | 浏览模式下的最大返回数                       |
| `FACETS_CACHE_TTL_SECONDS`   | 60     | Facet 聚合结果缓存时间                       |
| `FILTER_VOCAB_CACHE_TTL_SECONDS` | 300 | 过滤词表缓存时间                            |

### `import_to_es.py` 中的关键常量

| 常量                           | 值     | 说明                                       |
|--------------------------------|--------|--------------------------------------------|
| `BULK_BATCH_SIZE`              | 100    | Bulk API 每批文档数                        |
| `SECTION_SEMANTIC_CHAR_BUDGET` | 512    | 每个 section 证据的最大字符数              |
| `SKILLS_SEMANTIC_CHAR_BUDGET`  | 256    | 技能证据的最大字符数                       |
| `PROFILE_LEXICAL_CHAR_BUDGET`  | 768    | 档案证据的最大字符数                       |
| `VECTOR_EVIDENCE_SECTION_TYPES`| `{project, internship}` | 需要向量化的证据类型；技能标签和教育经历只走词面检索 |

### `embedding_service.py` 中的关键常量

| 常量          | 默认值     | 说明                     |
|---------------|------------|--------------------------|
| `EMBEDDING_PROVIDER` | `LOCAL_PROVIDER` | embedding 后端；改为 `DOUBAO_PROVIDER` 即切换到豆包 API |
| `MODEL_ID`    | `IEITYuan/Yuan-embedding-2.0-zh` | 当前 provider 的模型标识；豆包模式下为 `doubao:<接入点>` |
| `VECTOR_DIMS` | 1792       | 当前 provider 的向量维度；豆包模式默认 2048 |
| `DOUBAO_API_KEY` | 已硬编码 | 豆包 API key |
| `DOUBAO_API_BASE` | `https://ark.cn-beijing.volces.com/api/v3` | 豆包 OpenAI-compatible API base URL |
| `DOUBAO_MODEL_ID` | `ep-20260412051954-zl5fm` | 豆包 embedding 接入点 / model id |
| `DOUBAO_VECTOR_DIMS` | 2048 | 豆包 embedding 向量维度 |
| `DOUBAO_BATCH_SIZE` | 64 | 豆包 embedding 每批请求条数 |
| `DOUBAO_TIMEOUT_SECONDS` | 60 | 豆包 embedding 请求超时时间 |
| `DOUBAO_MULTIMODAL` | `True` | 使用 `/embeddings/multimodal` 接口 |
| `DOUBAO_SEND_DIMENSIONS` | `True` | 请求体中带 `dimensions` 参数 |

---

## 10. 与旧 README 的差异说明

旧版 README 中有部分内容与当前代码实现已存在较大出入，以下列出主要差异：

| 旧 README 描述                         | 实际代码实现                                                |
|----------------------------------------|-------------------------------------------------------------|
| 主索引上的 BM25 检索作为独立路线        | 当前已移除主索引上的独立 BM25 路线，所有词面检索统一在证据索引上进行 |
| 候选人级别的整文档向量（`semantic_profile_vector`、`role_vector` 等） | 已标记为 obsolete，导入时主动删除。当前向量化只在证据索引的切片级别进行 |
| 4 个候选人向量字段（`skills_vector`、`projects_vector` 等） | 已标记为 `LEGACY_CANDIDATE_VECTOR_FIELDS`，不再生成和使用 |
| 多路 BM25 + 多路 Dense 的检索架构       | 简化为 Evidence BM25 + Evidence Dense 两路                   |
| RRF 直接使用各路排名                    | 增加了 lexical_tier 加分和 term_coverage 加分机制            |

---

## 11. 当前实现说明与后续优化方向

### 当前已实现功能

- ✅ HTML 格式 `.doc` 简历解析，提取 12+ 类结构化字段
- ✅ 双索引架构：候选人主索引 + 证据切片索引
- ✅ 证据切片级别的向量化（非整文档向量化）
- ✅ 中文 IK 分词 + 多层权重 BM25 检索
- ✅ kNN 向量近邻检索
- ✅ 手动 RRF 融合排序（含 tier 加分 + coverage 加分）
- ✅ 自由文本中的结构化约束自动识别（年限、城市、学历）
- ✅ Facet 聚合驱动的动态筛选面板
- ✅ 完整的检索排序可解释性 Debug 面板
- ✅ 检索质量评估框架 (P@K / R@K / MRR / NDCG)
- ✅ 单元测试覆盖

### 已知设计边界与可优化方向

**数据处理层面**：

1. **简历解析仅支持 HTML 格式 `.doc`**——不支持 PDF、`.docx`、纯文本等格式。如需支持更多格式，可引入 Apache Tika 或 python-docx。
2. **工作经验仅从实习经历计算**——项目经历不纳入工时统计。如需更精确的经验估算，可考虑加权不同类型的经历。
3. **模型下载依赖网络**——生产环境应预先下载模型到本地或内网镜像。

**检索层面**：

4. **Query 解析依赖词表精确匹配**——城市、学历的识别依赖 ES 中已有的词表。对于新城市名或学历别名（如"研究生"→"硕士"），需要手动扩充。
5. **无 Query Rewriting 或同义词扩展**——"ML"不会被扩展为"Machine Learning"/"机器学习"。可引入同义词词典或 LLM 辅助改写。
6. **无 Re-ranking 阶段**——当前是 RRF 一次排序即最终结果。可在 RRF 之后增加一个 Cross-Encoder 精排阶段来提升头部排序质量。
7. **filter 子句在证据索引上基于冗余字段**——如果冗余字段与主索引不一致（理论上不应发生，但增量更新时需注意），可能导致过滤遗漏。

**工程层面**：

8. **ES 地址硬编码**——`ES_URL` 在代码中写死为 `localhost:9200`，生产部署需通过环境变量或配置文件管理。
9. **无认证机制**——API 和 ES 连接均无认证，仅适用于内网开发环境。
10. **embedding 推理在应用进程内**——模型加载占用约 4GB 内存。高并发场景应将 embedding 推理拆分为独立服务（如 ONNX Runtime Serving 或 Triton）。
11. **Facet 和词表使用内存缓存**——多实例部署时各实例缓存不一致。可改用 Redis 或直接每次查询（ES 聚合查询通常很快）。
12. **前端无分页**——默认返回最多 10,000 条结果一次性渲染。数据量大时需引入虚拟滚动或分页加载。
