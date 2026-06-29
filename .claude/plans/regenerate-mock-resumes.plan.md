# 重新生成 200 份贴合真实字段的模拟简历

## 背景与目标

当前 `data/ai_generated.jsonl`(100 份,`parser_version: llm-diverse-v2`)与两份真实简历(`html-doc-v1`)字段结构有明显出入:
- mock 缺 `languages` / `awards` / `offer_internship` / `it_skill_items`,全为 null
- mock 把 `candidate.years_experience` 写死(如 7.0);真实简历**没有**这个字段,由 `import_to_es._estimate_years_experience` 从实习经历估算
- mock 的 `education` 条目缺 `research_direction` / `lab_name` / `paper_level` / `college`(这些喂给 education 证据切片)
- mock 的 `section_text` 缺 `languages` / `awards` / `offer_internship` 等键

后果:`app._rerank_document` 实际读取 `languages/awards/offer_internship`,但这些信号在当前评测里从未被测到。

**目标**:生成 200 份字段对齐真实结构、岗位贴合奇安信安全业务、含一批 hard negative 的模拟简历,替换 `data/ai_generated.jsonl`。

## 已确认的决策

1. **字段对齐**:对齐"检索相关字段"——补全 `languages` / `awards` / `offer_internship` / `it_skill_items` / 完整 `education` 子字段;`years_experience` 不写死(交给 import 估算)。省略 `ethnicity/identity_info/political_status` 等纯档案噪声字段(真实简历有,但不参与检索且含敏感信息;用占位即可,见下)。
2. **岗位主题**:贴合奇安信真实安全业务岗位(以两份真实简历的"机器学习工程师/测试工程师"+ 奇安信安全产品线为主线)。
3. **hard negative**:200 份中专门安排一批"沾边但不对"的简历。
4. **老评测集**:全部重生,`eval_queries.jsonl` 的 qrels 会失效,**本任务不重标**(后续单独任务)。

## 字段 Schema(以真实 `html-doc-v1` 为准)

每份记录的结构(顶层键与真实解析输出一致):
```
resume_id, parse_status="ok", parse_errors=[], parser_version="mock-realistic-v1",
file{ name, sha256, detected_type, encoding },   # 占位元数据
application{ candidate_no, apply_time, company="奇安信集团", position_code, position_name, wishes[], expected_work_cities[] },
candidate{ name, gender, birth_date, current_city, highest_degree, graduation_date,
           school, major, phone, email,
           # 不含 years_experience(交给 import 估算);
           # ethnicity/nationality/political_status 等用统一占位或省略 },
education[]{ start_date, end_date, school, college, major, education_level, degree,
            research_direction, lab_name, paper_level, is_current },
internships[]{ start_date, end_date, company, company_type, department, title, work_type, description, is_current },
projects[]{ start_date, end_date, name, description, responsibility, is_current },
skills[],                       # 技能标签列表(大小写规范化)
it_skill_items[]{ skill_name, duration, proficiency, primary_languages, other_languages },
languages{ english_exam_score, english_spoken_level },
awards[]{ has_award, name, level, description },
offer_internship{ post_graduation_intention, can_intern, available_start_date, weekly_workdays, internship_period },
section_text{ personal_info, education, internships, projects, it_skills, languages, awards, offer_internship, expected_work_city }
```

关键正确性约束(对齐 import_to_es 消费逻辑):
- `years_experience` **不写入** candidate;让 import 从 internships 时间跨度估算(部分简历无实习→无年限,这是真实分布)
- 实体清洗:项目/实习 description 里**不要**塞公司名/学校名/城市(import 的 `_semantic_text` 会清洗实体,但生成侧也应避免,保持语义纯净)
- `skills` 用规范化标签(与 `CANONICAL_SKILL_LABELS` 一致:Python/C++/MySQL/PyTorch...)
- ID 方案:沿用 `M2026XXXX`(`M20260001`–`M20260200`),保持与现有前端/工具兼容

## 岗位族设计(200 份分布)

以奇安信安全业务为主线,覆盖 eval 现有 9 类主题的技能面,便于后续重标:

| 岗位族 | 份数 | 技能/语义主题 |
|---|---|---|
| 安全研究(漏洞挖掘/Fuzzing/逆向/0day) | ~20 | 二进制、IDA、内核、符号执行 |
| 蓝队/应急响应/SOC | ~18 | 日志取证、ATT&CK、威胁狩猎、SIEM |
| 红队/渗透测试 | ~18 | AD 域、内网横向、C2、Web 渗透 |
| DevSecOps/安全平台 | ~15 | SAST/SCA/IAST、CI/CD、容器安全 |
| 机器学习/LLM/RAG(对齐真实简历①) | ~22 | RAG、向量检索、PyTorch、NLP、大模型 |
| 后端 Go/云原生 | ~18 | K8s、gRPC、微服务、Operator |
| 后端 Java/架构 | ~18 | Spring、分布式事务、JVM、高并发 |
| 后端 C++/低延迟 | ~14 | RDMA、无锁、高频、内核旁路 |
| 前端/可视化 | ~16 | Vue3/React、WebGL、Three.js、大屏 |
| 数据分析/增长 | ~15 | SQL、A/B 实验、指标体系、归因 |
| 测试工程(对齐真实简历②) | ~12 | 自动化测试、性能测试、MATLAB |
| 产品/TPM | ~10 | 安全产品、SASE/ZTNA、需求 |
| **hard negative** | ~24 | 见下 |

(份数为目标值,生成时按比例,总和=200)

### hard negative 设计(~24 份,信息量最高)

针对现有评测主题造"沾边但不对"的简历,用于考验精排和 `forbidden@10`:
- "RAG 工程师"族的 hard neg:**传统搜索/ES 运维**背景,简历含"检索""索引""向量数据库运维"但**没碰过大模型/LLM**
- "蓝队应急"族的 hard neg:**普通运维/网管**,含"日志""监控"但无取证/威胁狩猎
- "C++ 低延迟"族的 hard neg:**普通 C++ 业务开发**,含 C++ 但无 RDMA/高频/无锁
- "数据分析"族的 hard neg:**BI 报表开发**,含 SQL 但无 A/B 实验/因果推断
- 库外负例对应人才(量子计算/SAP/iOS/医学影像等)**不生成**,保持其为真负例

## 实现方式

### 选项 A:确定性 Python 生成器(推荐)
写 `generate_mock_resumes.py`:
- 按岗位族定义 `POSITION_PROFILES`(部门、研究方向、实验室、项目模板、实习模板、技能池、典型院校、城市、学位分布)
- 模板化采样组合,保证同族内有差异、跨族有区分度
- 确定性(固定 random seed)→ 可复现,便于后续重标 qrels
- 直接输出符合上述 schema 的 JSONL
- **不依赖 LLM**:避免"LLM 生成 + LLM 评测"的同源偏差(与之前讨论的评测集可信度原则一致)

### 选项 B:LLM 生成
调 DeepSeek 批量生成。更自然但:① 成本/耗时;② 同源偏差;③ 不可复现。**不推荐**。

→ 采用 **选项 A**。

## 落地步骤

1. **写 `generate_mock_resumes.py`**:岗位族 profile 表 + 采样逻辑 + schema 组装 + hard negative 注入,固定 seed。
2. **生成并自检**:输出到 `data/ai_generated.jsonl`(覆盖前先备份为 `data/ai_generated.legacy100.jsonl`)。自检:200 条、ID 唯一、`parse_status=ok`、字段完整性(languages/awards/offer 非空率符合真实分布)、技能标签规范化。
3. **用 resume_parser/import 的契约验证**:跑一个轻量校验脚本,确认 `_enrich_doc` / `_resume_evidence_docs` 能正常消费(years_experience 估算、证据切片生成、skills_text)。不实际写 ES,只验证不抛异常 + 切片数合理。
4. **更新单元测试**(如有对数据格式的断言):检查 `tests/` 是否有依赖旧 100 份 mock 字段的测试,按需调整。
5. **不重标 eval_queries**(本任务范围外),但在 `PROJECT_REVIEW.md` 记一条:数据集已换 `mock-realistic-v1`,qrels 待重标。

## 交付物

- `generate_mock_resumes.py`(确定性生成器,可复现)
- `data/ai_generated.jsonl`(200 份,`mock-realistic-v1`)
- `data/ai_generated.legacy100.jsonl`(旧 100 份备份)
- 生成自检输出(份数/分布/字段覆盖率打印)

## 不做的事

- 不重新导入 ES(用户后续自行 `python import_to_es.py data/ai_generated.jsonl`)
- 不重标 `eval_queries.jsonl`(后续单独任务)
- 不生成库外负例对应的真实简历(保持真负例)
- 不写入敏感真实 PII(身份证/手机用占位生成)
