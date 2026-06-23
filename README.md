# 简历检索系统

这是一个基于 FastAPI 和 Elasticsearch 的简历检索原型，用于验证简历场景下的关键词检索、连续短语匹配、结构化过滤、向量检索和 RRF 混合排序。

本文重点说明检索字段如何分工，以及 exact keyword、phrase、BM25、filter、dense vector、应用层手写 RRF 在完整 use case 中分别解决什么问题。开发过程中的踩坑、错误决策和复盘见 [PROJECT_REVIEW.md](./PROJECT_REVIEW.md)。

## 检索设计原则

简历检索里有两类完全不同的需求：

- 精确匹配：用户搜索某个学校、公司、姓名、城市、岗位编号、技能标签。
- 语义匹配：用户描述一个能力、项目经验、实习职责、研究方向。

因此字段不能全部塞进向量，也不能全部依赖 BM25。

核心原则：

```text
keyword / phrase / BM25 检索“字段上写了什么”。
向量检索“这个人做过什么、能力像什么”。
RRF 把精确匹配和语义匹配结合起来。
```

当前检索架构：

```text
用户查询
  ├─ 轻量结构化解析
  │    └─ 从自然语言中识别明确年限、城市、学历、技能筛选
  ├─ BM25 / keyword / phrase / filter 检索
  │    └─ 负责实体、连续短语、关键词、结构化条件，并统计多词覆盖度
  ├─ dense vector 检索
  │    └─ 负责能力、职责、项目、经历语义
  └─ RRF 融合排序
       └─ 先按 exact/phrase 词面证据分层，再融合 BM25 / dense 名次
```

当前实现有几个边界：

- 当前只有一个主语义向量字段：`semantic_profile_vector`。表格里的“是否进入向量”表示字段内容是否会参与这个语义 profile 的文本拼接，不表示每个字段都有独立向量。
- 当前代码只支持最新 mapping，不兼容缺少 `semantic_profile_vector`、`.keyword` 或 `.phrase` 子字段的旧索引。mapping 变化后必须重建新索引。
- RRF 在应用层手写实现：后端并发请求 BM25 和 kNN，再按 `weight / (rank_constant + rank)` 合并名次，不依赖 Elasticsearch 内置 `retriever.rrf`。
- 候选人编号、岗位编号、手机号、邮箱、学校、公司等精确查询优先走 keyword/BM25，不触发 dense vector。
- 关键 text 字段同时保留 `ik_max_word` 的宽召回字段和 `ik_smart` 的 `.phrase` 子字段。连续短语查询走 `.phrase`，避免 `ik_max_word` 的重叠 token 破坏 `match_phrase` 位置匹配。
- `自然语言处理`、`深度学习`、`推荐召回`、`模型落地` 这类短能力表达会触发 dense vector，不再要求必须是长句。
- 前端技能筛选是 AND 语义，选择 `Python` 和 `NLP` 表示候选人必须同时具备两个技能。
- 用户直接输入 `0.5年以上 北京 本科 推荐系统` 时，会基于索引里的城市、学历、技能词表和明确年限模式解析成 filter + 剩余 query；普通 `推荐系统 NLP SQL` 仍作为宽召回文本查询，不强行拆成多个硬过滤。
- 多词文本查询会优先排序“覆盖更多查询词”的候选人。例如搜索 `A B` 时，同时命中 `A` 和 `B` 的候选人会排在只高频命中 `A` 或只高频命中 `B` 的候选人前面。
- 默认搜索不再只返回 20 条；未显式传 `limit` 时，空查询、筛选浏览和关键词混合检索都会尽量返回当前 ES 结果窗口内的全部候选人。当前窗口由 `MAX_BROWSE_RESULT_SIZE` 控制，默认是 10000。

## 检索字段分工

下表按当前简历结构和 Elasticsearch mapping 组织。当前未索引的字段不代表业务上永远不能搜索，只表示当前 mapping 尚未把它纳入可检索字段。后续如果有业务需求，可以补 mapping、导入逻辑和查询逻辑。“是否进入向量”指是否进入 `semantic_profile_vector` 的抽取式语义文本。

| 模块 | 字段 | 检索方式 | 是否进入向量 | 典型用途 |
| --- | --- | --- | --- | --- |
| 文档标识 | `resume_id` | keyword 精确查 | 否 | 文档详情、更新、删除 |
| 文件信息 | `file.path` | keyword / 管理字段 | 否 | 排查来源文件 |
| 文件信息 | `file.name` | keyword / 管理字段 | 否 | 排查来源文件 |
| 文件信息 | `file.sha256` | keyword / 管理字段 | 否 | 文件去重、导入校验 |
| 文件信息 | `file.size` | long | 否 | 导入校验 |
| 文件信息 | `file.mtime` | date | 否 | 导入校验、排序辅助 |
| 文件信息 | `file.detected_type` | keyword | 否 | 解析来源类型 |
| 文件信息 | `file.encoding` | keyword | 否 | 编码排查 |
| 投递信息 | `application.candidate_no` | keyword 精确查 | 否 | 候选人编号查询 |
| 投递信息 | `application.apply_time` | date 排序/过滤 | 否 | 按投递时间排序 |
| 投递信息 | `application.company` | keyword 精确查 | 否 | 投递公司过滤 |
| 投递信息 | `application.position_code` | keyword 精确查 | 否 | 岗位编号查询 |
| 投递信息 | `application.position_name` | keyword + phrase + BM25 | 是，末尾补充 | 岗位名称搜索、岗位语义 |
| 投递信息 | `application.expected_work_cities` | keyword filter | 否 | 期望城市筛选 |
| 投递志愿 | `application.wishes.rank` | integer | 否 | 志愿排序 |
| 投递志愿 | `application.wishes.position_name` | nested phrase + BM25 | 否 | 志愿岗位搜索 |
| 投递志愿 | `application.wishes.company` | nested keyword | 否 | 志愿公司过滤 |
| 候选人 | `candidate.name` | keyword + BM25 | 否 | 姓名精确查询 |
| 候选人 | `candidate.gender` | keyword filter | 否 | 性别筛选 |
| 候选人 | `candidate.birth_date` | date filter | 否 | 年龄筛选 |
| 候选人 | `candidate.current_city` | keyword filter | 否 | 当前城市筛选 |
| 候选人 | `candidate.highest_degree` | keyword filter | 否 | 最高学历筛选 |
| 候选人 | `candidate.graduation_date` | date filter | 否 | 毕业时间筛选 |
| 候选人 | `candidate.school` | keyword + phrase + BM25 | 否 | 毕业院校精确查询 |
| 候选人 | `candidate.major` | keyword + phrase + BM25 | 是，末尾补充 | 专业搜索、专业语义 |
| 候选人 | `candidate.phone` | keyword 精确查 | 否 | 联系方式查询 |
| 候选人 | `candidate.email` | keyword 精确查 | 否 | 联系方式查询 |
| 候选人 | `candidate.years_experience` | range filter | 否 | 工作/实习年限筛选 |
| 教育经历 | `education.start_date` | nested date | 否 | 教育时间过滤 |
| 教育经历 | `education.end_date` | nested date | 否 | 教育时间过滤 |
| 教育经历 | `education.school` | nested keyword + phrase + BM25 | 否 | 学校精确查询 |
| 教育经历 | `education.college` | nested phrase + BM25 | 否 | 学院查询 |
| 教育经历 | `education.major` | nested keyword + phrase + BM25 | 是 | 专业语义 |
| 教育经历 | `education.education_level` | nested keyword/filter | 否 | 本科、硕士等背景 |
| 教育经历 | `education.degree` | nested keyword/filter | 否 | 学士、硕士等背景 |
| 教育经历 | `education.research_direction` | nested phrase + BM25 | 是 | 研究方向语义 |
| 教育经历 | `education.lab_name` | nested phrase + BM25 | 是，但清理学校/学院实体 | 实验室方向语义 |
| 教育经历 | `education.paper_level` | keyword/filter | 否 | 科研背景参考 |
| 实习经历 | `internships.company` | nested keyword + phrase + BM25 | 否 | 实习公司精确查询 |
| 实习经历 | `internships.department` | nested phrase + BM25 | 是 | 部门方向语义 |
| 实习经历 | `internships.title` | nested phrase + BM25 | 是 | 实习职位语义 |
| 实习经历 | `internships.work_type` | keyword/filter | 否 | 实习性质 |
| 实习经历 | `internships.description` | nested phrase + BM25 | 是 | 工作内容和职责语义 |
| 项目经历 | `projects.name` | nested keyword + phrase + BM25 | 是 | 项目名称、项目主题 |
| 项目经历 | `projects.description` | nested phrase + BM25 | 是 | 项目背景和业务场景 |
| 项目经历 | `projects.responsibility` | nested phrase + BM25 | 是 | 项目职责和能力语义 |
| 技能 | `skills` | keyword 精确查/filter | 是 | 技能标签精确过滤和语义 |
| 技能 | `skills_text` | BM25 | 是 | 多技能组合检索 |
| 语言能力 | `languages.english_exam_score` | keyword/filter | 否 | 英语等级筛选 |
| 语言能力 | `languages.english_spoken_level` | keyword/filter | 否 | 英语口语筛选 |
| 分段文本 | `section_text.education` | phrase + BM25 / highlight | 否 | 教育片段高亮 |
| 分段文本 | `section_text.internships` | phrase + BM25 / highlight | 否 | 实习片段高亮 |
| 分段文本 | `section_text.projects` | phrase + BM25 / highlight | 否 | 项目片段高亮 |
| 原始文本 | `raw_text` | 不索引，仅详情/调试 | 否 | 调试和详情回溯 |

## 完整候选人样例

下面是一份结构化候选人样例。它覆盖投递、候选人、教育、实习、项目、技能和语言能力。

```yaml
resume_id: M20260001

application:
  candidate_no: M20260001
  apply_time: 2026-03-12
  company: 奇安信集团
  position_code: A0009
  position_name: 机器学习工程师
  expected_work_cities:
    - 北京
    - 上海
  wishes:
    - rank: 1
      position_name: 机器学习工程师
      company: 奇安信集团

candidate:
  name: 孔泽宇
  gender: 男
  birth_date: 2001-04-16
  current_city: 杭州
  highest_degree: 本科
  graduation_date: 2026-06-25
  school: 北京交通大学
  major: 网络空间安全
  phone: 138xxxxxxx
  email: mock001@example.com
  years_experience: 0.9

education:
  - school: 北京交通大学
    college: 计算机与信息技术学院
    major: 网络空间安全
    education_level: 本科
    degree: 学士
    research_direction: 数据挖掘
    lab_name: 认知计算实验室
    paper_level: EI

internships:
  - company: 百度在线网络技术
    department: 智能安全实验室
    title: 机器学习实习生
    work_type: 实习
    description: 在智能安全实验室担任机器学习实习生，负责将离线模型封装为批处理推理服务；使用推荐系统、NLP、SQL 完成实现和验证。

projects:
  - name: 弱监督风险样本挖掘平台
    description: 使用规则标注和主动学习扩充训练集，降低人工标注成本。
    responsibility: 清洗告警语料并训练文本分类模型，分析误召回样本。

skills:
  - 推荐系统
  - NLP
  - SQL
  - 机器学习
  - Linux

languages:
  english_exam_score: "CET 6: 520"
  english_spoken_level: 可技术面试
```

## BM25 / keyword / phrase / filter Use Case

### 查询：`M20260001`

这是候选人编号查询，应走 `application.candidate_no` 的 keyword 精确匹配，不需要向量参与。

### 查询：`A0009`

这是岗位编号查询，应走 `application.position_code` 的 keyword 精确匹配。岗位编号是结构化实体，不应该靠 BM25 分词或向量相似度猜。

### 查询：`北京交通大学`

这是学校实体查询。它应该主要依赖 keyword、phrase 和 nested BM25。

命中字段：

```text
candidate.school: 北京交通大学
education.school: 北京交通大学
section_text.education: 学校: 北京交通大学
```

检索方式：

```text
candidate.school.keyword 精确命中
candidate.school.phrase 连续短语命中
nested education.school.keyword 精确命中
nested education.school.phrase 连续短语命中
section_text.education BM25 命中
```

这种查询不应该主要靠向量判断，因为用户要找的是“这个学校”本身。

### 查询：`计算机科学`

这是连续短语查询，不应该只按 `计算机` 和 `科学` 两个拆开的词排序。系统把 lexical 证据分为四层：

```text
1. exact field：字段整体等于 query，例如 candidate.major.keyword = 计算机科学与技术
2. phrase：query 作为连续短语出现，例如 candidate.major.phrase 命中 计算机科学
3. all terms：query 分词后的所有词都命中同一个字段
4. broad terms：普通 BM25 宽召回
```

其中 `.phrase` 子字段使用 `ik_smart` 建索引，专门服务 `match_phrase`。主字段仍使用 `ik_max_word`，保证普通 BM25 召回足够宽。

因此搜索 `计算机科学` 时，`计算机科学与技术` 会优先于只分散命中 `计算机` 和 `科学` 的候选人；搜索 `计算机科学与技术` 时，`candidate.major.keyword` 或 `education.major.keyword` 完整命中的候选人会进入最高 exact tier。

### 查询：`百度在线网络技术`

这是公司实体查询。应优先匹配实习公司；如果查询的是投递公司，比如 `奇安信集团`，则匹配 `application.company` 和 `application.wishes.company`。

命中字段：

```text
internships.company: 百度在线网络技术
section_text.internships: 企业名称: 百度在线网络技术
```

检索方式：

```text
application.company keyword 精确命中
application.wishes.company nested keyword 精确命中
nested internships.company.keyword 精确命中
nested internships.company.phrase 连续短语命中
section_text.internships BM25 命中
```

公司名不进入向量。原因是公司名是实体，不是能力语义。如果放进向量，搜索公司名时可能召回语义相近但实体不匹配的候选人。

### 查询：`机器学习工程师`

这是岗位查询，BM25 和向量都可以参与。

BM25 命中字段：

```text
application.position_name: 机器学习工程师
application.wishes.position_name: 机器学习工程师
internships.title: 机器学习实习生
section_text.internships: 机器学习实习生
```

向量语义来源：

```text
实习职位：机器学习实习生
实习描述：担任机器学习实习生，负责离线模型批处理推理服务
目标岗位：机器学习工程师
```

这个查询既有明确岗位词，也有岗位语义，因此混合检索可以发挥作用。

### 查询：`推荐系统 NLP SQL`

这是技能和能力组合查询。

BM25 命中字段：

```text
skills: 推荐系统
skills: NLP
skills: SQL
skills_text: 推荐系统 NLP SQL
internships.description: 使用推荐系统、NLP、SQL 完成实现和验证
```

向量语义来源：

```text
项目职责：使用推荐系统、NLP、SQL、机器学习完成方案实现、联调和评估
实习描述：使用推荐系统、NLP、SQL 完成实现和验证
能力标签：推荐系统，NLP，SQL，机器学习，Linux
```

这种查询是混合检索的典型场景。BM25 能捕捉明确技能词，向量能理解技能组合和职责场景。

### 查询：`0.5年以上 北京 本科 推荐系统`

这是全文查询加结构化过滤。

系统会拆成：

```text
query: 推荐系统

filters:
  candidate.years_experience >= 0.5
  application.expected_work_cities contains 北京
  candidate.highest_degree = 本科
  skills must contain 推荐系统
```

执行方式：

```text
BM25 在过滤后的候选人里查 推荐系统
dense vector 在过滤后的候选人里查 推荐系统 相关语义
RRF 融合两路结果
```

城市、学历、年限、技能标签作为 filter 生效，不应该被拼进向量里靠相似度判断。

注意：硬过滤会提高精确性，但可能返回 0 条。比如当前 mock 数据中如果没有候选人同时满足 `0.5 年以上 + 北京 + 本科 + 推荐系统`，结果就是 0，这比用向量强行补满结果页更符合筛选语义。

## 向量检索 Use Case

当前方案使用一个主语义向量：

```text
semantic_profile_vector
```

这个向量不是整份简历全文，也不是 LLM 生成摘要，而是抽取式语义 profile。

### `semantic_profile_vector` 输入示例

```text
项目名称：弱监督风险样本挖掘平台
项目描述：使用规则标注和主动学习扩充训练集，降低人工标注成本。
项目职责：担任机器学习工程师相关角色，负责清洗告警语料并训练文本分类模型；使用推荐系统、NLP、SQL、机器学习完成方案实现、联调和评估，Top10 召回率提升 12%。
项目名称：简历技能画像抽取模型
项目描述：从教育、项目和实习文本中抽取技能实体，并生成候选人能力标签。
项目职责：担任机器学习工程师相关角色，负责设计特征统计报表并分析误召回样本；使用推荐系统、NLP、SQL、机器学习完成方案实现、联调和评估，人工复核样本量下降 22%。
实习部门：智能安全实验室
实习职位：机器学习实习生
实习描述：在智能安全实验室担任机器学习实习生，主要负责将离线模型封装为批处理推理服务；使用推荐系统、NLP、SQL完成实现和验证，重复告警合并率提升 18%。
能力标签：推荐系统，NLP，SQL，机器学习，Linux
教育专业：网络空间安全
研究方向：数据挖掘
实验室方向：认知计算实验室
目标岗位：机器学习工程师
专业背景：网络空间安全
```

### 明确不进入向量的字段

```text
孔泽宇
北京交通大学
北京
上海
杭州
百度在线网络技术
138xxxxxxx
mock001@example.com
M20260001
A0009
```

这些字段仍然可以被搜索，只是它们应该走 BM25、keyword、filter 或 nested phrase，而不是进入 dense vector。

### 查询：`做过推荐系统召回和 NLP 模型落地的人`

BM25 可能命中：

```text
skills: 推荐系统
skills: NLP
internships.description: 推荐系统、NLP
```

但 BM25 对下面这些表达可能不够稳定：

```text
召回
模型落地
```

向量可以匹配：

```text
负责将离线模型封装为批处理推理服务
使用推荐系统、NLP、SQL 完成实现和验证
```

因此 dense vector 可以把这个候选人召回或排高。

### 查询：`有文本分类和样本挖掘项目经验`

BM25 命中：

```text
projects.name: 弱监督风险样本挖掘平台
projects.responsibility: 训练文本分类模型
```

向量匹配：

```text
弱监督风险样本挖掘平台
主动学习扩充训练集
清洗告警语料
训练文本分类模型
分析误召回样本
```

这种查询既有项目关键词，也有经验语义，BM25 和 dense 都有价值。

### 查询：`机器学习实习经历，做过离线模型服务`

BM25 命中：

```text
internships.title: 机器学习实习生
internships.description: 离线模型
```

向量匹配：

```text
机器学习实习生
负责将离线模型封装为批处理推理服务
```

这类自然语言查询更依赖向量补充表达差异。

## 混合检索 Use Case

用户输入：

```text
做过推荐系统召回和 NLP 模型落地的人
```

系统并行执行两路召回：

```text
BM25 retriever:
  查 application.candidate_no
  查 application.position_code
  查 application.company
  查 application.position_name.keyword / application.position_name.phrase / application.position_name
  查 application.wishes.position_name
  查 application.wishes.company
  查 candidate.name
  查 candidate.phone / candidate.email
  查 candidate.school.keyword / candidate.school.phrase / candidate.school
  查 candidate.major.keyword / candidate.major.phrase / candidate.major
  查 skills_text
  查 skills
  查 section_text.projects
  查 section_text.internships
  查 nested projects.name.keyword / projects.name.phrase / projects.name
  查 nested projects.description
  查 nested projects.responsibility
  查 nested internships.title
  查 nested internships.description
  查 nested education.major.keyword / education.major.phrase / education.major
  查 nested education.research_direction

Dense retriever:
  将 query 编码成向量
  查 semantic_profile_vector
```

假设两路结果如下：

```text
BM25 排名：
1. 候选人 A：skills 命中 推荐系统、NLP
2. 候选人 B：项目职责命中 NLP
3. 孔泽宇：skills 和实习描述命中 推荐系统、NLP

Dense 排名：
1. 孔泽宇：语义匹配 模型落地、推荐系统、NLP
2. 候选人 C：语义匹配 推荐模型
3. 候选人 A：技能匹配
```

RRF 融合后：

```text
孔泽宇：
  BM25 rank = 3
  Dense rank = 1
  两路都命中，综合排名上升

候选人 A：
  BM25 rank = 1
  Dense rank = 3
  两路都命中，也靠前

候选人 C：
  Dense rank = 2
  BM25 没明显命中
  只有 dense 相似度足够高时才进入结果
```

RRF 的价值在于它不直接比较 BM25 分数和向量相似度分数，而是融合排名：

```text
rrf_score = 1 / (rank_constant + bm25_rank)
          + 1 / (rank_constant + dense_rank)
```

这样可以避免 BM25 分数和 cosine similarity 分数尺度不同导致的排序异常。

### 向量排名如何进入最终排序

向量检索的 `_score` 不会直接和 BM25 `_score` 相加。两者分数尺度不同，直接相加会让排序不可控。当前实现只使用向量结果的两个信息：

```text
1. dense _score:
   只用于 dense 内部排序，以及判断 BM25 没命中的 dense-only 候选是否足够可信。

2. dense rank:
   dense 候选通过高置信门槛后，用 rank 参与 RRF 融合。
```

完整链路如下：

```text
query
  ├─ BM25 retriever 返回按词面相关性排序的 top-k
  └─ Dense retriever 返回按 semantic_profile_vector 相似度排序的 top-k

对每个候选：
  如果同时出现在 BM25 和 Dense：
    最终 rrf_score = BM25 rank 贡献 + Dense rank 贡献

  如果只出现在 Dense：
    先判断 dense-only gate
    通过后才获得 Dense rank 的 RRF 贡献
```

RRF 贡献只和 rank 有关：

```text
单路贡献 = weight / (rank_constant + rank)

当前配置：
  BM25 weight = 1.0
  Dense weight = 1.0
  rank_constant = 60
```

例如：

```text
BM25 rank = 1  -> 1 / (60 + 1)  = 0.01639
Dense rank = 1 -> 1 / (60 + 1)  = 0.01639
Dense rank = 5 -> 1 / (60 + 5)  = 0.01538
```

如果候选人 A 同时被 BM25 和 Dense 找到：

```text
A:
  BM25 rank = 3
  Dense rank = 5
  rrf_score = 1 / 63 + 1 / 65 = 0.03126
```

如果候选人 B 只被 BM25 找到：

```text
B:
  BM25 rank = 1
  rrf_score = 1 / 61 = 0.01639
```

所以“两路都支持”的候选会明显上升。

对于 BM25 没命中的 dense-only 候选，不能只因为进入 dense top-k 就进入最终结果。当前门槛是：

```text
dense top1 _score 必须 >= 0.855
dense-only 候选 _score 必须 >= max(0.855, dense_top1_score - 0.02)
每次最多补 8 个 dense-only 候选
```

这意味着向量召回只补充“高置信、并且接近 dense top1”的候选，避免语义 top-k 自动填满结果页。当前 `0.855` 是用仓库内评估集校准得到的：它相比 `0.845` 去掉了 dense-only 负例误召回，同时比 `0.875` 保留更多语义召回。

最终排序时，系统先按强词面证据分层，再按多词覆盖度和 RRF 排序：

```text
1. lexical_tier 越大越靠前
   3 = exact field，例如 major.keyword 完整等于 query
   2 = phrase，例如 major.phrase 连续短语命中 query
   1 = 其他 BM25 named evidence
   0 = 没有强词面标记
2. term_coverage 越大越靠前
3. exact / phrase tier 内先按 BM25 rank 排，保留字段权重和 phrase 权重的作用
4. 再按 rrf_score 越大越靠前
5. rrf_score 相同，再按最好的单路 rank 越小越靠前
6. 仍然相同，用文档 id 做稳定排序
```

因此 dense 的作用是：

```text
1. 给 BM25 已命中的候选增加一份 Dense rank 的 RRF 加分
2. 在高置信时补充 BM25 没命中的语义候选
3. 不会压过 exact、phrase 或多词覆盖更完整的强词面候选
```

### 多词覆盖度优先排序

普通 BM25 会受词频、字段长度、字段权重影响。也就是说，搜索 `A B` 时，一个候选人如果只写了很多次 `A`，可能会比另一个同时写了 `A` 和 `B`、但词频较低的候选人分数更高。简历检索里这通常不是用户想要的结果：多词查询更像是在表达多个条件或多个能力点，优先级应该是“覆盖了几个查询词”，然后才是“每个词出现得多不多”。

当前实现为多词查询增加了一层覆盖度排序：

```text
query: A B

query tokens:
  A
  B

BM25 侧额外加入 named constant_score 查询，用于打 `matched_queries` 标记：
  query_term:0 -> 判断候选人是否命中 A
  query_term:1 -> 判断候选人是否命中 B

ES 返回每个 hit 的 matched_queries:
  A B 0 0   -> ["query_term:0", "query_term:1"] -> term_coverage = 2
  A A A 0   -> ["query_term:0"]                 -> term_coverage = 1
  0 B B B   -> ["query_term:1"]                 -> term_coverage = 1
  0 0 0 0   -> []                               -> term_coverage = 0，如果没有其他召回证据通常不会进入结果
```

覆盖度查询不负责主召回，也不应该用高 boost 改写 BM25 原始排序。当前实现先要求候选人命中正常的 keyword、phrase、nested 或 multi_match 查询；coverage query 只在这些候选上统计命中了几个 query token。只命中一个词的候选人仍然可以返回，只是会排在同时命中多个词的候选人后面。最终排序规则是：

```text
1. exact / phrase 词面证据优先
2. term_coverage 越大越靠前
3. exact / phrase tier 内先按 BM25 rank 排
4. 再按 rrf_score 越大越靠前
5. rrf_score 相同，再按最好的单路 rank 越小越靠前
6. 仍然相同，用文档 id 做稳定排序
```

因此搜索 `A B` 时，期望顺序是：

```text
A B 0 0
A A A 0
0 B B B
0 0 0 0
```

实现细节：

- 只有有效 query token 至少 2 个时才启用覆盖度排序，单词查询不受影响。
- query token 会去重，并最多取前 8 个，避免用户输入很长时生成过大的 Elasticsearch DSL。
- 覆盖度查询会覆盖主要可检索字段，包括编号、姓名、学校、岗位、技能、分段文本，以及 education / internships / projects 等 nested 字段。
- coverage query 的 boost 很小，只用于保留 `matched_queries` 标记，避免它自己把弱相关文档召回或压过 BM25 字段权重。
- dense-only 候选没有 BM25 `matched_queries`，所以 `term_coverage = 0`。它们仍可在高置信语义匹配时进入结果，但不会压过有明确多词词面证据的候选。

实际实现中，dense retriever 虽然会向 ES 请求 top-k 最近邻，但这些最近邻不会无条件进入最终结果。规则是：

```text
BM25 命中的候选：
  可以获得 dense rank 的 RRF 加分

BM25 没命中的 dense-only 候选：
  只有当 dense _score 达到高置信门槛，并且接近 dense top1 时才保留
  同时限制 dense-only 候选数量，避免向量 top-k 自动把结果页填满
```

因此，有关键词搜索的最终返回数量可能小于 ES 中的业务命中总数。未显式传 `limit` 时，接口会使用 `MAX_BROWSE_RESULT_SIZE` 作为结果窗口，当前默认是 10000；如果调用方传了 `limit=20`，才会只返回 20 条。

接口返回数量字段含义：

```text
returned_count:
  当前响应实际返回的结果数

matched_total:
  BM25 / keyword / phrase / filter 侧的真实命中总数

candidate_total:
  参与 RRF 融合后被接受的候选数量

retrieval_warnings:
  embedding 生成或某一路 retriever 降级失败时的可观测信息
```

不要把 `candidate_total` 当作业务总命中数。dense vector 的内部 top-k 只是排序候选窗口，不等价于用户条件下的匹配总数。

## 反例：为什么 `北京大学` 不应误召回

用户输入：

```text
北京大学
```

BM25 / keyword 查询学校字段：

```text
candidate.school
education.school
section_text.education
```

如果候选人没有北京大学，BM25 不应该命中。

Dense 查询 `semantic_profile_vector`：

```text
semantic_profile_vector 不包含：
  北京交通大学
  北京
  上海
  杭州
  百度在线网络技术
```

所以 dense 不应该因为下面这些字段误召回：

```text
candidate.school: 北京交通大学
application.expected_work_cities: 北京
candidate.current_city: 杭州
internships.company: 百度在线网络技术
```

如果 `北京大学` 仍然召回了无关候选人，通常说明：

```text
1. 实体字段又混进了向量文本
2. 向量输入过长，重要信息被截断
3. dense 权重过高
4. mock 数据太同质化
5. 短实体 query 被 dense 过度泛化
```

## 常见问题

### embedding 和索引版本如何保持一致？

索引 mapping 的 `_meta` 会记录当前 embedding 契约：

```yaml
embedding_model_id: IEITYuan/Yuan-embedding-2.0-zh
embedding_vector_dims: 1792
embedding_normalized: true
semantic_profile_version: semantic-profile-v2
```

每条文档也会写入 `embedding.model_id`、`embedding.vector_dims`、`embedding.normalized` 和 `embedding.semantic_profile_version`，用于排查“查询向量和文档向量是否同源”。

下面这些变更都需要重建索引并重新生成向量：

```text
embedding 模型或 pooling / dense 层变化
VECTOR_DIMS 变化
semantic_profile_vector 输入字段或顺序变化
dense_vector mapping / HNSW 参数变化
analyzer、keyword 或 phrase 子字段 mapping 变化
```

重建索引时使用导入脚本。`data_path` 可以是单个 HTML `.doc` 简历文件，也可以是简历目录：

```bash
python import_to_es.py data_path --es-url http://localhost:9200 --index resumes_v1 --alias resumes_current
```

如果是增量导入到当前 alias：

```bash
python import_to_es.py data_path --no-recreate
```

### 为什么不把整份简历直接向量化？

因为 embedding 模型有 token 上限。当前模型的 `max_seq_length` 是 512，真实简历全文很容易超限。更稳妥的方式是构造受控长度的 `semantic_profile_vector` 输入：

```text
结构化字段抽取
+ 实体清洗
+ 字段标签
+ 长度控制
```

不要把 `raw_text`、完整 `section_text`、联系方式、学校、公司、城市等全部拼进去。

### 为什么不做 LLM 总结后再向量化？

当前阶段不建议默认做生成式总结。原因是 LLM 总结可能改写原意或引入原文没有的信息。

更可靠的方式是抽取式 profile：

```text
可以加字段标签
可以删除实体噪声
可以截断
但不要改写成原文没有的表达
```

### 未索引字段是不是不能搜？

当前不能直接搜，不代表业务上永远不能搜。比如民族、政治面貌、招聘来源、奖励经历、offer 后实习信息等字段，如果业务需要，可以后续加入 mapping、导入和查询 DSL。

## 本地验证命令

运行单元测试：

```bash
python -m pytest -q
```

查看 ES 健康状态和 alias：

```bash
curl -s 'http://localhost:9200/_cluster/health?pretty'
curl -s 'http://localhost:9200/_cat/aliases/resumes_current?h=alias,index,is_write_index'
curl -s 'http://localhost:9200/resumes_current/_count?pretty'
```

验证当前 mapping 里的 embedding 契约和向量字段：

```bash
python - <<'PY'
import requests

mapping = requests.get(
    'http://localhost:9200/resumes_current/_mapping',
    timeout=10,
).json()

for index, body in mapping.items():
    props = body['mappings']['properties']
    meta = body['mappings'].get('_meta', {})
    vector_fields = [
        name
        for name, field in props.items()
        if isinstance(field, dict) and field.get('type') == 'dense_vector'
    ]
    print(index)
    print(meta)
    print(vector_fields)
PY
```

验证关键 exact / phrase 子字段：

```bash
python - <<'PY'
import requests

mapping = requests.get(
    'http://localhost:9200/resumes_current/_mapping',
    timeout=10,
).json()
props = next(iter(mapping.values()))['mappings']['properties']

checks = {
    'candidate.major': props['candidate']['properties']['major'],
    'education.major': props['education']['properties']['major'],
    'projects.name': props['projects']['properties']['name'],
}

for field, spec in checks.items():
    print(field, spec.get('fields'))
PY
```

验证主向量覆盖率：

```bash
python - <<'PY'
import requests

for field in ['semantic_profile_vector']:
    result = requests.post(
        'http://localhost:9200/resumes_current/_count',
        json={'query': {'exists': {'field': field}}},
        timeout=10,
    ).json()
    print(field, result['count'])
PY
```

验证 IK analyzer：

```bash
python - <<'PY'
import requests

text = '自然语言处理 推荐系统 北京大学'
for analyzer in ['resume_text', 'resume_search']:
    result = requests.post(
        'http://localhost:9200/resumes_current/_analyze',
        json={'analyzer': analyzer, 'text': text},
        timeout=10,
    ).json()
    print(analyzer, [item['token'] for item in result['tokens']])
PY
```

验证接口级检索行为：

```bash
python - <<'PY'
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

queries = [
    '北京大学',
    '北京交通大学',
    'A0009',
    '计算机科学',
    '计算机科学与技术',
    '自然语言处理',
    '做过推荐系统召回和 NLP 模型落地的人',
    '0.5年以上 北京 本科 推荐系统',
]

for query in queries:
    data = client.get('/api/search', params={'q': query, 'limit': 5}).json()
    print(query, {
        'effective_query': data['effective_query'],
        'matched_total': data['matched_total'],
        'candidate_total': data['candidate_total'],
        'returned_count': data['returned_count'],
        'parsed_constraints': data['parsed_constraints'],
        'warnings': data['retrieval_warnings'],
    })
    for item in data['results'][:3]:
        print(' ', item['id'], item['candidate'].get('name'), item['retrieval_debug'])
PY
```

## 检索效果评估

单元测试只能证明检索策略没有明显回归，不能证明线上排序真的有效。实际效果需要维护一份小型标注集，按查询类型覆盖实体查询、能力语义查询、组合过滤查询和负例查询。

仓库里已经提供了一份 126 条的评估集 [eval_queries.jsonl](./eval_queries.jsonl)，覆盖实体查询、专业 exact/phrase 查询、编号查询、结构化过滤、技能组合、岗位+技能、自然语言语义查询和负例查询。标注项支持两种方式：

- `relevant_ids` / `forbidden_ids`：人工指定相关或不应出现的简历 ID。
- `relevant_es_query` / `forbidden_es_query`：用 ES DSL 从结构化字段动态生成相关集合，适合学校、岗位、技能、学历、城市这类明确条件。

示例：

```json
{"id":"entity_school_bjtu","type":"entity","query":"北京交通大学","relevant_es_query":{"term":{"candidate.school.keyword":"北京交通大学"}}}
{"id":"negative_school_peking","type":"negative_entity","query":"北京大学","relevant_ids":[],"expect_empty":true}
{"id":"sem_recsys_nlp_model","type":"semantic","query":"做过推荐系统召回和 NLP 模型落地的人","relevant_ids":["M20260061","M20260121","M20260101","M20260221","M20260011"],"forbidden_ids":["M20260138"]}
```

运行评估并扫描 dense-only 阈值：

```bash
python evaluate_search.py
```

查看每条 query 的 Top10、命中 ID 和 forbidden ID：

```bash
python evaluate_search.py --details
```

按 query 类型查看指标：

```bash
python evaluate_search.py --type-summary
```

也可以指定阈值范围：

```bash
python evaluate_search.py --thresholds 0.84,0.845,0.855,0.86,0.875
```

每次调整字段 boost、`RRF_RANK_WINDOW_SIZE`、`KNN_NUM_CANDIDATES`、dense-only 阈值或 embedding profile 后，固定跑同一批 query，至少看这些指标：

```text
Recall@5 / Recall@10
Precision@5 / Precision@10
MRR@10
NDCG@10
负例误召回率
p50 / p95 latency
```

同时保存每条结果的 `lexical_tier`、`bm25_rank`、`dense_rank`、`dense_score`、`rrf_score`、`term_coverage`。这样可以区分问题来自 exact/phrase 证据缺失、BM25 召回不足、向量相似度误召回、RRF 融合异常，还是结构化 filter 过严。

当前这份扩展评估集的 dense-only 阈值扫描结论是：

```text
0.820-0.845: NDCG@10 最高，但负例 empty_acc 下降，dense-only 误召回偏多
0.855: empty_acc 达到 1.0，同时保持较高的 NDCG@10=0.928 和 Recall@10=0.492
0.875+: forbidden 命中更少，但开始明显损伤自然语言语义查询召回
```

所以当前默认值使用 `DENSE_ONLY_MIN_SCORE = 0.855`。

当前专业 exact/phrase 回归结果：

```text
entity_major: P@5=1.000, R@10=1.000, NDCG@10=1.000
phrase_major: P@5=1.000, R@10=0.900, NDCG@10=0.927
```

## 开发复盘

这份 README 说明的是检索字段分工和 use case。开发过程中遇到的详细问题、错误决策、排查过程和面试表达，见：

- [PROJECT_REVIEW.md](./PROJECT_REVIEW.md)
