# 简历检索系统

这是一个基于 FastAPI 和 Elasticsearch 的简历检索原型，用于验证简历场景下的关键词检索、结构化过滤、向量检索和 RRF 混合排序。

本文重点说明检索字段如何分工，以及 BM25、keyword/filter、dense vector、RRF 在完整 use case 中分别解决什么问题。开发过程中的踩坑、错误决策和复盘见 [PROJECT_REVIEW.md](./PROJECT_REVIEW.md)。

## 检索设计原则

简历检索里有两类完全不同的需求：

- 精确匹配：用户搜索某个学校、公司、姓名、城市、岗位编号、技能标签。
- 语义匹配：用户描述一个能力、项目经验、实习职责、研究方向。

因此字段不能全部塞进向量，也不能全部依赖 BM25。

核心原则：

```text
BM25 / keyword 检索“字段上写了什么”。
向量检索“这个人做过什么、能力像什么”。
RRF 把精确匹配和语义匹配结合起来。
```

当前检索架构：

```text
用户查询
  ├─ 轻量结构化解析
  │    └─ 从自然语言中识别明确年限、城市、学历、技能筛选
  ├─ BM25 / keyword / filter 检索
  │    └─ 负责实体、关键词、结构化条件
  ├─ dense vector 检索
  │    └─ 负责能力、职责、项目、经历语义
  └─ RRF 融合排序
       └─ 两路都命中的候选人排序更靠前
```

当前实现有几个边界：

- 候选人编号、岗位编号、手机号、邮箱、学校、公司等精确查询优先走 keyword/BM25，不触发 dense vector。
- `自然语言处理`、`深度学习`、`推荐召回`、`模型落地` 这类短能力表达会触发 dense vector，不再要求必须是长句。
- 前端技能筛选是 AND 语义，选择 `Python` 和 `NLP` 表示候选人必须同时具备两个技能。
- 用户直接输入 `0.5年以上 北京 本科 推荐系统` 时，会基于索引里的城市、学历、技能词表和明确年限模式解析成 filter + 剩余 query；普通 `推荐系统 NLP SQL` 仍作为宽召回文本查询，不强行拆成多个硬过滤。
- 空查询和只有筛选条件的浏览场景会返回 ES 默认窗口内的全部候选人；有关键词的混合检索仍保留单次返回上限，因为 RRF/kNN 是排序候选窗口，不适合一次拉无限结果。

## 检索字段分工

下表按当前简历结构和 Elasticsearch mapping 组织。当前未索引的字段不代表业务上永远不能搜索，只表示当前 mapping 尚未把它纳入可检索字段。后续如果有业务需求，可以补 mapping、导入逻辑和查询逻辑。

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
| 投递信息 | `application.position_name` | BM25 + keyword | 是，末尾补充 | 岗位名称搜索、岗位语义 |
| 投递信息 | `application.expected_work_cities` | keyword filter | 否 | 期望城市筛选 |
| 投递志愿 | `application.wishes.rank` | integer | 否 | 志愿排序 |
| 投递志愿 | `application.wishes.position_name` | nested BM25 | 否 | 志愿岗位搜索 |
| 投递志愿 | `application.wishes.company` | nested keyword | 否 | 志愿公司过滤 |
| 候选人 | `candidate.name` | keyword + BM25 | 否 | 姓名精确查询 |
| 候选人 | `candidate.gender` | keyword filter | 否 | 性别筛选 |
| 候选人 | `candidate.birth_date` | date filter | 否 | 年龄筛选 |
| 候选人 | `candidate.current_city` | keyword filter | 否 | 当前城市筛选 |
| 候选人 | `candidate.highest_degree` | keyword filter | 否 | 最高学历筛选 |
| 候选人 | `candidate.graduation_date` | date filter | 否 | 毕业时间筛选 |
| 候选人 | `candidate.school` | keyword + BM25 phrase | 否 | 毕业院校精确查询 |
| 候选人 | `candidate.major` | BM25 | 是，末尾补充 | 专业搜索、专业语义 |
| 候选人 | `candidate.phone` | keyword 精确查 | 否 | 联系方式查询 |
| 候选人 | `candidate.email` | keyword 精确查 | 否 | 联系方式查询 |
| 候选人 | `candidate.years_experience` | range filter | 否 | 工作/实习年限筛选 |
| 教育经历 | `education.start_date` | nested date | 否 | 教育时间过滤 |
| 教育经历 | `education.end_date` | nested date | 否 | 教育时间过滤 |
| 教育经历 | `education.school` | nested keyword + BM25 phrase | 否 | 学校精确查询 |
| 教育经历 | `education.college` | nested BM25 | 否 | 学院查询 |
| 教育经历 | `education.major` | nested BM25 | 是 | 专业语义 |
| 教育经历 | `education.education_level` | nested keyword/filter | 否 | 本科、硕士等背景 |
| 教育经历 | `education.degree` | nested keyword/filter | 否 | 学士、硕士等背景 |
| 教育经历 | `education.research_direction` | nested BM25 | 是 | 研究方向语义 |
| 教育经历 | `education.lab_name` | nested BM25 | 是，但清理学校/学院实体 | 实验室方向语义 |
| 教育经历 | `education.paper_level` | keyword/filter | 否 | 科研背景参考 |
| 实习经历 | `internships.company` | nested keyword + BM25 phrase | 否 | 实习公司精确查询 |
| 实习经历 | `internships.department` | nested BM25 | 是 | 部门方向语义 |
| 实习经历 | `internships.title` | nested BM25 | 是 | 实习职位语义 |
| 实习经历 | `internships.work_type` | keyword/filter | 否 | 实习性质 |
| 实习经历 | `internships.description` | nested BM25 | 是 | 工作内容和职责语义 |
| 项目经历 | `projects.name` | nested BM25 | 是 | 项目名称、项目主题 |
| 项目经历 | `projects.description` | nested BM25 | 是 | 项目背景和业务场景 |
| 项目经历 | `projects.responsibility` | nested BM25 | 是 | 项目职责和能力语义 |
| 技能 | `skills` | keyword 精确查/filter | 是 | 技能标签精确过滤和语义 |
| 技能 | `skills_text` | BM25 | 是 | 多技能组合检索 |
| 语言能力 | `languages.english_exam_score` | keyword/filter | 否 | 英语等级筛选 |
| 语言能力 | `languages.english_spoken_level` | keyword/filter | 否 | 英语口语筛选 |
| 分段文本 | `section_text.education` | BM25 / highlight | 否 | 教育片段高亮 |
| 分段文本 | `section_text.internships` | BM25 / highlight | 否 | 实习片段高亮 |
| 分段文本 | `section_text.projects` | BM25 / highlight | 否 | 项目片段高亮 |
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

## BM25 / keyword / filter Use Case

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
candidate.school phrase 命中
nested education.school.keyword 精确命中
nested education.school phrase 命中
section_text.education BM25 命中
```

这种查询不应该主要靠向量判断，因为用户要找的是“这个学校”本身。

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
nested internships.company phrase 命中
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
  查 application.position_name
  查 application.wishes.position_name
  查 application.wishes.company
  查 candidate.name
  查 candidate.phone / candidate.email
  查 candidate.school
  查 skills_text
  查 skills
  查 section_text.projects
  查 section_text.internships
  查 nested projects.name
  查 nested projects.description
  查 nested projects.responsibility
  查 nested internships.title
  查 nested internships.description
  查 nested education.major
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

实际实现中，dense retriever 虽然会向 ES 请求 top-k 最近邻，但这些最近邻不会无条件进入最终结果。规则是：

```text
BM25 命中的候选：
  可以获得 dense rank 的 RRF 加分

BM25 没命中的 dense-only 候选：
  只有当 dense _score 达到高置信门槛，并且接近 dense top1 时才保留
  同时限制 dense-only 候选数量，避免向量 top-k 自动把结果页填满
```

因此，有关键词搜索的最终返回数量可能小于页面默认上限 20。20 只是关键词搜索的默认最多返回条数，不代表每次搜索都应该有 20 个相关候选。空查询或只有筛选条件时属于浏览场景，会返回 ES 默认结果窗口内的全部候选人。

接口返回数量字段含义：

```text
returned_count:
  当前响应实际返回的结果数

matched_total:
  BM25 / keyword / filter 侧的真实命中总数

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
analyzer 或 keyword 字段 mapping 变化
```

本地 mock 数据重建命令：

```bash
python generate_mock_resumes.py --count 50 --seed 20260616 --recreate --no-output
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

## 开发复盘

这份 README 说明的是检索字段分工和 use case。开发过程中遇到的详细问题、错误决策、排查过程和面试表达，见：

- [PROJECT_REVIEW.md](./PROJECT_REVIEW.md)
