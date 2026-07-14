# 奖励系数应用时机 A/B 实验报告

> 层级与覆盖度奖励系数（`1.0 + 0.15×层级 + 0.05×覆盖数`，上限约 ×1.45）应放在 Rerank **之前**还是**之后**？

- 数据集：`resumes_current`（1621 篇简历）+ `resume_evidence_current`（1567 个证据切片）
- 查询集：`eval_queries.jsonl`，共 77 条，覆盖 9 种查询类型
- 评测指标：P@5/P@10、R@5/R@10/R@50/R@100、MRR@10、NDCG@5/@10、success@1、离域拒识率
- 实验脚本：`experiments/ab_reward_stage.py`；逐 case 明细：`experiments/ab_result.json`

---

## 一、两种方案的定义与实现

| | 方案 A（当前实现，默认） | 方案 B（候选方案） |
|---|---|---|
| 流水线 | RRF → **奖励系数** → Rerank | RRF → Rerank → **奖励系数** |
| 奖励作用对象 | RRF 原始分（`raw_rrf × multiplier`） | 相关性主分（rerank 命中者用 cross-encoder 分，未命中者用 raw RRF 分） |
| `REWARD_STAGE` 取值 | `pre_rerank` | `post_rerank` |

两方案共用同一套检索/规划/精排代码，**唯一差异是奖励系数乘入排序主分的时机**，由 `resume_search/config.py` 的 `REWARD_STAGE` 开关控制：

- 方案 A（`retrieval.py`）：奖励折进 RRF 主分后再排序、切 rerank 窗口。
- 方案 B（`reranking._apply_reward_multiplier` + `search.py`）：奖励作为**最后一步**乘到主分上再统一分桶重排；rerank 阶段与低相关度判定完全只看纯 cross-encoder 分。

> 实现要点：方案 B 的低相关度标记（`low_relevance`）仍基于**纯** cross-encoder 分判定，不被奖励系数抬过地板——否则"层级高但语义无关"的候选会被误判为高相关。实验已验证两方案的离域拒识决策逐条完全一致（见 §3.4）。

---

## 二、方案 A 现状的机制解剖（实验前必须澄清的关键事实）

在跑数之前，先读代码定位"奖励系数在方案 A 里究竟改变了什么"，这决定了 A/B 差异可能出现在哪里：

1. **奖励折进 RRF 主分**（`retrieval._rrf_merge`）：`final_scores[doc] = rrf_scores[doc] × multiplier`。
2. **切 rerank 窗口**：`RERANK_TOP_N = 20`，但候选池规模实测 145~160。奖励在此**门控**——它能把某候选人推入/推出被精排的 20 个名额。
3. **rerank 覆盖分数**（`reranking._rerank_results`）：窗口内 `item["score"]` 被 cross-encoder 分**覆盖**，随后按 rerank 分分桶重排。**奖励系数在窗口内仅作为同分桶内 `pre_rank` tie-break 残留**，不再直接进入 top-K 排序主分。

**结论**：方案 A 下，奖励对最终 top-K 排序的影响只剩两条弱通道——(a) 决定谁进 rerank 窗口；(b) rerank 同分桶内的 tie-break。真正决定 top-K 顺序的是 cross-encoder 分。

方案 B 则相反：`final = rerank_score × multiplier` 直接进入排序主分，奖励与 cross-encoder 分**乘性耦合**，能真正改写 top-K 顺序。这正是本实验要检验的核心分歧点。

---

## 三、实验设计：如何做到严格控制变量

### 3.1 混杂因素与消除

第一版实验用两个独立进程分别以 `REWARD_STAGE=pre_rerank` 和 `post_rerank` 跑 `evaluate_search.py`，发现**两个混杂因素**导致结果不可信：

1. **LLM planner 漂移**：query planner 走 LLM（`temperature=0` 但跨进程无缓存），两次运行对 6 条 query 产出了不同的 `lexical_query/semantic_query`（如 `jd_java_finance`、`jd_go_cloud`），导致检索输入本身就不同——这与奖励时机无关。
2. **检索/精排的进程间差异**：虽然 ES 与 rerank 对相同输入确定，但只要上游 plan 变了，下游全变。

对照实验的正确做法是**单进程内控制变量**（`experiments/ab_reward_stage.py`）：

- 每条 query **只规划一次**（预热 planner 进程内缓存，TTL 300s），保证 A、B 两次调用复用**同一 QueryPlan**；
- 随后仅翻转 `REWARD_STAGE`，各跑一次 `search()`；
- ES 检索、cross-encoder 精排对相同输入均确定（已实测 rerank 分逐位可复现）；
- 因此**唯一变量就是奖励系数的应用时机**。

### 3.2 一致性护栏

脚本对每条 case 校验两次调用的 `query_plan`（lexical/semantic/intent/dense/rerank）是否逐字段相同，统计 `plan_inconsistent`。最终实验：

```
cases=77  judged=69  plan_inconsistent=0  top10_changed=22
```

`plan_inconsistent=0` 证明混杂因素已被彻底消除；`top10_changed=22` 是**纯粹由奖励时机**引起的 top-10 顺序变化。

### 3.3 整体指标对比（A=pre_rerank，B=post_rerank）

| 指标 | 方案 A | 方案 B | ΔB−A |
|---|---|---|---|
| P@5 | 0.8406 | 0.8406 | +0.0000 |
| P@10 | 0.7348 | 0.7348 | +0.0000 |
| R@5 | 0.4874 | 0.4874 | +0.0000 |
| R@10 | 0.6251 | 0.6251 | +0.0000 |
| R@50 | 0.8815 | 0.8815 | +0.0000 |
| R@100 | 0.9620 | 0.9620 | +0.0000 |
| MRR@10 | 1.0000 | 1.0000 | +0.0000 |
| NDCG@5 | 0.9948 | 0.9948 | +0.0000 |
| **NDCG@10** | **0.9944** | **0.9938** | **−0.0006** |
| success@1 | 1.0000 | 1.0000 | +0.0000 |
| 拒识率 | 0.875 | 0.875 | +0.0000 |

**唯一非零差异是 NDCG@10 下降 0.0006，方向对 B 不利；其余全部指标完全相同。**

- **召回类指标（R@K）全等**：符合预期——奖励只重排已召回的候选，不改变召回集合，且 rerank 窗口成员在两方案下相同（见 §5）。
- **MRR@10 / success@1 恒为 1.0**：top-1 在任何 case 都没被奖励时机改变过。
- **NDCG@10 的微降**集中在极少数带分级相关性（graded relevance）的语义 case，下节详解。

### 3.4 离域拒识一致性（negative_semantic）

| case | reject_A | reject_B | 一致 |
|---|---|---|---|
| neg_quantum_chip | True | True | ✅ |
| neg_sap_abap | True | True | ✅ |
| neg_ios_swift_av | True | True | ✅ |
| neg_medical_dicom | True | True | ✅ |
| neg_slam_robotics | True | True | ✅ |
| neg_unity_game | True | True | ✅ |
| neg_oracle_dba | True | True | ✅ |
| neg_embedded_rtos | False | False | ✅ |

8/8 完全一致。因为方案 B 的 `low_relevance` 判定基于纯 cross-encoder 分，奖励系数不参与，离域拒识能力不受影响。

---

## 四、分查询类型表现

| 类型 | 条数 | top10 变化数 | A NDCG@10 | B NDCG@10 | ΔB−A |
|---|---|---|---|---|---|
| cross_language | 8 | 2 | 0.9920 | 0.9920 | +0.0000 |
| entity_exact | 8 | 0 | 0.9725 | 0.9725 | +0.0000 |
| exact_lookup | 8 | 0 | 1.0000 | 1.0000 | +0.0000 |
| jd_match | 13 | 8 | 1.0000 | 1.0000 | +0.0000 |
| major_query | 8 | 0 | 1.0000 | 1.0000 | +0.0000 |
| negative_semantic | 8 | 0 | 0.0000* | 0.0000* | +0.0000 |
| **semantic_capability** | 8 | 7 | **1.0000** | **0.9948** | **−0.0052** |
| skill_combo | 8 | 4 | 0.9875 | 0.9873 | −0.0002 |
| structured_filter | 8 | 1 | 1.0000 | 1.0000 | +0.0000 |

（* negative_semantic 无相关文档，NDCG 记 0，看拒识率不看 NDCG。）

按类型看结论非常清晰：

- **实体/精确/筛选/学校/专业类**（entity_exact、exact_lookup、major_query、structured_filter、lookup）：这些查询 **rerank 基本不触发**（intent 为 keyword/lookup），方案 B 退化为"RRF 后乘奖励"，与方案 A 数学等价，**指标逐位相同、0 变化**。这说明 A/B 之争只在"rerank 会触发"的语义类查询上才有意义。
- **jd_match（长 JD）**：8/13 条 top10 顺序变了，但**全部是等相关文档之间的换位**（这些 case 的相关集较大且多为同级相关），NDCG@10 不变。
- **skill_combo（多技能）**：4/8 变化，其中 3 条等相关换位、1 条（`skill_pentest_redteam`）NDCG@10 −0.0015。
- **semantic_capability（能力描述）**：变化最集中（7/8），也是**唯一出现实质性 NDCG 下降**的类型（−0.0052），根源是 1 条 case（`sem_go_cloud_native`，−0.0419）。

**没有任何类型、任何单条 case 上方案 B 优于方案 A。** 差异要么为 0，要么 B 略差。

---

## 五、Case Study：逐条剖析排名变化的成因

### Case 1｜`sem_go_cloud_native`（最关键的反例，B 明显更差）

- Query：`需要 Go 云原生底层能力，做过 Kubernetes Operator、服务网格和高并发网关`
- 分级相关性：16 个 grade-3（高度相关）+ 29 个 grade-2（相关），共 45 个相关文档
- NDCG@10：A=1.0000（前 10 全是 grade-3），B=0.9581，**ΔB−A = −0.0419**

| 候选 | 相关档 | rerank 分 | 奖励系数 | A 排名 | B 排名 | A 分 | B 分（rerank×系数） |
|---|---|---|---|---|---|---|---|
| M20260077 | **g3** | **0.9709** | 1.00 | **1** | **5** | 0.9709 | 0.9709 |
| M20260189 | g3 | 0.9204 | 1.40 | 2 | **1** | 0.9204 | **1.2886** |
| M20260085 | g3 | 0.9191 | 1.40 | 3 | 2 | 0.9191 | 1.2867 |
| M20260056 | g3 | 0.8719 | 1.35 | 4 | 3 | 0.8719 | 1.1771 |
| M20260080 | g3 | 0.8521 | 1.35 | 5 | 4 | 0.8521 | 1.1503 |
| M20260065 | g3 | 0.7564 | 1.00 | 6 | **8** | 0.7564 | 0.7564 |
| M20260069 | g3 | 0.7099 | 1.00 | 7 | **9** | 0.7099 | 0.7099 |
| M20260137 | g3 | 0.7088 | 1.35 | 8 | 6 | 0.7088 | 0.9569 |
| M20260075 | g2 | 0.6263 | 1.30 | >10 | **7** | 0.6263 | 0.8142 |
| M20260036 | **g3** | 0.6889 | 1.00 | **10** | **>10** | 0.6889 | 0.6889 |

**成因分析（这是"奖励系数覆盖语义判断"的教科书式案例）：**

1. cross-encoder 认为 **M20260077 最相关**（rerank=0.9709，且是 grade-3）。方案 A 尊重这一判断，把它排第 1。
2. 方案 B 中，`M20260189` 的 rerank 只有 0.9204，但因覆盖了 5 个 query 词拿到 ×1.40，最终分 `0.9204×1.40 = 1.2886 > 0.9709`，**反超语义最相关的候选**，把 M20260077 从第 1 挤到第 5。
3. 更糟的是末位：grade-3 的 **M20260036**（rerank=0.6889，奖励系数仅 1.0）在 A 中位列第 10（仍在 top-10 内）；方案 B 里，被奖励抬分的 grade-2 候选 `M20260075`（`0.6263×1.30 = 0.8142`）反超了它，把这个**高相关**文档挤出了 top-10，换进来一个**次相关**文档。
4. 这一进一出（grade-3 出、grade-2 进）正是 NDCG@10 下降的直接来源。

**本质**：奖励系数是"层级/覆盖度"这类**结构性偏好**，与"语义是否真的匹配"是两个维度。方案 B 让结构性偏好乘性地覆盖了 cross-encoder 的语义排序，结果把语义更相关的人往下压。

### Case 2｜`skill_pentest_redteam`（B 轻微更差）

- Query：`内网渗透 Cobalt Strike 横向移动 提权`；NDCG@10 ΔB−A = −0.0015
- top-10 集合两方案相同，仅内部换位：A 的第 5 位 `M20260168`（rerank=0.8291，系数 1.0）在 B 中被系数 1.30~1.35 的 `M20260174/M20260153/M20260051` 越过，掉到第 10。
- 由于这些都是 grade-2/3 相关文档，集合未变，NDCG 仅因高相关文档的位次略微后移而微降。

### Case 3｜`jd_*` 长 JD（8 条变化，NDCG 全不变）

- 以 `jd_pentest`、`jd_java_finance` 等为代表：top-10 顺序在 A/B 间有换位，但**都是同级相关文档之间**，NDCG@10 保持不变。
- 原因：长 JD 的相关集大、且顶部候选多为同级相关，奖励时机改变了它们的相对次序，却没有让"更相关"和"更不相关"的相对顺序发生翻转。这类 case 是"变了但无害"。

### Case 4｜`major_软件工程` / lookup / 精确类（0 变化）

- rerank 未触发（intent=keyword/lookup），方案 B 退化为"RRF→乘奖励"，与方案 A 完全等价，输出逐位相同。
- 这印证了 §2 的判断：A/B 分歧只存在于 rerank 触发的语义查询。

---

## 六、两种方案排名变化的原理性归因

**为什么方案 A 的奖励几乎不动 top-K，而方案 B 会？**

- 方案 A 把奖励折进 RRF 分，但紧接着 rerank 会用 cross-encoder 分**覆盖**排序主分——奖励在窗口内只剩 tie-break 作用（§2）。所以在"候选池远大于窗口"的现实下（145~160 vs 20），奖励在 A 里的真实作用退化为**窗口门控 + 同分桶 tie-break**，对 top-K 的语义顺序几乎无扰动。
- 方案 B 把奖励**乘**到 cross-encoder 分上直接排序，`multiplier` 跨度 1.0~1.45（相差 45%），而语义相近候选的 rerank 分差常常只有几个百分点。于是**奖励系数的量纲盖过了 rerank 分的语义分辨力**，出现"低语义分×高系数 > 高语义分×低系数"的翻转。

**量纲错配是根因**：cross-encoder 分是带绝对语义含义的相关性分（0.35 地板、0.27~0.49 空白带都是模型自身校准的），而奖励系数是人为设定的结构性乘子。二者相乘缺乏理论依据——一个语义相关性 0.92 的人，不会因为"多覆盖两个词"就真的变得比 0.97 的人更适配岗位。方案 A 把奖励留在 rerank 之前、让 cross-encoder 有最终裁决权，恰好避开了这个错配。

**为什么召回和 MRR 不受影响？** 奖励只重排召回集内部，不改召回边界（R@K 全等）；而 top-1 在所有 case 都是"语义分最高且通常也高覆盖"的候选，A/B 都把它排第一（MRR/success@1 恒 1.0）。差异只发生在**中部名次的高相关 vs 高结构分之间的竞争**，因此仅 NDCG@10 这种对中部位次敏感的指标才捕捉到。

---

## 七、优缺点总结

### 方案 A（RRF → 奖励 → Rerank，当前实现）

**优点**
- Top-K 语义排序质量最高：cross-encoder 是排序链路的最后裁决者，语义相关性不被结构性偏好覆盖（NDCG@10 全类型 ≥ B）。
- 奖励仍在起作用，但作用在"安全"的位置——窗口门控（帮高层级/高覆盖的候选进入精排）+ 同分桶 tie-break，不喧宾夺主。
- 离域拒识、召回、MRR 均为最优或并列最优。

**缺点**
- 奖励对最终 top-K 的影响被 rerank 覆盖后**偏弱**——如果产品明确希望"层级/覆盖度"在最终排序里有更强的显性话语权，A 无法直接满足。
- 奖励对结果的影响不够可解释（藏在窗口门控里）。

### 方案 B（RRF → Rerank → 奖励）

**优点**
- 奖励对最终排序的影响**显式、可控、可解释**（final = rerank × 系数，直接可读）。
- 若未来业务把"层级/覆盖度"提到与语义相关性同等重要，B 提供了直接的调节旋钮。

**缺点**
- Top-K 语义质量下降：奖励乘性覆盖 cross-encoder 判断，会把语义更相关的候选往下压，甚至把高相关文档挤出 top-10（Case 1）。
- 量纲错配：结构性乘子与绝对语义分相乘无理论依据，×1.45 的跨度盖过了 rerank 分的语义分辨力。
- 全实验无一 case 受益，semantic_capability 类 NDCG@10 −0.0052、最差单条 −0.0419。

---

## 八、推荐方案

**推荐维持方案 A（RRF → 奖励 → Rerank，即 `REWARD_STAGE=pre_rerank`）。**

依据（均来自实验数据与排序原理，非经验判断）：

1. **数据**：在严格控制变量（`plan_inconsistent=0`）的对照实验中，方案 B 在整体指标上**无一项优于 A**，唯一非零差异 NDCG@10 −0.0006 对 B 不利；分类型看，B 在 semantic_capability 上 −0.0052、单条最差 −0.0419，其余持平。
2. **原理**：cross-encoder 分是带绝对语义含义的校准分，应当是排序链路的最终裁决者。把人为结构性乘子（跨度达 45%）乘在其上会造成量纲错配，用"覆盖了几个词"覆盖"语义是否真的匹配"，与检索系统"语义相关性优先"的目标相悖。
3. **风险**：方案 A 是当前线上实现，120 条单测全绿；切 B 有确定的下行风险而无收益。

---

## 九、更优实现方式的探讨（超越 A/B 的候选）

实验揭示的真正问题不是"奖励放前还是放后"，而是**奖励系数与语义分的耦合方式**。以下方案值得进一步实验：

### 9.1 保持 A，但把奖励从"隐性 tie-break"升级为"分桶内显式重排"（低风险，推荐优先试）

现状 A 里奖励在 rerank 后只剩 tie-break。可改为：**在 rerank 分分桶（`score_bucket`，4 位精度）内**，用奖励系数（而非当前的 quality_score，或与之组合）做桶内排序。这样：
- **语义相关性仍是主序**（跨桶不可被奖励翻转，天然避免 Case 1 的错配）；
- 奖励只在"cross-encoder 认为语义实质等同"的候选间起作用，正是它该起作用的地方；
- 可解释性优于现状。
- 可行性：改动局限在 `reranking._rerank_results` 的排序键，风险低，可直接用本实验框架验证。

### 9.2 只对 Top-N 施加奖励（B 的收敛版）

对 B 的顾虑主要来自"高相关文档被挤出 top-10"。若限制奖励只在 rerank 后的 top-N（如 top-5）内做**受限重排**、且**禁止跨相关性档位翻转**，可保留部分显性调节能力。但本质仍是 §9.1 的分桶思路的弱化版，且引入 N 这个新超参，性价比不如 9.1。

### 9.3 动态/加性奖励系数（治本方向）

根因是**乘性 + 固定跨度**。两个改法：
- **加性小偏移**：`final = rerank_score + α×(0.15×tier + 0.05×cov)`，取 α 使偏移量远小于典型 rerank 分差（如 α≤0.02），奖励只在语义近似平手时才决定顺序，从数学上杜绝 Case 1 的大翻转。
- **动态系数**：跨度随 query 类型收缩——多技能/JD 类语义信号强，系数跨度应更小（如上限 ×1.1）；实体/覆盖度主导的查询可保留较大跨度。
- 可行性：加性方案改动小、可解释、可用本框架直接扫 α 网格，是**最值得投入的下一步实验**。

### 9.4 分阶段融合（重架构，暂不推荐）

把 RRF、rerank、结构分作为三路特征喂给一个轻量 learning-to-rank 融合器（如线性/LambdaMART），用 `eval_queries.jsonl` 的分级相关性做训练。理论上最优，但需要更大的标注集、引入训练/线上一致性复杂度，当前数据规模（69 judged query）不足以稳健训练，**列为长期方向**。

---

## 十、复现方式

```bash
# 受控单进程 A/B（唯一变量=奖励时机），产出 experiments/ab_result.json
PYTHONPATH="$PWD" python3 experiments/ab_reward_stage.py --output experiments/ab_result.json

# 单独跑某一方案的完整评测
REWARD_STAGE=pre_rerank  python3 evaluate_search.py --output reports/ab_pre_rerank.json
REWARD_STAGE=post_rerank python3 evaluate_search.py --output reports/ab_post_rerank.json

# 回归测试（默认 pre_rerank，120 passed）
python3 -m pytest tests/ -q
```

默认 `REWARD_STAGE=pre_rerank`，线上行为不变；方案 B 仅在显式设置 `REWARD_STAGE=post_rerank` 时启用，供后续实验使用。


