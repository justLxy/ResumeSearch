# 院校档位筛选 — 设计文档

日期：2026-06-30
状态：已批准设计，待写实现计划

## 1. 需求

在左侧筛选面板"学历"下方新增一组"院校"筛选，支持按院校档位筛选候选人：

- **985**
- **211**
- **双一流**
- **C9**
- **海外QS50**（海外 QS 前 50 院校）
- **其他**（不属于以上任何名单的院校）

约束：

- **不做特征工程**：不给简历打标签、不重建索引、不在文档里烘焙派生字段。
- 命中口径：**任一学历命中即算**。一个人多段教育（如本科双非、硕士 985），只要任意一所学校属于该档位，即命中该档位。

## 2. 核心设计思想

所有档位最终都归约为同一个操作，与现有"学历"筛选完全同构：

> **档位标签 → 一组学校名 → 一行 ES `terms` 过滤。**

这与现有学历筛选机制一致：现有学历筛选是"硕士 → `candidate.highest_degree = 硕士` 一行过滤"，不给文档打标签。院校筛选照搬这一哲学。

关键事实（已核实）：

- 每份简历已存在字段 `candidate.all_schools`（在 `import_to_es.py:913` 构建），包含该候选人**所有**就读学校的名字数组，且有 `.keyword` 子字段（`import_to_es.py:541`）。对该字段做 `terms` 过滤天然实现"任一学历命中即算"。
- 现有过滤机制 `_build_filters()`（`app.py:320`）把每个筛选条件拼成 `bool.filter` 里的一条 ES 查询；自然语言查询走平行路径 `_filters_from_llm_constraints()`（`app.py:805`）。
- 现有学历筛选已采用"查询期标签展开"模式（`DEGREE_ALIASES` / `_normalize_degree_list`），即标签 → 具体值集合 → 一条 `terms`，不依赖文档侧派生字段。院校筛选沿用同一模式。

### 为什么所有档位都是封闭名单

- 985 / 211 / 双一流 / C9 本身就是**权威封闭名单**（全国固定数十至上百所），属于参考数据（类比国家代码表 / 城市代码表），不是"特征工程"。
- 海外档定义为 **QS 前 50 的海外院校**，同样做成**封闭名单**（中英文校名都列入），避免"海外"泛指出国带来的歧义。
- "其他" = 不在以上任何名单中的院校，用补集（`must_not`）实现。

因此**全程不需要 LLM 对学校分类，零重建索引，零打标签**。

### 命名说明

海外档对外显示名为 **"海外QS50"**，按钮文字 `海外QS50`，悬停提示"海外 QS 前 50 院校"。理由：直接叫"海外"会让用户误以为只要出国就能筛到，而实际只筛 QS 前 50；显式标注档位口径避免歧义。名单 key 为 `qs50_overseas`。

## 3. 数据模型

新建静态参考数据文件 `school_tiers.json`：

```json
{
  "985":    ["清华大学", "北京大学", "上海交通大学", "..."],
  "211":    ["..."],
  "双一流":  ["..."],
  "c9":     ["清华大学", "北京大学", "..."],
  "qs50_overseas": ["Stanford University", "麻省理工学院", "..."]
}
```

说明：

- 各名单存在天然包含关系（C9 ⊂ 985 ⊂ 211 ⊂ 双一流），但**不做嵌套推导**——每个档位直接列全量校名，简单、可读、改名单不易出错。
- 海外名单中英文校名都列入，以匹配数据中可能出现的任意一种写法。
- 该文件是纯参考数据，可人工维护；新增院校只需编辑此文件。

档位 → key 映射：

| UI 显示 | 名单 key | 实现 |
|---|---|---|
| 985 | `985` | `terms` 名单 |
| 211 | `211` | `terms` 名单 |
| 双一流 | `双一流` | `terms` 名单 |
| C9 | `c9` | `terms` 名单 |
| 海外QS50 | `qs50_overseas` | `terms` 名单 |
| 其他 | （补集） | `must_not terms`（所有名单并集） |

## 4. 后端设计（app.py）

### 4.1 名单加载

启动时（或懒加载 + 缓存）从 `school_tiers.json` 读入，构建：

- `SCHOOL_TIERS: dict[str, set[str]]` — 各档位 key → 校名集合。
- `SCHOOL_TIERS_KNOWN: set[str]` — 所有名单的并集（供"其他"补集使用）。

### 4.2 过滤器构建

新增 `_school_tier_filter(tier: str) -> dict | None`：

```python
def _school_tier_filter(tier: str) -> dict | None:
    if not tier:
        return None
    if tier == "其他":
        return {"bool": {"must_not": {"terms": {"candidate.all_schools.keyword": sorted(SCHOOL_TIERS_KNOWN)}}}}
    names = SCHOOL_TIERS.get(_normalize_school_tier(tier))
    if not names:
        return None
    return {"terms": {"candidate.all_schools.keyword": sorted(names)}}
```

- 校名匹配大小写：海外英文校名以名单中存储的规范写法为准；对 `all_schools.keyword`（精确 keyword）做 `terms`。后续如发现数据中大小写不一致，可在名单层补齐变体（不引入文档侧处理）。
- 集成到 `_build_filters()`：新增参数 `school_tier: str = ""`，命中则把 `_school_tier_filter` 的结果 append 进 filters 列表，与学历/城市/技能/年限叠加。

### 4.3 search 入口

`@app.get("/api/search")`（`app.py:173`）新增查询参数 `school_tier: str = ""`，透传给 `_build_filters()`。

### 4.4 LLM Query Planner 扩展

让自然语言查询也能触发院校筛选，与左侧按钮走同一后端过滤路径：

- Planner 输出的 `constraints` 新增字段 `school_tier`（单值字符串，取值 ∈ {985, 211, 双一流, c9, qs50_overseas, 其他} 或空）。
- prompt 增加少量示例："找个985的"→`school_tier:"985"`；"海外回来的 / QS前50"→`"qs50_overseas"`。
- 沿用现有 degree 的"绝不臆测"原则：**只在 query 字面提及院校档位时才填**，不因岗位/专业"看起来高端"就臆测档位。
- `_filters_from_llm_constraints()`（`app.py:805`）解析该字段并调用 `_school_tier_filter` 生成过滤条件。

## 5. Facet 计数（可选增强）

在 `_load_facets()`（`app.py:2051`）的 aggs 中，为每个档位加一个 `filter` 聚合（用对应名单集合 / "其他"用 must_not），算出每档命中人数，使按钮显示如 `985 (47)`，与现有 city/skill facet 风格一致。

## 6. 前端设计

### 6.1 index.html

在"学历" radio 组下方新增"院校" radio 组（与学历样式一致）：

```
院校  ○不限 ○985 ○211 ○双一流 ○C9 ○海外QS50 ○其他
```

`海外QS50` 选项加 `title="海外 QS 前 50 院校"` 悬停提示。

### 6.2 app.js

- `state` 新增 `schoolTier: ""`。
- `params()`（`app.js:47`）：`if (state.schoolTier) query.set("school_tier", state.schoolTier)`。
- 绑定 radio change 事件 → 更新 `state.schoolTier` → 触发 `runSearch()`（与学历 radio 一致）。
- 若做 facet 计数，在 `renderFacets` 对应位置渲染计数。

## 7. 错误处理与边界

- `school_tiers.json` 缺失或解析失败：记录日志，院校筛选降级为不可用（返回空过滤），不影响其余检索。
- 未知 `school_tier` 值：`_school_tier_filter` 返回 `None`，等同不筛选。
- "其他"在名单并集为空时：`must_not` 空集 = 不过滤（安全降级）。

## 8. 测试

`tests/test_search_logic.py` 新增：

- 封闭档（如 985）→ 生成 `terms` 过滤，名字集合正确。
- "其他" → 生成 `must_not terms`，使用名单并集。
- 未知 / 空档位 → 返回 None（不筛选）。
- `_build_filters` 叠加：院校 + 学历 + 城市 同时存在时，filters 列表包含全部条件。
- （若实现 Planner 扩展）`_filters_from_llm_constraints` 解析 `school_tier` 字段。

## 9. 改动清单

| 文件 | 改动 |
|---|---|
| `school_tiers.json`（新） | 静态名单数据 |
| `app.py` | 加载名单；`_school_tier_filter()`；`_build_filters` / `search` 加 `school_tier` 参数；Planner prompt + 解析；（可选）facet 聚合 |
| `web/index.html` | 院校 radio 按钮组 |
| `web/app.js` | `state.schoolTier`；`params()`；事件绑定；（可选）facet 计数渲染 |
| `tests/test_search_logic.py` | 院校过滤单测 |

## 10. 非目标（YAGNI）

- 不做文档侧 `school_tiers` 字段或重建索引。
- 不做 LLM 对学校的运行时分类（已用封闭名单替代）。
- 不做档位多选（首版单选，与学历 radio 一致）；如需多选可后续扩展为 `terms` 并集。
- 不做学历段 × 院校档的交叉精确口径（如"只看硕士那段是不是985"）；首版统一"任一学历命中即算"。
