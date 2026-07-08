"""候选人质量分：匹配相关性相同时的 tie-break 维度。

这里的分【不】参与召回或相关性主链路。它只在两个候选人的检索主分分桶到同一
精度（见 _score_bucket）时才起作用——相关性有实质差异时绝不会被质量翻转。
五个子维度（学校层次 / 奖项含金量 / 实习 / 项目 / 完整度）各自归一化到 [0,1]，
加权求和后 quality_score 也落在 [0,1]，量纲无需与 RRF/rerank 分对齐。

纯函数，无 IO（学校名单复用 normalization 的进程内缓存）。被 retrieval / reranking
的排序键与 formatting 的结果装配引用。
"""
from __future__ import annotations

from datetime import date
from typing import Any

from resume_search.services.normalization import _load_school_tiers

# --- 子维度权重（对应产品优先级：学校 > 奖项 > 实习 > 项目 > 完整度）---
QUALITY_WEIGHTS = {
    "school": 0.30,
    "awards": 0.25,
    "internships": 0.20,
    "projects": 0.15,
    "completeness": 0.10,
}

# 院校档位 → 序数分。C9 是 985 子集，单独给最高分体现顶尖梯队；
# 海外 QS100 与 985 同档。取候选人命中的最高档。
SCHOOL_TIER_SCORE = {
    "c9": 1.0,
    "985": 0.85,
    "qs50_overseas": 0.85,
    "211": 0.6,
    "双一流": 0.5,
}
SCHOOL_TIER_OTHER_SCORE = 0.1

# 奖项含金量：level 枚举 → 权重（见 data 中实际出现的 level 取值）。
# 加权求和后用上限压缩，既奖励含金量也奖励数量，又防止刷奖线性膨胀。
AWARD_LEVEL_WEIGHT = {
    "一等奖": 1.0,
    "金奖": 1.0,
    "金牌": 1.0,
    "top 5%": 0.9,
    "二等奖": 0.7,
    "银奖": 0.7,
    "银牌": 0.7,
    "meritorious": 0.6,
    "三等奖": 0.5,
    "铜奖": 0.5,
    "铜牌": 0.5,
    "决赛圈": 0.4,
    "优胜奖": 0.3,
    "优秀个人": 0.3,
    "优秀学员": 0.3,
}
AWARD_LEVEL_DEFAULT_WEIGHT = 0.2
AWARD_SCORE_SATURATION = 2.0  # 加权和达到此值即封顶（约两个一等奖）

# 各子维度的"饱和阈值"：达到即视为满分，防止长尾无限加分。
INTERNSHIP_COUNT_SATURATION = 3
INTERNSHIP_MONTHS_SATURATION = 18.0
INTERNSHIP_DESC_SATURATION = 150.0
PROJECT_COUNT_SATURATION = 3
PROJECT_DESC_SATURATION = 200.0

# 主分分桶精度：检索主分（RRF / rerank）量化到此精度后相等，才认为"相关性实质
# 相同"，此时质量分才作为 tie-break 生效。相关性有实质差异时绝不被质量翻转。
SCORE_BUCKET_DIGITS = 4

# 完整度考察的关键字段填充率。
COMPLETENESS_FIELDS = (
    "school",
    "major",
    "highest_degree",
    "years_experience",
    "skills",
    "education",
    "internships",
    "projects",
    "awards",
    "languages",
)


def score_bucket(score: float | None) -> float:
    """把检索主分量化到 tie-break 精度。浮点精确相等几乎不触发，分桶后
    相关性实质相同的候选人才会进入质量比较。"""
    return round(float(score or 0.0), SCORE_BUCKET_DIGITS)


def compute_quality_score(source: dict[str, Any]) -> float:
    """候选人质量综合分，落在 [0,1]。source 是 ES 的 _source。"""
    if not source:
        return 0.0
    subscores = {
        "school": _school_score(source),
        "awards": _awards_score(source),
        "internships": _internships_score(source),
        "projects": _projects_score(source),
        "completeness": _completeness_score(source),
    }
    total = sum(QUALITY_WEIGHTS[name] * value for name, value in subscores.items())
    return round(_clamp01(total), 6)


def _school_score(source: dict[str, Any]) -> float:
    tiers, _known = _load_school_tiers()
    candidate = source.get("candidate") or {}
    schools = set(candidate.get("all_schools") or [])
    if not schools:
        school = candidate.get("school")
        if school:
            schools = {str(school).strip()}
    if not schools:
        return 0.0
    best = 0.0
    matched_any = False
    for tier_key, names in tiers.items():
        tier_names = set(names)
        if schools & tier_names:
            matched_any = True
            best = max(best, SCHOOL_TIER_SCORE.get(tier_key, 0.0))
    # 有学校但不在任何名单里 → "其他"档，给低保底分而非 0，与空学校区分。
    return best if matched_any else SCHOOL_TIER_OTHER_SCORE


def _awards_score(source: dict[str, Any]) -> float:
    awards = source.get("awards") or []
    raw = 0.0
    for award in awards:
        if not isinstance(award, dict):
            continue
        if award.get("has_award") in (None, "否", False) or not award.get("name"):
            continue
        level = str(award.get("level") or "").strip().lower()
        raw += AWARD_LEVEL_WEIGHT.get(level, AWARD_LEVEL_DEFAULT_WEIGHT)
    return _clamp01(raw / AWARD_SCORE_SATURATION)


def _internships_score(source: dict[str, Any]) -> float:
    internships = [i for i in (source.get("internships") or []) if isinstance(i, dict)]
    if not internships:
        return 0.0
    count_score = _clamp01(len(internships) / INTERNSHIP_COUNT_SATURATION)
    total_months = sum(_span_months(i) for i in internships)
    duration_score = _clamp01(total_months / INTERNSHIP_MONTHS_SATURATION)
    avg_desc = _avg_len(internships, ("description",))
    density_score = _clamp01(avg_desc / INTERNSHIP_DESC_SATURATION)
    return 0.5 * count_score + 0.3 * duration_score + 0.2 * density_score


def _projects_score(source: dict[str, Any]) -> float:
    # 只衡量项目经历本身的丰富与充实度；项目与 query 的相关性已由 RRF/rerank 表达，
    # 不在静态质量分里重复计分。
    projects = [p for p in (source.get("projects") or []) if isinstance(p, dict)]
    if not projects:
        return 0.0
    count_score = _clamp01(len(projects) / PROJECT_COUNT_SATURATION)
    avg_desc = _avg_len(projects, ("description", "responsibility"))
    density_score = _clamp01(avg_desc / PROJECT_DESC_SATURATION)
    return 0.6 * count_score + 0.4 * density_score


def _completeness_score(source: dict[str, Any]) -> float:
    candidate = source.get("candidate") or {}
    filled = 0
    for field in COMPLETENESS_FIELDS:
        value = candidate.get(field)
        if value in (None, "", [], {}):
            value = source.get(field)
        if value not in (None, "", [], {}):
            filled += 1
    return filled / len(COMPLETENESS_FIELDS)


def _span_months(record: dict[str, Any]) -> float:
    start = _parse_date(record.get("start_date"))
    end = _parse_date(record.get("end_date"))
    if not start or not end or end < start:
        return 0.0
    return (end.year - start.year) * 12 + (end.month - start.month) + (end.day - start.day) / 30.0


def _avg_len(records: list[dict[str, Any]], fields: tuple[str, ...]) -> float:
    if not records:
        return 0.0
    total = 0
    for record in records:
        total += sum(len(str(record.get(field) or "")) for field in fields)
    return total / len(records)


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    text = str(value).strip()[:10]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            from datetime import datetime

            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
