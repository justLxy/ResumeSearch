"""把结构化骨架 + 自由文本组装成一份贴合真实字段结构的简历文档。

真实字段结构对齐 html-doc-v1 解析输出（见 indexing/resume_parser.py 对两份奇安信
真实简历的解析）。只产出参与 ES 索引/检索/召回的字段，其余（证件、政治面貌、导师、
企业性质等不进 mapping 的字段）按需求精简掉，减少无意义数据与生成成本。

generate(count, seed) 确定性、可复现：结构化字段是评测 ground-truth 的来源。
自由文本经 narrative 模块（缓存 + 可选 LLM）填充，默认冷启动走确定性回退。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from resume_gen import narrative, personas
from resume_gen import reference as ref

PARSER_VERSION = "mock-realistic-v3"
COMPANY = "奇安信集团"
SEED = 20260629

APPLY_WINDOW_START = date(2026, 3, 1)
APPLY_WINDOW_END = date(2026, 6, 20)
APPLY_YEAR = 2026

INTERN_PERIODS = ["3个月以上", "6个月以上", "实习期不限"]


def _rng_date(rng: random.Random, start: date, end: date) -> date:
    delta = max((end - start).days, 0)
    return start + timedelta(days=rng.randint(0, delta))


# --- 实习 / 项目槽位规划（结构化，确定性）---------------------------------
def _plan_internships(
    rng: random.Random,
    profile: dict[str, Any],
    degree: str,
    grad_year: int,
    apply_time: date,
) -> list[dict[str, Any]]:
    """规划实习槽位（公司、部门、职位、起止日期），文本稍后由 narrative 填充。

    实习驱动 years_experience 估算；应届/本科偏少，高学历/往届偏多，部分人无实习。
    """
    roles = profile.get("internship_roles") or []
    if not roles:
        return []
    max_n = 2 if degree in ("硕士", "博士") else 1
    n = rng.randint(0, max_n)
    if n == 0:
        return []

    catalog = ref.company_catalog(profile_family(profile))
    used_companies: set[str] = set()
    base = min(apply_time - timedelta(days=rng.randint(20, 120)), date(grad_year, 5, 31))
    chosen_roles = rng.sample(roles, k=min(n, len(roles)))
    slots: list[dict[str, Any]] = []
    for i, (dept, title) in enumerate(chosen_roles):
        available = [c for c in catalog if c[0] not in used_companies] or catalog
        company, _company_type = rng.choice(available)
        used_companies.add(company)
        start = base - timedelta(days=rng.randint(70, 360) * (i + 1))
        end = min(start + timedelta(days=rng.randint(90, 360)), apply_time - timedelta(days=rng.randint(1, 25)))
        if end < start:
            end = start + timedelta(days=rng.randint(30, 90))
        is_current = rng.random() < 0.15 and i == 0
        slots.append({
            "start_date": start.isoformat(),
            "end_date": None if is_current else end.isoformat(),
            "company": company,
            "department": dept,
            "title": title,
            "work_type": "实习",
            "is_current": is_current,
        })
    return slots


def _plan_projects(
    rng: random.Random,
    profile: dict[str, Any],
    grad_year: int,
    apply_time: date,
) -> tuple[list[str], list[dict[str, Any]]]:
    """规划项目槽位：挑选不同主题 + 起止日期。返回 (themes, date_slots)。"""
    themes_pool = profile["project_themes"]
    n = min(len(themes_pool), rng.randint(1, 3))
    themes = rng.sample(themes_pool, k=n)
    slots: list[dict[str, Any]] = []
    for _ in themes:
        latest_end = min(apply_time - timedelta(days=rng.randint(5, 80)), date(grad_year, 6, 30))
        start = latest_end - timedelta(days=rng.randint(120, 420))
        slots.append({"start_date": start.isoformat(), "end_date": latest_end.isoformat(), "is_current": False})
    return themes, slots


def profile_family(profile: dict[str, Any]) -> str:
    return profile["_family"]


# --- 组装单份简历 ----------------------------------------------------------
def _build_one(
    rng: random.Random,
    family: str,
    profile: dict[str, Any],
    idx: int,
    used_names: set[str],
    *,
    use_llm: bool,
    request_sink: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    resume_id = f"M2026{idx:04d}"
    gender = rng.choice(["男", "女"])
    name = ref.dedup_name(rng, ref.sample_name(rng, gender), used_names, gender)
    degree = personas.choose_degree(rng, profile)
    grad_year = personas.choose_grad_year(rng, APPLY_YEAR)
    apply_time = _rng_date(rng, APPLY_WINDOW_START, APPLY_WINDOW_END)
    position = rng.choice(profile["positions"])
    position_code = rng.choice(profile["position_codes"])

    current_city = ref.sample_city(rng)
    expected_cities = ref.sample_expected_cities(rng, current_city)

    education = personas.build_education(rng, profile, degree, grad_year, APPLY_YEAR)
    birth_date = personas.build_birth_date(rng, degree, grad_year)
    skills, it_items = personas.build_skills(rng, profile)
    english_exam, english_spoken = ref.sample_english(rng)

    intern_slots = _plan_internships(rng, profile, degree, grad_year, apply_time)
    project_themes, project_slots = _plan_projects(rng, profile, grad_year, apply_time)

    # 自由文本：一份简历一次请求，画像完整喂入，缓存/LLM/回退三级。
    grad_edu = education[-1]
    request = {
        "resume_id": resume_id,  # 保证每个人的文本独立、缓存独立 → 千人千面、无重复
        "domain_label": profile["domain_label"],
        "degree": degree,
        "position": position,
        "skills": skills,
        "research_direction": grad_edu.get("research_direction") if grad_edu.get("research_direction") != "无" else None,
        "project_themes": project_themes,
        "internships": [{"department": s["department"], "title": s["title"]} for s in intern_slots],
        "is_hard_negative": family.startswith("hn_"),
        "negative_note": profile.get("negative_note"),
    }
    if request_sink is not None:
        request_sink.append(request)
    text = narrative.generate_narrative(request, use_llm=use_llm)

    internships = []
    for slot, body in zip(intern_slots, text.get("internships", [])):
        internships.append({
            "start_date": slot["start_date"],
            "end_date": slot["end_date"],
            "company": slot["company"],
            "department": slot["department"],
            "title": slot["title"],
            "work_type": slot["work_type"],
            "description": body.get("description", ""),
            "is_current": slot["is_current"],
        })

    projects = []
    for slot, body in zip(project_slots, text.get("projects", [])):
        projects.append({
            "start_date": slot["start_date"],
            "end_date": slot["end_date"],
            "name": body.get("name", ""),
            "description": body.get("description", ""),
            "responsibility": body.get("responsibility", ""),
            "is_current": slot["is_current"],
        })

    awards = _build_awards(rng, profile)
    offer = _build_offer(rng, apply_time)

    doc: dict[str, Any] = {
        "resume_id": resume_id,
        "parse_status": "ok",
        "parser_version": PARSER_VERSION,
        "file": {
            "name": f"{COMPANY}-{position}({position_code})-{name}({resume_id}).doc",
            "sha256": hashlib.sha256(resume_id.encode()).hexdigest(),
            "detected_type": "synthetic",
            "encoding": "utf-8",
        },
        "application": {
            "candidate_no": resume_id,
            "apply_time": apply_time.isoformat(),
            "company": COMPANY,
            "position_code": position_code,
            "position_name": position,
            "wishes": [{"rank": 1, "position_name": position, "company": COMPANY}],
            "expected_work_cities": expected_cities,
        },
        "candidate": {
            "name": name,
            "gender": gender,
            "birth_date": birth_date,
            "current_city": current_city,
            "highest_degree": degree,
            "graduation_date": f"{grad_year}-07-01",
            "school": grad_edu["school"],
            "major": grad_edu["major"],
            "phone": ref.sample_phone(rng),
            "email": ref.sample_email(rng, resume_id + name),
            # 不写 years_experience：交给 indexing.enrichment 从实习跨度估算
        },
        "education": education,
        "internships": internships,
        "projects": projects,
        "skills": skills,
        "it_skill_items": it_items,
        "languages": {"english_exam_score": english_exam, "english_spoken_level": english_spoken},
        "awards": awards,
        "offer_internship": offer,
        "_family": family,  # 评测 ground-truth 标签；写出前剔除
    }
    doc["section_text"] = _section_text(doc)
    return doc


def _build_awards(rng: random.Random, profile: dict[str, Any]) -> list[dict[str, Any]]:
    themes = profile.get("award_themes") or []
    if not themes or rng.random() < 0.45:
        return []
    name, levels = rng.choice(themes)
    level = rng.choice(levels)
    return [{
        "has_award": "是",
        "name": name,
        "level": level,
        "description": f"{name}，获{level}。",
    }]


def _build_offer(rng: random.Random, apply_time: date) -> dict[str, Any]:
    can_intern = rng.random() < 0.5
    return {
        "post_graduation_intention": rng.choice([None, "全职", "继续深造", "暂无明确意向"]),
        "can_intern": "是" if can_intern else "否",
        "available_start_date": _rng_date(rng, apply_time, APPLY_WINDOW_END + timedelta(days=200)).isoformat() if can_intern else None,
        "weekly_workdays": str(rng.choice([3, 4, 5])) if can_intern else None,
        "internship_period": rng.choice(INTERN_PERIODS) if can_intern else None,
    }


def _section_text(doc: dict[str, Any]) -> dict[str, str]:
    """只拼装 ES mapping 里实际存在的三段 section_text（education/internships/projects），
    其余段落 mapping 为 dynamic:false 会被丢弃，不再浪费生成。"""
    def join_items(items: list[dict[str, Any]], fields: list[str]) -> str:
        parts = []
        for it in items:
            seg = " ".join(str(it.get(f)) for f in fields if it.get(f))
            if seg:
                parts.append(seg)
        return "\n".join(parts)

    return {
        "education": join_items(doc["education"], ["school", "college", "major", "degree", "research_direction", "lab_name"]),
        "internships": join_items(doc["internships"], ["department", "title", "description"]),
        "projects": join_items(doc["projects"], ["name", "description", "responsibility"]),
    }


# --- 名额分配与顶层入口 ----------------------------------------------------
def _weighted_plan(profiles: dict[str, dict[str, Any]], total: int) -> list[str]:
    """按 weight 把 total 个名额分配到各族。"""
    if total <= 0:
        return []
    weights = {k: v["weight"] for k, v in profiles.items()}
    wsum = sum(weights.values())
    plan: list[str] = []
    for fam, w in weights.items():
        plan += [fam] * round(total * w / wsum)
    while len(plan) < total:
        plan.append(max(weights, key=weights.get))
    return plan[:total]


def generate(
    count: int = 200,
    seed: int = SEED,
    *,
    use_llm: bool = False,
    request_sink: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """确定性生成 count 份简历。use_llm=True 时对缓存未命中的简历实际调 LLM 并写缓存。

    request_sink 非 None 时，把每份简历的自由文本请求收集进去（供并发 warm 用），
    不影响返回的简历本身。
    """
    rng = random.Random(seed)
    n_hard = round(count * 0.12)
    n_normal = count - n_hard
    plan = _weighted_plan(personas.FAMILY_PROFILES, n_normal) + _weighted_plan(personas.HARD_NEGATIVE_PROFILES, n_hard)
    rng.shuffle(plan)

    docs: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for i, fam in enumerate(plan, start=1):
        profile = {**personas.ALL_PROFILES[fam], "_family": fam}
        docs.append(_build_one(rng, fam, profile, i, used_names, use_llm=use_llm, request_sink=request_sink))
    return docs


# --- 质量自检统计（供测试 / main 使用）------------------------------------
def _quality_stats(docs: list[dict[str, Any]]) -> dict[str, Any]:
    names = [doc["candidate"]["name"] for doc in docs]
    companies = [
        item["company"]
        for doc in docs
        for item in doc.get("internships") or []
        if item.get("company")
    ]
    project_signatures = [
        (item.get("name"), item.get("description"), item.get("responsibility"))
        for doc in docs
        for item in doc.get("projects") or []
    ]
    internship_signatures = [
        (item.get("company"), item.get("department"), item.get("title"), item.get("description"))
        for doc in docs
        for item in doc.get("internships") or []
    ]
    project_counts = Counter(project_signatures)
    internship_counts = Counter(internship_signatures)
    fake_companies = sorted({c for c in companies if "某" in c})
    return {
        "unique_names": len(set(names)),
        "total_names": len(names),
        "unique_companies": len(set(companies)),
        "total_internships": len(companies),
        "fake_companies": fake_companies,
        "duplicate_project_signatures": sum(n - 1 for n in project_counts.values() if n > 1),
        "duplicate_internship_signatures": sum(n - 1 for n in internship_counts.values() if n > 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成贴合真实字段的模拟简历 JSONL")
    parser.add_argument("-o", "--output", default="data/ai_generated.jsonl")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--warm", action="store_true", help="对缓存未命中的简历并发调用 LLM 生成自由文本并写缓存")
    parser.add_argument("--warm-workers", type=int, default=6, help="warm 时的并发数")
    args = parser.parse_args()

    if args.warm:
        # 两段式 warm：先确定性收集全部文本请求（走回退、零网络、快），
        # 再并发把缓存填满，最后从暖缓存组装 —— 比逐份串行调用快很多。
        sink: list[dict[str, Any]] = []
        generate(args.count, args.seed, use_llm=False, request_sink=sink)
        stats_warm = narrative.warm_many(sink, max_workers=args.warm_workers)
        print(f"warm 完成：{stats_warm}")

    docs = generate(args.count, args.seed, use_llm=False)
    stats = _quality_stats(docs)
    assert stats["unique_names"] == stats["total_names"], "候选人姓名重复过多"
    assert not stats["fake_companies"], f"存在占位公司名: {stats['fake_companies']}"
    if stats["total_internships"]:
        assert stats["unique_companies"] >= min(20, stats["total_internships"]), "真实公司池多样性不足"
    assert stats["duplicate_project_signatures"] == 0, "项目经历存在整段重复"
    assert stats["duplicate_internship_signatures"] == 0, "实习经历存在整段重复"

    families: dict[str, int] = {}
    cov = {"languages": 0, "awards": 0, "offer_can_intern": 0, "internships": 0}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            families[doc["_family"]] = families.get(doc["_family"], 0) + 1
            if doc["languages"].get("english_exam_score"):
                cov["languages"] += 1
            if doc["awards"]:
                cov["awards"] += 1
            if doc["offer_internship"].get("can_intern") == "是":
                cov["offer_can_intern"] += 1
            if doc["internships"]:
                cov["internships"] += 1
            doc.pop("_family", None)
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    ids = [d["resume_id"] for d in docs]
    assert len(ids) == len(set(ids)) == args.count, "resume_id 不唯一或数量不符"
    print(f"已写出 {len(docs)} 份到 {out_path}" + ("（LLM warm）" if args.warm else "（缓存/回退）"))
    print("岗位族分布：")
    for fam, n in sorted(families.items(), key=lambda kv: -kv[1]):
        tag = " [hard-neg]" if fam.startswith("hn_") else ""
        print(f"  {fam:20} {n:>3}{tag}")
    print("字段覆盖率：")
    for k, n in cov.items():
        print(f"  {k:20} {n}/{len(docs)} ({n / len(docs):.0%})")
    print("多样性自检：")
    print(f"  unique_names          {stats['unique_names']}/{stats['total_names']}")
    print(f"  unique_companies      {stats['unique_companies']}/{stats['total_internships']}")
    print(f"  duplicate_projects    {stats['duplicate_project_signatures']}")
    print(f"  duplicate_internships {stats['duplicate_internship_signatures']}")
