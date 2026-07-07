"""简历自由文本（项目 / 实习描述）的生成与缓存。

设计目标：让 generate() 默认零网络依赖且可复现，同时允许用 LLM 产出自然、
千人千面的文本。

- 结构化"骨架"由 builder/personas 确定性产出；本模块只负责把每份简历的
  project/internship 文本槽位填成自然语言。
- 每份简历一次 LLM 调用：把完整人物画像喂给模型，一次性生成该人所有项目与实习
  文本，保证篇内连贯、彼此呼应，避免逐段生成的拼接感。
- 结果按"画像内容哈希"缓存到 data/resume_text_cache.json。缓存命中即复现，
  与 seed 一起保证确定性。
- 冷启动（无缓存、不联网）走确定性回退文本，保证测试 / CI / 评测链路可跑。
- Hard-negative 族不喂给 LLM：它们的"沾边但不对"是评测 forbidden@10 的锚点，
  必须受控，故始终用回退模板并带上明确的能力边界说明。
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = PROJECT_ROOT / "data" / "resume_text_cache.json"


def _endpoint() -> tuple[str, str, str, int]:
    """LLM 端点配置（惰性读取，便于测试覆盖 env）。默认复用项目里已跑通的
    OpenAI 兼容 chat 端点（query parser 同款），可用 RESUME_GEN_* 覆盖。"""
    model = os.environ.get("RESUME_GEN_MODEL_ID", os.environ.get("QUERY_PARSER_MODEL_ID", "qwen3.5-flash"))
    url = os.environ.get("RESUME_GEN_API_URL", os.environ.get("QUERY_PARSER_API_URL", ""))
    key = os.environ.get("RESUME_GEN_API_KEY", os.environ.get("QUERY_PARSER_API_KEY", ""))
    timeout = int(os.environ.get("RESUME_GEN_TIMEOUT", "60"))
    return url, key, model, timeout


# --- 缓存 ------------------------------------------------------------------
_CACHE: dict[str, Any] | None = None


def _load_cache() -> dict[str, Any]:
    global _CACHE
    if _CACHE is None:
        if CACHE_PATH.exists():
            try:
                _CACHE = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                _CACHE = {}
        else:
            _CACHE = {}
    return _CACHE


def _save_cache() -> None:
    if _CACHE is None:
        return
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(_CACHE, ensure_ascii=False, indent=1), encoding="utf-8")


def request_key(request: dict[str, Any]) -> str:
    """按画像内容做稳定哈希：画像变了缓存自然失效，无需手动清缓存。"""
    canonical = json.dumps(request, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


# --- 公开入口 --------------------------------------------------------------
def generate_narrative(request: dict[str, Any], *, use_llm: bool = False) -> dict[str, Any]:
    """返回 {"projects": [{name,description,responsibility}...], "internships": [{description}...]}。

    use_llm=False（默认）：命中缓存用缓存，否则确定性回退，绝不联网、绝不写盘。
    use_llm=True（--warm）：命中缓存用缓存，否则调 LLM 并写缓存；LLM 失败则回退且不缓存。
    """
    cache = _load_cache()
    key = request_key(request)
    if key in cache:
        return cache[key]

    url, api_key, _model, _timeout = _endpoint()
    if use_llm and not request.get("is_hard_negative") and url and api_key:
        try:
            result = _llm_narrative_checked(request)
            cache[key] = result
            _save_cache()
            return result
        except Exception as exc:  # noqa: BLE001 - 任何失败都回退，不阻塞生成
            print(f"[narrative] LLM 生成失败，回退模板: {exc}")

    return _fallback_narrative(request)


def cache_hit(request: dict[str, Any]) -> bool:
    return request_key(request) in _load_cache()


def warm_many(requests: list[dict[str, Any]], *, max_workers: int = 6) -> dict[str, int]:
    """并发为一批简历的自由文本填充缓存（跳过已缓存与 hard-negative）。

    LLM 调用并发执行，缓存整体写盘一次；单条失败只记为 failed，稍后由回退兜底。
    返回 {cached, warmed, failed, skipped} 计数。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache = _load_cache()
    url, api_key, _model, _timeout = _endpoint()
    stats = {"cached": 0, "warmed": 0, "failed": 0, "skipped": 0}

    pending: dict[str, dict[str, Any]] = {}
    for req in requests:
        key = request_key(req)
        if key in cache:
            stats["cached"] += 1
        elif req.get("is_hard_negative") or not url or not api_key:
            stats["skipped"] += 1
        else:
            pending.setdefault(key, req)

    if not pending:
        return stats

    total = len(pending)
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_llm_narrative_checked, req): key for key, req in pending.items()}
        for fut in as_completed(futures):
            key = futures[fut]
            done += 1
            try:
                result = fut.result()
                cache[key] = result
                stats["warmed"] += 1
            except Exception as exc:  # noqa: BLE001 - 单条失败不阻塞整体
                stats["failed"] += 1
                if stats["failed"] <= 5:
                    print(f"[narrative] warm 失败({key}): {exc}")
            if done % 20 == 0 or done == total:
                print(f"[narrative] warm 进度 {done}/{total}")
                _save_cache()
    _save_cache()
    return stats


# --- LLM 路径 --------------------------------------------------------------
def _system_prompt() -> str:
    return (
        "你是一名资深 HR，正在按候选人的真实背景撰写中文简历里的项目经历与实习描述。"
        "要求：\n"
        "1. 文本要具体、专业、可信，体现真实的技术细节、规模数字与量化结果，像真人写的简历，"
        "不要出现明显的模板腔或 AI 腔。\n"
        "2. 紧扣候选人的方向、技能栈与学历层次；不同项目之间要有区分度，措辞不要重复。\n"
        "3. 不要在文本里出现公司名、学校名、城市名或候选人姓名（这些由系统单独维护）。\n"
        "4. 项目职责写成 '1. …；2. …；3. …' 的分条形式，动词开头，落到具体做法与产出。\n"
        "5. 只输出 JSON，不要多余解释。"
    )


def _user_prompt(request: dict[str, Any]) -> str:
    lines = [
        f"候选人方向：{request['domain_label']}",
        f"最高学历：{request['degree']}",
        f"投递岗位：{request['position']}",
        f"技能栈：{'、'.join(request['skills'])}",
    ]
    if request.get("research_direction"):
        lines.append(f"研究方向：{request['research_direction']}")
    themes = request.get("project_themes") or []
    interns = request.get("internships") or []
    lines.append("")
    lines.append(f"请为以下 {len(themes)} 个项目主题各写一段项目经历：")
    for i, theme in enumerate(themes, 1):
        lines.append(f"  项目{i}主题：{theme}")
    if interns:
        lines.append(f"再为以下 {len(interns)} 段实习各写一段工作描述（60~120 字，突出个人贡献与成果）：")
        for i, it in enumerate(interns, 1):
            lines.append(f"  实习{i}：部门『{it['department']}』岗位『{it['title']}』")
    lines.append("")
    lines.append(
        "输出 JSON，结构：\n"
        '{"projects": [{"name": "项目名称(可与主题不同,更像真实项目名)", '
        '"description": "项目背景与目标,60~120字", '
        '"responsibility": "个人职责,分条,80~160字"}], '
        '"internships": [{"description": "实习工作描述"}]}'
    )
    return "\n".join(lines)


def _llm_narrative(request: dict[str, Any]) -> dict[str, Any]:
    import requests

    url, api_key, model, timeout = _endpoint()
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": _user_prompt(request)},
        ],
        "response_format": {"type": "json_object"},
        "enable_thinking": False,
        "thinking": {"type": "disabled"},
        "temperature": 0.9,
        "top_p": 0.95,
        "max_tokens": 2000,
        "stream": False,
    }
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"empty LLM content: {data}")
    return json.loads(_strip_json_fence(content))


def _strip_json_fence(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _llm_narrative_checked(request: dict[str, Any], attempts: int = 3) -> dict[str, Any]:
    """调 LLM 并校验，失败重试若干次（temperature>0，重试天然会换一份输出）。"""
    last: Exception | None = None
    for _ in range(attempts):
        try:
            result = _llm_narrative(request)
            _validate(result, request)
            return result
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise last if last else RuntimeError("LLM 生成失败")


def _validate(result: dict[str, Any], request: dict[str, Any]) -> None:
    projects = result.get("projects")
    interns = result.get("internships")
    if not isinstance(projects, list) or len(projects) != len(request.get("project_themes") or []):
        raise ValueError("projects 数量与主题不符")
    for p in projects:
        if not isinstance(p, dict) or not (p.get("name") and p.get("description") and p.get("responsibility")):
            raise ValueError("project 字段缺失或结构异常")
    expected_interns = len(request.get("internships") or [])
    if expected_interns:
        if not isinstance(interns, list) or len(interns) != expected_interns:
            raise ValueError("internships 数量不符")
        for it in interns:
            if not isinstance(it, dict) or not it.get("description"):
                raise ValueError("internship description 缺失或结构异常")


# --- 确定性回退 ------------------------------------------------------------
# 冷启动用。文本质量不如 LLM，但确定性、可复现，且比旧版模板多一些结构变化。
_SCALE_BANK = [
    "覆盖 {n}+ 个业务模块", "接入 {n} 类数据源", "服务 {n} 个团队", "日均处理 {n} 万条记录",
    "支撑 {n} 万级请求", "沉淀 {n}+ 条规则/用例", "管理 {n}+ 个节点", "梳理 {n}+ 个需求项",
]
_OUTCOME_BANK = [
    "将处理耗时降低 {p}%", "把人工成本压降 {p}%", "将关键指标提升 {p}%", "把错误率下降 {p}%",
    "使交付周期缩短 {p}%", "将稳定性提升到 99.9% 以上", "把误报率降低 {p}%",
]
_ACTION_BANK = [
    "完成整体方案设计与技术选型", "搭建核心链路并推动上线", "补齐监控、告警与复盘机制",
    "编写自动化脚本提升效率", "梳理关键风险点并给出加固建议", "沉淀可复用的组件与文档",
    "主导性能剖析与瓶颈优化", "对接上下游系统完成联调",
]


def _fallback_narrative(request: dict[str, Any]) -> dict[str, Any]:
    rng = random.Random(int(request_key(request), 16) % (2**32))
    skills = request.get("skills") or ["工程实践"]
    neg = request.get("negative_note")
    domain_label = request["domain_label"]
    # 名称后缀用去掉括注的干净方向名，避免出现嵌套括号。
    clean_label = re.sub(r"[（(].*?[）)]", "", domain_label).strip(" /")

    projects = []
    for theme in request.get("project_themes") or []:
        n = rng.choice([3, 4, 5, 6, 8, 10, 12, 20, 30, 50])
        p = rng.choice([12, 15, 18, 20, 22, 25, 28, 30, 35, 40])
        scale = rng.choice(_SCALE_BANK).format(n=n)
        outcome = rng.choice(_OUTCOME_BANK).format(p=p)
        acts = rng.sample(_ACTION_BANK, k=3)
        skill_hint = "、".join(rng.sample(skills, k=min(2, len(skills))))
        desc = f"面向{domain_label}方向的{theme}，{scale}。"
        if neg:
            desc += f"（{neg}）"
        resp = (
            f"1. {acts[0]}；2. 基于 {skill_hint} {acts[1]}；"
            f"3. {acts[2]}，{outcome}。"
        )
        # 项目名做一点变化，避免与主题一字不差。
        name = theme if rng.random() < 0.6 else f"{theme}（{clean_label}）"
        projects.append({"name": name, "description": desc, "responsibility": resp})

    internships = []
    for it in request.get("internships") or []:
        p = rng.choice([15, 20, 25, 28, 30, 35])
        outcome = rng.choice(_OUTCOME_BANK).format(p=p)
        act = rng.choice(_ACTION_BANK)
        skill_hint = "、".join(rng.sample(skills, k=min(2, len(skills))))
        desc = f"在{it['title']}岗位参与{request['domain_label']}相关工作，使用 {skill_hint} {act}，{outcome}。"
        if neg:
            desc += f"（{neg}）"
        internships.append({"description": desc})

    return {"projects": projects, "internships": internships}
