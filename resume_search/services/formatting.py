"""检索结果格式化：把 ES hit / 证据 debug 转成前端所需的候选人卡片结构。

负责高亮片段抽取、教育摘要、经验年限展示、reranker 文档拼装等展示层逻辑。
不发起 IO；依赖 query_builder 的 _is_dense_retriever 做检索路判定。
"""
from __future__ import annotations

import html
import re
from typing import Any

from resume_search.services.query_builder import _is_dense_retriever


def _format_hit(hit: dict[str, Any], rrf_score: float | None = None) -> dict[str, Any]:
    source = hit.get("_source", {})
    score = round(rrf_score, 4) if rrf_score is not None else round(hit.get("_score") or 0, 3)
    candidate = source.get("candidate", {})
    education = source.get("education") or []
    projects = source.get("projects") or []
    internships = source.get("internships") or []
    highlight = hit.get("highlight", {})
    retrieval_debug = hit.get("_retrieval_debug", {})

    snippets = _highlight_snippets(highlight)
    if not snippets:
        evidence_snippets, evidence_block_count, dense_block_count = _debug_evidence_snippets(
            retrieval_debug
        )
        if evidence_snippets:
            snippets = evidence_snippets
            # 单一事实源：debug 面板两条"命中 N 段证据"必须等于卡片实际渲染的高亮块数。
            # *_support_count 来自检索池（统计召回的切片数），而切片共享去规范化的
            # skills_text，且向量 top 命中常与词面证据是同一份切片；渲染侧按文本去重后
            # 会折叠这些重复，两条路径天然不一致。此处把上报计数回填为去重后真正展示的
            # 块数，让 debug 与 UI 永远同源。仅在确有对应 debug 数据时覆盖，不动打分数学。
            reconciled = dict(retrieval_debug)
            if retrieval_debug.get("evidence_support_count") is not None:
                reconciled["evidence_support_count"] = evidence_block_count
            if retrieval_debug.get("dense_support_count") is not None:
                reconciled["dense_support_count"] = dense_block_count
            retrieval_debug = reconciled
        else:
            snippets = [_default_snippet(projects, internships)]
    years_experience = candidate.get("years_experience")

    return {
        "id": hit.get("_id"),
        "score": score,
        "candidate": candidate,
        "application": source.get("application", {}),
        "education_summary": _education_summary(candidate, education),
        "project_snippet": _safe_snippet("".join(f'<div style="margin-bottom: 6px;">{s}</div>' for s in snippets)) if snippets else "",
        "skills": source.get("skills", []),
        "years_experience": years_experience,
        "experience_display": _experience_display(years_experience),
        "retrieval_debug": retrieval_debug,
        "source": source,
    }


def _debug_evidence_snippets(debug: dict[str, Any]) -> tuple[list[str], int, int]:
    """从证据/向量 debug 匹配里抽取去重后的高亮块。

    返回 (snippets, evidence_block_count, dense_block_count)：snippets 是最终渲染的
    全部块（词面 + 向量）。两个计数分别是词面证据、向量证据贡献的去重后存活块数——
    它们就是 debug 面板两条"命中 N 段证据"应显示的数字，与 UI 实际展示的块严格一致。
    去重跨两路共享（seen_clean_texts）：向量 top 命中常与词面证据是同一份切片，
    渲染时会被折叠，计数必须随之折叠，否则 label 与卡片再次对不上。
    """
    seen_evidence_ids: set[str] = set()
    seen_clean_texts: set[str] = set()
    snippets: list[str] = []

    import re
    def _clean(s: str) -> str:
        s = re.sub(r'<span class="snippet-label[^>]*>.*?</span>', '', s)
        s = re.sub(r'\[.*?\]', '', s)
        s = re.sub(r'<[^>]+>', '', s)
        return re.sub(r'\W+', '', s)

    def _add_matches(matches: list[dict[str, Any]]) -> int:
        blocks_added = 0
        for item in matches:
            ev_id = item.get("evidence_id")
            snippet = str(item.get("snippet") or "").strip()
            if snippet and ev_id not in seen_evidence_ids:
                sub_snippets = [s.strip() for s in snippet.split('<span class="snippet-sep">|</span>')]
                surviving_subs = []
                for sub in sub_snippets:
                    if not sub:
                        continue
                    clean_text = _clean(sub)
                    is_duplicate = False
                    if clean_text:
                        for seen in seen_clean_texts:
                            if clean_text in seen or seen in clean_text:
                                is_duplicate = True
                                break
                    if not is_duplicate:
                        surviving_subs.append(sub)
                        if clean_text:
                            seen_clean_texts.add(clean_text)
                if surviving_subs:
                    snippets.append(' <span class="snippet-sep">|</span> '.join(surviving_subs))
                    blocks_added += 1
                if ev_id:
                    seen_evidence_ids.add(ev_id)
        return blocks_added

    evidence_block_count = _add_matches(debug.get("evidence_matches") or [])
    dense_block_count = _add_matches(debug.get("dense_matches") or [])
    return snippets, evidence_block_count, dense_block_count


def _highlight_snippets(highlight: dict[str, list[str]]) -> list[str]:
    fields = (
        "application.position_name",
        "candidate.major",
        "candidate.school",
        "skills_text",
        "section_text.internships",
        "section_text.projects",
        "section_text.education",
    )
    snippets: list[str] = []
    seen: set[str] = set()
    for field in fields:
        for fragment in highlight.get(field, []):
            text = str(fragment).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            snippets.append(text)
    return snippets


def _education_summary(candidate: dict[str, Any], education: list[dict[str, Any]]) -> str:
    if education:
        parts: list[str] = []
        for edu in education:
            school = edu.get("school")
            degree = edu.get("degree") or edu.get("education_level")
            major = edu.get("major")
            if school or degree or major:
                parts.append(
                    " / ".join(item for item in [school, degree, major] if item)
                )
        if parts:
            return " · ".join(parts)

    if candidate.get("school") or candidate.get("highest_degree") or candidate.get("major"):
        return " / ".join(
            item
            for item in [
                candidate.get("school"),
                candidate.get("highest_degree"),
                candidate.get("major"),
            ]
            if item
        )
    return "教育信息待补充"


def _default_snippet(projects: list[dict[str, Any]], internships: list[dict[str, Any]]) -> str:
    lines: list[str] = []

    proj_names = [p.get("name") for p in projects if p.get("name")]
    if proj_names:
        lines.append(f"项目经历：{' | '.join(proj_names)}")

    intern_names: list[str] = []
    for intern in internships:
        company = intern.get("company")
        title = intern.get("title")
        if company and title:
            intern_names.append(f"{company} / {title}")
        elif title:
            intern_names.append(title)
        elif company:
            intern_names.append(company)
    if intern_names:
        lines.append(f"实习经历：{' | '.join(intern_names)}")

    if lines:
        text = "\n".join(lines)
        return html.escape(text)
    return "暂无项目摘要"


def _experience_display(years_experience: float | None) -> str:
    if years_experience is None:
        return "无工作经验"
    return f"{years_experience:.1f} 年工作经验"


def _safe_snippet(snippet: str) -> str:
    allowed = (
        snippet.replace("&lt;mark&gt;", "<mark>")
        .replace("&lt;/mark&gt;", "</mark>")
    )
    while "</mark><mark>" in allowed:
        allowed = allowed.replace("</mark><mark>", "")
    return allowed


def _evidence_snippet(hit: dict[str, Any], retriever_name: str | None = None) -> str:
    highlight = hit.get("highlight") or {}
    snippets: list[str] = []
    
    field_labels = {
        "title": "标题",
        "text": "正文",
        "skills_text": "技能词",
    }
    
    import re
    def _clean(s: str) -> str:
        return re.sub(r'\W+', '', re.sub(r'<[^>]+>', '', s))
        
    seen_fragments: set[str] = set()
    
    for field, label in field_labels.items():
        if field in highlight and highlight[field]:
            fragments = [str(item).strip() for item in highlight[field] if str(item).strip()]
            if fragments:
                joined_fragments = " ... ".join(fragments)
                clean_frag = _clean(joined_fragments)
                is_duplicate = False
                for seen in seen_fragments:
                    if clean_frag in seen or seen in clean_frag:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    snippets.append(f'<span class="snippet-label">{label}</span> ' + joined_fragments)
                    seen_fragments.add(clean_frag)
                
    if snippets:
        return _safe_snippet(" <span class=\"snippet-sep\">|</span> ".join(snippets))
    if retriever_name and _is_dense_retriever(retriever_name):
        source = hit.get("_source") or {}
        title = str(source.get("title") or "").strip()
        text = str(source.get("text") or "").strip()
        if title and text.startswith(title):
            raw = text
        else:
            raw = f"{title}：{text}" if title else text
        escaped_raw = html.escape(raw[:100] + ("..." if len(raw) > 100 else ""))
        return f'<span class="snippet-label dense-label">Dense 匹配</span> {escaped_raw}'
        
    # 无高亮的 BM25 命中意味着它只匹配到了候选人的全局属性字段，
    # 返回空串以避免无关的切片文本污染 UI。
    return ""


def _evidence_match_debug(
    hit: dict[str, Any],
    retriever_name: Any,
    rank: int,
) -> dict[str, Any]:
    source = hit.get("_source") or {}
    return {
        "retriever": retriever_name,
        "rank": rank,
        "score": round(float(hit.get("_score") or 0), 4),
        "evidence_id": source.get("evidence_id"),
        "section_type": source.get("section_type"),
        "title": source.get("title"),
        "snippet": _evidence_snippet(hit, retriever_name),
    }


def _append_doc_line(lines: list[str], label: str, value: Any) -> None:
    text = _clean_doc_text(value)
    if text:
        lines.append(f"{label}: {text}")


def _clean_doc_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _strip_html(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value)).strip()


