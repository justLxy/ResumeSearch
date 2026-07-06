"""证据切片构建与向量化：把整份简历拆成 project/internship/skills/profile 等切片文档。

"证据优先"检索的数据基础——每个可检索片段独立成 doc，附语义文本与（部分类型的）
稠密向量。向量化调用 embedding_service 批量编码。
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any

from resume_search.infrastructure.embedding_service import (
    MODEL_ID,
    VECTOR_DIMS,
    encode_batch,
)

from indexing.enrichment import _collect_all_schools
from indexing.mappings import (
    EMBEDDING_NORMALIZED,
    EVIDENCE_VECTOR_FIELD,
    SEMANTIC_PROFILE_VERSION,
    VECTOR_EVIDENCE_SECTION_TYPES,
)


def add_evidence_embeddings(docs: list[dict[str, Any]]) -> None:
    texts = [str(doc.get("text") or "").strip() for doc in docs]
    entries = [
        (doc, text)
        for doc, text in zip(docs, texts)
        if text and doc.get("section_type") in VECTOR_EVIDENCE_SECTION_TYPES
    ]
    vectors = encode_batch([text for _, text in entries])
    for (doc, _), vector in zip(entries, vectors):
        if len(vector) != VECTOR_DIMS:
            raise RuntimeError(
                f"embedding dimension mismatch for {EVIDENCE_VECTOR_FIELD}: "
                f"expected {VECTOR_DIMS}, got {len(vector)}"
            )
        doc[EVIDENCE_VECTOR_FIELD] = vector


def _build_evidence_docs(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence_docs: list[dict[str, Any]] = []
    for doc in docs:
        evidence_docs.extend(_resume_evidence_docs(doc))
    return evidence_docs


def _resume_evidence_docs(doc: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    resume_id = str(doc.get("resume_id") or "").strip()
    if not resume_id:
        return items

    profile_text = _profile_lexical_text(doc)
    if profile_text:
        items.append(
            _evidence_doc(
                doc,
                "profile",
                0,
                "候选人档案",
                profile_text,
                vector_enabled=False,
            )
        )

    skills_text = _skills_semantic_text(doc)
    if skills_text:
        items.append(
            _evidence_doc(
                doc,
                "skills",
                0,
                "能力标签",
                skills_text,
                vector_enabled=False,
            )
        )

    for index, item in enumerate(doc.get("projects") or [], start=1):
        text = _semantic_text(
            doc,
            [
                _profile_line("项目名称", item.get("name")),
                _profile_line("项目描述", item.get("description")),
                _profile_line("项目职责", item.get("responsibility")),
            ],
        )
        if text:
            items.append(_evidence_doc(doc, "project", index, item.get("name") or "项目经历", text))

    for index, item in enumerate(doc.get("internships") or [], start=1):
        text = _semantic_text(
            doc,
            [
                _profile_line("实习部门", item.get("department")),
                _profile_line("实习职位", item.get("title")),
                _profile_line("实习描述", item.get("description")),
            ],
        )
        title = " / ".join(value for value in [item.get("company"), item.get("title")] if value)
        if text:
            items.append(_evidence_doc(doc, "internship", index, title or "实习经历", text))

    for index, item in enumerate(doc.get("education") or [], start=1):
        text = _semantic_text(
            doc,
            [
                _profile_line("教育专业", item.get("major")),
                _profile_line("研究方向", item.get("research_direction")),
                _profile_line("实验室方向", item.get("lab_name")),
            ],
        )
        title = " / ".join(value for value in [item.get("school"), item.get("major")] if value)
        if text:
            items.append(
                _evidence_doc(
                    doc,
                    "education",
                    index,
                    title or "教育经历",
                    text,
                    vector_enabled=False,
                )
            )

    for index, item in enumerate(doc.get("awards") or [], start=1):
        text = _semantic_text(
            doc,
            [
                _profile_line("获奖名称", item.get("name")),
                _profile_line("获奖级别", item.get("level")),
                _profile_line("获奖描述", item.get("description")),
            ],
        )
        if text:
            items.append(
                _evidence_doc(
                    doc,
                    "awards",
                    index,
                    item.get("name") or "获奖经历",
                    text,
                    vector_enabled=False,
                )
            )

    offer = doc.get("offer_internship") or {}
    offer_lines = []
    if offer.get("post_graduation_intention"):
        offer_lines.append(_profile_line("毕业后意向", offer["post_graduation_intention"]))
    if offer.get("can_intern"):
        offer_lines.append(_profile_line("是否可以实习", offer["can_intern"]))
    if offer.get("available_start_date"):
        offer_lines.append(_profile_line("可开始工作日期", offer["available_start_date"]))
    if offer.get("weekly_workdays"):
        offer_lines.append(_profile_line("每周可实习天数", offer["weekly_workdays"]))
    if offer.get("internship_period"):
        offer_lines.append(_profile_line("可实习周期", offer["internship_period"]))
    offer_text = _compact_join(offer_lines)
    if offer_text:
        items.append(
            _evidence_doc(
                doc,
                "offer",
                0,
                "实习与入职意向",
                offer_text,
                vector_enabled=False,
            )
        )

    return items


def _evidence_doc(
    doc: dict[str, Any],
    section_type: str,
    ordinal: int,
    title: Any,
    text: str,
    *,
    vector_enabled: bool = True,
) -> dict[str, Any]:
    resume_id = str(doc.get("resume_id") or "").strip()
    candidate = doc.get("candidate") or {}
    application = doc.get("application") or {}

    # 从教育经历中提取所有学校（与主索引 candidate.all_schools 保持一致）
    all_schools = _collect_all_schools(doc)
                
    evidence = {
        "evidence_id": f"{resume_id}:{section_type}:{ordinal}",
        "resume_id": resume_id,
        "section_type": section_type,
        "ordinal": ordinal,
        "title": str(title or "").strip() or section_type,
        "text": text,
        "skills_text": " ".join(doc.get("skills") or []),
        "skills": doc.get("skills") or [],
        "candidate": {
            "name": candidate.get("name"),
            "highest_degree": candidate.get("highest_degree"),
            "years_experience": candidate.get("years_experience"),
            "major": candidate.get("major"),
            "school": candidate.get("school"),
            "all_schools": all_schools,
            "phone": candidate.get("phone"),
            "email": candidate.get("email"),
        },
        "application": {
            "candidate_no": application.get("candidate_no"),
            "company": application.get("company"),
            "position_code": application.get("position_code"),
            "position_name": application.get("position_name"),
            "expected_work_cities": application.get("expected_work_cities") or [],
        },
    }
    if vector_enabled:
        evidence["embedding"] = {
            "model_id": MODEL_ID,
            "vector_dims": VECTOR_DIMS,
            "normalized": EMBEDDING_NORMALIZED,
            "semantic_profile_version": SEMANTIC_PROFILE_VERSION,
        }
    return evidence


def _semantic_text(doc: dict[str, Any], lines: list[Any]) -> str:
    cleaned = _strip_semantic_exclusions(_compact_join(lines), _semantic_exclusions(doc))
    return cleaned


def _skills_semantic_text(doc: dict[str, Any]) -> str:
    return _semantic_text(
        doc,
        [_profile_line("能力标签", "，".join(doc.get("skills") or []))],
    )


def _profile_lexical_text(doc: dict[str, Any]) -> str:
    candidate = doc.get("candidate") or {}
    application = doc.get("application") or {}
    wish_lines = [
        _profile_line(
            "志愿",
            " / ".join(
                str(value)
                for value in [item.get("company"), item.get("position_name")]
                if value
            ),
        )
        for item in application.get("wishes") or []
        if isinstance(item, dict)
    ]
    education_lines = [
        _profile_line(
            "教育",
            " / ".join(
                str(value)
                for value in [
                    item.get("school"),
                    item.get("college"),
                    item.get("major"),
                    item.get("education_level"),
                    item.get("degree"),
                ]
                if value
            ),
        )
        for item in doc.get("education") or []
        if isinstance(item, dict)
    ]
    language_lines = []
    languages = doc.get("languages") or {}
    if languages.get("english_exam_score"):
        language_lines.append(_profile_line("英语等级", languages["english_exam_score"]))
    if languages.get("english_spoken_level"):
        language_lines.append(_profile_line("英语口语", languages["english_spoken_level"]))

    lines = [
        _profile_line("候选人编号", application.get("candidate_no")),
        _profile_line("岗位编号", application.get("position_code")),
        _profile_line("候选人姓名", candidate.get("name")),
        _profile_line("手机号", candidate.get("phone")),
        _profile_line("邮箱", candidate.get("email")),
        _profile_line("招聘公司", application.get("company")),
        _profile_line("投递岗位", application.get("position_name")),
        _profile_line("最高学历", candidate.get("highest_degree")),
        _profile_line("毕业院校", candidate.get("school")),
        _profile_line("专业", candidate.get("major")),
        _profile_line("当前城市", candidate.get("current_city")),
        _profile_line("期望工作城市", application.get("expected_work_cities") or []),
        _profile_line("技能标签", "，".join(doc.get("skills") or [])),
        *language_lines,
        *wish_lines,
        *education_lines,
    ]
    return _compact_join(lines)


def _profile_line(label: str, value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        value = "，".join(str(item) for item in value if item)
    text = str(value).strip()
    if not text:
        return ""
    return f"{label}：{text}"


def _semantic_exclusions(doc: dict[str, Any]) -> tuple[str, ...]:
    application = doc.get("application") or {}
    candidate = doc.get("candidate") or {}
    values: list[Any] = [
        application.get("candidate_no"),
        application.get("company"),
        application.get("position_code"),
        candidate.get("name"),
        candidate.get("school"),
        candidate.get("current_city"),
        candidate.get("pre_college_residence_city"),
        candidate.get("interview_city"),
        candidate.get("phone"),
        candidate.get("email"),
        *(application.get("expected_work_cities") or []),
    ]
    for item in application.get("wishes") or []:
        values.append(item.get("company"))
    for item in doc.get("education") or []:
        values.extend([item.get("school"), item.get("college")])
    for item in doc.get("internships") or []:
        values.append(item.get("company"))

    return tuple(
        sorted(
            {
                str(value).strip()
                for value in values
                if value and len(str(value).strip()) >= 2
            },
            key=len,
            reverse=True,
        )
    )


def _strip_semantic_exclusions(text: str, exclusions: tuple[str, ...]) -> str:
    cleaned = text
    for value in exclusions:
        cleaned = cleaned.replace(value, "")
    return _compact_join(cleaned.splitlines())


def _compact_join(chunks: list[Any]) -> str:
    seen: set[str] = set()
    result: list[str] = []
    for chunk in chunks:
        if not chunk:
            continue
        text = str(chunk).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return "\n".join(result)


