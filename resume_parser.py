from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup


PARSER_VERSION = "html-doc-v1"

CHARSET_RE = re.compile(br"charset\s*=\s*[\"']?([A-Za-z0-9_\-]+)", re.I)
FILENAME_RE = re.compile(
    r"^(?P<company>.+)-(?P<position>.+)\((?P<position_code>[A-Za-z]\d+)\)-"
    r"(?P<name>.+)\((?P<candidate_no>\d+)\)\.doc$"
)

SECTION_MAP = {
    "个人信息": "personal_info",
    "教育经历": "education",
    "实习经历": "internships",
    "项目经验": "projects",
    "语言能力": "languages",
    "IT技能": "it_skills",
    "奖励、活动": "awards",
    "奖励活动": "awards",
    "接受offer之后可实习信息": "offer_internship",
    "上传简历": "uploaded_resume",
    "期望工作城市": "expected_work_city",
}

PERSONAL_FIELD_MAP = {
    "姓名": "name",
    "性别": "gender",
    "出生日期": "birth_date",
    "民族": "ethnicity",
    "国籍": "nationality",
    "证件信息": "identity_info",
    "政治面貌": "political_status",
    "大学入学前户口所在城市": "pre_college_residence_city",
    "目前所在城市": "current_city",
    "最高（在读）学位": "highest_degree",
    "最高(在读)学位": "highest_degree",
    "毕业时间": "graduation_date",
    "毕业（在读）院校": "school",
    "毕业(在读)院校": "school",
    "专业": "major",
    "移动电话": "phone",
    "紧急联系电话": "emergency_phone",
    "电子邮箱": "email",
    "是否接受调剂": "accept_transfer",
    "可面试城市": "interview_city",
    "从哪里知道招聘信息的": "recruiting_source",
    "内推码": "referral_code",
}

EDUCATION_FIELD_MAP = {
    "开始日期": "start_date",
    "取得毕业证时间": "end_date",
    "学校": "school",
    "学院": "college",
    "专业": "major",
    "学历": "education_level",
    "学位": "degree",
    "主修课程": "courses",
    "研究方向": "research_direction",
    "实验室名称": "lab_name",
    "导师姓名": "advisor_name",
    "导师联系方式": "advisor_contact",
    "论文发表等级": "paper_level",
    "GitHub": "github",
}

INTERNSHIP_FIELD_MAP = {
    "开始日期": "start_date",
    "结束日期": "end_date",
    "企业名称": "company",
    "企业性质": "company_type",
    "企业规模": "company_size",
    "所在部门": "department",
    "职位名称": "title",
    "工作性质": "work_type",
    "工作描述": "description",
}

PROJECT_FIELD_MAP = {
    "开始日期": "start_date",
    "结束日期": "end_date",
    "项目名称": "name",
    "项目描述": "description",
    "项目职责": "responsibility",
}

IT_SKILL_FIELD_MAP = {
    "IT技能": "skill_name",
    "使用时间": "duration",
    "熟练程度": "proficiency",
    "主要编程语言": "primary_languages",
    "其它编程语言": "other_languages",
}

LANGUAGE_FIELD_MAP = {
    "英语等级考试成绩": "english_exam_score",
    "英语口语水平": "english_spoken_level",
}

AWARD_FIELD_MAP = {
    "是否有参赛获奖经历": "has_award",
    "参赛奖项": "name",
    "参赛奖项级别": "level",
    "校内活动描述": "description",
}

OFFER_INTERNSHIP_FIELD_MAP = {
    "毕业后意向": "post_graduation_intention",
    "是否可以实习": "can_intern",
    "可开始工作日期": "available_start_date",
    "每周可实习工作日天数": "weekly_workdays",
    "可实习周期": "internship_period",
}

UPLOAD_FIELD_MAP = {
    "上传中文简历": "chinese_resume",
}


class ResumeParseError(Exception):
    """Raised when the file is not an HTML-based .doc resume."""


def parse_resume_file(path: str | Path) -> dict[str, Any]:
    resume_path = Path(path)
    raw_bytes = resume_path.read_bytes()
    html, encoding = _decode_html_doc(raw_bytes)
    soup = BeautifulSoup(html, "html.parser")
    _drop_non_content_nodes(soup)

    raw_text = _extract_raw_text(soup)
    filename_info = _parse_filename(resume_path.name)
    application_header = _parse_application_header(raw_text)
    section_pairs = _extract_section_pairs(soup)

    personal_raw = _pairs_to_dict(section_pairs.get("personal_info", []))
    candidate = _map_flat_fields(personal_raw, PERSONAL_FIELD_MAP)
    _normalize_candidate(candidate, filename_info)

    education = _map_repeating_section(
        section_pairs.get("education", []),
        start_keys={"开始日期"},
        field_map=EDUCATION_FIELD_MAP,
    )
    internships = _map_repeating_section(
        section_pairs.get("internships", []),
        start_keys={"开始日期"},
        field_map=INTERNSHIP_FIELD_MAP,
    )
    projects = _map_repeating_section(
        section_pairs.get("projects", []),
        start_keys={"开始日期"},
        field_map=PROJECT_FIELD_MAP,
    )
    it_skill_items = _map_repeating_section(
        section_pairs.get("it_skills", []),
        start_keys={"IT技能"},
        field_map=IT_SKILL_FIELD_MAP,
    )

    language_raw = _pairs_to_dict(section_pairs.get("languages", []))
    languages = _map_flat_fields(language_raw, LANGUAGE_FIELD_MAP)

    awards = _map_repeating_section(
        section_pairs.get("awards", []),
        start_keys={"是否有参赛获奖经历"},
        field_map=AWARD_FIELD_MAP,
    )

    offer_raw = _pairs_to_dict(section_pairs.get("offer_internship", []))
    offer_internship = _map_flat_fields(offer_raw, OFFER_INTERNSHIP_FIELD_MAP)

    uploaded_resume_raw = _pairs_to_dict(section_pairs.get("uploaded_resume", []))
    uploaded_resume = _map_flat_fields(uploaded_resume_raw, UPLOAD_FIELD_MAP)

    expected_city_raw = _pairs_to_dict(section_pairs.get("expected_work_city", []))
    expected_work_cities = _split_values(expected_city_raw.get("期望工作城市"))

    application = _build_application(
        filename_info=filename_info,
        header=application_header,
        expected_work_cities=expected_work_cities,
    )
    resume_id = _resume_id(raw_bytes, filename_info, application)

    section_text = _build_section_text(section_pairs)
    skills = _collect_skills(it_skill_items)

    return {
        "resume_id": resume_id,
        "parse_status": "ok",
        "parse_errors": [],
        "parser_version": PARSER_VERSION,
        "file": _file_metadata(resume_path, raw_bytes, encoding),
        "application": application,
        "candidate": candidate,
        "education": education,
        "internships": internships,
        "projects": projects,
        "languages": languages,
        "it_skill_items": it_skill_items,
        "skills": skills,
        "awards": awards,
        "offer_internship": offer_internship,
        "uploaded_resume": uploaded_resume,
        "raw_sections": {
            section: _pairs_to_dict(pairs) for section, pairs in section_pairs.items()
        },
        "section_text": section_text,
        "raw_text": raw_text,
    }


def parse_resume_batch(paths: list[str | Path]) -> list[dict[str, Any]]:
    docs = []
    for path in paths:
        try:
            docs.append(parse_resume_file(path))
        except Exception as exc:
            resume_path = Path(path)
            raw_bytes = resume_path.read_bytes() if resume_path.exists() else b""
            docs.append(
                {
                    "resume_id": _stable_resume_id(raw_bytes)
                    if raw_bytes
                    else str(resume_path),
                    "parse_status": "failed",
                    "parse_errors": [str(exc)],
                    "parser_version": PARSER_VERSION,
                    "file": {
                        "path": str(resume_path.resolve()),
                        "name": resume_path.name,
                    },
                }
            )
    return docs


def discover_doc_files(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root] if _is_resume_doc_file(root) else []
    return sorted(item for item in root.rglob("*.doc") if _is_resume_doc_file(item))


def _is_resume_doc_file(path: Path) -> bool:
    name = path.name
    return path.is_file() and not name.startswith(".") and not name.startswith("~")


def _decode_html_doc(raw_bytes: bytes) -> tuple[str, str]:
    lower = raw_bytes[:4096].lower()
    if b"<html" not in lower and b"<!doctype html" not in lower:
        raise ResumeParseError("only HTML-based .doc files are supported")

    declared = None
    match = CHARSET_RE.search(raw_bytes[:4096])
    if match:
        declared = match.group(1).decode("ascii", errors="ignore")

    candidates = []
    if declared:
        candidates.append(_normalize_encoding_name(declared))
    candidates.extend(["utf-8-sig", "utf-8", "gb18030"])

    tried = set()
    for encoding in candidates:
        if encoding in tried:
            continue
        tried.add(encoding)
        try:
            return raw_bytes.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise ResumeParseError("could not decode HTML .doc as utf-8 or gb18030")


def _normalize_encoding_name(name: str) -> str:
    lowered = name.strip().lower()
    if lowered in {"gb2312", "gbk", "gb18030"}:
        return "gb18030"
    if lowered in {"utf8", "utf-8"}:
        return "utf-8"
    return lowered


def _drop_non_content_nodes(soup: BeautifulSoup) -> None:
    for tag in soup(["script", "style"]):
        tag.decompose()


def _extract_raw_text(soup: BeautifulSoup) -> str:
    lines = []
    for line in soup.get_text("\n", strip=True).splitlines():
        cleaned = _clean_text(line)
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def _extract_section_pairs(soup: BeautifulSoup) -> dict[str, list[tuple[str, str | None]]]:
    sections: dict[str, list[tuple[str, str | None]]] = OrderedDict()
    current_section: str | None = None

    for row in soup.find_all("tr"):
        cells = row.find_all(["td", "th"], recursive=False)
        if not cells:
            continue

        cell_texts = [_clean_text(cell.get_text(" ", strip=True)) for cell in cells]
        row_text = _clean_text(" ".join(text for text in cell_texts if text))
        section = _canonical_section(row_text)
        if section:
            current_section = section
            sections.setdefault(current_section, [])
            continue

        if current_section and len(cells) >= 2:
            key = _clean_label(cell_texts[0])
            value = _clean_value(" ".join(cell_texts[1:]))
            if key:
                sections.setdefault(current_section, []).append((key, value))

    return sections


def _canonical_section(text: str) -> str | None:
    compact = re.sub(r"\s+", "", text)
    return SECTION_MAP.get(compact)


def _clean_label(text: str | None) -> str:
    if not text:
        return ""
    text = _clean_text(text)
    text = text.strip(" :：")
    text = re.sub(r"\s+", "", text)
    return text


def _clean_value(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = _clean_text(text)
    if not cleaned:
        return None
    return cleaned


def _clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = text.replace("\u3000", " ")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"^\s*[·•]\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _pairs_to_dict(pairs: list[tuple[str, str | None]]) -> dict[str, Any]:
    result: dict[str, Any] = OrderedDict()
    for key, value in pairs:
        if key in result:
            existing = result[key]
            if isinstance(existing, list):
                existing.append(value)
            else:
                result[key] = [existing, value]
        else:
            result[key] = value
    return result


def _map_flat_fields(raw: dict[str, Any], field_map: dict[str, str]) -> dict[str, Any]:
    mapped: dict[str, Any] = {"raw_fields": raw}
    for source_key, target_key in field_map.items():
        value = _normalize_field_value(target_key, raw.get(source_key))
        if target_key not in mapped or mapped[target_key] in (None, "", [], {}):
            mapped[target_key] = value
    return mapped


def _map_repeating_section(
    pairs: list[tuple[str, str | None]],
    start_keys: set[str],
    field_map: dict[str, str],
) -> list[dict[str, Any]]:
    raw_records = _split_repeating_pairs(pairs, start_keys=start_keys)
    records = []
    for raw in raw_records:
        mapped = _map_flat_fields(raw, field_map)
        _annotate_date_raw_fields(mapped, raw, field_map)
        if _has_meaningful_values(mapped):
            records.append(mapped)
    return records


def _split_repeating_pairs(
    pairs: list[tuple[str, str | None]], start_keys: set[str]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] = OrderedDict()

    for key, value in pairs:
        if key in start_keys and current:
            records.append(current)
            current = OrderedDict()

        if key in current:
            existing = current[key]
            if isinstance(existing, list):
                existing.append(value)
            else:
                current[key] = [existing, value]
        else:
            current[key] = value

    if current:
        records.append(current)
    return records


def _has_meaningful_values(record: dict[str, Any]) -> bool:
    for key, value in record.items():
        if key in {"raw_fields", "is_current"}:
            continue
        if isinstance(value, list) and any(item for item in value):
            return True
        if value not in (None, "", [], {}):
            return True
    return False


def _normalize_field_value(key: str, value: Any) -> Any:
    if isinstance(value, list):
        value = [_clean_value(str(item)) for item in value if item]
        return [item for item in value if item] or None
    if value is None:
        return None

    text = _clean_value(str(value))
    if not text:
        return None

    if key in {
        "birth_date",
        "graduation_date",
        "start_date",
        "end_date",
        "available_start_date",
    }:
        return _normalize_date(text)
    if key in {"phone", "emergency_phone"}:
        return re.sub(r"\s+", "", text)
    if key == "email":
        return text.replace(" ", "")
    return text


def _annotate_date_raw_fields(
    mapped: dict[str, Any],
    raw: dict[str, Any],
    field_map: dict[str, str],
) -> None:
    for source_key, target_key in field_map.items():
        if target_key not in {"start_date", "end_date"}:
            continue
        raw_value = raw.get(source_key)
        if isinstance(raw_value, list):
            raw_value = next((item for item in raw_value if item), None)
        raw_text = _clean_value(str(raw_value)) if raw_value else None
        mapped[f"{target_key}_raw"] = raw_text

    end_date_raw = mapped.get("end_date_raw")
    mapped["is_current"] = bool(
        isinstance(end_date_raw, str)
        and end_date_raw.strip().lower() in {"至今", "现在", "当前", "present"}
    )


def _normalize_date(value: str | None) -> str | None:
    if not value:
        return None
    text = _clean_text(value)
    if text in {"至今", "现在", "当前", "present", "Present"}:
        return None

    match = re.search(r"(\d{4})[-/.年](\d{1,2})(?:[-/.月](\d{1,2}))?", text)
    if not match:
        return text

    year = int(match.group(1))
    month = int(match.group(2))
    day = int(match.group(3) or 1)
    try:
        return datetime(year, month, day).date().isoformat()
    except ValueError:
        return text


def _normalize_candidate(candidate: dict[str, Any], filename_info: dict[str, Any]) -> None:
    if not candidate.get("name"):
        candidate["name"] = filename_info.get("candidate_name")
    if not candidate.get("highest_degree"):
        candidate["highest_degree"] = None
    if not candidate.get("school"):
        candidate["school"] = None
    if not candidate.get("major"):
        candidate["major"] = None


def _parse_filename(filename: str) -> dict[str, Any]:
    match = FILENAME_RE.match(filename)
    if not match:
        return {}
    info = match.groupdict()
    return {
        "company": info["company"],
        "position_name": info["position"],
        "position_code": info["position_code"].upper(),
        "candidate_name": info["name"],
        "candidate_no": info["candidate_no"],
    }


def _parse_application_header(raw_text: str) -> dict[str, Any]:
    normalized = _clean_text(raw_text.replace("\n", " "))
    result: dict[str, Any] = {}

    wish_match = re.search(
        r"第\s*(?P<rank>\d+)\s*志愿\s*[-－—]{2}\s*"
        r"(?P<position>.*?)\s*[-－—]{2}\s*"
        r"(?P<company>.*?)(?=\s*(?:个人编号|申请时间|$))",
        normalized,
    )
    if wish_match:
        result["wishes"] = [
            {
                "rank": int(wish_match.group("rank")),
                "position_name": _clean_text(wish_match.group("position")),
                "company": _clean_text(wish_match.group("company")),
            }
        ]

    candidate_no_match = re.search(r"个人编号[:：]\s*([0-9\s]+)", normalized)
    if candidate_no_match:
        result["candidate_no"] = re.sub(r"\s+", "", candidate_no_match.group(1))

    apply_time_match = re.search(r"申请时间[:：]\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})", normalized)
    if apply_time_match:
        result["apply_time"] = _normalize_date(apply_time_match.group(1))

    return result


def _build_application(
    filename_info: dict[str, Any],
    header: dict[str, Any],
    expected_work_cities: list[str],
) -> dict[str, Any]:
    wishes = header.get("wishes") or []
    first_wish = wishes[0] if wishes else {}
    position_name = first_wish.get("position_name") or filename_info.get("position_name")
    company = first_wish.get("company") or filename_info.get("company")

    return {
        "candidate_no": header.get("candidate_no") or filename_info.get("candidate_no"),
        "apply_time": header.get("apply_time"),
        "company": company,
        "position_code": filename_info.get("position_code"),
        "position_name": position_name,
        "wishes": wishes,
        "expected_work_cities": expected_work_cities,
    }


def _split_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        values = value
    else:
        values = re.split(r"[,，、;/；\s]+", str(value))
    return [_clean_text(str(item)) for item in values if _clean_text(str(item))]


def _collect_skills(skill_items: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    skills: list[str] = []

    for item in skill_items:
        for key in ("skill_name", "primary_languages", "other_languages"):
            for value in _split_values(item.get(key)):
                if value and value not in seen:
                    seen.add(value)
                    skills.append(value)
    return skills


def _build_section_text(
    section_pairs: dict[str, list[tuple[str, str | None]]]
) -> dict[str, str]:
    section_text = {}
    for section, pairs in section_pairs.items():
        lines = []
        for key, value in pairs:
            if value:
                lines.append(f"{key}: {value}")
            else:
                lines.append(f"{key}:")
        section_text[section] = "\n".join(lines)
    return section_text


def _stable_resume_id(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


def _resume_id(
    raw_bytes: bytes,
    filename_info: dict[str, Any],
    application: dict[str, Any],
) -> str:
    candidate_no = application.get("candidate_no") or filename_info.get("candidate_no")
    if candidate_no:
        return str(candidate_no)
    return _stable_resume_id(raw_bytes)


def _file_metadata(path: Path, raw_bytes: bytes, encoding: str) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "name": path.name,
        "sha256": _stable_resume_id(raw_bytes),
        "size": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "detected_type": "html_doc",
        "encoding": encoding,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse HTML-based .doc resumes to JSON.")
    parser.add_argument("input", nargs="?", default="data", help="A .doc file or directory.")
    parser.add_argument("-o", "--output", help="Write JSON output to this file.")
    parser.add_argument("--jsonl", action="store_true", help="Emit one JSON document per line.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args()

    docs = parse_resume_batch(discover_doc_files(args.input))
    if args.jsonl:
        output = "\n".join(json.dumps(doc, ensure_ascii=False) for doc in docs)
    else:
        indent = 2 if args.pretty else None
        output = json.dumps(docs, ensure_ascii=False, indent=indent)

    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
