from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import requests

from embedding_service import MODEL_ID, VECTOR_DIMS, encode_batch
from resume_parser import discover_doc_files, parse_resume_batch


DEFAULT_ES_URL = "http://localhost:9200"
DEFAULT_INDEX = "resumes_v1"
DEFAULT_ALIAS = "resumes_current"
BULK_BATCH_SIZE = 100
REQUEST_TIMEOUT_SECONDS = 90
SEMANTIC_PROFILE_CHAR_BUDGET = 512
SEMANTIC_PROFILE_VERSION = "semantic-profile-v2"
EMBEDDING_NORMALIZED = True
VECTOR_FIELDS = ("semantic_profile_vector",)


def _dense_vector_mapping() -> dict[str, Any]:
    return {
        "type": "dense_vector",
        "dims": VECTOR_DIMS,
        "similarity": "cosine",
        "index": True,
        "index_options": {
            "type": "hnsw",
            "m": 32,
            "ef_construction": 300,
        },
    }


INDEX_BODY: dict[str, Any] = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "resume_text": {
                    "type": "custom",
                    "tokenizer": "ik_max_word",
                    "filter": ["lowercase"],
                },
                "resume_search": {
                    "type": "custom",
                    "tokenizer": "ik_smart",
                    "filter": ["lowercase"],
                },
            },
        },
    },
    "mappings": {
        "dynamic": False,
        "_meta": {
            "embedding_model_id": MODEL_ID,
            "embedding_vector_dims": VECTOR_DIMS,
            "embedding_normalized": EMBEDDING_NORMALIZED,
            "semantic_profile_version": SEMANTIC_PROFILE_VERSION,
        },
        "properties": {
            "resume_id": {"type": "keyword"},
            "parse_status": {"type": "keyword"},
            "parser_version": {"type": "keyword"},
            "embedding": {
                "properties": {
                    "model_id": {"type": "keyword"},
                    "vector_dims": {"type": "integer"},
                    "normalized": {"type": "boolean"},
                    "semantic_profile_version": {"type": "keyword"},
                }
            },
            "file": {
                "properties": {
                    "path": {"type": "keyword"},
                    "name": {"type": "keyword"},
                    "sha256": {"type": "keyword"},
                    "size": {"type": "long"},
                    "mtime": {"type": "date"},
                    "detected_type": {"type": "keyword"},
                    "encoding": {"type": "keyword"},
                }
            },
            "application": {
                "properties": {
                    "candidate_no": {"type": "keyword"},
                    "apply_time": {"type": "date"},
                    "company": {"type": "keyword"},
                    "position_code": {"type": "keyword"},
                    "position_name": {
                        "type": "text",
                        "analyzer": "resume_text",
                        "search_analyzer": "resume_search",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "phrase": {
                                "type": "text",
                                "analyzer": "resume_search",
                                "search_analyzer": "resume_search",
                            },
                        },
                    },
                    "expected_work_cities": {"type": "keyword"},
                    "wishes": {
                        "type": "nested",
                        "properties": {
                            "rank": {"type": "integer"},
                            "position_name": {
                                "type": "text",
                                "analyzer": "resume_text",
                                "search_analyzer": "resume_search",
                                "fields": {
                                    "phrase": {
                                        "type": "text",
                                        "analyzer": "resume_search",
                                        "search_analyzer": "resume_search",
                                    },
                                },
                            },
                            "company": {"type": "keyword"},
                        },
                    },
                }
            },
            "candidate": {
                "properties": {
                    "name": {
                        "type": "text",
                        "analyzer": "resume_text",
                        "search_analyzer": "resume_search",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "gender": {"type": "keyword"},
                    "birth_date": {"type": "date"},
                    "current_city": {"type": "keyword"},
                    "highest_degree": {"type": "keyword"},
                    "graduation_date": {"type": "date"},
                    "school": {
                        "type": "text",
                        "analyzer": "resume_text",
                        "search_analyzer": "resume_search",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "phrase": {
                                "type": "text",
                                "analyzer": "resume_search",
                                "search_analyzer": "resume_search",
                            },
                        },
                    },
                    "major": {
                        "type": "text",
                        "analyzer": "resume_text",
                        "search_analyzer": "resume_search",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "phrase": {
                                "type": "text",
                                "analyzer": "resume_search",
                                "search_analyzer": "resume_search",
                            },
                        },
                    },
                    "phone": {"type": "keyword"},
                    "email": {"type": "keyword"},
                    "years_experience": {"type": "float"},
                }
            },
            "education": {
                "type": "nested",
                "properties": {
                    "start_date": {"type": "date"},
                    "end_date": {"type": "date"},
                    "start_date_raw": {"type": "keyword"},
                    "end_date_raw": {"type": "keyword"},
                    "is_current": {"type": "boolean"},
                    "school": {
                        "type": "text",
                        "analyzer": "resume_text",
                        "search_analyzer": "resume_search",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "phrase": {
                                "type": "text",
                                "analyzer": "resume_search",
                                "search_analyzer": "resume_search",
                            },
                        },
                    },
                    "college": {
                        "type": "text",
                        "analyzer": "resume_text",
                        "search_analyzer": "resume_search",
                        "fields": {
                            "phrase": {
                                "type": "text",
                                "analyzer": "resume_search",
                                "search_analyzer": "resume_search",
                            },
                        },
                    },
                    "major": {
                        "type": "text",
                        "analyzer": "resume_text",
                        "search_analyzer": "resume_search",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "phrase": {
                                "type": "text",
                                "analyzer": "resume_search",
                                "search_analyzer": "resume_search",
                            },
                        },
                    },
                    "education_level": {"type": "keyword"},
                    "degree": {"type": "keyword"},
                    "research_direction": {
                        "type": "text",
                        "analyzer": "resume_text",
                        "search_analyzer": "resume_search",
                        "fields": {
                            "phrase": {
                                "type": "text",
                                "analyzer": "resume_search",
                                "search_analyzer": "resume_search",
                            },
                        },
                    },
                    "lab_name": {
                        "type": "text",
                        "analyzer": "resume_text",
                        "search_analyzer": "resume_search",
                        "fields": {
                            "phrase": {
                                "type": "text",
                                "analyzer": "resume_search",
                                "search_analyzer": "resume_search",
                            },
                        },
                    },
                    "paper_level": {"type": "keyword"},
                },
            },
            "internships": {
                "type": "nested",
                "properties": {
                    "start_date": {"type": "date"},
                    "end_date": {"type": "date"},
                    "start_date_raw": {"type": "keyword"},
                    "end_date_raw": {"type": "keyword"},
                    "is_current": {"type": "boolean"},
                    "company": {
                        "type": "text",
                        "analyzer": "resume_text",
                        "search_analyzer": "resume_search",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "phrase": {
                                "type": "text",
                                "analyzer": "resume_search",
                                "search_analyzer": "resume_search",
                            },
                        },
                    },
                    "department": {
                        "type": "text",
                        "analyzer": "resume_text",
                        "search_analyzer": "resume_search",
                        "fields": {
                            "phrase": {
                                "type": "text",
                                "analyzer": "resume_search",
                                "search_analyzer": "resume_search",
                            },
                        },
                    },
                    "title": {
                        "type": "text",
                        "analyzer": "resume_text",
                        "search_analyzer": "resume_search",
                        "fields": {
                            "phrase": {
                                "type": "text",
                                "analyzer": "resume_search",
                                "search_analyzer": "resume_search",
                            },
                        },
                    },
                    "work_type": {"type": "keyword"},
                    "description": {
                        "type": "text",
                        "analyzer": "resume_text",
                        "search_analyzer": "resume_search",
                        "fields": {
                            "phrase": {
                                "type": "text",
                                "analyzer": "resume_search",
                                "search_analyzer": "resume_search",
                            },
                        },
                    },
                },
            },
            "projects": {
                "type": "nested",
                "properties": {
                    "start_date": {"type": "date"},
                    "end_date": {"type": "date"},
                    "start_date_raw": {"type": "keyword"},
                    "end_date_raw": {"type": "keyword"},
                    "is_current": {"type": "boolean"},
                    "name": {
                        "type": "text",
                        "analyzer": "resume_text",
                        "search_analyzer": "resume_search",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "phrase": {
                                "type": "text",
                                "analyzer": "resume_search",
                                "search_analyzer": "resume_search",
                            },
                        },
                    },
                    "description": {
                        "type": "text",
                        "analyzer": "resume_text",
                        "search_analyzer": "resume_search",
                        "fields": {
                            "phrase": {
                                "type": "text",
                                "analyzer": "resume_search",
                                "search_analyzer": "resume_search",
                            },
                        },
                    },
                    "responsibility": {
                        "type": "text",
                        "analyzer": "resume_text",
                        "search_analyzer": "resume_search",
                        "fields": {
                            "phrase": {
                                "type": "text",
                                "analyzer": "resume_search",
                                "search_analyzer": "resume_search",
                            },
                        },
                    },
                },
            },
            "skills": {"type": "keyword"},
            "skills_text": {
                "type": "text",
                "analyzer": "resume_search",
            },
            "languages": {
                "properties": {
                    "english_exam_score": {"type": "keyword"},
                    "english_spoken_level": {"type": "keyword"},
                }
            },
            "section_text": {
                "dynamic": False,
                "properties": {
                    "education": {
                        "type": "text",
                        "analyzer": "resume_text",
                        "search_analyzer": "resume_search",
                        "fields": {
                            "phrase": {
                                "type": "text",
                                "analyzer": "resume_search",
                                "search_analyzer": "resume_search",
                            },
                        },
                    },
                    "internships": {
                        "type": "text",
                        "analyzer": "resume_text",
                        "search_analyzer": "resume_search",
                        "fields": {
                            "phrase": {
                                "type": "text",
                                "analyzer": "resume_search",
                                "search_analyzer": "resume_search",
                            },
                        },
                    },
                    "projects": {
                        "type": "text",
                        "analyzer": "resume_text",
                        "search_analyzer": "resume_search",
                        "fields": {
                            "phrase": {
                                "type": "text",
                                "analyzer": "resume_search",
                                "search_analyzer": "resume_search",
                            },
                        },
                    },
                }
            },
            "raw_text": {
                "type": "text",
                "index": False,
            },
            "search_text": {
                "type": "text",
                "index": False,
            },
            "semantic_profile_vector": _dense_vector_mapping(),
        }
    },
}


def import_resumes(
    data_path: str | Path,
    es_url: str = DEFAULT_ES_URL,
    index: str = DEFAULT_INDEX,
    alias: str = DEFAULT_ALIAS,
    recreate: bool = True,
    delete_missing: bool = False,
) -> dict[str, Any]:
    docs = parse_resume_batch(discover_doc_files(data_path))
    docs = [_enrich_doc(doc) for doc in docs if doc.get("parse_status") == "ok"]
    if recreate and not docs:
        raise RuntimeError("no parsed documents; aborting index rebuild")
    add_doc_embeddings(docs)

    target_index = _versioned_index_name(index) if recreate else _write_target(es_url, index, alias)
    if recreate:
        _request("PUT", f"{es_url}/{target_index}", json_body=INDEX_BODY, ok_statuses={200})
    elif not _target_exists(es_url, target_index):
        _request("PUT", f"{es_url}/{target_index}", json_body=INDEX_BODY, ok_statuses={200})

    if docs:
        _bulk_index(es_url, target_index, docs)
        _request("POST", f"{es_url}/{target_index}/_refresh", ok_statuses={200})

    if delete_missing and not recreate:
        _delete_missing_docs(es_url, target_index, {doc["resume_id"] for doc in docs})

    count = _request("GET", f"{es_url}/{target_index}/_count", ok_statuses={200})["count"]
    if recreate and count != len(docs):
        raise RuntimeError(f"indexed count mismatch: expected {len(docs)}, got {count}")

    if recreate or not _target_exists(es_url, alias):
        _switch_alias(es_url, target_index, alias)

    alias_count = _request("GET", f"{es_url}/{alias}/_count", ok_statuses={200})["count"]
    return {
        "index": target_index,
        "alias": alias,
        "parsed": len(docs),
        "indexed": count,
        "alias_count": alias_count,
    }


def _enrich_doc(doc: dict[str, Any]) -> dict[str, Any]:
    candidate = doc.setdefault("candidate", {})
    years_experience = _estimate_years_experience(doc)
    if years_experience is None:
        candidate.pop("years_experience", None)
    else:
        candidate["years_experience"] = years_experience
    doc["skills_text"] = " ".join(doc.get("skills") or [])
    doc["search_text"] = _build_search_text(doc)
    doc["embedding"] = {
        "model_id": MODEL_ID,
        "vector_dims": VECTOR_DIMS,
        "normalized": EMBEDDING_NORMALIZED,
        "semantic_profile_version": SEMANTIC_PROFILE_VERSION,
    }
    _drop_index_debug_fields(doc)
    return doc


def add_doc_embeddings(docs: list[dict[str, Any]]) -> None:
    entries: list[tuple[dict[str, Any], str, str]] = []
    for doc in docs:
        for field, text in _embedding_inputs(doc).items():
            if text.strip():
                entries.append((doc, field, text))

    vectors = encode_batch([text for _, _, text in entries])
    for (doc, field, _), vector in zip(entries, vectors):
        if len(vector) != VECTOR_DIMS:
            raise RuntimeError(
                f"embedding dimension mismatch for {field}: expected {VECTOR_DIMS}, got {len(vector)}"
            )
        doc[field] = vector


def _embedding_inputs(doc: dict[str, Any]) -> dict[str, str]:
    return {"semantic_profile_vector": doc.get("search_text", "")}


def _drop_index_debug_fields(value: Any) -> Any:
    if isinstance(value, dict):
        value.pop("raw_fields", None)
        value.pop("raw_sections", None)
        for item in list(value.values()):
            _drop_index_debug_fields(item)
    elif isinstance(value, list):
        for item in value:
            _drop_index_debug_fields(item)
    return value


def _estimate_years_experience(doc: dict[str, Any]) -> float | None:
    reference_end = _resume_reference_date(doc)
    spans = []
    for item in doc.get("internships", []):
        start = _parse_date(item.get("start_date"))
        end = _parse_date(item.get("end_date")) or reference_end
        if reference_end and end and end > reference_end:
            end = reference_end
        if start and end and end >= start:
            spans.append((start, end))

    if not spans:
        return None
    days = sum((end - start).days for start, end in _merge_spans(spans))
    return round(days / 365, 1)


def _resume_reference_date(doc: dict[str, Any]) -> date:
    application = doc.get("application") or {}
    file_meta = doc.get("file") or {}
    return (
        _parse_date(application.get("apply_time"))
        or _parse_date(file_meta.get("mtime"))
        or date.today()
    )


def _merge_spans(spans: list[tuple[date, date]]) -> list[tuple[date, date]]:
    merged: list[list[date]] = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
            continue
        if end > merged[-1][1]:
            merged[-1][1] = end
    return [(start, end) for start, end in merged]


def _build_search_text(doc: dict[str, Any]) -> str:
    application = doc.get("application") or {}
    candidate = doc.get("candidate") or {}
    lines = [
        _project_semantic_text(doc),
        _internship_semantic_text(doc),
        _profile_line("能力标签", "，".join(doc.get("skills") or [])),
        _education_semantic_text(doc),
        _profile_line("目标岗位", application.get("position_name")),
        _profile_line("专业背景", candidate.get("major")),
    ]
    cleaned = _strip_semantic_exclusions(_compact_join(lines), _semantic_exclusions(doc))
    return _budgeted_join(cleaned.splitlines(), SEMANTIC_PROFILE_CHAR_BUDGET)


def _education_semantic_text(doc: dict[str, Any]) -> str:
    lines: list[str] = []
    for item in doc.get("education") or []:
        lines.extend(
            [
                _profile_line("教育专业", item.get("major")),
                _profile_line("研究方向", item.get("research_direction")),
                _profile_line("实验室方向", item.get("lab_name")),
            ]
        )
    return _compact_join(lines)


def _internship_semantic_text(doc: dict[str, Any]) -> str:
    lines: list[str] = []
    for item in doc.get("internships") or []:
        lines.extend(
            [
                _profile_line("实习部门", item.get("department")),
                _profile_line("实习职位", item.get("title")),
                _profile_line("实习描述", item.get("description")),
            ]
        )
    return _compact_join(lines)


def _project_semantic_text(doc: dict[str, Any]) -> str:
    lines: list[str] = []
    for item in doc.get("projects") or []:
        lines.extend(
            [
                _profile_line("项目名称", item.get("name")),
                _profile_line("项目描述", item.get("description")),
                _profile_line("项目职责", item.get("responsibility")),
            ]
        )
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


def _budgeted_join(lines: list[str], max_chars: int) -> str:
    result: list[str] = []
    used = 0
    for line in _compact_join(lines).splitlines():
        extra = len(line) + (1 if result else 0)
        if used + extra <= max_chars:
            result.append(line)
            used += extra
            continue

        remaining = max_chars - used - (1 if result else 0)
        if remaining > 20:
            result.append(line[: remaining - 3].rstrip() + "...")
        break
    return "\n".join(result)


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


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _versioned_index_name(base_index: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    return f"{base_index}_{stamp}"


def _write_target(es_url: str, index: str, alias: str) -> str:
    if _target_exists(es_url, alias):
        return alias
    return index


def _target_exists(es_url: str, target: str) -> bool:
    response = requests.head(f"{es_url}/{target}", timeout=10)
    return response.status_code == 200


def _bulk_index(es_url: str, index: str, docs: list[dict[str, Any]]) -> None:
    for start in range(0, len(docs), BULK_BATCH_SIZE):
        batch = docs[start : start + BULK_BATCH_SIZE]
        lines = []
        for doc in batch:
            lines.append(
                json.dumps(
                    {"index": {"_index": index, "_id": doc["resume_id"]}},
                    ensure_ascii=False,
                )
            )
            lines.append(json.dumps(doc, ensure_ascii=False))
        response = _request(
            "POST",
            f"{es_url}/_bulk",
            data=("\n".join(lines) + "\n").encode("utf-8"),
            headers={"Content-Type": "application/x-ndjson"},
            ok_statuses={200},
        )
        if response.get("errors"):
            failures = [
                item
                for item in response.get("items", [])
                if item.get("index", {}).get("error")
            ]
            raise RuntimeError(f"bulk import failed: {failures[:3]}")


def _delete_missing_docs(es_url: str, target: str, live_ids: set[str]) -> None:
    if not live_ids:
        return
    _request(
        "POST",
        f"{es_url}/{target}/_delete_by_query?refresh=true",
        json_body={
            "query": {
                "bool": {
                    "must_not": {
                        "ids": {"values": sorted(live_ids)}
                    }
                }
            }
        },
        ok_statuses={200},
    )


def _switch_alias(es_url: str, index: str, alias: str) -> None:
    _request(
        "POST",
        f"{es_url}/_aliases",
        json_body={
            "actions": [
                {"remove": {"index": "*", "alias": alias}},
                {"add": {"index": index, "alias": alias, "is_write_index": True}},
            ]
        },
        ok_statuses={200},
    )


def _request(
    method: str,
    url: str,
    *,
    json_body: dict[str, Any] | None = None,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    ok_statuses: set[int],
) -> dict[str, Any]:
    response = requests.request(
        method,
        url,
        json=json_body,
        data=data,
        headers=headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code not in ok_statuses:
        raise RuntimeError(f"{method} {url} -> {response.status_code}: {response.text[:800]}")
    if response.text:
        return response.json()
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Import parsed resumes into Elasticsearch.")
    parser.add_argument("data_path", nargs="?", default="data")
    parser.add_argument("--es-url", default=DEFAULT_ES_URL)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--alias", default=DEFAULT_ALIAS)
    parser.add_argument("--no-recreate", action="store_true")
    parser.add_argument(
        "--delete-missing",
        action="store_true",
        help="When importing into an existing index, delete documents not present in this import set.",
    )
    args = parser.parse_args()

    result = import_resumes(
        data_path=args.data_path,
        es_url=args.es_url,
        index=args.index,
        alias=args.alias,
        recreate=not args.no_recreate,
        delete_missing=args.delete_missing,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
