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
DEFAULT_EVIDENCE_INDEX = "resume_evidence_v1"
DEFAULT_EVIDENCE_ALIAS = "resume_evidence_current"
BULK_BATCH_SIZE = 100
REQUEST_TIMEOUT_SECONDS = 90
SECTION_SEMANTIC_CHAR_BUDGET = 512
SKILLS_SEMANTIC_CHAR_BUDGET = 256
PROFILE_LEXICAL_CHAR_BUDGET = 768
SEMANTIC_PROFILE_VERSION = "semantic-profile-v5"
EMBEDDING_NORMALIZED = True
LEGACY_CANDIDATE_VECTOR_FIELDS = (
    "skills_vector",
    "projects_vector",
    "internships_vector",
    "education_vector",
)
EVIDENCE_VECTOR_FIELD = "evidence_vector"
VECTOR_EVIDENCE_SECTION_TYPES = {"skills", "project", "internship", "education"}
OBSOLETE_VECTOR_FIELDS = (
    "semantic_profile_vector",
    "role_vector",
    *LEGACY_CANDIDATE_VECTOR_FIELDS,
)


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
            "index_role": "candidate_profile",
            "semantic_profile_version": SEMANTIC_PROFILE_VERSION,
            "embedding_vector_fields": [],
        },
        "properties": {
            "resume_id": {"type": "keyword"},
            "parse_status": {"type": "keyword"},
            "parser_version": {"type": "keyword"},
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
        }
    },
}


EVIDENCE_INDEX_BODY: dict[str, Any] = {
    "settings": INDEX_BODY["settings"],
    "mappings": {
        "dynamic": False,
        "_meta": {
            "embedding_model_id": MODEL_ID,
            "embedding_vector_dims": VECTOR_DIMS,
            "embedding_normalized": EMBEDDING_NORMALIZED,
            "semantic_profile_version": SEMANTIC_PROFILE_VERSION,
            "embedding_vector_fields": [EVIDENCE_VECTOR_FIELD],
            "vectorized_section_types": sorted(VECTOR_EVIDENCE_SECTION_TYPES),
        },
        "properties": {
            "evidence_id": {"type": "keyword"},
            "resume_id": {"type": "keyword"},
            "section_type": {"type": "keyword"},
            "ordinal": {"type": "integer"},
            "title": {
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
            "text": {
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
            "skills_text": {
                "type": "text",
                "analyzer": "resume_search",
            },
            "skills": {"type": "keyword"},
            "candidate": {
                "properties": {
                    "name": {
                        "type": "text",
                        "analyzer": "resume_text",
                        "search_analyzer": "resume_search",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "highest_degree": {"type": "keyword"},
                    "years_experience": {"type": "float"},
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
                    "phone": {"type": "keyword"},
                    "email": {"type": "keyword"},
                }
            },
            "application": {
                "properties": {
                    "candidate_no": {"type": "keyword"},
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
                }
            },
            "embedding": {
                "properties": {
                    "model_id": {"type": "keyword"},
                    "vector_dims": {"type": "integer"},
                    "normalized": {"type": "boolean"},
                    "semantic_profile_version": {"type": "keyword"},
                }
            },
            EVIDENCE_VECTOR_FIELD: _dense_vector_mapping(),
        },
    },
}


def import_resumes(
    data_path: str | Path,
    es_url: str = DEFAULT_ES_URL,
    index: str = DEFAULT_INDEX,
    alias: str = DEFAULT_ALIAS,
    evidence_index: str = DEFAULT_EVIDENCE_INDEX,
    evidence_alias: str = DEFAULT_EVIDENCE_ALIAS,
    recreate: bool = True,
    delete_missing: bool = False,
) -> dict[str, Any]:
    docs = _load_resume_docs(data_path)
    docs = [_enrich_doc(doc) for doc in docs if doc.get("parse_status") == "ok"]
    if recreate and not docs:
        raise RuntimeError("no parsed documents; aborting index rebuild")
    evidence_docs = _build_evidence_docs(docs)
    add_evidence_embeddings(evidence_docs)

    target_index = _versioned_index_name(index) if recreate else _write_target(es_url, index, alias)
    target_evidence_index = (
        _versioned_index_name(evidence_index)
        if recreate
        else _write_target(es_url, evidence_index, evidence_alias)
    )
    if recreate:
        _request("PUT", f"{es_url}/{target_index}", json_body=INDEX_BODY, ok_statuses={200})
        _request(
            "PUT",
            f"{es_url}/{target_evidence_index}",
            json_body=EVIDENCE_INDEX_BODY,
            ok_statuses={200},
        )
    elif not _target_exists(es_url, target_index):
        _request("PUT", f"{es_url}/{target_index}", json_body=INDEX_BODY, ok_statuses={200})
    if not recreate and not _target_exists(es_url, target_evidence_index):
        _request(
            "PUT",
            f"{es_url}/{target_evidence_index}",
            json_body=EVIDENCE_INDEX_BODY,
            ok_statuses={200},
        )

    if docs:
        _bulk_index(es_url, target_index, docs, id_field="resume_id")
        _request("POST", f"{es_url}/{target_index}/_refresh", ok_statuses={200})
    if evidence_docs:
        _bulk_index(es_url, target_evidence_index, evidence_docs, id_field="evidence_id")
        _request("POST", f"{es_url}/{target_evidence_index}/_refresh", ok_statuses={200})

    if delete_missing and not recreate:
        live_ids = {doc["resume_id"] for doc in docs}
        _delete_missing_docs(es_url, target_index, live_ids)
        _delete_missing_evidence_docs(es_url, target_evidence_index, live_ids)

    count = _request("GET", f"{es_url}/{target_index}/_count", ok_statuses={200})["count"]
    if recreate and count != len(docs):
        raise RuntimeError(f"indexed count mismatch: expected {len(docs)}, got {count}")
    evidence_count = _request(
        "GET",
        f"{es_url}/{target_evidence_index}/_count",
        ok_statuses={200},
    )["count"]
    if recreate and evidence_count != len(evidence_docs):
        raise RuntimeError(
            f"indexed evidence count mismatch: expected {len(evidence_docs)}, got {evidence_count}"
        )

    if recreate or not _target_exists(es_url, alias):
        _switch_alias(es_url, target_index, alias)
    if recreate or not _target_exists(es_url, evidence_alias):
        _switch_alias(es_url, target_evidence_index, evidence_alias)

    alias_count = _request("GET", f"{es_url}/{alias}/_count", ok_statuses={200})["count"]
    evidence_alias_count = _request(
        "GET",
        f"{es_url}/{evidence_alias}/_count",
        ok_statuses={200},
    )["count"]
    return {
        "index": target_index,
        "alias": alias,
        "evidence_index": target_evidence_index,
        "evidence_alias": evidence_alias,
        "parsed": len(docs),
        "indexed": count,
        "evidence_indexed": evidence_count,
        "alias_count": alias_count,
        "evidence_alias_count": evidence_alias_count,
    }


def _load_resume_docs(data_path: str | Path) -> list[dict[str, Any]]:
    path = Path(data_path)
    if path.is_file() and path.suffix.lower() == ".jsonl":
        return _load_jsonl_docs(path)
    if path.is_file() and path.suffix.lower() == ".json":
        return _load_json_docs(path)
    return parse_resume_batch(discover_doc_files(path))


def _load_jsonl_docs(path: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_no} must contain a JSON object per line")
        docs.append(item)
    return docs


def _load_json_docs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError(f"{path} must contain a list of JSON objects")
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"{path} must contain a JSON object or a list of JSON objects")


def _enrich_doc(doc: dict[str, Any]) -> dict[str, Any]:
    for field in OBSOLETE_VECTOR_FIELDS:
        doc.pop(field, None)
    doc.pop("search_text", None)
    doc.pop("embedding", None)
    candidate = doc.setdefault("candidate", {})
    years_experience = _estimate_years_experience(doc)
    if years_experience is None:
        candidate.pop("years_experience", None)
    else:
        candidate["years_experience"] = years_experience
    doc["skills_text"] = " ".join(doc.get("skills") or [])
    _drop_index_debug_fields(doc)
    return doc


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
        items.append(_evidence_doc(doc, "skills", 0, "能力标签", skills_text))

    for index, item in enumerate(doc.get("projects") or [], start=1):
        text = _semantic_text(
            doc,
            [
                _profile_line("项目名称", item.get("name")),
                _profile_line("项目描述", item.get("description")),
                _profile_line("项目职责", item.get("responsibility")),
            ],
            SECTION_SEMANTIC_CHAR_BUDGET,
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
            SECTION_SEMANTIC_CHAR_BUDGET,
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
            SECTION_SEMANTIC_CHAR_BUDGET,
        )
        title = " / ".join(value for value in [item.get("school"), item.get("major")] if value)
        if text:
            items.append(_evidence_doc(doc, "education", index, title or "教育经历", text))

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


def _semantic_text(doc: dict[str, Any], lines: list[Any], max_chars: int) -> str:
    cleaned = _strip_semantic_exclusions(_compact_join(lines), _semantic_exclusions(doc))
    return _budgeted_join(cleaned.splitlines(), max_chars)


def _skills_semantic_text(doc: dict[str, Any]) -> str:
    return _semantic_text(
        doc,
        [_profile_line("能力标签", "，".join(doc.get("skills") or []))],
        SKILLS_SEMANTIC_CHAR_BUDGET,
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
        *wish_lines,
        *education_lines,
    ]
    return _budgeted_join(_compact_join(lines).splitlines(), PROFILE_LEXICAL_CHAR_BUDGET)


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


def _bulk_index(
    es_url: str,
    index: str,
    docs: list[dict[str, Any]],
    *,
    id_field: str,
) -> None:
    for start in range(0, len(docs), BULK_BATCH_SIZE):
        batch = docs[start : start + BULK_BATCH_SIZE]
        lines = []
        for doc in batch:
            doc_id = doc.get(id_field)
            if not doc_id:
                raise ValueError(f"document is missing required id field: {id_field}")
            lines.append(
                json.dumps(
                    {"index": {"_index": index, "_id": str(doc_id)}},
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


def _delete_missing_evidence_docs(es_url: str, target: str, live_resume_ids: set[str]) -> None:
    if not live_resume_ids:
        return
    _request(
        "POST",
        f"{es_url}/{target}/_delete_by_query?refresh=true",
        json_body={
            "query": {
                "bool": {
                    "must_not": {
                        "terms": {"resume_id": sorted(live_resume_ids)}
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
    parser.add_argument("--evidence-index", default=DEFAULT_EVIDENCE_INDEX)
    parser.add_argument("--evidence-alias", default=DEFAULT_EVIDENCE_ALIAS)
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
        evidence_index=args.evidence_index,
        evidence_alias=args.evidence_alias,
        recreate=not args.no_recreate,
        delete_missing=args.delete_missing,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
