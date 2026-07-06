"""ES 索引 mapping 与导入相关常量。

集中定义候选人索引 (INDEX_BODY) 与证据切片索引 (EVIDENCE_INDEX_BODY) 的 mapping、
IK 分词器配置、稠密向量字段、默认索引/别名名与批量参数。纯数据，无 IO。
"""
from __future__ import annotations

import os
from typing import Any

from resume_search.infrastructure.embedding_service import MODEL_ID, VECTOR_DIMS

DEFAULT_ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
DEFAULT_INDEX = "resumes_v1"
DEFAULT_ALIAS = "resumes_current"
DEFAULT_EVIDENCE_INDEX = "resume_evidence_v1"
DEFAULT_EVIDENCE_ALIAS = "resume_evidence_current"
BULK_BATCH_SIZE = 100
REQUEST_TIMEOUT_SECONDS = 90
SEMANTIC_PROFILE_VERSION = "semantic-profile-v6"
EMBEDDING_NORMALIZED = True
LEGACY_CANDIDATE_VECTOR_FIELDS = (
    "skills_vector",
    "projects_vector",
    "internships_vector",
    "education_vector",
)
EVIDENCE_VECTOR_FIELD = "evidence_vector"
VECTOR_EVIDENCE_SECTION_TYPES = {"project", "internship"}
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
                    "all_schools": {
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
            "awards": {
                "type": "nested",
                "properties": {
                    "has_award": {"type": "keyword"},
                    "name": {"type": "keyword"},
                    "level": {"type": "keyword"},
                    "description": {"type": "text", "analyzer": "resume_search"},
                    "is_current": {"type": "boolean"},
                },
            },
            "it_skill_items": {
                "type": "nested",
                "properties": {
                    "skill_name": {"type": "keyword"},
                    "duration": {"type": "keyword"},
                    "proficiency": {"type": "keyword"},
                    "primary_languages": {"type": "keyword"},
                    "other_languages": {"type": "keyword"},
                    "is_current": {"type": "boolean"},
                },
            },
            "offer_internship": {
                "properties": {
                    "post_graduation_intention": {"type": "keyword"},
                    "can_intern": {"type": "keyword"},
                    "available_start_date": {"type": "date"},
                    "weekly_workdays": {"type": "keyword"},
                    "internship_period": {"type": "keyword"},
                },
            },
            "uploaded_resume": {
                "properties": {
                    "chinese_resume": {"type": "keyword"},
                },
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
                    "all_schools": {
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


