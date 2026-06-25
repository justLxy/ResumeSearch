import json
import unittest

from app import (
    BM25_RETRIEVER,
    DENSE_RETRIEVER,
    _build_filters,
    _default_snippet,
    _format_hit,
    _hybrid_total,
    _infer_dense_routes,
    _lexical_query,
    _lexical_total,
    _merge_case_insensitive_skill_buckets,
    _normalize_limit,
    _parse_query_constraints,
    _rrf_merge,
    _use_dense,
)
from import_to_es import (
    EMBEDDING_NORMALIZED,
    SEMANTIC_PROFILE_VERSION,
    VECTOR_DIMS,
    VECTOR_FIELDS,
    INDEX_BODY,
    _embedding_inputs,
    _estimate_years_experience,
)


class SearchLogicTests(unittest.TestCase):
    def test_search_limit_defaults_to_full_result_window(self) -> None:
        self.assertEqual(_normalize_limit(None), 10_000)
        self.assertEqual(_normalize_limit(0), 10_000)
        self.assertEqual(_normalize_limit(20), 20)
        self.assertEqual(_normalize_limit(500), 500)
        self.assertEqual(_normalize_limit(50_000), 10_000)

    def test_experience_uses_apply_time_for_internships_only(self) -> None:
        doc = {
            "application": {"apply_time": "2019-09-11"},
            "internships": [
                {
                    "start_date": "2018-10-11",
                    "end_date": None,
                }
            ],
            "projects": [
                {
                    "start_date": "2018-12-25",
                    "end_date": "2019-08-01",
                },
                {
                    "start_date": "2019-06-01",
                    "end_date": "2019-07-15",
                },
            ],
        }
        self.assertEqual(_estimate_years_experience(doc), 0.9)

    def test_experience_ignores_project_duration_without_internship(self) -> None:
        doc = {
            "application": {"apply_time": "2019-09-05"},
            "projects": [
                {
                    "start_date": "2018-08-17",
                    "end_date": None,
                }
            ],
        }
        self.assertIsNone(_estimate_years_experience(doc))

    def test_default_snippet_lists_project_names(self) -> None:
        snippet = _default_snippet(
            [
                {
                    "start_date": "2018-12-25",
                    "end_date": "2019-08-01",
                    "name": "机器人智能导诊项目",
                    "responsibility": "负责医疗知识库和实体抽取。",
                },
                {
                    "start_date": "2019-06-01",
                    "end_date": "2019-07-15",
                    "name": "医疗问答系统",
                    "description": "负责问句分析与结果排序。",
                },
            ],
            [],
        )
        self.assertIn("医疗问答系统", snippet)
        self.assertIn("机器人智能导诊项目", snippet)
        self.assertNotIn("负责问句分析与结果排序。", snippet)

    def test_format_hit_combines_highlights_from_multiple_fields(self) -> None:
        hit = _hit("candidate-1", "候选人")
        hit["highlight"] = {
            "application.position_name": ["<mark>机器学习</mark>工程师"],
            "section_text.internships": [
                "企业名称: <mark>百度</mark>在线网络技术 职位名称: <mark>机器学习</mark>实习生"
            ],
        }

        formatted = _format_hit(hit)

        self.assertIn("<mark>机器学习</mark>", formatted["project_snippet"])
        self.assertIn("<mark>百度</mark>", formatted["project_snippet"])

    def test_hybrid_merge_allows_dense_only_for_semantic_queries(self) -> None:
        vector_response = _response(
            DENSE_RETRIEVER,
            [
                _hit("vector-only", "向量候选"),
            ],
        )

        self.assertEqual(_hybrid_total([vector_response], allow_dense_only=True), 1)
        self.assertEqual(_rrf_merge([vector_response], 10, allow_dense_only=True)[0]["id"], "vector-only")

    def test_hybrid_merge_blocks_dense_only_for_short_entity_queries(self) -> None:
        bm25_response = _response(
            BM25_RETRIEVER,
            [
                _hit("lexical-1", "词面第一"),
            ],
            total=1,
        )
        vector_response = _response(
            DENSE_RETRIEVER,
            [
                _hit("vector-only", "向量候选"),
            ],
        )

        results = _rrf_merge([bm25_response, vector_response], 10, allow_dense_only=False)

        self.assertFalse(_use_dense("北京大学"))
        self.assertFalse(_use_dense("奇安信集团"))
        self.assertFalse(_use_dense("M20260001"))
        self.assertFalse(_use_dense("A0009"))
        self.assertTrue(_use_dense("自然语言处理"))
        self.assertTrue(_use_dense("推荐召回"))
        self.assertTrue(_use_dense("做过推荐系统召回和 NLP 模型落地的人"))
        self.assertEqual(_hybrid_total([bm25_response, vector_response], allow_dense_only=False), 1)
        self.assertEqual([item["id"] for item in results], ["lexical-1"])

    def test_hybrid_merge_combines_bm25_and_dense_with_rrf(self) -> None:
        bm25_response = _response(
            BM25_RETRIEVER,
            [
                _hit("lexical-1", "词面第一"),
                _hit("lexical-2", "词面第二"),
            ],
            weight=1.0,
            total=7,
        )
        vector_response = _response(
            DENSE_RETRIEVER,
            [
                _hit("lexical-2", "词面第二"),
                _hit("vector-only", "向量候选"),
            ],
        )

        results = _rrf_merge([bm25_response, vector_response], 10, allow_dense_only=True)

        self.assertEqual(_hybrid_total([bm25_response, vector_response], allow_dense_only=True), 3)
        self.assertEqual([item["id"] for item in results], ["lexical-2", "lexical-1", "vector-only"])
        self.assertEqual(results[0]["retrieval_debug"]["retrieval_sources"], [BM25_RETRIEVER, DENSE_RETRIEVER])

    def test_multi_term_coverage_boost_is_reflected_in_final_score(self) -> None:
        bm25_response = _response(
            BM25_RETRIEVER,
            [
                _hit("a-many", "A A A 0", matched_queries=["query_term:0"]),
                _hit("b-many", "0 B B B", matched_queries=["query_term:1"]),
                _hit("both", "A B 0 0", matched_queries=["query_term:0", "query_term:1"]),
            ],
        )
        vector_response = _response(
            DENSE_RETRIEVER,
            [
                _hit("a-many", "A A A 0", score=1.0),
                _hit("zero", "0 0 0 0", score=0.99),
            ],
        )

        results = _rrf_merge(
            [bm25_response, vector_response],
            10,
            allow_dense_only=True,
            query_text="A B",
        )

        self.assertEqual([item["id"] for item in results], ["a-many", "both", "b-many", "zero"])
        self.assertGreaterEqual(results[0]["retrieval_debug"]["rrf_score"], results[1]["retrieval_debug"]["rrf_score"])
        self.assertEqual(results[0]["retrieval_debug"]["term_coverage"], 1)
        self.assertEqual(results[1]["retrieval_debug"]["term_coverage"], 2)

    def test_final_score_orders_phrase_against_dense_rank(self) -> None:
        bm25_response = _response(
            BM25_RETRIEVER,
            [
                _hit("split-and-dense", "计算机 与 科学", matched_queries=["query_term:0"]),
                _hit("phrase", "计算机科学", matched_queries=["lexical_phrase:candidate_major"]),
            ],
        )
        vector_response = _response(
            DENSE_RETRIEVER,
            [
                _hit("split-and-dense", "计算机 与 科学", score=0.9),
            ],
        )

        results = _rrf_merge(
            [bm25_response, vector_response],
            10,
            allow_dense_only=True,
            query_text="计算机科学",
        )

        self.assertEqual([item["id"] for item in results], ["split-and-dense", "phrase"])
        self.assertGreaterEqual(results[0]["retrieval_debug"]["rrf_score"], results[1]["retrieval_debug"]["rrf_score"])
        self.assertEqual(results[1]["retrieval_debug"]["lexical_tier"], 2)

    def test_final_score_can_include_dense_support_for_phrase_hits(self) -> None:
        bm25_response = _response(
            BM25_RETRIEVER,
            [
                _hit("major-phrase", "专业短语", matched_queries=["lexical_phrase:candidate_major"]),
                _hit("weak-field-phrase", "弱字段短语", matched_queries=["lexical_phrase:section_education"]),
            ],
        )
        vector_response = _response(
            DENSE_RETRIEVER,
            [
                _hit("weak-field-phrase", "弱字段短语", score=0.9),
            ],
        )

        results = _rrf_merge(
            [bm25_response, vector_response],
            10,
            allow_dense_only=True,
            query_text="计算机科学",
        )

        self.assertEqual([item["id"] for item in results], ["weak-field-phrase", "major-phrase"])
        self.assertGreaterEqual(results[0]["retrieval_debug"]["rrf_score"], results[1]["retrieval_debug"]["rrf_score"])

    def test_dense_only_hits_need_similarity_gate(self) -> None:
        bm25_response = _response(
            BM25_RETRIEVER,
            [
                _hit("lexical-1", "词面第一"),
            ],
            total=1,
        )
        vector_response = _response(
            DENSE_RETRIEVER,
            [
                _hit("weak-vector-only", "弱语义候选", score=0.79),
                _hit("lexical-1", "词面第一", score=0.78),
            ],
        )

        results = _rrf_merge([bm25_response, vector_response], 10, allow_dense_only=True)

        self.assertEqual(_hybrid_total([bm25_response, vector_response], allow_dense_only=True), 1)
        self.assertEqual([item["id"] for item in results], ["lexical-1"])
        self.assertEqual(results[0]["retrieval_debug"]["dense_rank"], 1)
        self.assertEqual(results[0]["retrieval_debug"]["dense_route_rank"], 2)
        self.assertIsNone(results[0]["retrieval_debug"]["dense_only_threshold"])

    def test_dense_only_gate_keeps_high_confidence_semantic_hits(self) -> None:
        vector_response = _response(
            DENSE_RETRIEVER,
            [
                _hit("vector-1", "强语义候选", score=0.88),
                _hit("vector-2", "近邻语义候选", score=0.865),
                _hit("vector-3", "低置信语义候选", score=0.84),
            ],
        )

        results = _rrf_merge([vector_response], 10, allow_dense_only=True)

        self.assertEqual([item["id"] for item in results], ["vector-1", "vector-2"])
        self.assertEqual(results[0]["retrieval_debug"]["dense_only_threshold"], 0.86)
        self.assertTrue(results[0]["retrieval_debug"]["dense_only_accepted"])

    def test_dense_routes_are_group_normalized_before_outer_rrf(self) -> None:
        bm25_response = _response(
            BM25_RETRIEVER,
            [
                _hit("lexical-1", "词面候选"),
            ],
            weight=1.5,
            total=1,
        )
        dense_projects_response = _response(
            "dense:projects",
            [
                _hit("vector-1", "向量候选", score=0.92),
            ],
            weight=2.0,
        )
        dense_skills_response = _response(
            "dense:skills",
            [
                _hit("vector-1", "向量候选", score=0.91),
            ],
            weight=2.0,
        )

        results = _rrf_merge(
            [bm25_response, dense_projects_response, dense_skills_response],
            10,
            allow_dense_only=True,
        )

        self.assertEqual([item["id"] for item in results], ["lexical-1", "vector-1"])
        vector_debug = results[1]["retrieval_debug"]
        self.assertEqual(vector_debug["retrieval_sources"], [DENSE_RETRIEVER])
        self.assertEqual(vector_debug["dense_rank"], 1)
        self.assertGreater(vector_debug["dense_inner_score"], vector_debug["raw_rrf_score"])

    def test_vector_text_excludes_entity_and_location_noise(self) -> None:
        doc = {
            "application": {
                "position_name": "机器学习工程师",
                "expected_work_cities": ["北京"],
            },
            "candidate": {
                "name": "张三",
                "school": "北京大学",
                "current_city": "北京",
                "major": "人工智能",
            },
            "skills": ["Python", "机器学习"],
            "education": [
                {
                    "school": "北京大学",
                    "college": "计算机学院",
                    "major": "人工智能",
                    "education_level": "本科",
                    "degree": "学士",
                    "research_direction": "自然语言处理",
                    "lab_name": "智能计算实验室",
                    "paper_level": "EI",
                }
            ],
            "internships": [
                {
                    "company": "百度在线网络技术",
                    "department": "搜索策略组",
                    "title": "算法实习生",
                    "work_type": "实习",
                    "description": "在百度在线网络技术北京团队负责推荐系统召回实验和离线评估。",
                }
            ],
            "projects": [
                {
                    "name": "医疗问答系统",
                    "description": "构建医疗知识库。",
                    "responsibility": "负责实体抽取和排序模型。",
                }
            ],
        }

        embedding_inputs = _embedding_inputs(doc)
        all_embedding_text = "\n".join(embedding_inputs.values())

        self.assertNotIn("北京大学", all_embedding_text)
        self.assertNotIn("北京", all_embedding_text)
        self.assertNotIn("张三", all_embedding_text)
        self.assertNotIn("百度在线网络技术", all_embedding_text)
        self.assertNotIn("学士", all_embedding_text)
        self.assertNotIn("semantic_profile_vector", embedding_inputs)
        self.assertIn("机器学习", embedding_inputs["skills_vector"])
        self.assertIn("自然语言处理", embedding_inputs["education_vector"])
        self.assertIn("推荐系统", embedding_inputs["internships_vector"])
        self.assertIn("医疗问答系统", embedding_inputs["projects_vector"])
        self.assertNotIn("role_vector", embedding_inputs)
        self.assertNotIn("机器学习工程师", all_embedding_text)

    def test_dense_routes_prioritize_query_intent(self) -> None:
        skill_routes = _infer_dense_routes(
            "Python SQL 推荐系统",
            {"Python", "SQL", "推荐系统"},
        )
        self.assertEqual(skill_routes[0]["field"], "skills_vector")
        self.assertIn("projects_vector", [route["field"] for route in skill_routes])

        project_routes = _infer_dense_routes(
            "做过推荐系统召回和 NLP 模型落地的人",
            {"推荐系统", "NLP"},
        )
        self.assertEqual(project_routes[0]["field"], "projects_vector")
        self.assertIn("skills_vector", [route["field"] for route in project_routes])

        role_like_routes = _infer_dense_routes(
            "机器学习工程师 Python 推荐系统",
            {"Python", "推荐系统"},
        )
        role_like_fields = [route["field"] for route in role_like_routes]
        self.assertNotIn("role_vector", role_like_fields)
        self.assertIn("skills_vector", role_like_fields)
        self.assertTrue(all(route["weight"] == 1.0 for route in role_like_routes))
        self.assertTrue(all("priority" in route for route in role_like_routes))

    def test_skill_filters_are_and_terms(self) -> None:
        filters = _build_filters("", [], ["Python", "NLP", "Python"], 0)
        self.assertEqual(
            filters,
            [
                {"term": {"skills": "Python"}},
                {"term": {"skills": "NLP"}},
            ],
        )

    def test_skill_filter_matches_case_variants(self) -> None:
        filters = _build_filters(
            "",
            [],
            ["Java", "JAVA", "LINUX"],
            0,
            skill_vocab={"JAVA", "Java", "Linux", "LINUX"},
        )

        self.assertEqual(
            filters,
            [
                {"terms": {"skills": ["Java", "JAVA"]}},
                {"terms": {"skills": ["Linux", "LINUX"]}},
            ],
        )

    def test_skill_facets_merge_case_variants(self) -> None:
        facets = _merge_case_insensitive_skill_buckets(
            [
                {"key": "JAVA", "doc_count": 2},
                {"key": "Java", "doc_count": 5},
                {"key": "LINUX", "doc_count": 1},
                {"key": "Linux", "doc_count": 3},
                {"key": "c", "doc_count": 1},
                {"key": "C", "doc_count": 2},
            ],
            30,
        )

        self.assertEqual(
            facets,
            [
                {"key": "Java", "count": 7},
                {"key": "Linux", "count": 4},
                {"key": "C", "count": 3},
            ],
        )

    def test_mixed_query_constraints_are_parsed_from_facets(self) -> None:
        parsed = _parse_query_constraints(
            "0.5年以上 北京 本科 推荐系统",
            facets={
                "degrees": [{"key": "本科"}],
                "cities": [{"key": "北京"}],
                "skills": [{"key": "推荐系统"}],
            },
        )

        self.assertEqual(parsed["query_text"], "推荐系统")
        self.assertEqual(
            parsed["constraints"],
            {
                "min_years": 0.5,
                "degree": "本科",
                "cities": ["北京"],
                "skills": ["推荐系统"],
            },
        )
        self.assertIn({"term": {"skills": "推荐系统"}}, parsed["filters"])

    def test_mixed_query_skill_constraints_are_case_insensitive(self) -> None:
        parsed = _parse_query_constraints(
            "0.5年以上 北京 本科 java",
            facets={
                "degrees": [{"key": "本科"}],
                "cities": [{"key": "北京"}],
                "skills": [{"key": "Java"}],
            },
        )

        self.assertEqual(parsed["constraints"]["skills"], ["Java"])
        self.assertIn({"term": {"skills": "Java"}}, parsed["filters"])

    def test_plain_skill_query_stays_broad(self) -> None:
        parsed = _parse_query_constraints(
            "推荐系统 NLP SQL",
            facets={
                "degrees": [{"key": "本科"}],
                "cities": [{"key": "北京"}],
                "skills": [{"key": "推荐系统"}, {"key": "NLP"}, {"key": "SQL"}],
            },
        )

        self.assertEqual(parsed["query_text"], "推荐系统 NLP SQL")
        self.assertEqual(parsed["filters"], [])

    def test_lexical_query_covers_exact_recruiting_fields(self) -> None:
        query_json = json.dumps(_lexical_query("A0009"), ensure_ascii=False)

        self.assertIn("application.candidate_no", query_json)
        self.assertIn("application.position_code", query_json)
        self.assertIn("application.company", query_json)
        self.assertIn("application.wishes", query_json)

    def test_lexical_query_prioritizes_exact_major_and_phrase_evidence(self) -> None:
        query_json = json.dumps(_lexical_query("计算机科学"), ensure_ascii=False)

        self.assertIn("candidate.major.keyword", query_json)
        self.assertIn("education.major.keyword", query_json)
        self.assertIn("projects.name.keyword", query_json)
        self.assertIn("candidate.major.phrase", query_json)
        self.assertIn("education.major.phrase", query_json)
        self.assertIn("lexical_exact:candidate_major", query_json)
        self.assertIn("lexical_phrase:candidate_major", query_json)
        self.assertIn("lexical_phrase:education_major", query_json)
        self.assertIn('"operator": "and"', query_json)

    def test_lexical_query_adds_named_term_coverage_for_multi_term_queries(self) -> None:
        single_term_json = json.dumps(_lexical_query("A"), ensure_ascii=False)
        multi_term_json = json.dumps(_lexical_query("A B"), ensure_ascii=False)

        self.assertNotIn("query_term:0", single_term_json)
        self.assertIn("constant_score", multi_term_json)
        self.assertIn("query_term:0", multi_term_json)
        self.assertIn("query_term:1", multi_term_json)

    def test_term_coverage_does_not_act_as_primary_recall_clause(self) -> None:
        query = _lexical_query("A B")
        bool_query = query["bool"]

        self.assertIn("must", bool_query)
        self.assertIn("should", bool_query)
        scoring_query_json = json.dumps(bool_query["must"], ensure_ascii=False)
        coverage_query_json = json.dumps(bool_query["should"], ensure_ascii=False)

        self.assertNotIn("query_term:0", scoring_query_json)
        self.assertIn("query_term:0", coverage_query_json)

    def test_lexical_total_uses_bm25_total_not_candidate_window(self) -> None:
        bm25_response = _response(
            BM25_RETRIEVER,
            [
                _hit("lexical-1", "词面第一"),
            ],
            total=125,
        )
        vector_response = _response(
            DENSE_RETRIEVER,
            [
                _hit("vector-1", "向量第一"),
            ],
        )

        self.assertEqual(_lexical_total([bm25_response, vector_response]), 125)
        self.assertEqual(_hybrid_total([bm25_response, vector_response], allow_dense_only=True), 2)

    def test_index_mapping_records_embedding_contract(self) -> None:
        meta = INDEX_BODY["mappings"]["_meta"]

        self.assertEqual(meta["embedding_vector_dims"], VECTOR_DIMS)
        self.assertEqual(meta["semantic_profile_version"], SEMANTIC_PROFILE_VERSION)
        self.assertEqual(meta["embedding_normalized"], EMBEDDING_NORMALIZED)
        self.assertEqual(tuple(meta["embedding_vector_fields"]), VECTOR_FIELDS)
        self.assertIn("embedding", INDEX_BODY["mappings"]["properties"])
        props = INDEX_BODY["mappings"]["properties"]
        for field in VECTOR_FIELDS:
            self.assertIn(field, props)
        self.assertNotIn("semantic_profile_vector", props)
        self.assertNotIn("role_vector", props)
        self.assertIn("keyword", props["candidate"]["properties"]["major"]["fields"])
        self.assertIn("phrase", props["candidate"]["properties"]["major"]["fields"])
        self.assertIn("keyword", props["education"]["properties"]["major"]["fields"])
        self.assertIn("phrase", props["education"]["properties"]["major"]["fields"])
        self.assertIn("keyword", props["projects"]["properties"]["name"]["fields"])
        self.assertIn("phrase", props["projects"]["properties"]["name"]["fields"])


def _response(
    retriever_name: str,
    hits: list[dict],
    weight: float = 1.0,
    total: int | None = None,
) -> dict:
    return {
        "_retriever_name": retriever_name,
        "_rrf_weight": weight,
        "hits": {
            "total": {"value": len(hits) if total is None else total, "relation": "eq"},
            "hits": hits,
        },
    }


def _hit(
    doc_id: str,
    name: str,
    score: float = 1.0,
    matched_queries: list[str] | None = None,
) -> dict:
    hit = {
        "_id": doc_id,
        "_score": score,
        "_source": {
            "candidate": {"name": name},
            "application": {},
            "education": [],
            "projects": [],
            "internships": [],
            "skills": [],
        },
    }
    if matched_queries is not None:
        hit["matched_queries"] = matched_queries
    return hit


if __name__ == "__main__":
    unittest.main()
