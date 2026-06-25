import json
import unittest
from unittest.mock import patch

from app import (
    DENSE_RETRIEVER,
    EVIDENCE_DENSE_RETRIEVER,
    EVIDENCE_RETRIEVER,
    _build_filters,
    _default_snippet,
    _evidence_lexical_query,
    _format_hit,
    _hybrid_total,
    _lexical_query,
    _lexical_total,
    _merge_case_insensitive_skill_buckets,
    _normalize_limit,
    _parse_query_constraints,
    _rrf_merge,
    _run_hybrid_search,
    _use_dense,
)
from import_to_es import (
    EMBEDDING_NORMALIZED,
    EVIDENCE_INDEX_BODY,
    EVIDENCE_VECTOR_FIELD,
    SEMANTIC_PROFILE_VERSION,
    VECTOR_DIMS,
    INDEX_BODY,
    _resume_evidence_docs,
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

    def test_hybrid_merge_keeps_independent_dense_hits(self) -> None:
        vector_response = _response(
            EVIDENCE_DENSE_RETRIEVER,
            [
                _hit("vector-only", "向量候选"),
            ],
        )

        self.assertEqual(_hybrid_total([vector_response]), 1)
        self.assertEqual(_rrf_merge([vector_response], 10)[0]["id"], "vector-only")

    def test_hybrid_search_uses_only_evidence_retrievers(self) -> None:
        calls: list[str] = []

        def fake_es(_method: str, path: str, _body: dict) -> dict:
            calls.append(path)
            return {"hits": {"total": {"value": 0}, "hits": []}}

        with patch("app._es", side_effect=fake_es):
            responses, warnings = _run_hybrid_search(
                "推荐系统",
                [0.1, 0.2, 0.3],
                [],
                10,
                use_dense=True,
            )

        self.assertEqual(warnings, [])
        self.assertEqual(
            {response["_retriever_name"] for response in responses},
            {EVIDENCE_RETRIEVER, EVIDENCE_DENSE_RETRIEVER},
        )
        self.assertTrue(all("resume_evidence_current" in path for path in calls))

    def test_exact_entity_queries_skip_dense_before_merge(self) -> None:
        lexical_response = _response(
            EVIDENCE_RETRIEVER,
            [
                _hit("lexical-1", "词面第一"),
            ],
            total=1,
        )
        vector_response = _response(
            EVIDENCE_DENSE_RETRIEVER,
            [
                _hit("vector-only", "向量候选"),
            ],
        )

        results = _rrf_merge([lexical_response, vector_response], 10)

        self.assertFalse(_use_dense("北京大学"))
        self.assertFalse(_use_dense("奇安信集团"))
        self.assertFalse(_use_dense("M20260001"))
        self.assertFalse(_use_dense("A0009"))
        self.assertTrue(_use_dense("自然语言处理"))
        self.assertTrue(_use_dense("推荐召回"))
        self.assertTrue(_use_dense("做过推荐系统召回和 NLP 模型落地的人"))
        self.assertEqual(_hybrid_total([lexical_response, vector_response]), 2)
        self.assertEqual([item["id"] for item in results], ["lexical-1", "vector-only"])

    def test_hybrid_merge_combines_evidence_lexical_and_dense_with_rrf(self) -> None:
        lexical_response = _response(
            EVIDENCE_RETRIEVER,
            [
                _hit("lexical-1", "词面第一"),
                _hit("lexical-2", "词面第二"),
            ],
            weight=1.0,
            total=7,
        )
        vector_response = _response(
            EVIDENCE_DENSE_RETRIEVER,
            [
                _hit("lexical-2", "词面第二"),
                _hit("vector-only", "向量候选"),
            ],
        )

        results = _rrf_merge([lexical_response, vector_response], 10)

        self.assertEqual(_hybrid_total([lexical_response, vector_response]), 3)
        self.assertEqual([item["id"] for item in results], ["lexical-2", "lexical-1", "vector-only"])
        self.assertEqual(results[0]["retrieval_debug"]["retrieval_sources"], [EVIDENCE_RETRIEVER, DENSE_RETRIEVER])

    def test_evidence_hits_are_collapsed_to_resume_results(self) -> None:
        evidence_response = _response(
            EVIDENCE_RETRIEVER,
            [
                _evidence_hit(
                    "resume-1:project:1",
                    "resume-1",
                    "推荐系统召回项目",
                    "项目职责：负责推荐系统召回和 NLP 模型落地。",
                    matched_queries=["evidence_phrase:text:W10"],
                )
            ],
            weight=1.2,
        )

        with patch(
            "app._fetch_resume_hits_for_evidence",
            return_value={"resume-1": _hit("resume-1", "候选人")},
        ):
            results = _rrf_merge([evidence_response], 10)

        self.assertEqual([item["id"] for item in results], ["resume-1"])
        self.assertIn(EVIDENCE_RETRIEVER, results[0]["retrieval_debug"]["retrieval_sources"])
        self.assertEqual(
            results[0]["retrieval_debug"]["evidence_matches"][0]["section_type"],
            "project",
        )
        self.assertEqual(results[0]["retrieval_debug"]["matched_queries"], ["evidence_phrase:text:W10"])
        self.assertEqual(results[0]["retrieval_debug"]["lexical_tier"], 2)
        self.assertIn("<mark>推荐系统</mark>", results[0]["project_snippet"])
        self.assertIn("召回", results[0]["project_snippet"])

    def test_multi_term_coverage_boost_is_reflected_in_final_score(self) -> None:
        lexical_response = _response(
            EVIDENCE_RETRIEVER,
            [
                _hit("a-many", "A A A 0", matched_queries=["query_term:0"]),
                _hit("b-many", "0 B B B", matched_queries=["query_term:1"]),
                _hit("both", "A B 0 0", matched_queries=["query_term:0", "query_term:1"]),
            ],
        )
        vector_response = _response(
            EVIDENCE_DENSE_RETRIEVER,
            [
                _hit("a-many", "A A A 0", score=1.0),
                _hit("zero", "0 0 0 0", score=0.99),
            ],
        )

        results = _rrf_merge(
            [lexical_response, vector_response],
            10,
            query_text="A B",
        )

        self.assertEqual([item["id"] for item in results], ["a-many", "both", "b-many", "zero"])
        self.assertGreaterEqual(results[0]["retrieval_debug"]["rrf_score"], results[1]["retrieval_debug"]["rrf_score"])
        self.assertEqual(results[0]["retrieval_debug"]["term_coverage"], 1)
        self.assertEqual(results[1]["retrieval_debug"]["term_coverage"], 2)

    def test_final_score_orders_phrase_against_dense_rank(self) -> None:
        lexical_response = _response(
            EVIDENCE_RETRIEVER,
            [
                _hit("split-and-dense", "计算机 与 科学", matched_queries=["query_term:0"]),
                _hit("phrase", "计算机科学", matched_queries=["lexical_phrase:candidate_major"]),
            ],
        )
        vector_response = _response(
            EVIDENCE_DENSE_RETRIEVER,
            [
                _hit("split-and-dense", "计算机 与 科学", score=0.9),
            ],
        )

        results = _rrf_merge(
            [lexical_response, vector_response],
            10,
            query_text="计算机科学",
        )

        self.assertEqual([item["id"] for item in results], ["split-and-dense", "phrase"])
        self.assertGreaterEqual(results[0]["retrieval_debug"]["rrf_score"], results[1]["retrieval_debug"]["rrf_score"])
        self.assertEqual(results[1]["retrieval_debug"]["lexical_tier"], 2)

    def test_final_score_can_include_dense_support_for_phrase_hits(self) -> None:
        lexical_response = _response(
            EVIDENCE_RETRIEVER,
            [
                _hit("major-phrase", "专业短语", matched_queries=["lexical_phrase:candidate_major"]),
                _hit("weak-field-phrase", "弱字段短语", matched_queries=["lexical_phrase:section_education"]),
            ],
        )
        vector_response = _response(
            EVIDENCE_DENSE_RETRIEVER,
            [
                _hit("weak-field-phrase", "弱字段短语", score=0.9),
            ],
        )

        results = _rrf_merge(
            [lexical_response, vector_response],
            10,
            query_text="计算机科学",
        )

        self.assertEqual([item["id"] for item in results], ["weak-field-phrase", "major-phrase"])
        self.assertGreaterEqual(results[0]["retrieval_debug"]["rrf_score"], results[1]["retrieval_debug"]["rrf_score"])

    def test_dense_hits_do_not_need_lexical_support_or_similarity_gate(self) -> None:
        lexical_response = _response(
            EVIDENCE_RETRIEVER,
            [
                _hit("lexical-1", "词面第一"),
            ],
            total=1,
        )
        vector_response = _response(
            EVIDENCE_DENSE_RETRIEVER,
            [
                _hit("weak-vector-only", "弱语义候选", score=0.79),
                _hit("lexical-1", "词面第一", score=0.78),
            ],
        )

        results = _rrf_merge([lexical_response, vector_response], 10)

        self.assertEqual(_hybrid_total([lexical_response, vector_response]), 2)
        self.assertEqual([item["id"] for item in results], ["lexical-1", "weak-vector-only"])
        self.assertEqual(results[0]["retrieval_debug"]["dense_rank"], 2)
        self.assertEqual(results[0]["retrieval_debug"]["dense_route_rank"], 2)
        self.assertEqual(results[1]["retrieval_debug"]["retrieval_sources"], [DENSE_RETRIEVER])
        self.assertNotIn("dense_only_threshold", results[1]["retrieval_debug"])

    def test_dense_hits_are_not_threshold_filtered(self) -> None:
        vector_response = _response(
            EVIDENCE_DENSE_RETRIEVER,
            [
                _hit("vector-1", "强语义候选", score=0.88),
                _hit("vector-2", "近邻语义候选", score=0.865),
                _hit("vector-3", "低置信语义候选", score=0.84),
            ],
        )

        results = _rrf_merge([vector_response], 10)

        self.assertEqual([item["id"] for item in results], ["vector-1", "vector-2", "vector-3"])
        self.assertEqual(results[2]["retrieval_debug"]["dense_rank"], 3)
        self.assertNotIn("dense_only_accepted", results[0]["retrieval_debug"])

    def test_evidence_dense_is_grouped_before_outer_rrf(self) -> None:
        evidence_response = _response(
            EVIDENCE_RETRIEVER,
            [
                _evidence_hit(
                    "lexical-1:project:1",
                    "lexical-1",
                    "词面证据",
                    "项目职责：负责推荐系统召回。",
                ),
            ],
            weight=1.2,
            total=1,
        )
        dense_response = _response(
            EVIDENCE_DENSE_RETRIEVER,
            [
                _evidence_hit(
                    "vector-1:project:1",
                    "vector-1",
                    "向量证据",
                    "项目职责：负责语义召回和排序模型。",
                    score=0.92,
                ),
            ],
            weight=1.0,
        )
        dense_response["_vector_field"] = EVIDENCE_VECTOR_FIELD

        with patch(
            "app._fetch_resume_hits_for_evidence",
            return_value={
                "lexical-1": _hit("lexical-1", "词面候选"),
                "vector-1": _hit("vector-1", "向量候选"),
            },
        ):
            results = _rrf_merge(
                [evidence_response, dense_response],
                10,
            )

        self.assertEqual([item["id"] for item in results], ["lexical-1", "vector-1"])
        vector_debug = results[1]["retrieval_debug"]
        self.assertEqual(vector_debug["retrieval_sources"], [DENSE_RETRIEVER])
        self.assertEqual(vector_debug["dense_rank"], 1)
        self.assertEqual(vector_debug["dense_retriever"], EVIDENCE_DENSE_RETRIEVER)
        self.assertEqual(vector_debug["dense_field"], EVIDENCE_VECTOR_FIELD)
        dense_match = vector_debug["dense_matches"][0]
        self.assertEqual(dense_match["section_type"], "project")
        self.assertEqual(dense_match["title"], "向量证据")
        self.assertIn("语义召回", dense_match["snippet"])

    def test_dense_group_rank_uses_global_vector_rank_not_filtered_rank(self) -> None:
        evidence_response = _response(
            EVIDENCE_RETRIEVER,
            [
                _evidence_hit(
                    "target:project:1",
                    "target",
                    "词面证据",
                    "项目职责：负责目标系统。",
                ),
            ],
            weight=1.2,
        )
        dense_hits = [
            _evidence_hit(
                f"other-{index}:project:1",
                f"other-{index}",
                f"其它证据 {index}",
                "项目职责：其它候选人的相似证据。",
                score=0.9 - index * 0.001,
            )
            for index in range(1, 59)
        ]
        dense_hits.append(
            _evidence_hit(
                "target:project:2",
                "target",
                "目标向量证据",
                "项目职责：负责目标系统的语义召回。",
                score=0.75,
            )
        )
        dense_response = _response(EVIDENCE_DENSE_RETRIEVER, dense_hits)
        dense_response["_vector_field"] = EVIDENCE_VECTOR_FIELD

        with patch(
            "app._fetch_resume_hits_for_evidence",
            return_value={"target": _hit("target", "目标候选人")},
        ):
            results = _rrf_merge(
                [evidence_response, dense_response],
                10,
            )

        debug = results[0]["retrieval_debug"]
        self.assertEqual(debug["dense_route_rank"], 59)
        self.assertEqual(debug["dense_rank"], 59)
        self.assertEqual(debug["dense_group_rank"], 59)
        self.assertEqual(debug["dense_rrf_contribution"], round(1 / (60 + 59), 6))

    def test_vector_text_excludes_entity_and_location_noise(self) -> None:
        doc = {
            "resume_id": "resume-1",
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

        evidence_docs = _resume_evidence_docs(doc)
        all_embedding_text = "\n".join(
            item["text"]
            for item in evidence_docs
            if item["section_type"] != "profile"
        )
        profile_text = "\n".join(
            item["text"]
            for item in evidence_docs
            if item["section_type"] == "profile"
        )

        self.assertNotIn("北京大学", all_embedding_text)
        self.assertNotIn("北京", all_embedding_text)
        self.assertNotIn("张三", all_embedding_text)
        self.assertNotIn("百度在线网络技术", all_embedding_text)
        self.assertNotIn("学士", all_embedding_text)
        self.assertIn("机器学习", all_embedding_text)
        self.assertIn("自然语言处理", all_embedding_text)
        self.assertIn("推荐系统", all_embedding_text)
        self.assertIn("医疗问答系统", all_embedding_text)
        self.assertNotIn("机器学习工程师", all_embedding_text)
        self.assertIn("北京大学", profile_text)
        self.assertIn("机器学习工程师", profile_text)

    def test_resume_evidence_docs_are_chunked_and_semantic_cleaned(self) -> None:
        doc = {
            "resume_id": "resume-1",
            "application": {
                "candidate_no": "M0001",
                "position_code": "A0001",
                "position_name": "机器学习工程师",
                "expected_work_cities": ["北京"],
            },
            "candidate": {
                "name": "张三",
                "school": "北京大学",
                "current_city": "北京",
                "highest_degree": "本科",
                "major": "人工智能",
                "years_experience": 0.8,
            },
            "skills": ["Python", "推荐系统"],
            "education": [
                {
                    "school": "北京大学",
                    "college": "计算机学院",
                    "major": "人工智能",
                    "research_direction": "自然语言处理",
                }
            ],
            "internships": [
                {
                    "company": "百度在线网络技术",
                    "department": "搜索策略组",
                    "title": "算法实习生",
                    "description": "在百度在线网络技术北京团队负责推荐系统召回实验。",
                }
            ],
            "projects": [
                {
                    "name": "推荐系统召回项目",
                    "description": "构建推荐系统召回链路。",
                    "responsibility": "负责 NLP 特征和离线评估。",
                }
            ],
        }

        evidence_docs = _resume_evidence_docs(doc)
        section_types = {item["section_type"] for item in evidence_docs}
        vector_text = "\n".join(
            item["text"]
            for item in evidence_docs
            if item["section_type"] != "profile"
        )
        profile_docs = [item for item in evidence_docs if item["section_type"] == "profile"]

        self.assertEqual(section_types, {"profile", "skills", "project", "internship", "education"})
        self.assertTrue(all(item["resume_id"] == "resume-1" for item in evidence_docs))
        self.assertNotIn("北京大学", vector_text)
        self.assertNotIn("百度在线网络技术", vector_text)
        self.assertIn("推荐系统召回", vector_text)
        self.assertIn("自然语言处理", vector_text)
        self.assertEqual(len(profile_docs), 1)
        self.assertIn("北京大学", profile_docs[0]["text"])
        self.assertIn("机器学习工程师", profile_docs[0]["text"])
        self.assertNotIn(EVIDENCE_VECTOR_FIELD, profile_docs[0])

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

    def test_evidence_lexical_query_covers_profile_fields(self) -> None:
        query_json = json.dumps(_evidence_lexical_query("A0009"), ensure_ascii=False)

        self.assertIn("application.candidate_no", query_json)
        self.assertIn("application.position_code", query_json)
        self.assertIn("candidate.name.keyword", query_json)
        self.assertIn("candidate.all_schools.keyword", query_json)
        self.assertIn("candidate.major.keyword", query_json)
        self.assertIn("evidence_exact:position_code", query_json)

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

    def test_lexical_total_counts_evidence_candidate_window(self) -> None:
        evidence_response = _response(
            EVIDENCE_RETRIEVER,
            [
                _hit("lexical-1", "词面第一"),
            ],
            total=125,
        )
        vector_response = _response(
            EVIDENCE_DENSE_RETRIEVER,
            [
                _hit("vector-1", "向量第一"),
            ],
        )

        self.assertEqual(_lexical_total([evidence_response, vector_response]), 1)
        self.assertEqual(_hybrid_total([evidence_response, vector_response]), 2)

    def test_lexical_total_deduplicates_multiple_evidence_sources(self) -> None:
        first_response = _response(
            EVIDENCE_RETRIEVER,
            [
                _hit("candidate-1", "候选人一"),
                _hit("candidate-2", "候选人二"),
            ],
            total=125,
        )
        evidence_response = _response(
            EVIDENCE_RETRIEVER,
            [
                _evidence_hit(
                    "candidate-1:project:1",
                    "candidate-1",
                    "推荐系统",
                    "项目职责：负责推荐系统召回。",
                ),
                _evidence_hit(
                    "candidate-3:project:1",
                    "candidate-3",
                    "NLP 系统",
                    "项目职责：负责 NLP 模型落地。",
                ),
            ],
            total=80,
        )

        self.assertEqual(_lexical_total([first_response, evidence_response]), 3)

    def test_index_mapping_records_embedding_contract(self) -> None:
        meta = INDEX_BODY["mappings"]["_meta"]

        self.assertEqual(meta["index_role"], "candidate_profile")
        self.assertEqual(meta["semantic_profile_version"], SEMANTIC_PROFILE_VERSION)
        self.assertEqual(meta["embedding_vector_fields"], [])
        props = INDEX_BODY["mappings"]["properties"]
        self.assertNotIn("embedding", props)
        self.assertNotIn("skills_vector", props)
        self.assertNotIn("projects_vector", props)
        self.assertNotIn("internships_vector", props)
        self.assertNotIn("education_vector", props)
        self.assertNotIn("semantic_profile_vector", props)
        self.assertNotIn("role_vector", props)
        self.assertIn("keyword", props["candidate"]["properties"]["major"]["fields"])
        self.assertIn("phrase", props["candidate"]["properties"]["major"]["fields"])
        self.assertIn("keyword", props["education"]["properties"]["major"]["fields"])
        self.assertIn("phrase", props["education"]["properties"]["major"]["fields"])
        self.assertIn("keyword", props["projects"]["properties"]["name"]["fields"])
        self.assertIn("phrase", props["projects"]["properties"]["name"]["fields"])
        evidence_meta = EVIDENCE_INDEX_BODY["mappings"]["_meta"]
        self.assertEqual(evidence_meta["embedding_vector_dims"], VECTOR_DIMS)
        self.assertEqual(evidence_meta["semantic_profile_version"], SEMANTIC_PROFILE_VERSION)
        self.assertEqual(evidence_meta["embedding_normalized"], EMBEDDING_NORMALIZED)
        self.assertEqual(evidence_meta["embedding_vector_fields"], [EVIDENCE_VECTOR_FIELD])
        self.assertEqual(
            evidence_meta["vectorized_section_types"],
            ["education", "internship", "project", "skills"],
        )
        evidence_props = EVIDENCE_INDEX_BODY["mappings"]["properties"]
        self.assertIn(EVIDENCE_VECTOR_FIELD, evidence_props)
        self.assertIn("embedding", evidence_props)
        self.assertIn("phrase", evidence_props["text"]["fields"])
        self.assertIn("keyword", evidence_props["title"]["fields"])
        self.assertIn("name", evidence_props["candidate"]["properties"])
        self.assertIn("company", evidence_props["application"]["properties"])


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


def _evidence_hit(
    evidence_id: str,
    resume_id: str,
    title: str,
    text: str,
    score: float = 1.0,
    matched_queries: list[str] | None = None,
) -> dict:
    section_type = evidence_id.split(":")[1]
    hit = {
        "_id": evidence_id,
        "_score": score,
        "_source": {
            "evidence_id": evidence_id,
            "resume_id": resume_id,
            "section_type": section_type,
            "title": title,
            "text": text,
        },
        "highlight": {
            "text": [text.replace("推荐系统", "<mark>推荐系统</mark>")]
        },
    }
    if matched_queries is not None:
        hit["matched_queries"] = matched_queries
    return hit


if __name__ == "__main__":
    unittest.main()
