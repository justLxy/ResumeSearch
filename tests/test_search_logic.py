import json
import unittest
from unittest.mock import patch

from app import (
    DENSE_RETRIEVER,
    EVIDENCE_DENSE_RETRIEVER,
    EVIDENCE_RETRIEVER,
    INTENT_KEYWORD,
    INTENT_LOOKUP,
    INTENT_SEMANTIC,
    QUERY_PARSER_MODEL_ID,
    _build_filters,
    _call_query_parser_llm,
    _default_snippet,
    _evidence_lexical_query,
    _format_hit,
    _hybrid_total,
    _lexical_total,
    _lookup_fast_path,
    _merge_case_insensitive_skill_buckets,
    _min_years_filter,
    _normalize_limit,
    _normalize_offset,
    _parse_query_with_llm,
    _plan_query,
    _query_parser_system_prompt,
    _rerank_document,
    _rerank_results,
    _rrf_merge,
    _run_hybrid_search,
)
from import_to_es import (
    EMBEDDING_NORMALIZED,
    EVIDENCE_INDEX_BODY,
    EVIDENCE_VECTOR_FIELD,
    SEMANTIC_PROFILE_VERSION,
    VECTOR_DIMS,
    INDEX_BODY,
    _enrich_doc,
    _resume_evidence_docs,
    _estimate_years_experience,
)
from evaluate_search import evaluate_query_plan


class SearchLogicTests(unittest.TestCase):
    def test_query_parser_disables_thinking(self) -> None:
        calls: list[dict] = []

        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "intent": "semantic",
                                        "lexical_query": "RAG 向量检索",
                                        "semantic_query": "RAG 向量检索",
                                        "constraints": {},
                                        "enable_dense": True,
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ]
                }

        def fake_post(url, *, headers, json, timeout):
            calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
            return FakeResponse()

        with patch("app.requests.post", side_effect=fake_post):
            plan = _call_query_parser_llm(
                "找做过 RAG 和向量检索的人",
                facets={
                    "degrees": [{"key": "本科"}],
                    "cities": [{"key": "北京"}],
                    "skills": [{"key": "RAG"}],
                },
            )

        self.assertEqual(plan["intent"], "semantic")
        self.assertEqual(calls[0]["json"]["model"], QUERY_PARSER_MODEL_ID)
        # 两种关闭思考的参数都发：qwen 用 enable_thinking，deepseek/豆包用 thinking。
        self.assertFalse(calls[0]["json"]["enable_thinking"])
        self.assertEqual(calls[0]["json"]["thinking"], {"type": "disabled"})
        self.assertFalse(calls[0]["json"]["stream"])

    def test_query_parser_prompt_routes_structured_skill_queries_to_semantic(self) -> None:
        prompt = _query_parser_system_prompt()

        self.assertIn("学历/城市/年限只是结构化 filter，不决定 intent", prompt)
        self.assertIn("lexical_query 不要复读原句", prompt)
        self.assertIn("北京 硕士 4年以上 RAG LangChain", prompt)
        self.assertIn('"intent":"semantic","lexical_query":"RAG LangChain"', prompt)
        self.assertIn(
            '"lexical_query":"LLM RAG 企业知识库问答 文档解析 向量检索 召回排序 Prompt 模型微调 Python PyTorch LangChain LlamaIndex RAG评测 长文本处理 ToB知识库"',
            prompt,
        )

    def test_planner_eval_reports_field_mismatches(self) -> None:
        planner_eval = evaluate_query_plan(
            {
                "intent": "keyword",
                "lexical_query": "RAG LangChain",
                "semantic_query": "RAG LangChain",
                "enable_dense": False,
                "enable_rerank": False,
            },
            {
                "intent": "semantic",
                "lexical_query": "RAG LangChain",
                "semantic_query": "RAG LangChain",
                "enable_dense": True,
                "enable_rerank": True,
            },
        )

        self.assertFalse(planner_eval["exact_match"])
        self.assertEqual(
            planner_eval["mismatched_fields"],
            ["intent", "enable_dense", "enable_rerank"],
        )
        self.assertTrue(planner_eval["field_matches"]["lexical_query"])

    def test_planner_eval_lexical_uses_token_subset(self) -> None:
        # 期望的核心词都出现在实际（被 LLM 压缩/扩展的）lexical_query 里 → 视为通过，
        # 不要求逐字相等。
        planner_eval = evaluate_query_plan(
            {
                "intent": "semantic",
                "lexical_query": "LLM RAG 企业知识库 向量检索 召回排序 Python PyTorch",
                "semantic_query": "岗位：LLM/RAG 应用工程师……",
                "enable_dense": True,
                "enable_rerank": True,
            },
            {
                "intent": "semantic",
                "lexical_query": "RAG 向量检索",
                "semantic_query": "岗位：LLM/RAG 应用工程师……",
                "enable_dense": True,
                "enable_rerank": True,
            },
        )

        self.assertTrue(planner_eval["exact_match"])
        self.assertTrue(planner_eval["field_matches"]["lexical_query"])

    def test_planner_eval_lexical_flags_dropped_core_term(self) -> None:
        # 期望的核心词没有全部出现在实际 lexical_query 里 → 判失配（核心词被丢）。
        planner_eval = evaluate_query_plan(
            {
                "intent": "semantic",
                "lexical_query": "RAG 向量检索",
                "semantic_query": "x",
                "enable_dense": True,
                "enable_rerank": True,
            },
            {
                "intent": "semantic",
                "lexical_query": "RAG 向量检索 知识图谱",
                "semantic_query": "x",
                "enable_dense": True,
                "enable_rerank": True,
            },
        )

        self.assertFalse(planner_eval["field_matches"]["lexical_query"])
        self.assertIn("lexical_query", planner_eval["mismatched_fields"])

    def test_search_limit_defaults_to_full_result_window(self) -> None:
        self.assertEqual(_normalize_limit(None), 100)
        self.assertEqual(_normalize_limit(0), 100)
        self.assertEqual(_normalize_limit(20), 20)
        self.assertEqual(_normalize_limit(500), 500)
        self.assertEqual(_normalize_limit(50_000), 1000)

    def test_search_offset_is_capped_to_viewable_window(self) -> None:
        self.assertEqual(_normalize_offset(None), 0)
        self.assertEqual(_normalize_offset(0), 0)
        self.assertEqual(_normalize_offset(200), 200)
        self.assertEqual(_normalize_offset(50_000), 1000)

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

    def test_enrich_doc_preserves_explicit_years_experience(self) -> None:
        doc = {
            "candidate": {"years_experience": 4.0},
            "application": {"apply_time": "2026-06-14"},
            "internships": [
                {
                    "start_date": "2022-07-01",
                    "end_date": "2026-06-01",
                }
            ],
        }

        enriched = _enrich_doc(doc)

        self.assertEqual(enriched["candidate"]["years_experience"], 4.0)
        self.assertEqual(_estimate_years_experience(doc), 3.9)

    def test_enrich_doc_normalizes_string_years_experience(self) -> None:
        doc = {
            "candidate": {"years_experience": "4年以上"},
            "application": {"apply_time": "2026-06-14"},
            "internships": [],
        }

        enriched = _enrich_doc(doc)

        self.assertEqual(enriched["candidate"]["years_experience"], 4.0)

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

    def test_hybrid_merge_keeps_dense_candidates_when_dense_route_is_present(self) -> None:
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
                    matched_queries=["evidence_phrase:text:W10", "evidence_term:all_terms:W4"],
                ),
                _evidence_hit(
                    "resume-1:internship:1",
                    "resume-1",
                    "推荐系统排序实习",
                    "实习描述：负责召回策略。",
                    matched_queries=["evidence_phrase:title:W12", "evidence_term:all_terms:W4"],
                ),
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
        self.assertEqual(
            results[0]["retrieval_debug"]["matched_queries"],
            [
                "evidence_phrase:text:W10",
                "evidence_term:all_terms:W4",
                "evidence_phrase:title:W12",
            ],
        )
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
                _hit("phrase", "计算机科学", matched_queries=["evidence_phrase:candidate_major"]),
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
                _hit("major-phrase", "专业短语", matched_queries=["evidence_phrase:candidate_major"]),
                _hit("weak-field-phrase", "弱字段短语", matched_queries=["evidence_phrase:section_education"]),
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

    def test_dense_pooling_reranks_route_without_extra_rrf_score(self) -> None:
        dense_response = _response(
            EVIDENCE_DENSE_RETRIEVER,
            [
                _evidence_hit("single:project:1", "single", "单证据候选人", "项目职责：负责推荐系统召回。"),
                _evidence_hit("pooled:project:1", "pooled", "第一段向量证据", "项目职责：负责推荐系统召回。"),
                _evidence_hit("pooled:internship:1", "pooled", "第二段向量证据", "实习描述：负责排序模型。"),
                _evidence_hit("pooled:project:2", "pooled", "第三段向量证据", "项目职责：负责自然语言处理。"),
            ],
        )
        dense_response["_vector_field"] = EVIDENCE_VECTOR_FIELD

        with patch(
            "app._fetch_resume_hits_for_evidence",
            return_value={
                "single": _hit("single", "单证据候选人"),
                "pooled": _hit("pooled", "多证据候选人"),
            },
        ):
            results = _rrf_merge([dense_response], 10)

        self.assertEqual([item["id"] for item in results], ["pooled", "single"])
        debug = results[0]["retrieval_debug"]
        base_contribution = 1 / (60 + 1)
        self.assertEqual(debug["dense_route_rank"], 2)
        self.assertEqual(debug["dense_group_rank"], 1)
        self.assertEqual(debug["dense_support_count"], 3)
        self.assertEqual(debug["dense_rrf_contribution"], round(base_contribution, 6))
        self.assertNotIn("dense_support_bonus", debug)
        self.assertEqual(len(debug["dense_matches"]), 3)

    def test_bm25_evidence_pooling_reranks_route_without_extra_rrf_score(self) -> None:
        evidence_response = _response(
            EVIDENCE_RETRIEVER,
            [
                _evidence_hit("single:project:1", "single", "单证据候选人", "项目职责：负责推荐系统召回。"),
                _evidence_hit("pooled:project:1", "pooled", "第一段词面证据", "项目职责：负责推荐系统召回。"),
                _evidence_hit("pooled:internship:1", "pooled", "第二段词面证据", "实习描述：负责排序模型。"),
                _evidence_hit("pooled:education:1", "pooled", "第三段词面证据", "研究方向：自然语言处理。"),
            ],
            weight=1.2,
        )

        with patch(
            "app._fetch_resume_hits_for_evidence",
            return_value={
                "single": _hit("single", "单证据候选人"),
                "pooled": _hit("pooled", "多证据候选人"),
            },
        ):
            results = _rrf_merge([evidence_response], 10)

        self.assertEqual([item["id"] for item in results], ["pooled", "single"])
        debug = results[0]["retrieval_debug"]
        self.assertEqual(debug["evidence_rank"], 2)
        self.assertEqual(debug["evidence_group_rank"], 1)
        self.assertEqual(debug["evidence_support_count"], 3)
        self.assertEqual(debug["evidence_rrf_contribution"], round(1.2 / (60 + 1), 6))

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
        evidence_text = "\n".join(
            item["text"]
            for item in evidence_docs
            if item["section_type"] != "profile"
        )
        vector_text = "\n".join(
            item["text"]
            for item in evidence_docs
            if item.get("embedding")
        )
        profile_text = "\n".join(
            item["text"]
            for item in evidence_docs
            if item["section_type"] == "profile"
        )

        self.assertNotIn("北京大学", evidence_text)
        self.assertNotIn("北京", evidence_text)
        self.assertNotIn("张三", evidence_text)
        self.assertNotIn("百度在线网络技术", evidence_text)
        self.assertNotIn("学士", evidence_text)
        self.assertIn("机器学习", evidence_text)
        self.assertIn("自然语言处理", evidence_text)
        self.assertIn("推荐系统", vector_text)
        self.assertIn("医疗问答系统", vector_text)
        self.assertNotIn("自然语言处理", vector_text)
        self.assertNotIn("机器学习工程师", evidence_text)
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
        non_profile_text = "\n".join(
            item["text"]
            for item in evidence_docs
            if item["section_type"] != "profile"
        )
        profile_docs = [item for item in evidence_docs if item["section_type"] == "profile"]
        skill_docs = [item for item in evidence_docs if item["section_type"] == "skills"]
        education_docs = [item for item in evidence_docs if item["section_type"] == "education"]

        self.assertEqual(section_types, {"profile", "skills", "project", "internship", "education"})
        self.assertTrue(all(item["resume_id"] == "resume-1" for item in evidence_docs))
        self.assertNotIn("北京大学", non_profile_text)
        self.assertNotIn("百度在线网络技术", non_profile_text)
        self.assertIn("推荐系统召回", non_profile_text)
        self.assertIn("自然语言处理", non_profile_text)
        self.assertEqual(len(profile_docs), 1)
        self.assertEqual(len(skill_docs), 1)
        self.assertEqual(len(education_docs), 1)
        self.assertIn("北京大学", profile_docs[0]["text"])
        self.assertIn("机器学习工程师", profile_docs[0]["text"])
        self.assertNotIn("embedding", skill_docs[0])
        self.assertNotIn("embedding", education_docs[0])
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

    def test_evidence_partial_terms_requires_query_coverage(self) -> None:
        body = _evidence_lexical_query("Python Java C++")
        partial_query = _find_multi_match_by_name(body, "evidence_term:partial_terms:W1")

        self.assertEqual(partial_query["operator"], "or")
        self.assertEqual(partial_query["minimum_should_match"], "70%")

    def test_lookup_lexical_query_uses_exact_profile_fields_only(self) -> None:
        query_json = json.dumps(
            _evidence_lexical_query("zhangwei_mock@example.com", query_intent=INTENT_LOOKUP),
            ensure_ascii=False,
        )

        self.assertIn("candidate.email", query_json)
        self.assertIn("application.candidate_no", query_json)
        self.assertNotIn("partial_terms", query_json)
        self.assertNotIn("multi_match", query_json)

    def test_entity_field_match_uses_phrase_not_partial_or(self) -> None:
        # 实体字段用 match_phrase（按序连续）而非 70% OR-token，避免"哥伦比亚大学"
        # 被切成 [哥伦比亚, 大学] 后仅凭泛词"大学"匹配全库学校。
        clause = _find_clause_with_name_prefix(
            _evidence_lexical_query("哥伦比亚大学"), "evidence_match:candidate_school"
        )
        self.assertIsNotNone(clause, "应存在实体学校匹配子句")
        # 子句类型应为 match_phrase，而不是带 minimum_should_match 的 match
        self.assertIn("match_phrase", clause)
        self.assertNotIn("match", {k for k in clause if k != "match_phrase"})

    def test_keyword_intent_drops_partial_terms_route(self) -> None:
        # keyword 实体查询不应启用 partial_terms（70% OR-token）——否则泛词直通全库。
        keyword_json = json.dumps(
            _evidence_lexical_query("哥伦比亚大学", query_intent=INTENT_KEYWORD),
            ensure_ascii=False,
        )
        self.assertNotIn("partial_terms", keyword_json)

        # 非 keyword（如多技能 semantic）仍保留 partial_terms 做部分覆盖召回。
        semantic_json = json.dumps(
            _evidence_lexical_query("Python NLP SQL", query_intent=INTENT_SEMANTIC),
            ensure_ascii=False,
        )
        self.assertIn("partial_terms", semantic_json)

    def test_query_plan_uses_llm_keyword_constraints_without_rerank(self) -> None:
        with patch(
            "app._call_query_parser_llm",
            return_value={
                "intent": "keyword",
                "lexical_query": "推荐系统",
                "semantic_query": "推荐系统",
                "constraints": {
                    "min_years": 0.5,
                    "degree": "本科",
                    "cities": ["北京"],
                    "skills": ["推荐系统"],
                },
                "enable_dense": False,
                "enable_rerank": False,
            },
        ):
            plan = _plan_query(
                "0.5年以上 北京 本科 推荐系统",
                [],
                size=10,
                facets={
                    "degrees": [{"key": "本科"}],
                    "cities": [{"key": "北京"}],
                    "skills": [{"key": "推荐系统"}],
                },
            )

        self.assertEqual(plan.intent, INTENT_KEYWORD)
        self.assertEqual(plan.lexical_query, "推荐系统")
        self.assertEqual(plan.semantic_query, "推荐系统")
        self.assertFalse(plan.enable_dense)
        self.assertFalse(plan.enable_rerank)
        self.assertEqual(plan.constraints["skills"], ["推荐系统"])
        self.assertNotIn({"term": {"skills": "推荐系统"}}, plan.filters)
        self.assertIn({"term": {"candidate.highest_degree": "本科"}}, plan.filters)
        self.assertIn({"terms": {"application.expected_work_cities": ["北京"]}}, plan.filters)
        self.assertIn({"range": {"candidate.years_experience": {"gte": 0.0}}}, plan.filters)
        debug_plan = plan.to_debug_dict()
        self.assertEqual(debug_plan["raw_query"], "0.5年以上 北京 本科 推荐系统")
        self.assertEqual(debug_plan["lexical_query"], "推荐系统")
        self.assertEqual(debug_plan["constraints"]["skills"], ["推荐系统"])

    def test_query_plan_treats_degree_floor_as_range_filter(self) -> None:
        with patch(
            "app._call_query_parser_llm",
            return_value={
                "intent": "semantic",
                "lexical_query": "大模型 RAG Agent",
                "semantic_query": "大模型 RAG Agent",
                "constraints": {
                    "degree": "本科",
                    "skills": ["RAG"],
                },
                "enable_dense": True,
                "enable_rerank": True,
            },
        ):
            plan = _plan_query(
                "本科及以上 大模型 RAG Agent",
                [],
                size=10,
                facets={
                    "degrees": [{"key": "本科"}, {"key": "硕士"}, {"key": "博士"}],
                    "cities": [],
                    "skills": [{"key": "RAG"}],
                },
            )

        self.assertEqual(plan.constraints["min_degree"], "本科")
        self.assertNotIn({"term": {"candidate.highest_degree": "本科"}}, plan.filters)
        self.assertIn({"terms": {"candidate.highest_degree": ["本科", "硕士", "博士"]}}, plan.filters)

    def test_query_plan_keeps_llm_skill_constraints_soft(self) -> None:
        with patch(
            "app._call_query_parser_llm",
            return_value={
                "intent": "keyword",
                "lexical_query": "java",
                "semantic_query": "java",
                "constraints": {"skills": ["java"]},
                "enable_dense": False,
                "enable_rerank": False,
            },
        ):
            plan = _plan_query(
                "0.5年以上 北京 本科 java",
                [],
                size=10,
                facets={
                    "degrees": [{"key": "本科"}],
                    "cities": [{"key": "北京"}],
                    "skills": [{"key": "Java"}],
                },
            )

        self.assertEqual(plan.constraints["skills"], ["java"])
        self.assertNotIn({"term": {"skills": "java"}}, plan.filters)
        self.assertNotIn({"term": {"skills": "Java"}}, plan.filters)

    def test_query_plan_disables_rerank_for_lookup_queries(self) -> None:
        with patch(
            "app._call_query_parser_llm",
            return_value={
                "intent": "lookup",
                "lexical_query": "M20260013",
                "semantic_query": "M20260013",
                "constraints": {},
                "enable_dense": False,
                "enable_rerank": True,
            },
        ):
            plan = _plan_query(
                "M20260013",
                [],
                size=10,
                facets={"degrees": [], "cities": [], "skills": []},
            )

        self.assertEqual(plan.intent, INTENT_LOOKUP)
        self.assertFalse(plan.enable_dense)
        self.assertFalse(plan.enable_rerank)

    def test_query_plan_routes_semantic_as_hybrid_query(self) -> None:
        with patch(
            "app._call_query_parser_llm",
            return_value={
                "intent": "semantic",
                "lexical_query": "推荐系统 NLP SQL",
                "semantic_query": "推荐系统 NLP SQL",
                "constraints": {},
                "enable_dense": True,
                "enable_rerank": False,
            },
        ):
            plan = _plan_query(
                "推荐系统 NLP SQL",
                [],
                size=10,
                facets={
                    "degrees": [{"key": "本科"}],
                    "cities": [{"key": "北京"}],
                    "skills": [{"key": "推荐系统"}, {"key": "NLP"}, {"key": "SQL"}],
                },
            )

        self.assertEqual(plan.intent, INTENT_SEMANTIC)
        self.assertEqual(plan.filters, [])
        self.assertTrue(plan.enable_dense)
        self.assertTrue(plan.enable_rerank)

    def test_query_plan_disables_dense_for_keyword_queries(self) -> None:
        with patch(
            "app._call_query_parser_llm",
            return_value={
                "intent": "keyword",
                "lexical_query": "北京大学",
                "semantic_query": "",
                "constraints": {},
                "enable_dense": False,
                "enable_rerank": False,
            },
        ):
            plan = _plan_query(
                "北京大学",
                [],
                size=10,
                facets={
                    "degrees": [{"key": "本科"}],
                    "cities": [{"key": "北京"}],
                    "skills": [{"key": "Python"}],
                },
            )

        self.assertEqual(plan.intent, INTENT_KEYWORD)
        self.assertFalse(plan.enable_dense)
        self.assertFalse(plan.enable_rerank)

    def test_query_plan_routes_natural_language_to_semantic(self) -> None:
        with patch(
            "app._call_query_parser_llm",
            return_value={
                "intent": "semantic",
                "lexical_query": "推荐系统召回 NLP 模型落地",
                "semantic_query": "做过推荐系统召回和 NLP 模型落地的人",
                "constraints": {},
                "enable_dense": True,
                "enable_rerank": True,
            },
        ):
            plan = _plan_query(
                "做过推荐系统召回和 NLP 模型落地的人",
                [],
                size=10,
                facets={
                    "degrees": [{"key": "本科"}],
                    "cities": [{"key": "北京"}],
                    "skills": [{"key": "NLP"}],
                },
            )

        self.assertEqual(plan.intent, INTENT_SEMANTIC)
        self.assertTrue(plan.enable_dense)
        self.assertTrue(plan.enable_rerank)

    def test_rerank_results_reorders_top_window_and_keeps_tail(self) -> None:
        results = [
            _formatted_result("a", "RRF 第一", "Vue3 后台管理系统", 0.9),
            _formatted_result("b", "RRF 第二", "企业知识库 RAG 向量检索系统", 0.8),
            _formatted_result("c", "RRF 第三", "普通 Java 后端系统", 0.7),
        ]

        with patch(
            "app._score_rerank_documents",
            return_value=[0.05, 0.95],
        ):
            reranked, warnings = _rerank_results("需要 RAG 和向量检索经验", results, top_n=2)

        self.assertEqual(warnings, [])
        self.assertEqual([item["id"] for item in reranked], ["b", "a", "c"])
        self.assertEqual(reranked[0]["score"], 0.95)
        self.assertEqual(reranked[0]["retrieval_debug"]["pre_rerank_rank"], 2)
        self.assertEqual(reranked[0]["retrieval_debug"]["rerank_rank"], 1)
        self.assertTrue(reranked[0]["retrieval_debug"]["rerank_applied"])
        self.assertEqual(reranked[0]["retrieval_debug"]["rerank_window_size"], 2)
        self.assertFalse(reranked[2]["retrieval_debug"]["rerank_applied"])
        self.assertEqual(reranked[2]["retrieval_debug"]["rerank_skip_reason"], "outside_top_n")
        self.assertEqual(reranked[2]["retrieval_debug"]["pre_rerank_rank"], 3)
        self.assertNotIn("rerank_rank", reranked[2]["retrieval_debug"])

    def test_rerank_results_abstains_when_top_score_below_floor(self) -> None:
        results = [
            _formatted_result("a", "弱相关一", "Oracle DBA 数据库巡检", 0.9),
            _formatted_result("b", "弱相关二", "SAP 财务实施", 0.8),
        ]

        # Both candidates score below the relevance floor -> abstain (empty).
        with patch(
            "app._score_rerank_documents",
            return_value=[0.28, 0.31],
        ):
            reranked, warnings = _rerank_results("量子计算芯片设计经验", results, top_n=2)

        self.assertEqual(reranked, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("abstained", warnings[0])

    def test_rerank_results_keeps_window_when_one_candidate_clears_floor(self) -> None:
        results = [
            _formatted_result("a", "弱相关", "Oracle DBA 数据库巡检", 0.9),
            _formatted_result("b", "强相关", "企业知识库 RAG 向量检索系统", 0.8),
        ]

        # One candidate clears the floor -> keep the window (no abstain).
        with patch(
            "app._score_rerank_documents",
            return_value=[0.28, 0.91],
        ):
            reranked, warnings = _rerank_results("需要 RAG 经验", results, top_n=2)

        self.assertEqual(warnings, [])
        self.assertEqual([item["id"] for item in reranked], ["b", "a"])

    def test_rerank_document_uses_resume_evidence_fields(self) -> None:
        document = _rerank_document(
            _formatted_result(
                "rag-1",
                "RAG 候选",
                "企业级RAG知识问答系统",
                0.8,
                skills=["RAG", "LangChain"],
            )
        )

        self.assertIn("应聘岗位: RAG 候选", document)
        self.assertIn("技能: RAG、LangChain", document)
        self.assertIn("项目: 企业级RAG知识问答系统", document)

    def test_rerank_document_does_not_truncate_available_project_or_internship_text(self) -> None:
        result = _formatted_result("long-1", "算法候选", "项目1", 0.8)
        result["source"]["projects"] = [
            {"name": f"项目{i}", "description": f"描述{i}", "responsibility": f"职责{i}"}
            for i in range(1, 7)
        ]
        result["source"]["internships"] = [
            {"company": f"公司{i}", "department": "算法部", "title": "算法实习生", "description": f"实习描述{i}"}
            for i in range(1, 7)
        ]

        document = _rerank_document(result)

        self.assertIn("项目: 项目6 描述6 职责6", document)
        self.assertIn("经历: 公司6 算法部 算法实习生 实习描述6", document)

    def test_evidence_lexical_query_covers_profile_fields(self) -> None:
        query_json = json.dumps(_evidence_lexical_query("A0009"), ensure_ascii=False)

        self.assertIn("application.candidate_no", query_json)
        self.assertIn("application.position_code", query_json)
        self.assertIn("candidate.name.keyword", query_json)
        self.assertIn("candidate.all_schools.keyword", query_json)
        self.assertIn("candidate.major.keyword", query_json)
        self.assertIn("evidence_exact:position_code", query_json)

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
            ["internship", "project"],
        )
        evidence_props = EVIDENCE_INDEX_BODY["mappings"]["properties"]
        self.assertIn(EVIDENCE_VECTOR_FIELD, evidence_props)
        self.assertIn("embedding", evidence_props)
        self.assertIn("phrase", evidence_props["text"]["fields"])
        self.assertIn("keyword", evidence_props["title"]["fields"])
        self.assertIn("name", evidence_props["candidate"]["properties"])
        self.assertIn("company", evidence_props["application"]["properties"])


def _find_multi_match_by_name(node: object, name: str) -> dict:
    if isinstance(node, dict):
        multi_match = node.get("multi_match")
        if isinstance(multi_match, dict) and multi_match.get("_name") == name:
            return multi_match
        for value in node.values():
            try:
                return _find_multi_match_by_name(value, name)
            except AssertionError:
                continue
    elif isinstance(node, list):
        for value in node:
            try:
                return _find_multi_match_by_name(value, name)
            except AssertionError:
                continue
    raise AssertionError(f"multi_match query not found: {name}")


def _find_clause_with_name_prefix(node: object, name_prefix: str):
    """返回内层 `_name` 以 name_prefix 开头的查询包装子句（如 {"match_phrase": {...}}）。"""
    if isinstance(node, dict):
        for clause_type, params in node.items():
            if isinstance(params, dict):
                nm = params.get("_name")
                if isinstance(nm, str) and nm.startswith(name_prefix):
                    return {clause_type: params}
                for field_params in params.values():
                    if isinstance(field_params, dict):
                        fnm = field_params.get("_name")
                        if isinstance(fnm, str) and fnm.startswith(name_prefix):
                            return {clause_type: params}
        for value in node.values():
            found = _find_clause_with_name_prefix(value, name_prefix)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_clause_with_name_prefix(value, name_prefix)
            if found is not None:
                return found
    return None


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


def _formatted_result(
    doc_id: str,
    position_name: str,
    project_name: str,
    score: float,
    *,
    skills: list[str] | None = None,
) -> dict:
    skills = skills or []
    return {
        "id": doc_id,
        "score": score,
        "candidate": {
            "highest_degree": "硕士",
            "school": "测试大学",
            "major": "计算机科学与技术",
            "years_experience": 3,
        },
        "application": {
            "position_name": position_name,
            "expected_work_cities": ["北京"],
        },
        "skills": skills,
        "retrieval_debug": {"rrf_score": score},
        "source": {
            "candidate": {
                "highest_degree": "硕士",
                "school": "测试大学",
                "major": "计算机科学与技术",
                "years_experience": 3,
            },
            "application": {
                "position_name": position_name,
                "expected_work_cities": ["北京"],
            },
            "skills": skills,
            "projects": [
                {
                    "name": project_name,
                    "description": "项目描述",
                    "responsibility": "核心职责",
                }
            ],
            "internships": [],
        },
    }


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


class QueryPlanFastPathAndCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        import app

        app._query_plan_cache.clear()

    def test_email_lookup_short_circuits_llm(self) -> None:
        with patch("app._call_query_parser_llm") as mock_llm:
            parsed = _parse_query_with_llm("zhangwei_mock@example.com")

        mock_llm.assert_not_called()
        self.assertEqual(parsed["intent"], INTENT_LOOKUP)
        self.assertEqual(parsed["parser"], "regex_fast_path")
        self.assertFalse(parsed["enable_dense"])

    def test_phone_and_candidate_no_fast_path(self) -> None:
        self.assertEqual(_lookup_fast_path("13800138000")["intent"], INTENT_LOOKUP)
        self.assertEqual(_lookup_fast_path("M20260013")["intent"], INTENT_LOOKUP)
        self.assertEqual(_lookup_fast_path("A0009")["intent"], INTENT_LOOKUP)

    def test_multi_token_and_entity_queries_skip_fast_path(self) -> None:
        self.assertIsNone(_lookup_fast_path("北京大学"))
        self.assertIsNone(_lookup_fast_path("M20260013 推荐系统"))
        self.assertIsNone(_lookup_fast_path("Python"))

    def test_repeated_query_is_served_from_cache(self) -> None:
        payload = {
            "intent": "semantic",
            "lexical_query": "RAG 向量检索",
            "semantic_query": "RAG 向量检索",
            "constraints": {},
            "enable_dense": True,
        }
        with patch(
            "app._call_query_parser_llm", return_value=payload
        ) as mock_llm:
            first = _parse_query_with_llm("找做过 RAG 的人")
            second = _parse_query_with_llm("找做过 RAG 的人")

        self.assertEqual(mock_llm.call_count, 1)
        self.assertEqual(first["intent"], "semantic")
        self.assertEqual(second["intent"], "semantic")
        # Cached copies must be independent so downstream mutation can't leak.
        first["constraints"]["mutated"] = True
        self.assertNotIn("mutated", second["constraints"])

    def test_cache_key_normalizes_whitespace_and_case(self) -> None:
        payload = {
            "intent": "keyword",
            "lexical_query": "北京大学",
            "semantic_query": "",
            "constraints": {},
            "enable_dense": False,
        }
        with patch(
            "app._call_query_parser_llm", return_value=payload
        ) as mock_llm:
            _parse_query_with_llm("北京大学")
            _parse_query_with_llm("  北京大学  ")

        self.assertEqual(mock_llm.call_count, 1)

    def test_failed_parse_is_not_cached(self) -> None:
        with patch(
            "app._call_query_parser_llm", side_effect=RuntimeError("boom")
        ) as mock_llm:
            first = _parse_query_with_llm("做过推荐系统的人")
            second = _parse_query_with_llm("做过推荐系统的人")

        self.assertEqual(mock_llm.call_count, 2)
        self.assertIn("parser_warning", first["constraints"])
        self.assertIn("parser_warning", second["constraints"])


class MinYearsToleranceTests(unittest.TestCase):
    def test_min_years_filter_applies_soft_tolerance(self) -> None:
        # "4 年以上" should still admit a 3.9y candidate. tolerance =
        # max(4*0.1, 0.5) = 0.5, so gte=3.5.
        gte = _min_years_filter(4)["range"]["candidate.years_experience"]["gte"]
        self.assertAlmostEqual(gte, 3.5, places=3)
        self.assertLess(gte, 3.9)

    def test_large_min_years_uses_ratio_band(self) -> None:
        # For large thresholds the 10% ratio dominates the 0.5y floor.
        gte = _min_years_filter(8)["range"]["candidate.years_experience"]["gte"]
        self.assertAlmostEqual(gte, 7.2, places=3)

    def test_small_min_years_uses_flat_floor(self) -> None:
        # For small thresholds the 0.5y floor dominates the 10% ratio.
        gte = _min_years_filter(2)["range"]["candidate.years_experience"]["gte"]
        self.assertAlmostEqual(gte, 1.5, places=3)

    def test_min_years_filter_never_goes_negative(self) -> None:
        gte = _min_years_filter(0.3)["range"]["candidate.years_experience"]["gte"]
        self.assertEqual(gte, 0.0)


if __name__ == "__main__":
    unittest.main()
