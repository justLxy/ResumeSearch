import unittest

from app import BM25_RETRIEVER, DENSE_RETRIEVER, _default_snippet, _hybrid_total, _rrf_merge, _use_dense
from import_to_es import _build_search_text, _embedding_inputs, _estimate_years_experience


class SearchLogicTests(unittest.TestCase):
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
        self.assertEqual(results[0]["retrieval_debug"]["dense_rank"], 2)
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

        search_text = _build_search_text(doc)
        embedding_inputs = _embedding_inputs({"search_text": search_text, **doc})

        self.assertNotIn("北京大学", search_text)
        self.assertNotIn("北京", search_text)
        self.assertNotIn("张三", search_text)
        self.assertNotIn("百度在线网络技术", embedding_inputs["semantic_profile_vector"])
        self.assertNotIn("学士", embedding_inputs["semantic_profile_vector"])
        self.assertLessEqual(len(embedding_inputs["semantic_profile_vector"]), 512)
        self.assertIn("机器学习", search_text)
        self.assertIn("自然语言处理", embedding_inputs["semantic_profile_vector"])
        self.assertIn("推荐系统", embedding_inputs["semantic_profile_vector"])
        self.assertIn("医疗问答系统", embedding_inputs["semantic_profile_vector"])
        self.assertLess(search_text.index("项目名称"), search_text.index("实习职位"))


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


def _hit(doc_id: str, name: str, score: float = 1.0) -> dict:
    return {
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


if __name__ == "__main__":
    unittest.main()
