from resume_search.infrastructure import rerank_service


def test_build_payload_contains_query_documents_and_top_n() -> None:
    payload = rerank_service._build_payload(
        "需要做过 RAG 的候选人",
        [" 项目：企业知识库问答 \n\n 使用 LangChain 和向量检索 "],
    )

    assert payload["model"] == "qwen3-rerank"
    assert payload["query"] == "需要做过 RAG 的候选人"
    assert payload["documents"] == ["项目：企业知识库问答\n使用 LangChain 和向量检索"]
    assert payload["parameters"]["return_documents"] is True
    assert payload["parameters"]["top_n"] == 1


def test_extract_scores_restores_original_candidate_order() -> None:
    scores = rerank_service._extract_scores(
        {
            "output": {
                "results": [
                    {"index": 1, "relevance_score": 0.1},
                    {"index": 0, "relevance_score": 0.9},
                ]
            }
        },
        2,
    )

    assert scores == [0.9, 0.1]


def test_extract_scores_accepts_top_level_results() -> None:
    scores = rerank_service._extract_scores(
        {
            "results": [
                {"index": 1, "relevance_score": 0.1},
                {"index": 0, "relevance_score": 0.9},
            ]
        },
        2,
    )

    assert scores == [0.9, 0.1]


def test_clean_document_drops_blank_lines_and_trims_text() -> None:
    assert rerank_service._clean_document("  A  \n\n B \n") == "A\nB"
