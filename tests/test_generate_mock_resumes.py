import build_eval_queries
import generate_mock_resumes as gen


def test_generated_resumes_use_real_companies_and_unique_experiences() -> None:
    docs = gen.generate(200, gen.SEED)
    stats = gen._quality_stats(docs)

    assert stats["unique_names"] == stats["total_names"]
    assert stats["unique_companies"] >= 20
    assert stats["fake_companies"] == []
    assert stats["duplicate_project_signatures"] == 0
    assert stats["duplicate_internship_signatures"] == 0


def test_structured_filter_cases_have_enough_ground_truth() -> None:
    families, by_id = build_eval_queries._load_ground_truth()

    for case in build_eval_queries._build_structured_filter_cases(families, by_id):
        assert len(case["relevance"]) >= 2
