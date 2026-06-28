from evaluate_search import (
    EvalCase,
    QueryResult,
    build_report,
    compare_reports,
    load_report,
    write_report,
)


def test_build_report_groups_metrics_by_query_type() -> None:
    results = [
        _result(
            case_id="exact_1",
            case_type="exact_code",
            query="A0001",
            relevant_ids={"resume-1"},
            returned_ids=["resume-1", "resume-2"],
            precision_at_5=0.2,
            precision_at_10=0.1,
            recall_at_5=1.0,
            recall_at_10=1.0,
            mrr_at_10=1.0,
            ndcg_at_10=1.0,
        ),
        _result(
            case_id="negative_1",
            case_type="negative_semantic",
            query="量子计算芯片设计经验",
            relevant_ids=set(),
            returned_ids=[],
            empty_success=True,
        ),
    ]

    report = build_report(results, qrels_path="eval_queries.jsonl", limit=10)

    assert report["overall"]["queries"] == 2
    assert report["overall"]["judged"] == 1
    assert report["overall"]["p5"] == 0.2
    assert report["overall"]["empty_accuracy"] == 1.0
    assert report["by_type"]["exact_code"]["ndcg10"] == 1.0
    assert report["by_type"]["negative_semantic"]["empty_accuracy"] == 1.0
    assert report["details"][0]["relevant_hits_at_10"] == ["resume-1"]


def test_report_json_round_trip(tmp_path) -> None:
    report = build_report(
        [
            _result(
                case_id="skill_1",
                case_type="skill_combo",
                query="Python NLP",
                relevant_ids={"resume-1"},
                returned_ids=["resume-1"],
                ndcg_at_10=1.0,
            )
        ],
        qrels_path="eval_queries.jsonl",
        limit=10,
    )
    path = tmp_path / "report.json"

    write_report(report, path)
    loaded = load_report(path)

    assert loaded["overall"]["queries"] == 1
    assert loaded["by_type"]["skill_combo"]["queries"] == 1


def test_compare_reports_returns_metric_deltas() -> None:
    previous = {
        "overall": {"queries": 2, "ndcg10": 0.5, "mrr10": 0.4, "r10": 0.3, "r100": 0.5, "forbidden10": 2},
        "by_type": {
            "semantic": {"queries": 1, "ndcg10": 0.2, "mrr10": 0.2, "r10": 0.2, "r100": 0.4, "forbidden10": 1}
        },
    }
    current = {
        "overall": {"queries": 3, "ndcg10": 0.7, "mrr10": 0.5, "r10": 0.4, "r100": 0.7, "forbidden10": 1},
        "by_type": {
            "semantic": {"queries": 1, "ndcg10": 0.6, "mrr10": 0.4, "r10": 0.5, "r100": 0.9, "forbidden10": 0},
            "entity": {"queries": 1, "ndcg10": 1.0, "mrr10": 1.0, "r10": 1.0, "r100": 1.0, "forbidden10": 0},
        },
    }

    comparison = compare_reports(current, previous)

    assert comparison["overall"]["queries"]["delta"] == 1
    assert comparison["overall"]["ndcg10"]["delta"] == 0.19999999999999996
    assert comparison["overall"]["forbidden10"]["delta"] == -1
    assert comparison["by_type"]["semantic"]["r10"]["delta"] == 0.3
    assert comparison["by_type"]["semantic"]["r100"]["delta"] == 0.5
    assert comparison["by_type"]["entity"]["ndcg10"]["delta"] is None


def _result(
    *,
    case_id: str,
    case_type: str,
    query: str,
    relevant_ids: set[str],
    returned_ids: list[str],
    precision_at_5: float = 0.0,
    precision_at_10: float = 0.0,
    recall_at_5: float = 0.0,
    recall_at_10: float = 0.0,
    recall_at_50: float = 0.0,
    recall_at_100: float = 0.0,
    mrr_at_10: float = 0.0,
    ndcg_at_10: float = 0.0,
    forbidden_at_10: int = 0,
    empty_success: bool | None = None,
) -> QueryResult:
    return QueryResult(
        case=EvalCase(
            case_id=case_id,
            query=query,
            case_type=case_type,
            relevant_ids=relevant_ids,
            forbidden_ids=set(),
            expect_empty=empty_success is not None,
        ),
        returned_ids=returned_ids,
        precision_at_5=precision_at_5,
        precision_at_10=precision_at_10,
        recall_at_5=recall_at_5,
        recall_at_10=recall_at_10,
        recall_at_50=recall_at_50,
        recall_at_100=recall_at_100,
        mrr_at_10=mrr_at_10,
        ndcg_at_10=ndcg_at_10,
        forbidden_at_10=forbidden_at_10,
        empty_success=empty_success,
    )
