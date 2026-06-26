from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import requests
from fastapi.testclient import TestClient

import app as search_app


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    query: str
    case_type: str
    relevant_ids: set[str]
    forbidden_ids: set[str]
    expect_empty: bool


@dataclass(frozen=True)
class SkippedCase:
    case_id: str
    line_no: int
    query: str
    case_type: str
    reason: str


@dataclass(frozen=True)
class EvalCaseSet:
    cases: list[EvalCase]
    skipped: list[SkippedCase]


@dataclass(frozen=True)
class QueryResult:
    case: EvalCase
    returned_ids: list[str]
    precision_at_5: float
    precision_at_10: float
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    ndcg_at_10: float
    forbidden_at_10: int
    empty_success: bool | None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate resume search quality for the current retrieval configuration."
    )
    parser.add_argument("--qrels", default="eval_queries.jsonl", help="JSONL qrels file.")
    parser.add_argument("--limit", type=int, default=10, help="Search result window.")
    parser.add_argument("--output", help="Write the full evaluation report as JSON.")
    parser.add_argument("--compare-to", help="Compare this run with a previous JSON report.")
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print per-query result ids.",
    )
    args = parser.parse_args()

    case_set = load_cases(Path(args.qrels))
    cases = case_set.cases
    if not cases:
        raise SystemExit(f"No eval cases found in {args.qrels}")

    client = TestClient(search_app.app)
    results = [evaluate_case(client, case, args.limit) for case in cases]
    report = build_report(
        results,
        qrels_path=args.qrels,
        limit=args.limit,
        skipped_cases=case_set.skipped,
    )

    print_summary_table(report["overall"])
    print_type_summary(report)
    if case_set.skipped:
        print_skipped_cases(case_set.skipped)
    if args.details:
        print_details(results)
    if args.output:
        write_report(report, Path(args.output))
    if args.compare_to:
        previous = load_report(Path(args.compare_to))
        print_comparison_report(compare_reports(report, previous), Path(args.compare_to))


def load_cases(path: Path) -> EvalCaseSet:
    cases: list[EvalCase] = []
    skipped: list[SkippedCase] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        case_id = raw["id"]
        relevant_ids = set(raw.get("relevant_ids") or [])
        forbidden_ids = set(raw.get("forbidden_ids") or [])

        if "relevant_es_query" in raw:
            relevant_ids.update(fetch_ids(raw["relevant_es_query"]))
        if "forbidden_es_query" in raw:
            forbidden_ids.update(fetch_ids(raw["forbidden_es_query"]))

        if relevant_ids & forbidden_ids:
            overlap = ", ".join(sorted(relevant_ids & forbidden_ids))
            raise ValueError(f"{path}:{line_no} has ids marked relevant and forbidden: {overlap}")
        if raw.get("expect_empty") and relevant_ids:
            raise ValueError(f"{path}:{line_no} expects empty results but has relevant ids")
        if not raw.get("expect_empty") and not relevant_ids:
            if "relevant_es_query" in raw and not raw.get("relevant_ids"):
                skipped.append(
                    SkippedCase(
                        case_id=case_id,
                        line_no=line_no,
                        query=raw["query"],
                        case_type=raw.get("type", "unknown"),
                        reason="relevant_es_query matched no documents in current index",
                    )
                )
                continue
            raise ValueError(f"{path}:{line_no} has no relevant ids")

        cases.append(
            EvalCase(
                case_id=case_id,
                query=raw["query"],
                case_type=raw.get("type", "unknown"),
                relevant_ids=relevant_ids,
                forbidden_ids=forbidden_ids,
                expect_empty=bool(raw.get("expect_empty", False)),
            )
        )
    return EvalCaseSet(cases=cases, skipped=skipped)


def fetch_ids(query: dict[str, Any]) -> set[str]:
    body = {
        "size": 10_000,
        "track_total_hits": True,
        "_source": False,
        "query": query,
    }
    url = f"{search_app.ES_URL}/{search_app.INDEX_ALIAS}/_search"
    response = requests.post(url, json=body, timeout=30)
    response.raise_for_status()
    hits = response.json().get("hits", {}).get("hits", [])
    return {hit["_id"] for hit in hits}


def evaluate_case(client: TestClient, case: EvalCase, limit: int) -> QueryResult:
    response = client.get("/api/search", params={"q": case.query, "limit": limit})
    response.raise_for_status()
    payload = response.json()
    returned_ids = [item["id"] for item in payload.get("results", [])]

    return QueryResult(
        case=case,
        returned_ids=returned_ids,
        precision_at_5=precision_at(returned_ids, case.relevant_ids, 5),
        precision_at_10=precision_at(returned_ids, case.relevant_ids, 10),
        recall_at_5=recall_at(returned_ids, case.relevant_ids, 5),
        recall_at_10=recall_at(returned_ids, case.relevant_ids, 10),
        mrr_at_10=mrr_at(returned_ids, case.relevant_ids, 10),
        ndcg_at_10=ndcg_at(returned_ids, case.relevant_ids, 10),
        forbidden_at_10=sum(1 for doc_id in returned_ids[:10] if doc_id in case.forbidden_ids),
        empty_success=(len(returned_ids) == 0) if case.expect_empty else None,
    )


def precision_at(returned_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return _hits_at(returned_ids, relevant_ids, k) / k


def recall_at(returned_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return _hits_at(returned_ids, relevant_ids, k) / len(relevant_ids)


def mrr_at(returned_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    for rank, doc_id in enumerate(returned_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1 / rank
    return 0.0


def ndcg_at(returned_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    dcg = 0.0
    for rank, doc_id in enumerate(returned_ids[:k], start=1):
        if doc_id in relevant_ids:
            dcg += 1 / math.log2(rank + 1)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def _hits_at(returned_ids: list[str], relevant_ids: set[str], k: int) -> int:
    return sum(1 for doc_id in returned_ids[:k] if doc_id in relevant_ids)


def build_report(
    results: list[QueryResult],
    *,
    qrels_path: str,
    limit: int,
    skipped_cases: list[SkippedCase] | None = None,
) -> dict[str, Any]:
    skipped_cases = skipped_cases or []
    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "qrels": qrels_path,
            "limit": limit,
            "index_alias": search_app.INDEX_ALIAS,
            "evidence_index_alias": search_app.EVIDENCE_INDEX_ALIAS,
            "skipped": len(skipped_cases),
        },
        "overall": summarize(results),
        "by_type": summarize_by_type(results),
        "details": [result_to_detail(result) for result in results],
        "skipped": [skipped_case_to_detail(case) for case in skipped_cases],
    }


def summarize(results: list[QueryResult]) -> dict[str, Any]:
    judged = [result for result in results if result.case.relevant_ids]
    empty_cases = [result for result in results if result.empty_success is not None]
    return {
        "queries": len(results),
        "judged": len(judged),
        "p5": _mean_metric(judged, "precision_at_5"),
        "p10": _mean_metric(judged, "precision_at_10"),
        "r5": _mean_metric(judged, "recall_at_5"),
        "r10": _mean_metric(judged, "recall_at_10"),
        "mrr10": _mean_metric(judged, "mrr_at_10"),
        "ndcg10": _mean_metric(judged, "ndcg_at_10"),
        "forbidden10": sum(result.forbidden_at_10 for result in results),
        "empty_accuracy": (
            mean(1.0 if result.empty_success else 0.0 for result in empty_cases)
            if empty_cases
            else None
        ),
    }


def summarize_by_type(results: list[QueryResult]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[QueryResult]] = {}
    for result in results:
        grouped.setdefault(result.case.case_type, []).append(result)
    return {
        case_type: summarize(rows)
        for case_type, rows in sorted(grouped.items())
    }


def result_to_detail(result: QueryResult) -> dict[str, Any]:
    return {
        "id": result.case.case_id,
        "type": result.case.case_type,
        "query": result.case.query,
        "returned_ids": result.returned_ids,
        "relevant_hits_at_10": [
            doc_id for doc_id in result.returned_ids[:10] if doc_id in result.case.relevant_ids
        ],
        "forbidden_hits_at_10": [
            doc_id for doc_id in result.returned_ids[:10] if doc_id in result.case.forbidden_ids
        ],
        "p5": result.precision_at_5,
        "p10": result.precision_at_10,
        "r5": result.recall_at_5,
        "r10": result.recall_at_10,
        "mrr10": result.mrr_at_10,
        "ndcg10": result.ndcg_at_10,
        "forbidden10": result.forbidden_at_10,
        "empty_success": result.empty_success,
    }


def skipped_case_to_detail(case: SkippedCase) -> dict[str, Any]:
    return {
        "id": case.case_id,
        "line_no": case.line_no,
        "type": case.case_type,
        "query": case.query,
        "reason": case.reason,
    }


def _mean_metric(results: list[QueryResult], attr: str) -> float:
    if not results:
        return 0.0
    return mean(float(getattr(result, attr)) for result in results)


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "overall" not in payload:
        raise ValueError(f"{path} is not an evaluation report JSON")
    return payload


COMPARISON_METRICS = (
    "p5",
    "p10",
    "r5",
    "r10",
    "mrr10",
    "ndcg10",
    "empty_accuracy",
    "forbidden10",
)


def compare_reports(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    current_by_type = current.get("by_type") or {}
    previous_by_type = previous.get("by_type") or {}
    case_types = sorted(set(current_by_type) | set(previous_by_type))
    return {
        "overall": compare_summary(current.get("overall") or {}, previous.get("overall") or {}),
        "by_type": {
            case_type: compare_summary(
                current_by_type.get(case_type) or {},
                previous_by_type.get(case_type) or {},
            )
            for case_type in case_types
        },
    }


def compare_summary(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "queries": {
            "previous": previous.get("queries", 0),
            "current": current.get("queries", 0),
            "delta": (current.get("queries", 0) or 0) - (previous.get("queries", 0) or 0),
        }
    }
    for metric in COMPARISON_METRICS:
        previous_value = previous.get(metric)
        current_value = current.get(metric)
        comparison[metric] = {
            "previous": previous_value,
            "current": current_value,
            "delta": _metric_delta(current_value, previous_value),
        }
    return comparison


def _metric_delta(current_value: Any, previous_value: Any) -> float | None:
    if current_value is None or previous_value is None:
        return None
    return float(current_value) - float(previous_value)


def print_summary_table(row: dict[str, Any]) -> None:
    print("\nSearch quality summary")
    print(
        "queries judged  P@5    P@10   R@5    R@10   MRR@10 NDCG@10 empty_acc forbidden@10"
    )
    empty = "-" if row["empty_accuracy"] is None else f"{row['empty_accuracy']:.3f}"
    print(
        f"{row['queries']:>7} "
        f"{row['judged']:>6}  "
        f"{row['p5']:.3f}  "
        f"{row['p10']:.3f}  "
        f"{row['r5']:.3f}  "
        f"{row['r10']:.3f}  "
        f"{row['mrr10']:.3f}  "
        f"{row['ndcg10']:.3f}  "
        f"{empty:>9}  "
        f"{row['forbidden10']:>12}"
    )


def print_type_summary(report: dict[str, Any]) -> None:
    print("\nType summary")
    print("type                    queries judged P@5   R@10  NDCG@10 empty_acc forbidden@10")
    for case_type, row in (report.get("by_type") or {}).items():
        empty = row.get("empty_accuracy")
        print(
            f"{case_type:<23} "
            f"{row['queries']:>7} "
            f"{row['judged']:>6} "
            f"{row['p5']:.3f} "
            f"{row['r10']:.3f} "
            f"{row['ndcg10']:.3f} "
            f"{'-' if empty is None else f'{empty:.3f}':>9} "
            f"{row['forbidden10']:>12}"
        )


def print_skipped_cases(skipped: list[SkippedCase]) -> None:
    print(f"\nSkipped {len(skipped)} eval cases")
    grouped: dict[str, int] = {}
    for case in skipped:
        grouped[case.case_type] = grouped.get(case.case_type, 0) + 1
    for case_type, count in sorted(grouped.items()):
        print(f"- {case_type}: {count}")


def print_comparison_report(comparison: dict[str, Any], previous_path: Path) -> None:
    print(f"\nComparison vs {previous_path}")
    print("scope                   queriesΔ NDCG@10Δ MRR@10Δ R@10Δ  emptyΔ  forbiddenΔ")
    print_comparison_row("overall", comparison["overall"])
    for case_type, row in (comparison.get("by_type") or {}).items():
        print_comparison_row(case_type, row)


def print_comparison_row(label: str, row: dict[str, Any]) -> None:
    print(
        f"{label:<23} "
        f"{_format_delta(row['queries']['delta'], digits=0):>8} "
        f"{_format_delta(row['ndcg10']['delta']):>8} "
        f"{_format_delta(row['mrr10']['delta']):>7} "
        f"{_format_delta(row['r10']['delta']):>6} "
        f"{_format_delta(row['empty_accuracy']['delta']):>7} "
        f"{_format_delta(row['forbidden10']['delta'], digits=0):>10}"
    )


def _format_delta(value: float | int | None, *, digits: int = 3) -> str:
    if value is None:
        return "-"
    if digits == 0:
        return f"{int(value):+d}"
    return f"{float(value):+.{digits}f}"


def print_details(results: list[QueryResult]) -> None:
    print("\nDetails")
    for result in results:
        relevant_hits = [doc_id for doc_id in result.returned_ids[:10] if doc_id in result.case.relevant_ids]
        forbidden_hits = [
            doc_id for doc_id in result.returned_ids[:10] if doc_id in result.case.forbidden_ids
        ]
        print(
            f"- {result.case.case_id} [{result.case.case_type}] "
            f"NDCG@10={result.ndcg_at_10:.3f} R@10={result.recall_at_10:.3f} "
            f"hits={relevant_hits} forbidden={forbidden_hits} returned={result.returned_ids[:10]}"
        )


if __name__ == "__main__":
    main()
