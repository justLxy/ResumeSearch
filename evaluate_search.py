from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
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
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print per-query result ids.",
    )
    parser.add_argument(
        "--type-summary",
        action="store_true",
        help="Print metrics grouped by eval case type.",
    )
    args = parser.parse_args()

    cases = load_cases(Path(args.qrels))
    if not cases:
        raise SystemExit(f"No eval cases found in {args.qrels}")

    client = TestClient(search_app.app)
    results = [evaluate_case(client, case, args.limit) for case in cases]
    summary = summarize(results)

    print_summary_table(summary)
    if args.type_summary:
        print_type_summary(results)
    if args.details:
        print_details(results)


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
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
    return cases


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


def _mean_metric(results: list[QueryResult], attr: str) -> float:
    if not results:
        return 0.0
    return mean(float(getattr(result, attr)) for result in results)


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


def print_type_summary(results: list[QueryResult]) -> None:
    grouped: dict[str, list[QueryResult]] = {}
    for result in results:
        grouped.setdefault(result.case.case_type, []).append(result)

    print("\nType summary")
    print("type                    queries judged P@5   R@10  NDCG@10 empty_acc forbidden@10")
    for case_type in sorted(grouped):
        rows = grouped[case_type]
        judged = [row for row in rows if row.case.relevant_ids]
        empty_cases = [row for row in rows if row.empty_success is not None]
        empty = (
            mean(1.0 if row.empty_success else 0.0 for row in empty_cases)
            if empty_cases
            else None
        )
        print(
            f"{case_type:<23} "
            f"{len(rows):>7} "
            f"{len(judged):>6} "
            f"{_mean_metric(judged, 'precision_at_5'):.3f} "
            f"{_mean_metric(judged, 'recall_at_10'):.3f} "
            f"{_mean_metric(judged, 'ndcg_at_10'):.3f} "
            f"{'-' if empty is None else f'{empty:.3f}':>9} "
            f"{sum(row.forbidden_at_10 for row in rows):>12}"
        )


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
