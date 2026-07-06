from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import requests
from fastapi.testclient import TestClient

from resume_search import config as _config
from resume_search.api import app as _fastapi_app


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    query: str
    case_type: str
    relevance: dict[str, float]
    relevant_ids: set[str]
    forbidden_ids: set[str]
    expect_empty: bool
    expected_plan: dict[str, Any] = field(default_factory=dict)
    api_params: dict[str, Any] = field(default_factory=dict)


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
    query_plan: dict[str, Any]
    planner_eval: dict[str, Any]
    precision_at_5: float
    precision_at_10: float
    recall_at_5: float
    recall_at_10: float
    recall_at_50: float
    recall_at_100: float
    mrr_at_10: float
    ndcg_at_5: float
    ndcg_at_10: float
    forbidden_at_10: int
    success_at_1: float
    empty_success: bool | None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate resume search quality for the current retrieval configuration."
    )
    parser.add_argument("--qrels", default="eval_queries.jsonl", help="JSONL qrels file.")
    parser.add_argument("--limit", type=int, default=100, help="Search result window.")
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

    client = TestClient(_fastapi_app)
    results = [evaluate_case(client, case, args.limit) for case in cases]
    report = build_report(
        results,
        qrels_path=args.qrels,
        limit=args.limit,
        skipped_cases=case_set.skipped,
    )

    print_summary_table(report["overall"])
    print_type_summary(report)
    print_planner_summary(report)
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
        relevance = _load_relevance(raw)
        relevant_ids = {doc_id for doc_id, grade in relevance.items() if grade > 0}
        forbidden_ids = set(raw.get("forbidden_ids") or [])

        if "relevant_es_query" in raw:
            for doc_id in fetch_ids(raw["relevant_es_query"]):
                relevance[doc_id] = max(relevance.get(doc_id, 0.0), 1.0)
            relevant_ids = {doc_id for doc_id, grade in relevance.items() if grade > 0}
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
                relevance=relevance,
                relevant_ids=relevant_ids,
                forbidden_ids=forbidden_ids,
                expect_empty=bool(raw.get("expect_empty", False)),
                expected_plan=_load_expected_plan(raw),
                api_params=_load_api_params(raw),
            )
        )
    return EvalCaseSet(cases=cases, skipped=skipped)


def _load_expected_plan(raw: dict[str, Any]) -> dict[str, Any]:
    expected_plan = raw.get("expected_plan") or {}
    if not isinstance(expected_plan, dict):
        raise ValueError(f"{raw.get('id', '<unknown>')} expected_plan must be an object")
    return expected_plan


def _load_api_params(raw: dict[str, Any]) -> dict[str, Any]:
    api_params = raw.get("api_params") or {}
    if not isinstance(api_params, dict):
        raise ValueError(f"{raw.get('id', '<unknown>')} api_params must be an object")
    return api_params


def fetch_ids(query: dict[str, Any]) -> set[str]:
    body = {
        "size": 10_000,
        "track_total_hits": True,
        "_source": False,
        "query": query,
    }
    url = f"{_config.ES_URL}/{_config.INDEX_ALIAS}/_search"
    response = requests.post(url, json=body, timeout=30)
    response.raise_for_status()
    hits = response.json().get("hits", {}).get("hits", [])
    return {hit["_id"] for hit in hits}


def evaluate_case(client: TestClient, case: EvalCase, limit: int) -> QueryResult:
    response = client.get("/api/search", params=_search_params(case, limit))
    response.raise_for_status()
    payload = response.json()
    returned_ids = [item["id"] for item in payload.get("results", [])]
    query_plan = payload.get("query_plan") if isinstance(payload.get("query_plan"), dict) else {}
    planner_eval = evaluate_query_plan(query_plan, case.expected_plan)

    return QueryResult(
        case=case,
        returned_ids=returned_ids,
        query_plan=query_plan,
        planner_eval=planner_eval,
        precision_at_5=precision_at(returned_ids, case.relevant_ids, 5),
        precision_at_10=precision_at(returned_ids, case.relevant_ids, 10),
        recall_at_5=recall_at(returned_ids, case.relevant_ids, 5),
        recall_at_10=recall_at(returned_ids, case.relevant_ids, 10),
        recall_at_50=recall_at(returned_ids, case.relevant_ids, 50),
        recall_at_100=recall_at(returned_ids, case.relevant_ids, 100),
        mrr_at_10=mrr_at(returned_ids, case.relevant_ids, 10),
        ndcg_at_5=ndcg_at(returned_ids, case.relevance, 5),
        ndcg_at_10=ndcg_at(returned_ids, case.relevance, 10),
        forbidden_at_10=sum(1 for doc_id in returned_ids[:10] if doc_id in case.forbidden_ids),
        success_at_1=(
            1.0 if (returned_ids and returned_ids[0] in case.relevant_ids) else 0.0
        ),
        empty_success=(len(returned_ids) == 0) if case.expect_empty else None,
    )


def _search_params(case: EvalCase, limit: int) -> list[tuple[str, str]]:
    params: list[tuple[str, str]] = [("q", case.query), ("limit", str(limit))]
    for key, value in case.api_params.items():
        if key in {"q", "limit"} or value in (None, ""):
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            if item not in (None, ""):
                params.append((str(key), str(item)))
    return params


def _load_relevance(raw: dict[str, Any]) -> dict[str, float]:
    relevance: dict[str, float] = {}
    for doc_id in raw.get("relevant_ids") or []:
        relevance[str(doc_id)] = max(relevance.get(str(doc_id), 0.0), 1.0)
    for doc_id, grade in (raw.get("relevance") or {}).items():
        grade_value = float(grade)
        if grade_value < 0:
            raise ValueError(f"relevance grade must be non-negative for {doc_id}")
        relevance[str(doc_id)] = max(relevance.get(str(doc_id), 0.0), grade_value)
    return relevance


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


def ndcg_at(returned_ids: list[str], relevance: dict[str, float], k: int) -> float:
    if not relevance:
        return 0.0
    dcg = 0.0
    for rank, doc_id in enumerate(returned_ids[:k], start=1):
        gain = relevance.get(doc_id, 0.0)
        if gain > 0:
            dcg += (2.0 ** gain - 1.0) / math.log2(rank + 1)
    ideal_gains = sorted((grade for grade in relevance.values() if grade > 0), reverse=True)[:k]
    idcg = sum((2.0 ** gain - 1.0) / math.log2(rank + 1) for rank, gain in enumerate(ideal_gains, start=1))
    return dcg / idcg if idcg else 0.0


def _hits_at(returned_ids: list[str], relevant_ids: set[str], k: int) -> int:
    return sum(1 for doc_id in returned_ids[:k] if doc_id in relevant_ids)


PLANNER_EVAL_FIELDS = (
    "intent",
    "lexical_query",
    "semantic_query",
    "enable_dense",
    "enable_rerank",
)
PLANNER_BOOL_FIELDS = {"enable_dense", "enable_rerank"}
# 这两个字段不做逐字比对：长 JD / 语义类的 lexical_query 由 LLM 压缩，不会逐字
# 等于评测集里写的期望文本。改为"子集/包含"判定——期望文本里的每个核心词都出现
# 在实际值里即算通过。这样既不产生假失配，又能测出"核心检索词有没有被丢掉"。
PLANNER_SUBSET_FIELDS = {"lexical_query", "semantic_query"}


def evaluate_query_plan(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    expected_fields = [field for field in PLANNER_EVAL_FIELDS if field in expected]
    if not expected_fields:
        return {
            "available": False,
            "fields": 0,
            "matched": 0,
            "exact_match": None,
            "field_matches": {},
            "mismatched_fields": [],
            "expected": {},
            "actual": {},
        }

    expected_values = {
        field: _normalize_planner_value(field, expected.get(field))
        for field in expected_fields
    }
    actual_values = {
        field: _normalize_planner_value(field, actual.get(field))
        for field in expected_fields
    }
    field_matches = {
        field: _planner_field_matches(field, expected_values[field], actual_values[field])
        for field in expected_fields
    }
    mismatched_fields = [
        field for field, matched in field_matches.items() if not matched
    ]
    return {
        "available": True,
        "fields": len(expected_fields),
        "matched": len(expected_fields) - len(mismatched_fields),
        "exact_match": not mismatched_fields,
        "field_matches": field_matches,
        "mismatched_fields": mismatched_fields,
        "expected": expected_values,
        "actual": actual_values,
    }


def _planner_field_matches(field: str, expected: Any, actual: Any) -> bool:
    """文本字段用"核心词包含"判定，其余字段逐字相等。

    期望文本里的每个核心词都作为子串出现在实际文本里即算通过。用子串而非
    token 集合，是因为中文分词粒度不一致（如期望"应急响应"，实际压缩成
    "蓝队应急响应"一个 token），子串能正确判定核心词没有丢。
    """
    if field in PLANNER_SUBSET_FIELDS:
        expected_terms = _planner_terms(expected)
        actual_text = "" if actual is None else str(actual).casefold()
        if not expected_terms:
            # 期望为空（如 keyword 纯筛选场景）时不约束实际值——该字段对检索无影响。
            return True
        return all(term in actual_text for term in expected_terms)
    return actual == expected


def _planner_terms(value: Any) -> list[str]:
    if not value:
        return []
    return [
        token.casefold()
        for token in re.split(r"[\s,，、;/；]+", str(value).strip())
        if token
    ]


def _normalize_planner_value(field: str, value: Any) -> Any:
    if field in PLANNER_BOOL_FIELDS:
        return None if value is None else bool(value)
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


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
            "index_alias": _config.INDEX_ALIAS,
            "evidence_index_alias": _config.EVIDENCE_INDEX_ALIAS,
            "skipped": len(skipped_cases),
        },
        "overall": summarize(results),
        "by_type": summarize_by_type(results),
        "planner": summarize_planner(results),
        "planner_by_type": summarize_planner_by_type(results),
        "details": [result_to_detail(result) for result in results],
        "skipped": [skipped_case_to_detail(case) for case in skipped_cases],
    }


def summarize(results: list[QueryResult]) -> dict[str, Any]:
    judged = [result for result in results if result.case.relevant_ids]
    empty_cases = [result for result in results if result.empty_success is not None]
    # 单目标口径：判定集里每条 query 恰好 1 个相关文档时，P@K 的上限是 1/K（数学假象），
    # 这类组改报 success@1（首位是否命中唯一目标）。
    single_target = bool(judged) and all(
        len(result.case.relevant_ids) == 1 for result in judged
    )
    return {
        "queries": len(results),
        "judged": len(judged),
        "single_target": single_target,
        "success_at_1": _mean_metric(judged, "success_at_1"),
        "p5": _mean_metric(judged, "precision_at_5"),
        "p10": _mean_metric(judged, "precision_at_10"),
        "r5": _mean_metric(judged, "recall_at_5"),
        "r10": _mean_metric(judged, "recall_at_10"),
        "r50": _mean_metric(judged, "recall_at_50"),
        "r100": _mean_metric(judged, "recall_at_100"),
        "mrr10": _mean_metric(judged, "mrr_at_10"),
        "ndcg5": _mean_metric(judged, "ndcg_at_5"),
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


def summarize_planner(results: list[QueryResult]) -> dict[str, Any]:
    evaluated = [result for result in results if result.planner_eval.get("available")]
    field_totals: dict[str, int] = {field: 0 for field in PLANNER_EVAL_FIELDS}
    field_hits: dict[str, int] = {field: 0 for field in PLANNER_EVAL_FIELDS}
    mismatches: list[dict[str, Any]] = []

    for result in evaluated:
        field_matches = result.planner_eval.get("field_matches") or {}
        for field, matched in field_matches.items():
            field_totals[field] = field_totals.get(field, 0) + 1
            if matched:
                field_hits[field] = field_hits.get(field, 0) + 1
        if not result.planner_eval.get("exact_match"):
            mismatches.append(
                {
                    "id": result.case.case_id,
                    "type": result.case.case_type,
                    "query": result.case.query,
                    "mismatched_fields": result.planner_eval.get("mismatched_fields") or [],
                    "expected": result.planner_eval.get("expected") or {},
                    "actual": result.planner_eval.get("actual") or {},
                }
            )

    field_accuracy = {
        field: (field_hits.get(field, 0) / total if total else None)
        for field, total in field_totals.items()
    }
    return {
        "evaluated": len(evaluated),
        "exact_match_accuracy": (
            mean(1.0 if result.planner_eval.get("exact_match") else 0.0 for result in evaluated)
            if evaluated
            else None
        ),
        "field_accuracy": field_accuracy,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def summarize_planner_by_type(results: list[QueryResult]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[QueryResult]] = {}
    for result in results:
        grouped.setdefault(result.case.case_type, []).append(result)
    return {
        case_type: summarize_planner(rows)
        for case_type, rows in sorted(grouped.items())
    }


def result_to_detail(result: QueryResult) -> dict[str, Any]:
    return {
        "id": result.case.case_id,
        "type": result.case.case_type,
        "query": result.case.query,
        "api_params": result.case.api_params,
        "expected_plan": result.case.expected_plan,
        "query_plan": result.query_plan,
        "planner_eval": result.planner_eval,
        "returned_ids": result.returned_ids,
        "relevant_hits_at_10": [
            doc_id for doc_id in result.returned_ids[:10] if doc_id in result.case.relevant_ids
        ],
        "relevant_grades_at_10": {
            doc_id: result.case.relevance[doc_id]
            for doc_id in result.returned_ids[:10]
            if doc_id in result.case.relevance
        },
        "forbidden_hits_at_10": [
            doc_id for doc_id in result.returned_ids[:10] if doc_id in result.case.forbidden_ids
        ],
        "p5": result.precision_at_5,
        "p10": result.precision_at_10,
        "r5": result.recall_at_5,
        "r10": result.recall_at_10,
        "r50": result.recall_at_50,
        "r100": result.recall_at_100,
        "mrr10": result.mrr_at_10,
        "ndcg5": result.ndcg_at_5,
        "ndcg10": result.ndcg_at_10,
        "forbidden10": result.forbidden_at_10,
        "success_at_1": result.success_at_1,
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
    "r50",
    "r100",
    "mrr10",
    "ndcg5",
    "ndcg10",
    "empty_accuracy",
    "forbidden10",
)


def compare_reports(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    current_by_type = current.get("by_type") or {}
    previous_by_type = previous.get("by_type") or {}
    case_types = sorted(set(current_by_type) | set(previous_by_type))
    current_planner_by_type = current.get("planner_by_type") or {}
    previous_planner_by_type = previous.get("planner_by_type") or {}
    planner_case_types = sorted(set(current_planner_by_type) | set(previous_planner_by_type))
    return {
        "overall": compare_summary(current.get("overall") or {}, previous.get("overall") or {}),
        "by_type": {
            case_type: compare_summary(
                current_by_type.get(case_type) or {},
                previous_by_type.get(case_type) or {},
            )
            for case_type in case_types
        },
        "planner": compare_planner_summary(current.get("planner") or {}, previous.get("planner") or {}),
        "planner_by_type": {
            case_type: compare_planner_summary(
                current_planner_by_type.get(case_type) or {},
                previous_planner_by_type.get(case_type) or {},
            )
            for case_type in planner_case_types
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


def compare_planner_summary(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    current_field_accuracy = current.get("field_accuracy") or {}
    previous_field_accuracy = previous.get("field_accuracy") or {}
    return {
        "evaluated": {
            "previous": previous.get("evaluated", 0),
            "current": current.get("evaluated", 0),
            "delta": (current.get("evaluated", 0) or 0) - (previous.get("evaluated", 0) or 0),
        },
        "exact_match_accuracy": {
            "previous": previous.get("exact_match_accuracy"),
            "current": current.get("exact_match_accuracy"),
            "delta": _metric_delta(
                current.get("exact_match_accuracy"),
                previous.get("exact_match_accuracy"),
            ),
        },
        "mismatch_count": {
            "previous": previous.get("mismatch_count"),
            "current": current.get("mismatch_count"),
            "delta": _metric_delta(
                current.get("mismatch_count"),
                previous.get("mismatch_count"),
            ),
        },
        "field_accuracy": {
            field: {
                "previous": previous_field_accuracy.get(field),
                "current": current_field_accuracy.get(field),
                "delta": _metric_delta(
                    current_field_accuracy.get(field),
                    previous_field_accuracy.get(field),
                ),
            }
            for field in PLANNER_EVAL_FIELDS
        },
    }


def print_summary_table(row: dict[str, Any]) -> None:
    print("\nSearch quality summary")
    print(
        "queries judged  P@5    P@10   R@5    R@10   R@50   R@100  MRR@10 NDCG@5 NDCG@10 empty_acc forbidden@10"
    )
    empty = "-" if row["empty_accuracy"] is None else f"{row['empty_accuracy']:.3f}"
    print(
        f"{row['queries']:>7} "
        f"{row['judged']:>6}  "
        f"{row['p5']:.3f}  "
        f"{row['p10']:.3f}  "
        f"{row['r5']:.3f}  "
        f"{row['r10']:.3f}  "
        f"{row['r50']:.3f}  "
        f"{row['r100']:.3f}  "
        f"{row['mrr10']:.3f}  "
        f"{row['ndcg5']:.3f}  "
        f"{row['ndcg10']:.3f}  "
        f"{empty:>9}  "
        f"{row['forbidden10']:>12}"
    )


def print_type_summary(report: dict[str, Any]) -> None:
    print("\nType summary")
    print("type                    queries judged P@5*  R@10  R@100 NDCG@5 NDCG@10 empty_acc forbidden@10")
    print("(* 单目标类型 P@5 列改报 success@1，标记 †)")
    for case_type, row in (report.get("by_type") or {}).items():
        empty = row.get("empty_accuracy")
        if row.get("single_target"):
            headline = row.get("success_at_1") or 0.0
            headline_str = f"{headline:.3f}†"
        else:
            headline_str = f"{row['p5']:.3f} "
        print(
            f"{case_type:<23} "
            f"{row['queries']:>7} "
            f"{row['judged']:>6} "
            f"{headline_str:>5} "
            f"{row['r10']:.3f} "
            f"{row['r100']:.3f} "
            f"{row['ndcg5']:.3f} "
            f"{row['ndcg10']:.3f} "
            f"{'-' if empty is None else f'{empty:.3f}':>9} "
            f"{row['forbidden10']:>12}"
        )


def print_planner_summary(report: dict[str, Any]) -> None:
    planner = report.get("planner") or {}
    if not planner.get("evaluated"):
        return
    field_accuracy = planner.get("field_accuracy") or {}
    exact = planner.get("exact_match_accuracy")
    print("\nQuery planner summary")
    print("evaluated exact  intent dense  rerank lexical semantic mismatches")
    print(
        f"{planner['evaluated']:>9} "
        f"{'-' if exact is None else f'{exact:.3f}':>5} "
        f"{_format_optional_accuracy(field_accuracy.get('intent')):>7} "
        f"{_format_optional_accuracy(field_accuracy.get('enable_dense')):>6} "
        f"{_format_optional_accuracy(field_accuracy.get('enable_rerank')):>7} "
        f"{_format_optional_accuracy(field_accuracy.get('lexical_query')):>7} "
        f"{_format_optional_accuracy(field_accuracy.get('semantic_query')):>8} "
        f"{planner.get('mismatch_count', 0):>10}"
    )


def _format_optional_accuracy(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def print_skipped_cases(skipped: list[SkippedCase]) -> None:
    print(f"\nSkipped {len(skipped)} eval cases")
    grouped: dict[str, int] = {}
    for case in skipped:
        grouped[case.case_type] = grouped.get(case.case_type, 0) + 1
    for case_type, count in sorted(grouped.items()):
        print(f"- {case_type}: {count}")


def print_comparison_report(comparison: dict[str, Any], previous_path: Path) -> None:
    print(f"\nComparison vs {previous_path}")
    print("scope                   queriesΔ NDCG@5Δ NDCG@10Δ MRR@10Δ R@10Δ R@100Δ emptyΔ  forbiddenΔ")
    print_comparison_row("overall", comparison["overall"])
    for case_type, row in (comparison.get("by_type") or {}).items():
        print_comparison_row(case_type, row)
    if (comparison.get("planner") or {}).get("evaluated", {}).get("current"):
        print_planner_comparison_report(comparison)


def print_comparison_row(label: str, row: dict[str, Any]) -> None:
    print(
        f"{label:<23} "
        f"{_format_delta(row['queries']['delta'], digits=0):>8} "
        f"{_format_delta(row['ndcg5']['delta']):>7} "
        f"{_format_delta(row['ndcg10']['delta']):>8} "
        f"{_format_delta(row['mrr10']['delta']):>7} "
        f"{_format_delta(row['r10']['delta']):>6} "
        f"{_format_delta(row['r100']['delta']):>7} "
        f"{_format_delta(row['empty_accuracy']['delta']):>7} "
        f"{_format_delta(row['forbidden10']['delta'], digits=0):>10}"
    )


def _format_delta(value: float | int | None, *, digits: int = 3) -> str:
    if value is None:
        return "-"
    if digits == 0:
        return f"{int(value):+d}"
    return f"{float(value):+.{digits}f}"


def print_planner_comparison_report(comparison: dict[str, Any]) -> None:
    print("\nQuery planner comparison")
    print("scope                   evalΔ exactΔ intentΔ denseΔ rerankΔ lexicalΔ semanticΔ mismatchΔ")
    print_planner_comparison_row("overall", comparison["planner"])
    for case_type, row in (comparison.get("planner_by_type") or {}).items():
        print_planner_comparison_row(case_type, row)


def print_planner_comparison_row(label: str, row: dict[str, Any]) -> None:
    field_accuracy = row.get("field_accuracy") or {}
    print(
        f"{label:<23} "
        f"{_format_delta(row['evaluated']['delta'], digits=0):>5} "
        f"{_format_delta(row['exact_match_accuracy']['delta']):>6} "
        f"{_format_delta((field_accuracy.get('intent') or {}).get('delta')):>7} "
        f"{_format_delta((field_accuracy.get('enable_dense') or {}).get('delta')):>6} "
        f"{_format_delta((field_accuracy.get('enable_rerank') or {}).get('delta')):>7} "
        f"{_format_delta((field_accuracy.get('lexical_query') or {}).get('delta')):>8} "
        f"{_format_delta((field_accuracy.get('semantic_query') or {}).get('delta')):>9} "
        f"{_format_delta(row['mismatch_count']['delta'], digits=0):>9}"
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
            f"NDCG@5={result.ndcg_at_5:.3f} NDCG@10={result.ndcg_at_10:.3f} R@10={result.recall_at_10:.3f} "
            f"R@100={result.recall_at_100:.3f} "
            f"planner_mismatch={result.planner_eval.get('mismatched_fields') or []} "
            f"hits={relevant_hits} forbidden={forbidden_hits} returned={result.returned_ids[:10]}"
        )


if __name__ == "__main__":
    main()
