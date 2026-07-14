from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

RERANK_PROVIDER = "dashscope_api"
RERANK_MODEL_ID = "qwen3-rerank"
RERANK_API_URL = os.environ.get(
    "RERANK_API_URL",
    "https://ws-nl8tvztfpss60i8t.cn-beijing.maas.aliyuncs.com/api/v1/services/"
    "rerank/text-rerank/text-rerank",
)
RERANK_API_KEY = os.environ.get("RERANK_API_KEY", "")
RERANK_BATCH_SIZE = 20
RERANK_TIMEOUT_SECONDS = 60
RERANK_INSTRUCT: str | None = "Given a recruitment query, retrieve relevant candidate resumes that match the required skills, experience, and qualifications."
# qwen3-rerank rejects overly long queries with a 400; cap the query text so a
# pasted full JD degrades to a truncated query instead of failing the request.
RERANK_MAX_QUERY_CHARS = 1024


@dataclass(frozen=True)
class RerankResult:
    index: int
    score: float
    text: str


def rerank(query: str, documents: list[str], *, top_n: int | None = None) -> list[RerankResult]:
    if not documents:
        return []
    scores = score_pairs(query, documents)
    results = [
        RerankResult(index=index, score=score, text=documents[index])
        for index, score in enumerate(scores)
    ]
    results.sort(key=lambda item: item.score, reverse=True)
    return results[:top_n] if top_n is not None else results


def score_pairs(query: str, documents: list[str]) -> list[float]:
    if not documents:
        return []
    if not query.strip():
        return [0.0 for _document in documents]

    scores = [0.0 for _document in documents]
    for start in range(0, len(documents), RERANK_BATCH_SIZE):
        batch = documents[start : start + RERANK_BATCH_SIZE]
        batch_scores = _score_batch(query, batch)
        for local_index, score in enumerate(batch_scores):
            scores[start + local_index] = score
    return scores


def _score_batch(query: str, documents: list[str]) -> list[float]:
    if not RERANK_API_KEY.strip():
        raise RuntimeError("RERANK_API_KEY is required to call the rerank service")
    payload = _build_payload(query, documents)
    response = requests.post(
        RERANK_API_URL,
        headers={
            "Authorization": f"Bearer {RERANK_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=RERANK_TIMEOUT_SECONDS,
    )
    if not response.ok:
        raise RuntimeError(
            f"rerank API {response.status_code} error: {response.text[:500]}"
        )
    data = response.json()
    return _extract_scores(data, len(documents))


def _build_payload(query: str, documents: list[str]) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "return_documents": True,
        "top_n": len(documents),
    }
    if RERANK_INSTRUCT:
        parameters["instruct"] = RERANK_INSTRUCT
    return {
        "model": RERANK_MODEL_ID,
        "query": query.strip()[:RERANK_MAX_QUERY_CHARS],
        "documents": [_clean_document(document) for document in documents],
        "parameters": parameters,
    }


def _extract_scores(data: dict[str, Any], expected_count: int) -> list[float]:
    results = data.get("results") or data.get("output", {}).get("results")
    if not isinstance(results, list):
        raise RuntimeError(f"rerank API response missing output.results: {data}")

    scores = [0.0 for _index in range(expected_count)]
    seen: set[int] = set()
    for item in results:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        if not isinstance(index, int) or index < 0 or index >= expected_count:
            continue
        score = item.get("relevance_score", item.get("score"))
        if score is None:
            continue
        scores[index] = float(score)
        seen.add(index)

    if len(seen) != expected_count:
        missing = sorted(set(range(expected_count)) - seen)
        raise RuntimeError(f"rerank API returned no score for candidate indexes: {missing}")
    return scores


def _clean_document(document: str) -> str:
    return os.linesep.join(line.strip() for line in document.splitlines() if line.strip())
