import pytest

import embedding_service


class FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_doubao_embedding_backend_posts_openai_compatible_request(monkeypatch) -> None:
    monkeypatch.setattr(embedding_service, "EMBEDDING_PROVIDER", "doubao")
    monkeypatch.setattr(embedding_service, "DOUBAO_API_KEY", "test-key")
    monkeypatch.setattr(embedding_service, "DOUBAO_MODEL_ID", "ep-test")
    monkeypatch.setattr(embedding_service, "DOUBAO_VECTOR_DIMS", 3)

    calls: list[dict] = []

    def fake_post(url, *, headers, json, timeout):
        calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        return FakeResponse(
            {
                "data": [
                    {"index": 1, "embedding": [0, 3, 4]},
                    {"index": 0, "embedding": [6, 8, 0]},
                ]
            }
        )

    monkeypatch.setattr(embedding_service.requests, "post", fake_post)

    vectors = embedding_service.encode_batch(["first", "second"])

    assert calls == [
        {
            "url": "https://ark.cn-beijing.volces.com/api/v3/embeddings",
            "headers": {
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
            },
            "json": {
                "model": "ep-test",
                "input": ["first", "second"],
                "encoding_format": "float",
                "dimensions": 3,
            },
            "timeout": 60,
        }
    ]
    assert vectors[0] == pytest.approx([0.6, 0.8, 0.0])
    assert vectors[1] == pytest.approx([0.0, 0.6, 0.8])


def test_doubao_embedding_backend_validates_vector_dimensions(monkeypatch) -> None:
    monkeypatch.setattr(embedding_service, "EMBEDDING_PROVIDER", "doubao")
    monkeypatch.setattr(embedding_service, "DOUBAO_API_KEY", "test-key")
    monkeypatch.setattr(embedding_service, "DOUBAO_VECTOR_DIMS", 2)

    def fake_post(url, *, headers, json, timeout):
        return FakeResponse({"data": [{"index": 0, "embedding": [1, 2, 3]}]})

    monkeypatch.setattr(embedding_service.requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="dimension mismatch"):
        embedding_service.encode_batch(["text"])


def test_encode_single_accepts_list_vectors_from_api_backend(monkeypatch) -> None:
    monkeypatch.setattr(embedding_service, "EMBEDDING_PROVIDER", "doubao")
    monkeypatch.setattr(
        embedding_service,
        "_encode_doubao_batch",
        lambda texts, *, normalize=True: [[1.0, 2.0, 3.0]],
    )

    assert embedding_service.encode_single("text") == [1.0, 2.0, 3.0]
