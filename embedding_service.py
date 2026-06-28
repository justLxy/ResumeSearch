from __future__ import annotations

import json
import math
import os
import shutil
from typing import TYPE_CHECKING

import requests
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    import numpy as np

LOCAL_PROVIDER = "local"
DOUBAO_PROVIDER = "doubao"
EMBEDDING_PROVIDER = DOUBAO_PROVIDER

LOCAL_MODEL_ID = "IEITYuan/Yuan-embedding-2.0-zh"
LOCAL_VECTOR_DIMS = 1792

DOUBAO_API_KEY = "b22a1ce8-9df9-4aec-9a94-a0a6be74cc86"
DOUBAO_API_BASE = "https://ark.cn-beijing.volces.com/api/v3"
DOUBAO_MODEL_ID = "ep-20260412051954-zl5fm"
DOUBAO_VECTOR_DIMS = 2048
DOUBAO_BATCH_SIZE = 64
DOUBAO_TIMEOUT_SECONDS = 60
DOUBAO_SEND_DIMENSIONS = True
DOUBAO_MULTIMODAL = True

_POOLING_CONFIG = json.dumps(
    {
        "word_embedding_dimension": 1024,
        "pooling_mode_cls_token": False,
        "pooling_mode_mean_tokens": True,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
        "pooling_mode_weightedmean_tokens": False,
        "pooling_mode_lasttoken": False,
        "include_prompt": True,
    },
    indent=1,
)
_DENSE_CONFIG = json.dumps(
    {
        "in_features": 1024,
        "out_features": 1792,
        "bias": True,
        "activation_function": "torch.nn.modules.linear.Identity",
    },
    indent=1,
)

_model: SentenceTransformer | None = None


def _current_provider() -> str:
    provider = EMBEDDING_PROVIDER.strip().lower()
    if provider in {"api", "ark", "volcengine"}:
        return DOUBAO_PROVIDER
    if provider in {LOCAL_PROVIDER, DOUBAO_PROVIDER}:
        return provider
    raise ValueError("EMBEDDING_PROVIDER must be one of: local, doubao")


def _current_vector_dims() -> int:
    if _current_provider() == DOUBAO_PROVIDER:
        return _positive_int(DOUBAO_VECTOR_DIMS, "DOUBAO_VECTOR_DIMS")
    return _positive_int(LOCAL_VECTOR_DIMS, "LOCAL_VECTOR_DIMS")


def _positive_int(raw_value: int | str, name: str) -> int:
    try:
        value = int(str(raw_value).strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return value


def _doubao_model_id() -> str:
    return DOUBAO_MODEL_ID.strip()


def _current_model_id() -> str:
    if _current_provider() == DOUBAO_PROVIDER:
        return f"{DOUBAO_PROVIDER}:{_doubao_model_id()}"
    return LOCAL_MODEL_ID


MODEL_ID = _current_model_id()
VECTOR_DIMS = _current_vector_dims()


def _download_from_modelscope() -> str:
    from modelscope import snapshot_download

    model_dir = snapshot_download(LOCAL_MODEL_ID)

    # ModelScope flattens all files to root level, but sentence_transformers
    # expects 1_Pooling/ and 2_Dense/ as subdirectories with their own config
    # and weight files. Reconstruct these subdirectories.
    pooling_dir = os.path.join(model_dir, "1_Pooling")
    os.makedirs(pooling_dir, exist_ok=True)
    with open(os.path.join(pooling_dir, "config.json"), "w") as f:
        f.write(_POOLING_CONFIG)

    dense_dir = os.path.join(model_dir, "2_Dense")
    os.makedirs(dense_dir, exist_ok=True)
    with open(os.path.join(dense_dir, "config.json"), "w") as f:
        f.write(_DENSE_CONFIG)
    # The Dense layer has its own small weight files (~7MB each), separate
    # from the root full-model weights. Download them from HuggingFace directly
    # into the 2_Dense/ subdirectory.
    _fetch_dense_weights(dense_dir)

    return model_dir


def _fetch_dense_weights(dense_dir: str) -> None:
    for filename in ("model.safetensors", "pytorch_model.bin"):
        dst = os.path.join(dense_dir, filename)
        if os.path.exists(dst):
            return
        try:
            hf_path = _try_hf_download(f"2_Dense/{filename}")
            shutil.copyfile(hf_path, dst)
            return
        except Exception:
            continue
    raise RuntimeError(
        "无法下载 2_Dense 权重文件（model.safetensors 或 pytorch_model.bin），"
        "请检查网络或设置 HF_ENDPOINT=https://hf-mirror.com 后重试"
    )


def _try_hf_download(filename: str) -> str:
    from huggingface_hub import hf_hub_download

    endpoints = [
        os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        "https://hf-mirror.com",
    ]
    for endpoint in endpoints:
        os.environ["HF_ENDPOINT"] = endpoint
        try:
            return hf_hub_download(repo_id=LOCAL_MODEL_ID, filename=filename)
        except Exception:
            continue
    raise RuntimeError(
        f"无法下载 {filename}，请设置 HF_ENDPOINT=https://hf-mirror.com 后重试"
    )


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        model_path = _download_from_modelscope()
        _model = SentenceTransformer(model_path, local_files_only=True)
    return _model


def encode(texts: str | list[str], *, normalize: bool = True) -> "np.ndarray | list[list[float]]":
    if isinstance(texts, str):
        texts = [texts]
    if _current_provider() == DOUBAO_PROVIDER:
        return _encode_doubao_batch(texts, normalize=normalize)
    return get_model().encode(texts, normalize_embeddings=normalize)


def encode_single(text: str) -> list[float]:
    vec = encode(text, normalize=True)
    first = vec[0]
    if hasattr(first, "tolist"):
        return first.tolist()
    return list(first)


def encode_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vecs = encode(texts, normalize=True)
    if hasattr(vecs, "tolist"):
        return vecs.tolist()
    return list(vecs)


def _encode_doubao_batch(texts: list[str], *, normalize: bool = True) -> list[list[float]]:
    api_key = _doubao_api_key()
    batch_size = _positive_int(DOUBAO_BATCH_SIZE, "DOUBAO_BATCH_SIZE")
    if DOUBAO_MULTIMODAL:
        batch_size = 1
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(
            _request_doubao_embeddings(texts[start : start + batch_size], api_key)
        )
    return [_normalize_vector(vector) for vector in vectors] if normalize else vectors


def _doubao_api_key() -> str:
    api_key = DOUBAO_API_KEY.strip()
    if not api_key:
        raise RuntimeError("DOUBAO_API_KEY is required when EMBEDDING_PROVIDER=doubao")
    return api_key


def _request_doubao_embeddings(texts: list[str], api_key: str) -> list[list[float]]:
    if not texts:
        return []
    base_url = DOUBAO_API_BASE.rstrip("/")
    model_id = _doubao_model_id()
    timeout = _positive_int(DOUBAO_TIMEOUT_SECONDS, "DOUBAO_TIMEOUT_SECONDS")
    input_value: object = texts
    api_path = "embeddings"
    if DOUBAO_MULTIMODAL:
        if len(texts) != 1:
            raise RuntimeError("Doubao multimodal embedding requests must contain one text")
        input_value = [{"type": "text", "text": texts[0]}]
        api_path = "embeddings/multimodal"

    body: dict[str, object] = {
        "model": model_id,
        "input": input_value,
        "encoding_format": "float",
    }
    if DOUBAO_SEND_DIMENSIONS:
        body["dimensions"] = _current_vector_dims()

    response = requests.post(
        f"{base_url}/{api_path}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=timeout,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Doubao embedding request failed: {response.status_code} "
            f"{response.text[:800]}"
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("Doubao embedding response is not valid JSON") from exc
    rows = payload.get("data")
    if isinstance(rows, dict):
        vector = rows.get("embedding")
        return _clean_doubao_vectors([vector], len(texts))
    if not isinstance(rows, list):
        raise RuntimeError("Doubao embedding response is missing data")
    if not all(isinstance(row, dict) for row in rows):
        raise RuntimeError("Doubao embedding response data[] contains a non-object item")
    ordered_rows = sorted(rows, key=lambda item: int(item.get("index", 0)))
    vectors = [row.get("embedding") for row in ordered_rows]
    return _clean_doubao_vectors(vectors, len(texts))


def _clean_doubao_vectors(vectors: list[object], expected_count: int) -> list[list[float]]:
    if len(vectors) != expected_count:
        raise RuntimeError(
            f"Doubao embedding response count mismatch: expected {expected_count}, got {len(vectors)}"
        )
    cleaned_vectors: list[list[float]] = []
    expected_dims = _current_vector_dims()
    for vector in vectors:
        if not isinstance(vector, list):
            raise RuntimeError("Doubao embedding response contains a non-list vector")
        cleaned_vector = [float(value) for value in vector]
        if len(cleaned_vector) != expected_dims:
            raise RuntimeError(
                f"Doubao embedding dimension mismatch: expected {expected_dims}, "
                f"got {len(cleaned_vector)}"
            )
        cleaned_vectors.append(cleaned_vector)
    return cleaned_vectors


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
