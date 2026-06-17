from __future__ import annotations

import json
import os
import shutil
from typing import TYPE_CHECKING

from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    import numpy as np

MODEL_ID = "IEITYuan/Yuan-embedding-2.0-zh"
VECTOR_DIMS = 1792

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


def _download_from_modelscope() -> str:
    from modelscope import snapshot_download

    model_dir = snapshot_download(MODEL_ID)

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
    for filename in ("pytorch_model.bin", "model.safetensors"):
        dst = os.path.join(dense_dir, filename)
        if os.path.exists(dst):
            continue
        hf_path = _try_hf_download(f"2_Dense/{filename}")
        shutil.copyfile(hf_path, dst)


def _try_hf_download(filename: str) -> str:
    from huggingface_hub import hf_hub_download

    endpoints = [
        os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        "https://hf-mirror.com",
    ]
    for endpoint in endpoints:
        os.environ["HF_ENDPOINT"] = endpoint
        try:
            return hf_hub_download(repo_id=MODEL_ID, filename=filename)
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


def encode(texts: str | list[str], *, normalize: bool = True) -> "np.ndarray":
    if isinstance(texts, str):
        texts = [texts]
    return get_model().encode(texts, normalize_embeddings=normalize)


def encode_single(text: str) -> list[float]:
    vec = encode(text, normalize=True)
    return vec[0].tolist()


def encode_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vecs = encode(texts, normalize=True)
    return vecs.tolist()
