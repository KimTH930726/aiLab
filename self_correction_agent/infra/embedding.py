"""
sentence-transformers 기반 다국어 임베딩 (384차원).

모델: paraphrase-multilingual-MiniLM-L12-v2
- 한국어 + 영어 동시 지원
- CPU 추론 실용적 (~470MB)
- L2 정규화된 벡터 → 코사인 유사도 검색 가능
"""
from __future__ import annotations

from sentence_transformers import SentenceTransformer

EMBED_DIM = 384
_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# 모듈 레벨 싱글톤 — 최초 import 시 로드
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def text_to_vector(text: str) -> list[float]:
    """384차원 다국어 임베딩 벡터 반환 (L2 정규화)."""
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()
