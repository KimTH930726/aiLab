"""
CPU 전용 경량 임베딩.

Bag-of-Words 방식으로 텍스트를 고정 차원 벡터로 변환한다.
프로덕션 교체: sentence-transformers (all-MiniLM-L6-v2) 또는
LanceDB 내장 embedding function.
"""
from __future__ import annotations

import numpy as np


_VOCAB = sorted({
    "gpt", "4o", "omni", "multimodal", "model", "text", "image", "audio",
    "sora", "video", "generation", "diffusion", "transformer",
    "api", "platform", "developer", "function", "assistant", "fine-tuning",
    "batch", "pricing", "structured", "json",
    "safety", "alignment", "rlhf", "red-team", "superalignment",
    "preparedness", "oversight",
    "enterprise", "business", "team", "chatgpt", "edu", "admin",
    "store", "custom", "builder", "creator", "monetize",
    "o1", "reasoning", "chain-of-thought", "thinking", "strawberry",
    "benchmark", "math", "science", "coding",
    "vision", "dall-e", "whisper", "voice", "speech", "clip",
    "open-source", "research", "paper", "triton", "evals",
    "scaling", "instruction",
    "partnership", "microsoft", "azure", "apple", "investment",
    "copilot", "valuation", "ios",
    "openai", "release", "update", "latest", "new", "launched",
})

EMBED_DIM = len(_VOCAB)
_VOCAB_LIST = list(_VOCAB)


def text_to_vector(text: str) -> list[float]:
    """
    Bag-of-Words 임베딩: 각 차원 = 어휘 출현 횟수, L2 정규화.

    GPU 불필요, 수 밀리초 내 완료.
    """
    text_lower = text.lower()
    vec = np.zeros(EMBED_DIM, dtype=np.float32)
    for i, word in enumerate(_VOCAB_LIST):
        vec[i] = float(text_lower.count(word))
    norm = np.linalg.norm(vec)
    if norm > 1e-9:
        vec /= norm
    return vec.tolist()
