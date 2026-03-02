"""
문자 기반 슬라이딩 윈도우 청킹.

한국어/영어 토크나이저 없이 문자 수 기준으로 분할한다.
기본값: 500자 윈도우, 50자 오버랩.
"""
from __future__ import annotations


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """
    문자 기반 슬라이딩 윈도우 청킹.

    - chunk_size 이하 문서는 청크 1개로 반환
    - 마지막 청크가 chunk_size // 4 미만이면 이전 청크에 흡수
    - 항상 최소 1개 청크 보장
    """
    if not text:
        return [""]
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap

    # 마지막 청크가 너무 짧으면 이전 청크에 흡수
    if len(chunks) > 1 and len(chunks[-1]) < chunk_size // 4:
        chunks.pop()

    return chunks if chunks else [text]
