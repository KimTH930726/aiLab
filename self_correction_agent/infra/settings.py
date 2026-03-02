"""
RAG 검색 설정 영속화.

alpha, distance_threshold, top_k 를 {DB_PATH}/settings.json 에 저장한다.
웹 UI 슬라이더와 에이전트 실행 간 설정을 공유한다.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_SETTINGS: dict = {
    "alpha": 0.7,            # 0=순수 키워드(BM25), 1=순수 의미(벡터)
    "distance_threshold": 1.5,  # 벡터 거리 임계값 (낮을수록 엄격)
    "top_k": 2,              # 쿼리당 반환 문서 수
}


def _settings_path(db_path: str) -> Path:
    return Path(db_path) / "settings.json"


def load_settings(db_path: str) -> dict:
    """settings.json 읽기. 없거나 손상되면 DEFAULT_SETTINGS 반환."""
    path = _settings_path(db_path)
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                saved = json.load(f)
            return {**DEFAULT_SETTINGS, **saved}
        except Exception:
            pass
    return dict(DEFAULT_SETTINGS)


def save_settings(db_path: str, settings: dict) -> None:
    """설정을 settings.json 에 저장. DB_PATH 디렉토리가 없으면 생성."""
    path = _settings_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = {**DEFAULT_SETTINGS, **settings}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
