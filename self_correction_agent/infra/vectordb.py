"""
LanceDB 래퍼: 로컬 벡터 저장소.

네트워크 의존성 제로 — 모든 데이터가 로컬 디스크에 저장된다.
프로덕션 교체: LanceDB Cloud, 또는 embedding function을 sentence-transformers로 변경.
"""
from __future__ import annotations

import warnings
from typing import Any

import lancedb

from .embedding import text_to_vector


class LocalRAG:
    """LanceDB 기반 로컬 벡터 검색 엔진."""

    TABLE_NAME = "knowledge_base"

    def __init__(self, db_path: str = "/tmp/lancedb_self_correction_agent"):
        self.db = lancedb.connect(db_path)
        self._table = None

    def _list_tables(self) -> list:
        """lancedb API 버전 호환 헬퍼."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return self.db.table_names()

    def seed(self, documents: list[dict[str, str]]) -> int:
        """
        문서를 벡터화하여 LanceDB에 인덱싱.
        각 doc: {"id", "text", "topic", "source"}
        """
        records = []
        for doc in documents:
            records.append({
                "id":     doc["id"],
                "text":   doc["text"],
                "topic":  doc.get("topic", "general"),
                "source": doc.get("source", "unknown"),
                "vector": text_to_vector(doc["text"]),
            })

        if self.TABLE_NAME in self._list_tables():
            self.db.drop_table(self.TABLE_NAME)

        self._table = self.db.create_table(self.TABLE_NAME, records)
        return len(records)

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """쿼리 벡터와 코사인 유사도 기반 검색."""
        if self._table is None:
            if self.TABLE_NAME in self._list_tables():
                self._table = self.db.open_table(self.TABLE_NAME)
            else:
                return []

        query_vec = text_to_vector(query)
        raw = self._table.search(query_vec).limit(top_k).to_list()

        return [
            {
                "text":     r["text"],
                "topic":    r["topic"],
                "source":   r["source"],
                "distance": round(float(r.get("_distance", 0.0)), 4),
            }
            for r in raw
        ]
