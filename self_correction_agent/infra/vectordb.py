"""
LanceDB 래퍼: 로컬 벡터 저장소.

네트워크 의존성 제로 — 모든 데이터가 로컬 디스크에 저장된다.
v2: initialize/add/delete/list CRUD 메서드 추가.
"""
from __future__ import annotations

import uuid
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

    def _open_or_none(self):
        """테이블이 존재하면 열고 없으면 None 반환."""
        if self._table is None:
            if self.TABLE_NAME in self._list_tables():
                self._table = self.db.open_table(self.TABLE_NAME)
        return self._table

    def _count(self) -> int:
        t = self._open_or_none()
        if t is None:
            return 0
        return len(t.to_arrow().to_pylist())

    # ── v2: initialize (테이블 없을 때만 생성) ─────────────────────────
    def initialize(self, seed_docs: list[dict[str, str]] | None = None) -> int:
        """
        테이블이 없을 때만 생성+시드.
        이미 존재하면 그냥 열어서 반환 (DROP 없음).
        """
        if self.TABLE_NAME in self._list_tables():
            self._table = self.db.open_table(self.TABLE_NAME)
            return self._count()
        if seed_docs:
            return self.seed(seed_docs)
        return 0

    # ── 기존 seed (강제 재초기화 — 관리자 전용) ────────────────────────
    def seed(self, documents: list[dict[str, str]]) -> int:
        """
        문서를 벡터화하여 LanceDB에 인덱싱.
        기존 테이블을 DROP하고 새로 만든다 (전체 재초기화).
        """
        records = []
        for doc in documents:
            records.append({
                "id":     doc.get("id", str(uuid.uuid4())),
                "text":   doc["text"],
                "topic":  doc.get("topic", "general"),
                "source": doc.get("source", "unknown"),
                "vector": text_to_vector(doc["text"]),
            })

        if self.TABLE_NAME in self._list_tables():
            self.db.drop_table(self.TABLE_NAME)

        self._table = self.db.create_table(self.TABLE_NAME, records)
        return len(records)

    # ── v2: 단일 문서 추가 ──────────────────────────────────────────────
    def add_document(self, doc: dict[str, str]) -> str:
        """단일 문서를 기존 테이블에 추가. 생성된 doc_id 반환."""
        doc_id = doc.get("id") or str(uuid.uuid4())
        record = {
            "id":     doc_id,
            "text":   doc["text"],
            "topic":  doc.get("topic", "general"),
            "source": doc.get("source", "unknown"),
            "vector": text_to_vector(doc["text"]),
        }
        t = self._open_or_none()
        if t is None:
            self._table = self.db.create_table(self.TABLE_NAME, [record])
        else:
            t.add([record])
        return doc_id

    # ── v2: 문서 삭제 ───────────────────────────────────────────────────
    def delete_document(self, doc_id: str) -> bool:
        """id 기준으로 문서 삭제. 성공 여부 반환."""
        t = self._open_or_none()
        if t is None:
            return False
        try:
            t.delete(f"id = '{doc_id}'")
            return True
        except Exception:
            return False

    # ── v2: 문서 목록 (페이지네이션) ────────────────────────────────────
    def list_documents(self, offset: int = 0, limit: int = 50) -> list[dict]:
        """모든 문서를 페이지네이션하여 반환 (vector 제외)."""
        t = self._open_or_none()
        if t is None:
            return []
        rows = t.to_arrow().to_pylist()
        return [
            {"id": r["id"], "topic": r["topic"], "source": r["source"], "text": r["text"]}
            for r in rows[offset: offset + limit]
        ]

    # ── 검색 ────────────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """쿼리 벡터와 코사인 유사도 기반 검색."""
        t = self._open_or_none()
        if t is None:
            return []

        query_vec = text_to_vector(query)
        raw = t.search(query_vec).limit(top_k).to_list()

        return [
            {
                "text":     r["text"],
                "topic":    r["topic"],
                "source":   r["source"],
                "distance": round(float(r.get("_distance", 0.0)), 4),
            }
            for r in raw
        ]
