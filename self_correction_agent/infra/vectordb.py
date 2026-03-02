"""
LanceDB 래퍼: 로컬 벡터 저장소 + 하이브리드 검색.

v3 변경사항:
- 스키마: parent_id, chunk_index 컬럼 추가
- 청킹: 모든 문서 입력에 chunker.chunk_text() 적용
- 하이브리드 검색: 벡터(ST 384차원) + BM25(rank-bm25) 결합
- 자동 마이그레이션: 구 BoW(48차원) 테이블 감지 시 DROP+재시드
"""
from __future__ import annotations

import uuid
import warnings
from typing import Any

import lancedb
from rank_bm25 import BM25Okapi

from .chunker import chunk_text
from .embedding import EMBED_DIM, text_to_vector


class LocalRAG:
    """LanceDB 기반 로컬 하이브리드 검색 엔진 (벡터 + BM25)."""

    TABLE_NAME = "knowledge_base"

    def __init__(self, db_path: str = "/tmp/lancedb_self_correction_agent"):
        self.db = lancedb.connect(db_path)
        self._table = None
        # BM25 인덱스 (in-memory)
        self._bm25_index: BM25Okapi | None = None
        self._bm25_ids: list[str] = []  # 청크 id 순서 (BM25 행과 1:1 대응)

    # ── 내부 헬퍼 ────────────────────────────────────────────────────

    def _list_tables(self) -> list:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return self.db.table_names()

    def _open_or_none(self):
        if self._table is None:
            if self.TABLE_NAME in self._list_tables():
                self._table = self.db.open_table(self.TABLE_NAME)
        return self._table

    def _count(self) -> int:
        t = self._open_or_none()
        if t is None:
            return 0
        return len(t.to_arrow().to_pylist())

    def _rebuild_bm25(self) -> None:
        """전체 청크에서 BM25 인덱스를 재구성한다."""
        t = self._open_or_none()
        if t is None:
            self._bm25_index = None
            self._bm25_ids = []
            return
        rows = t.to_arrow().to_pylist()
        if not rows:
            self._bm25_index = None
            self._bm25_ids = []
            return
        self._bm25_ids = [r["id"] for r in rows]
        tokenized = [r["text"].lower().split() for r in rows]
        self._bm25_index = BM25Okapi(tokenized)

    def _docs_to_records(self, doc: dict[str, str], parent_id: str) -> list[dict]:
        """단일 문서를 청킹하여 LanceDB 레코드 리스트로 변환한다."""
        text = doc["text"]
        topic = doc.get("topic", "general")
        source = doc.get("source", "unknown")
        chunks = chunk_text(text)
        records = []
        for i, chunk in enumerate(chunks):
            records.append({
                "id":          str(uuid.uuid4()),
                "parent_id":   parent_id,
                "chunk_index": i,
                "text":        chunk,
                "topic":       topic,
                "source":      source,
                "vector":      text_to_vector(chunk),
            })
        return records

    # ── 초기화 / 시드 ─────────────────────────────────────────────────

    def initialize(self, seed_docs: list[dict[str, str]] | None = None) -> int:
        """
        테이블이 없을 때만 생성+시드.
        이미 존재하면 벡터 차원을 확인하여 구 스키마(48차원)면 DROP+재시드.
        """
        if self.TABLE_NAME in self._list_tables():
            self._table = self.db.open_table(self.TABLE_NAME)
            # 차원 불일치 시 자동 마이그레이션 (BoW 48→ST 384)
            sample = self._table.to_arrow().slice(0, 1).to_pylist()
            if sample and len(sample[0].get("vector", [])) != EMBED_DIM:
                self.db.drop_table(self.TABLE_NAME)
                self._table = None
                if seed_docs:
                    return self.seed(seed_docs)
                return 0
            self._rebuild_bm25()
            return self._count()
        if seed_docs:
            return self.seed(seed_docs)
        return 0

    def seed(self, documents: list[dict[str, str]]) -> int:
        """
        강제 재초기화: 테이블을 DROP하고 문서를 청킹+임베딩하여 재생성.
        관리자 전용 (웹 UI '초기화' 버튼).
        """
        all_records: list[dict] = []
        for doc in documents:
            parent_id = doc.get("id", str(uuid.uuid4()))
            all_records.extend(self._docs_to_records(doc, parent_id))

        if self.TABLE_NAME in self._list_tables():
            self.db.drop_table(self.TABLE_NAME)

        self._table = self.db.create_table(self.TABLE_NAME, all_records)
        self._rebuild_bm25()
        return len(all_records)

    # ── CRUD ─────────────────────────────────────────────────────────

    def add_document(self, doc: dict[str, str]) -> str:
        """문서를 청킹+임베딩하여 추가. parent_id 반환."""
        parent_id = doc.get("id") or str(uuid.uuid4())
        records = self._docs_to_records(doc, parent_id)
        t = self._open_or_none()
        if t is None:
            self._table = self.db.create_table(self.TABLE_NAME, records)
        else:
            t.add(records)
        self._rebuild_bm25()
        return parent_id

    def delete_document(self, doc_id: str) -> bool:
        """parent_id 기준으로 해당 문서의 모든 청크를 삭제."""
        t = self._open_or_none()
        if t is None:
            return False
        try:
            t.delete(f"parent_id = '{doc_id}'")
            self._rebuild_bm25()
            return True
        except Exception:
            return False

    def list_documents(self, offset: int = 0, limit: int = 50) -> list[dict]:
        """
        원본 문서 목록 반환 (청크 0만, parent_id 중복 제거).
        id 컬럼은 parent_id를 반환 (삭제 API와 일치).
        """
        t = self._open_or_none()
        if t is None:
            return []
        rows = t.to_arrow().to_pylist()
        seen: set[str] = set()
        docs: list[dict] = []
        for r in rows:
            chunk_idx = r.get("chunk_index", 0)
            pid = r.get("parent_id", r["id"])
            if chunk_idx == 0 and pid not in seen:
                seen.add(pid)
                docs.append({
                    "id":     pid,
                    "topic":  r["topic"],
                    "source": r["source"],
                    "text":   r["text"],
                })
        return docs[offset: offset + limit]

    # ── 검색 ─────────────────────────────────────────────────────────

    def hybrid_search(
        self,
        query: str,
        top_k: int = 2,
        alpha: float = 0.7,
        distance_threshold: float = 1.5,
    ) -> list[dict[str, Any]]:
        """
        하이브리드 검색: alpha * 벡터유사도 + (1-alpha) * BM25유사도

        alpha=1.0 → 순수 벡터 검색
        alpha=0.0 → 순수 BM25 키워드 검색

        반환: [{"text", "topic", "source", "vector_score", "bm25_score", "combined_score", ...}]
        """
        t = self._open_or_none()
        if t is None:
            return []
        all_rows = t.to_arrow().to_pylist()
        if not all_rows:
            return []

        row_by_id = {r["id"]: r for r in all_rows}

        # 1. 벡터 검색: 거리 기준 후보 선택
        vec_sim: dict[str, float] = {}
        if alpha > 0:
            query_vec = text_to_vector(query)
            candidate_limit = min(len(all_rows), max(top_k * 20, 50))
            raw_vec = t.search(query_vec).limit(candidate_limit).to_list()
            dists = [r.get("_distance", 999.0) for r in raw_vec]
            max_dist = max(dists) if dists else 1.0
            for r in raw_vec:
                d = float(r.get("_distance", max_dist))
                if d <= distance_threshold:
                    # 거리 → 유사도 변환 (낮은 거리 = 높은 유사도)
                    vec_sim[r["id"]] = 1.0 - d / (max_dist + 1e-9)

        # 2. BM25 점수: 전체 청크 대상
        bm25_sim: dict[str, float] = {}
        if (1 - alpha) > 0 and self._bm25_index and self._bm25_ids:
            scores = self._bm25_index.get_scores(query.lower().split())
            max_s = float(max(scores)) if len(scores) > 0 and max(scores) > 0 else 1.0
            bm25_sim = {
                cid: float(s) / max_s
                for cid, s in zip(self._bm25_ids, scores)
            }

        # 3. 후보 집합 결정
        if alpha >= 1.0:
            candidates = set(vec_sim.keys())
        elif alpha <= 0.0:
            top_n = max(top_k * 10, 30)
            candidates = set(
                sorted(bm25_sim.keys(), key=lambda k: bm25_sim[k], reverse=True)[:top_n]
            )
        else:
            top_n = max(top_k * 10, 30)
            bm25_top = set(
                sorted(bm25_sim.keys(), key=lambda k: bm25_sim[k], reverse=True)[:top_n]
            )
            candidates = set(vec_sim.keys()) | bm25_top

        # 4. 점수 계산 및 수집
        results: list[dict] = []
        for cid in candidates:
            row = row_by_id.get(cid)
            if row is None:
                continue
            v = vec_sim.get(cid, 0.0)
            b = bm25_sim.get(cid, 0.0)
            combined = alpha * v + (1 - alpha) * b
            results.append({
                "text":           row["text"],
                "topic":          row["topic"],
                "source":         row["source"],
                "parent_id":      row.get("parent_id", cid),
                "chunk_index":    row.get("chunk_index", 0),
                "vector_score":   round(v, 4),
                "bm25_score":     round(b, 4),
                "combined_score": round(combined, 4),
            })

        # 5. parent_id 중복 제거 (최고 combined_score 청크만 유지)
        best_by_parent: dict[str, dict] = {}
        for r in results:
            pid = r["parent_id"]
            if pid not in best_by_parent or r["combined_score"] > best_by_parent[pid]["combined_score"]:
                best_by_parent[pid] = r

        # 6. 정렬 후 top_k 반환
        sorted_results = sorted(
            best_by_parent.values(),
            key=lambda x: x["combined_score"],
            reverse=True,
        )
        return sorted_results[:top_k]

    def search(self, query: str, top_k: int = 2) -> list[dict[str, Any]]:
        """하위호환 래퍼: hybrid_search 기본값(alpha=0.7)으로 위임."""
        return self.hybrid_search(query, top_k=top_k)
