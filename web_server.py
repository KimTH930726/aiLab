"""
FastAPI 웹 서버: Self-Correction Agent Web UI (v3)

엔드포인트:
  GET  /                      → static/index.html
  POST /run                   → 에이전트 실행 시작, session_id 반환
  GET  /stream/{id}           → SSE: 실시간 에이전트 이벤트 스트림
  GET  /db                    → LanceDB 상태 (count)
  GET  /db/documents          → 문서 목록 (페이지네이션 + 텍스트 검색)
  POST /db/documents          → 단일 문서 추가
  DELETE /db/documents/{id}   → 문서 삭제
  POST /db/upload             → .txt 파일 업로드 → 자동 임베딩
  POST /db/seed               → 초기 시드 데이터 강제 재적재
  GET  /db/settings           → 검색 설정 조회 (alpha, threshold, top_k)
  POST /db/settings           → 검색 설정 저장
  GET  /db/search             → 하이브리드 검색 테스트 (유사도 점수 반환)
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from self_correction_agent.orchestrator import run_agent
from self_correction_agent.infra.vectordb import LocalRAG
from self_correction_agent.infra.settings import load_settings, save_settings
from self_correction_agent.knowledge import KNOWLEDGE_BASE

# ── 설정 ──────────────────────────────────────────────────────────
DB_PATH = os.environ.get("LANCEDB_PATH", "/tmp/lancedb_self_correction_agent")
STATIC_DIR = Path(__file__).parent / "static"

# ── LocalRAG 앱 레벨 싱글톤 (BM25 인덱스 메모리 보존) ────────────
_rag: LocalRAG | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag
    _rag = LocalRAG(DB_PATH)
    _rag.initialize(KNOWLEDGE_BASE)
    yield


def get_rag() -> LocalRAG:
    """싱글톤 RAG 인스턴스 반환. 미초기화 시 503."""
    if _rag is None:
        raise HTTPException(status_code=503, detail="RAG not initialized")
    return _rag


app = FastAPI(title="Self-Correction Agent UI", lifespan=lifespan)

# 정적 파일 서빙 (index.html)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 진행 중인 SSE 세션 저장소: session_id -> asyncio.Queue
_sessions: dict[str, asyncio.Queue] = {}


# ── 요청/응답 모델 ─────────────────────────────────────────────────
class RunRequest(BaseModel):
    query: str
    model: Optional[str] = None


class RunResponse(BaseModel):
    session_id: str


class DocumentCreate(BaseModel):
    topic: str
    source: str
    text: str


class SettingsUpdate(BaseModel):
    alpha: Optional[float] = None
    distance_threshold: Optional[float] = None
    top_k: Optional[int] = None


# ── 라우트 ────────────────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/run", response_model=RunResponse)
async def start_run(req: RunRequest):
    """에이전트 실행을 시작하고 SSE 스트림용 session_id를 반환한다."""
    session_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    _sessions[session_id] = queue
    loop = asyncio.get_running_loop()

    def on_event(event: dict) -> None:
        """워커 스레드 → asyncio Queue 브릿지 (thread-safe)."""
        asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    def thread_target() -> None:
        try:
            run_agent(
                query=req.query,
                model_name=req.model or None,
                db_path=DB_PATH,
                verbose=True,
                on_event=on_event,
            )
        except Exception as exc:
            on_event({"type": "error", "message": str(exc)})
        finally:
            on_event({"type": "_sentinel"})

    threading.Thread(target=thread_target, daemon=True).start()
    return RunResponse(session_id=session_id)


@app.get("/stream/{session_id}")
async def stream(session_id: str):
    """SSE: 에이전트 이벤트를 실시간으로 브라우저에 전달한다."""
    queue = _sessions.get(session_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="Session not found")

    async def event_generator():
        try:
            while True:
                event = await queue.get()
                if event.get("type") == "_sentinel":
                    break
                yield {"data": json.dumps(event, ensure_ascii=False)}
        except asyncio.CancelledError:
            pass
        finally:
            _sessions.pop(session_id, None)

    return EventSourceResponse(event_generator())


@app.get("/db")
def get_db():
    """LanceDB 지식베이스 상태 (문서 수)."""
    try:
        rag = get_rag()
        rows = rag.list_documents(limit=10000)
        return {"count": len(rows)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── DB CRUD 엔드포인트 ────────────────────────────────────────────

@app.get("/db/documents")
def list_documents(
    q: Optional[str] = Query(None, description="텍스트 검색어"),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
):
    """문서 목록 조회 (페이지네이션 + 텍스트 검색)."""
    try:
        rag = get_rag()
        rows = rag.list_documents(offset=0, limit=10000)
        if q:
            ql = q.lower()
            rows = [r for r in rows if ql in r["text"].lower()
                    or ql in r["topic"].lower() or ql in r["source"].lower()]
        total = len(rows)
        return {"total": total, "offset": offset, "limit": limit,
                "documents": rows[offset: offset + limit]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/db/documents", status_code=201)
def add_document(doc: DocumentCreate):
    """단일 문서를 지식베이스에 추가 (청킹+임베딩 자동 처리)."""
    try:
        rag = get_rag()
        doc_id = rag.add_document({"topic": doc.topic, "source": doc.source, "text": doc.text})
        return {"id": doc_id, "topic": doc.topic, "source": doc.source}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/db/documents/{doc_id}")
def delete_document(doc_id: str):
    """문서 삭제 (해당 parent_id의 모든 청크 제거)."""
    try:
        rag = get_rag()
        ok = rag.delete_document(doc_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"deleted": doc_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/db/upload", status_code=201)
async def upload_file(file: UploadFile = File(...)):
    """.txt 파일 업로드 → 청킹+임베딩 후 지식베이스에 추가."""
    if not file.filename or not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported")
    try:
        content = (await file.read()).decode("utf-8", errors="ignore").strip()
        if not content:
            raise HTTPException(status_code=400, detail="File is empty")
        rag = get_rag()
        source = file.filename
        topic = Path(file.filename).stem
        doc_id = rag.add_document({"topic": topic, "source": source, "text": content})
        return {"id": doc_id, "topic": topic, "source": source, "chars": len(content)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/db/seed")
def reseed():
    """초기 시드 데이터로 강제 재적재 (기존 데이터 전체 삭제)."""
    try:
        rag = get_rag()
        n = rag.seed(KNOWLEDGE_BASE)
        return {"seeded": n}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── 검색 설정 ─────────────────────────────────────────────────────

@app.get("/db/settings")
def get_settings():
    """현재 검색 설정 반환 (alpha, distance_threshold, top_k)."""
    try:
        return load_settings(DB_PATH)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/db/settings")
def update_settings(body: SettingsUpdate):
    """검색 설정 저장. 미지정 필드는 기존값 유지."""
    try:
        current = load_settings(DB_PATH)
        if body.alpha is not None:
            current["alpha"] = max(0.0, min(1.0, body.alpha))
        if body.distance_threshold is not None:
            current["distance_threshold"] = max(0.1, body.distance_threshold)
        if body.top_k is not None:
            current["top_k"] = max(1, min(20, body.top_k))
        save_settings(DB_PATH, current)
        return current
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── 검색 테스트 ───────────────────────────────────────────────────

@app.get("/db/search")
def search_test(
    q: str = Query(..., description="검색 쿼리"),
    top_k: int = Query(5, ge=1, le=20),
    alpha: float = Query(0.7, ge=0.0, le=1.0),
    threshold: float = Query(1.5, ge=0.0),
):
    """
    하이브리드 검색 테스트. 유사도 점수(vector/bm25/combined)를 반환한다.
    """
    try:
        rag = get_rag()
        results = rag.hybrid_search(
            query=q, top_k=top_k, alpha=alpha, distance_threshold=threshold
        )
        return {"query": q, "alpha": alpha, "threshold": threshold, "results": results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
