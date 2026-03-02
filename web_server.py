"""
FastAPI 웹 서버: Self-Correction Agent Web UI (v2)

엔드포인트:
  GET  /                      → static/index.html
  POST /run                   → 에이전트 실행 시작, session_id 반환
  GET  /stream/{id}           → SSE: 실시간 에이전트 이벤트 스트림
  GET  /db                    → (구) LanceDB 전체 문서 목록 (호환용)
  GET  /db/documents          → v2: 페이지네이션 + 텍스트 검색
  POST /db/documents          → v2: 단일 문서 추가
  DELETE /db/documents/{id}   → v2: 문서 삭제
  POST /db/upload             → v2: .txt 파일 업로드 → 자동 임베딩
  POST /db/seed               → v2: 초기 시드 데이터 강제 재적재
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from self_correction_agent.orchestrator import run_agent
from self_correction_agent.infra.vectordb import LocalRAG
from self_correction_agent.knowledge import KNOWLEDGE_BASE

# ── 설정 ──────────────────────────────────────────────────────────
DB_PATH = os.environ.get("LANCEDB_PATH", "/tmp/lancedb_self_correction_agent")
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Self-Correction Agent UI")

# 정적 파일 서빙 (index.html)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 진행 중인 SSE 세션 저장소: session_id -> asyncio.Queue
_sessions: dict[str, asyncio.Queue] = {}


# ── 요청/응답 모델 ─────────────────────────────────────────────────
class RunRequest(BaseModel):
    query: str
    model: Optional[str] = None  # None → mock, "openai:bitnet" 등


class RunResponse(BaseModel):
    session_id: str


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
            # SSE 제너레이터 종료 신호
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
    """(구) LanceDB 지식베이스 문서 목록 — 호환용."""
    try:
        rag = LocalRAG(DB_PATH)
        rag.initialize(KNOWLEDGE_BASE)
        rows = rag.list_documents(limit=200)
        return {"count": len(rows), "documents": rows}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── v2: DB CRUD 엔드포인트 ────────────────────────────────────────────

class DocumentCreate(BaseModel):
    topic: str
    source: str
    text: str


def _get_rag() -> LocalRAG:
    """요청마다 공유 LocalRAG 인스턴스 반환 (initialize 보장)."""
    rag = LocalRAG(DB_PATH)
    rag.initialize(KNOWLEDGE_BASE)
    return rag


@app.get("/db/documents")
def list_documents(
    q: Optional[str] = Query(None, description="텍스트 검색어"),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
):
    """문서 목록 조회 (페이지네이션 + 텍스트 검색)."""
    try:
        rag = _get_rag()
        rows = rag.list_documents(offset=0, limit=10000)  # 전체 로드 후 필터
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
    """단일 문서를 지식베이스에 추가."""
    try:
        rag = _get_rag()
        doc_id = rag.add_document({"topic": doc.topic, "source": doc.source, "text": doc.text})
        return {"id": doc_id, "topic": doc.topic, "source": doc.source}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/db/documents/{doc_id}")
def delete_document(doc_id: str):
    """문서 삭제."""
    try:
        rag = _get_rag()
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
    """.txt 파일 업로드 → 자동 임베딩 후 지식베이스에 추가."""
    if not file.filename or not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported")
    try:
        content = (await file.read()).decode("utf-8", errors="ignore").strip()
        if not content:
            raise HTTPException(status_code=400, detail="File is empty")
        rag = _get_rag()
        source = file.filename
        topic = Path(file.filename).stem
        doc_id = rag.add_document({"topic": topic, "source": source, "text": content})
        return {"id": doc_id, "topic": topic, "source": source,
                "chars": len(content)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/db/seed")
def reseed():
    """초기 시드 데이터를 강제 재적재 (기존 데이터 전체 삭제)."""
    try:
        rag = LocalRAG(DB_PATH)
        n = rag.seed(KNOWLEDGE_BASE)
        return {"seeded": n}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
