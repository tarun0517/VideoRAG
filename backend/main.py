"""VideoRAG FastAPI application.

Run from the project root:
    uvicorn backend.main:app --reload
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from .config import get_settings
from .models import (
    ChatRequest,
    ChatResponse,
    IndexRequest,
    IndexResponse,
    LibraryVideo,
    VideoSummary,
)
from .services.rag_engine import LLMError, RagEngine
from .services.vector_store import VectorStore
from .services.youtube_loader import (
    TranscriptUnavailable,
    chunk_segments,
    fetch_transcript,
    resolve_url,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("videorag")

settings = get_settings()
store = VectorStore(settings)
engine = RagEngine(settings, store)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="VideoRAG", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Ingestion pipeline (shared by /api/index and /api/index/stream) ──────────
def _log(message: str, level: str = "info") -> dict[str, Any]:
    return {"type": "log", "level": level, "message": message}


async def run_ingestion(url: str, collection: str) -> AsyncIterator[dict[str, Any]]:
    yield _log("Resolving YouTube URL…")
    try:
        videos = await asyncio.to_thread(resolve_url, url, settings.max_playlist_videos)
    except ValueError as exc:
        yield {"type": "error", "status": 422, "message": str(exc)}
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("URL resolution failed")
        yield {"type": "error", "status": 502, "message": f"Could not read that URL from YouTube: {str(exc)[:250]}"}
        return

    total = len(videos)
    yield _log(f"Found {total} video{'s' if total != 1 else ''}.")
    yield {"type": "progress", "done": 0, "total": total}

    summaries: list[VideoSummary] = []
    total_chunks = 0

    for n, video in enumerate(videos, start=1):
        yield _log(f"[{n}/{total}] Fetching transcript: {video.title}")
        base = dict(video_id=video.video_id, title=video.title, url=video.url, thumbnail=video.thumbnail)
        try:
            segments = await asyncio.to_thread(fetch_transcript, video.video_id)
        except TranscriptUnavailable as exc:
            yield _log(f"Skipped “{video.title}”: {exc}", "warn")
            summaries.append(VideoSummary(**base, status="skipped", message=str(exc)))
            yield {"type": "progress", "done": n, "total": total}
            continue

        chunks = chunk_segments(
            video.video_id, video.title, segments,
            chunk_seconds=settings.chunk_seconds,
            overlap_seconds=settings.chunk_overlap_seconds,
            max_chars=settings.max_chunk_chars,
        )
        yield _log(f"Chunked {len(segments)} caption lines into {len(chunks)} chunks.")
        yield _log("Embedding and indexing in Qdrant…")
        try:
            count = await asyncio.to_thread(store.index_chunks, collection, video, chunks)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Indexing failed")
            yield {"type": "error", "status": 502, "message": f"Indexing failed: {str(exc)[:300]}"}
            return

        total_chunks += count
        summaries.append(VideoSummary(**base, chunk_count=count, status="indexed"))
        yield _log(f"Indexed {count} chunks for “{video.title}”.", "ok")
        yield {"type": "progress", "done": n, "total": total}

    indexed = sum(1 for s in summaries if s.status == "indexed")
    result = IndexResponse(
        collection=collection,
        videos=summaries,
        total_chunks=total_chunks,
        indexed=indexed,
        skipped=len(summaries) - indexed,
    )
    level = "ok" if indexed else "warn"
    yield _log(f"Done: {indexed} indexed, {result.skipped} skipped, {total_chunks} chunks.", level)
    yield {"type": "result", "data": result.model_dump(mode="json")}


# ── API routes ───────────────────────────────────────────────────────────────
@app.get("/api/status")
async def status() -> dict[str, Any]:
    health = await asyncio.to_thread(store.health)
    provider = engine.resolve_provider()
    return {
        "status": "ok" if health["connected"] else "degraded",
        "qdrant": health,
        "collection": settings.collection_name,
        "collection_exists": settings.collection_name in health["collections"],
        "embedding_model": settings.embedding_model,
        "llm": {
            "provider": provider or "none",
            "model": engine.model_for(provider),
            "configured": provider is not None,
        },
    }


@app.post("/api/index", response_model=IndexResponse)
async def index_videos(req: IndexRequest) -> IndexResponse:
    collection = req.collection or settings.collection_name
    result: dict[str, Any] | None = None
    async for event in run_ingestion(req.url, collection):
        if event["type"] == "error":
            raise HTTPException(status_code=event.get("status", 500), detail=event["message"])
        if event["type"] == "result":
            result = event["data"]
    if result is None:
        raise HTTPException(status_code=500, detail="Ingestion finished without a result.")
    return IndexResponse(**result)


@app.post("/api/index/stream")
async def index_videos_stream(req: IndexRequest) -> StreamingResponse:
    """Same as /api/index but streams newline-delimited JSON progress events."""
    collection = req.collection or settings.collection_name

    async def generator() -> AsyncIterator[str]:
        try:
            async for event in run_ingestion(req.url, collection):
                yield json.dumps(event) + "\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Streaming ingestion crashed")
            yield json.dumps({"type": "error", "status": 500, "message": str(exc)[:300]}) + "\n"

    return StreamingResponse(
        generator(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    collection = req.collection or settings.collection_name
    try:
        return await engine.answer(req.query, collection, req.top_k, req.video_ids)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(exc)[:300]}")


@app.get("/api/videos", response_model=list[LibraryVideo])
async def list_videos(collection: str | None = Query(default=None, max_length=64)) -> list[LibraryVideo]:
    try:
        return await asyncio.to_thread(store.list_videos, collection or settings.collection_name)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Could not read the library: {str(exc)[:300]}")


@app.delete("/api/videos/{video_id}")
async def delete_video(video_id: str, collection: str | None = Query(default=None, max_length=64)) -> dict[str, Any]:
    name = collection or settings.collection_name
    try:
        await asyncio.to_thread(store.delete_video, name, video_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Delete failed: {str(exc)[:300]}")
    return {"deleted": video_id, "collection": name}


@app.get("/", include_in_schema=False)
async def index_page() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
