"""Pydantic v2 data models shared across the backend."""
import re
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator

YOUTUBE_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
    "youtu.be", "www.youtu.be",
}
COLLECTION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def format_timestamp(seconds: float) -> str:
    """Seconds -> mm:ss or h:mm:ss."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_video_url(video_id: str, start: float = 0) -> str:
    return f"https://www.youtube.com/watch?v={video_id}&t={int(max(0, start))}s"


def thumbnail_url(video_id: str) -> str:
    return f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg"


def _clean_collection(v: str | None) -> str | None:
    if v is None:
        return None
    v = v.strip()
    if not v:
        return None
    if not COLLECTION_RE.match(v):
        raise ValueError("Collection names may only contain letters, numbers, '_' and '-' (max 64).")
    return v


# ── Internal domain models ───────────────────────────────────────────────────
class TranscriptSegment(BaseModel):
    text: str
    start: float
    duration: float = 0.0


class VideoInfo(BaseModel):
    video_id: str
    title: str
    url: str
    thumbnail: str


class Chunk(BaseModel):
    video_id: str
    title: str
    text: str
    start: float
    end: float
    chunk_index: int
    url: str


class RetrievedChunk(BaseModel):
    score: float
    video_id: str
    title: str
    text: str
    start: float
    end: float
    chunk_index: int
    url: str
    thumbnail: str


# ── API request models ───────────────────────────────────────────────────────
class IndexRequest(BaseModel):
    url: str = Field(..., min_length=5, max_length=500)
    collection: str | None = None

    @field_validator("url")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        v = v.strip()
        parsed = urlparse(v if "://" in v else f"https://{v}")
        if (parsed.hostname or "").lower() not in YOUTUBE_HOSTS:
            raise ValueError("Enter a YouTube video or playlist URL.")
        return v

    @field_validator("collection")
    @classmethod
    def _validate_collection(cls, v: str | None) -> str | None:
        return _clean_collection(v)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    collection: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    video_ids: list[str] | None = None

    @field_validator("query")
    @classmethod
    def _validate_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty.")
        return v

    @field_validator("collection")
    @classmethod
    def _validate_collection(cls, v: str | None) -> str | None:
        return _clean_collection(v)


# ── API response models ──────────────────────────────────────────────────────
class VideoSummary(BaseModel):
    video_id: str
    title: str
    url: str
    thumbnail: str
    chunk_count: int = 0
    status: str  # "indexed" | "skipped"
    message: str = ""


class IndexResponse(BaseModel):
    collection: str
    videos: list[VideoSummary]
    total_chunks: int
    indexed: int
    skipped: int


class Source(BaseModel):
    id: str
    video_id: str
    title: str
    start: float
    end: float
    timestamp: str
    end_timestamp: str
    url: str
    thumbnail: str
    score: float
    text: str
    cited: bool = False


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    collection: str
    provider: str
    model: str


class LibraryVideo(BaseModel):
    video_id: str
    title: str
    url: str
    thumbnail: str
    chunk_count: int
