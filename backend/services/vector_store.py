"""Embeddings + Qdrant vector store (cloud, local server, or embedded on-disk)."""
import logging
import threading
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from ..config import Settings
from ..models import Chunk, LibraryVideo, RetrievedChunk, VideoInfo, thumbnail_url

logger = logging.getLogger("videorag.vectors")

EMBED_BATCH = 64
UPSERT_BATCH = 128


class Embedder:
    """Wraps sentence-transformers (local) or OpenAI embeddings (text-embedding-*)."""

    def __init__(self, model_name: str, openai_api_key: str = ""):
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        self._st_model = None
        self._openai = None
        self._dim: int | None = None
        self._lock = threading.Lock()

    @property
    def uses_openai(self) -> bool:
        return self.model_name.startswith("text-embedding")

    def _load(self) -> None:
        with self._lock:
            if self.uses_openai:
                if self._openai is None:
                    if not self.openai_api_key:
                        raise RuntimeError("EMBEDDING_MODEL uses OpenAI but OPENAI_API_KEY is not set.")
                    from openai import OpenAI

                    self._openai = OpenAI(api_key=self.openai_api_key)
            elif self._st_model is None:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model %s", self.model_name)
                self._st_model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load()
        vectors: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i:i + EMBED_BATCH]
            if self.uses_openai:
                resp = self._openai.embeddings.create(model=self.model_name, input=batch)
                ordered = sorted(resp.data, key=lambda d: d.index)
                vectors.extend([list(d.embedding) for d in ordered])
            else:
                arr = self._st_model.encode(
                    batch, batch_size=EMBED_BATCH, normalize_embeddings=True, show_progress_bar=False
                )
                vectors.extend(arr.tolist())
        return vectors

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed(["dimension probe"])[0])
        return self._dim


class VectorStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = Embedder(settings.embedding_model, settings.openai_api_key)
        self._lock = threading.RLock()
        if settings.qdrant_url:
            self.mode = "remote"
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
                timeout=60,
            )
        else:
            self.mode = "local"
            Path(settings.qdrant_local_path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=settings.qdrant_local_path)

    # ── health ───────────────────────────────────────────────────────────────
    def health(self) -> dict:
        try:
            with self._lock:
                names = [c.name for c in self.client.get_collections().collections]
            return {"connected": True, "mode": self.mode, "collections": names, "error": None}
        except Exception as exc:  # noqa: BLE001
            return {"connected": False, "mode": self.mode, "collections": [], "error": str(exc)[:300]}

    def collection_exists(self, collection: str) -> bool:
        with self._lock:
            return bool(self.client.collection_exists(collection))

    # ── collection management ────────────────────────────────────────────────
    def ensure_collection(self, collection: str) -> None:
        dim = self.embedder.dim
        with self._lock:
            if not self.client.collection_exists(collection):
                try:
                    self.client.create_collection(
                        collection_name=collection,
                        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
                    )
                except Exception:  # noqa: BLE001
                    if not self.client.collection_exists(collection):
                        raise
                try:
                    self.client.create_payload_index(
                        collection_name=collection,
                        field_name="video_id",
                        field_schema=qm.PayloadSchemaType.KEYWORD,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Payload index skipped: %s", exc)
                return

            vectors = self.client.get_collection(collection).config.params.vectors
            if isinstance(vectors, qm.VectorParams) and vectors.size != dim:
                raise ValueError(
                    f"Collection '{collection}' stores {vectors.size}-dim vectors but "
                    f"'{self.embedder.model_name}' produces {dim}. Use a different collection name."
                )

    def delete_video(self, collection: str, video_id: str) -> None:
        with self._lock:
            if not self.client.collection_exists(collection):
                return
            self.client.delete(
                collection_name=collection,
                points_selector=qm.FilterSelector(
                    filter=qm.Filter(
                        must=[qm.FieldCondition(key="video_id", match=qm.MatchValue(value=video_id))]
                    )
                ),
            )

    # ── indexing ─────────────────────────────────────────────────────────────
    def index_chunks(self, collection: str, video: VideoInfo, chunks: list[Chunk]) -> int:
        """Embed and upsert a video's chunks (replacing any previous version). Returns count."""
        if not chunks:
            return 0
        self.ensure_collection(collection)
        texts = [f"{c.title}\n{c.text}" for c in chunks]
        vectors = self.embedder.embed(texts)

        points = [
            qm.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{c.video_id}:{c.chunk_index}")),
                vector=vec,
                payload={
                    "video_id": c.video_id,
                    "title": c.title,
                    "text": c.text,
                    "start": c.start,
                    "end": c.end,
                    "chunk_index": c.chunk_index,
                    "url": c.url,
                    "thumbnail": video.thumbnail,
                },
            )
            for c, vec in zip(chunks, vectors)
        ]
        self.delete_video(collection, video.video_id)
        with self._lock:
            for i in range(0, len(points), UPSERT_BATCH):
                self.client.upsert(
                    collection_name=collection, points=points[i:i + UPSERT_BATCH], wait=True
                )
        return len(points)

    # ── search ───────────────────────────────────────────────────────────────
    def search(
        self,
        collection: str,
        query: str,
        top_k: int = 6,
        video_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        if not self.collection_exists(collection):
            return []
        vector = self.embedder.embed([query])[0]
        flt = None
        if video_ids:
            flt = qm.Filter(
                must=[qm.FieldCondition(key="video_id", match=qm.MatchAny(any=video_ids))]
            )
        with self._lock:
            result = self.client.query_points(
                collection_name=collection,
                query=vector,
                limit=top_k,
                query_filter=flt,
                with_payload=True,
            )
        hits: list[RetrievedChunk] = []
        for p in result.points:
            pl = p.payload or {}
            vid = str(pl.get("video_id", ""))
            hits.append(
                RetrievedChunk(
                    score=float(p.score),
                    video_id=vid,
                    title=str(pl.get("title", vid)),
                    text=str(pl.get("text", "")),
                    start=float(pl.get("start", 0.0)),
                    end=float(pl.get("end", 0.0)),
                    chunk_index=int(pl.get("chunk_index", 0)),
                    url=str(pl.get("url", "")),
                    thumbnail=str(pl.get("thumbnail") or thumbnail_url(vid)),
                )
            )
        return hits

    # ── library ──────────────────────────────────────────────────────────────
    def list_videos(self, collection: str) -> list[LibraryVideo]:
        if not self.collection_exists(collection):
            return []
        found: dict[str, dict] = {}
        offset = None
        while True:
            with self._lock:
                points, offset = self.client.scroll(
                    collection_name=collection,
                    limit=512,
                    offset=offset,
                    with_payload=["video_id", "title", "thumbnail"],
                    with_vectors=False,
                )
            for p in points:
                pl = p.payload or {}
                vid = str(pl.get("video_id", ""))
                if not vid:
                    continue
                entry = found.setdefault(
                    vid,
                    {
                        "title": str(pl.get("title", vid)),
                        "thumbnail": str(pl.get("thumbnail") or thumbnail_url(vid)),
                        "count": 0,
                    },
                )
                entry["count"] += 1
            if offset is None:
                break
        return sorted(
            (
                LibraryVideo(
                    video_id=vid,
                    title=e["title"],
                    url=f"https://www.youtube.com/watch?v={vid}",
                    thumbnail=e["thumbnail"],
                    chunk_count=e["count"],
                )
                for vid, e in found.items()
            ),
            key=lambda v: v.title.lower(),
        )
