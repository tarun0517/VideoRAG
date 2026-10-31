"""YouTube URL parsing, playlist resolution, metadata and transcript fetching.

All functions here are synchronous (yt-dlp and youtube-transcript-api are blocking);
call them via `asyncio.to_thread` from async code.
"""
import html
import json
import logging
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass

import yt_dlp
from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

from ..models import (
    YOUTUBE_HOSTS,
    Chunk,
    TranscriptSegment,
    VideoInfo,
    build_video_url,
    thumbnail_url,
)

logger = logging.getLogger("videorag.youtube")

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
PLAYLIST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{10,64}$")
PREFERRED_LANGUAGES = ["en", "en-US", "en-GB", "en-IN", "en-CA", "en-AU"]
_NOISE_RE = re.compile(r"\[(?:[^\]]{0,30}?(?:music|applause|laughter|inaudible|silence)[^\]]{0,30}?)\]|♪+", re.I)


class TranscriptUnavailable(Exception):
    """Raised when a transcript cannot be obtained for a video."""


@dataclass
class ParsedUrl:
    kind: str  # "video" | "playlist"
    video_id: str | None = None
    playlist_id: str | None = None


# ── URL handling ─────────────────────────────────────────────────────────────
def parse_youtube_url(url: str) -> ParsedUrl:
    raw = url.strip()
    parsed = urllib.parse.urlparse(raw if "://" in raw else f"https://{raw}")
    host = (parsed.hostname or "").lower()
    if host not in YOUTUBE_HOSTS:
        raise ValueError("Not a YouTube URL.")

    qs = urllib.parse.parse_qs(parsed.query)
    path = parsed.path.rstrip("/")

    video_id: str | None = None
    if host in ("youtu.be", "www.youtu.be"):
        video_id = path.lstrip("/").split("/")[0] or None
    elif path == "/watch":
        video_id = (qs.get("v") or [None])[0]
    else:
        m = re.match(r"^/(?:shorts|embed|live|v)/([^/?]+)", path)
        if m:
            video_id = m.group(1)
    if video_id and not VIDEO_ID_RE.match(video_id):
        video_id = None

    playlist_id = (qs.get("list") or [None])[0]
    if playlist_id and not PLAYLIST_ID_RE.match(playlist_id):
        playlist_id = None

    # "RD..." lists are auto-generated mixes, not real playlists.
    if playlist_id and not playlist_id.startswith("RD"):
        return ParsedUrl(kind="playlist", playlist_id=playlist_id, video_id=video_id)
    if video_id:
        return ParsedUrl(kind="video", video_id=video_id)
    raise ValueError("Could not find a video or playlist ID in that URL.")


def _oembed_title(video_id: str) -> str | None:
    try:
        target = urllib.parse.quote(f"https://www.youtube.com/watch?v={video_id}", safe="")
        req = urllib.request.Request(
            f"https://www.youtube.com/oembed?url={target}&format=json",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8")).get("title")
    except Exception as exc:  # noqa: BLE001
        logger.debug("oEmbed lookup failed for %s: %s", video_id, exc)
        return None


def _make_info(video_id: str, title: str | None) -> VideoInfo:
    return VideoInfo(
        video_id=video_id,
        title=(title or video_id).strip() or video_id,
        url=build_video_url(video_id, 0).replace("&t=0s", ""),
        thumbnail=thumbnail_url(video_id),
    )


def get_video_info(video_id: str) -> VideoInfo:
    """Resolve a single video's title with yt-dlp, falling back to oEmbed, then the ID."""
    opts = {
        "quiet": True, "no_warnings": True, "skip_download": True,
        "noplaylist": True, "socket_timeout": 20,
    }
    title: str | None = None
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            title = (info or {}).get("title")
    except Exception as exc:  # noqa: BLE001
        logger.warning("yt-dlp metadata failed for %s: %s", video_id, exc)
    if not title:
        title = _oembed_title(video_id)
    return _make_info(video_id, title)


def get_playlist_videos(playlist_id: str, max_videos: int) -> list[VideoInfo]:
    opts = {
        "quiet": True, "no_warnings": True, "skip_download": True,
        "extract_flat": "in_playlist", "ignoreerrors": True,
        "playlistend": max_videos, "socket_timeout": 20,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/playlist?list={playlist_id}", download=False)
    entries = (info or {}).get("entries") or []
    videos: list[VideoInfo] = []
    seen: set[str] = set()
    for entry in entries:
        if not entry:
            continue
        vid = entry.get("id")
        title = entry.get("title") or ""
        if not vid or not VIDEO_ID_RE.match(vid) or vid in seen:
            continue
        if title.lower() in ("[private video]", "[deleted video]"):
            continue
        seen.add(vid)
        videos.append(_make_info(vid, title))
        if len(videos) >= max_videos:
            break
    return videos


def resolve_url(url: str, max_videos: int = 50) -> list[VideoInfo]:
    """Turn a video or playlist URL into a list of VideoInfo."""
    parsed = parse_youtube_url(url)
    if parsed.kind == "playlist" and parsed.playlist_id:
        videos = get_playlist_videos(parsed.playlist_id, max_videos)
        if not videos:
            raise ValueError("The playlist is empty, private, or could not be read.")
        return videos
    assert parsed.video_id is not None
    return [get_video_info(parsed.video_id)]


# ── Transcripts ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = _NOISE_RE.sub(" ", text)
    text = text.replace(">>", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def _first_line(exc: Exception) -> str:
    for line in str(exc).splitlines():
        line = line.strip()
        if line:
            return line[:220]
    return exc.__class__.__name__


def fetch_transcript(video_id: str) -> list[TranscriptSegment]:
    """Fetch a timestamped, cleaned transcript. Raises TranscriptUnavailable on failure."""
    api = YouTubeTranscriptApi()
    try:
        transcript_list = api.list(video_id)
        try:
            transcript = transcript_list.find_transcript(PREFERRED_LANGUAGES)
        except NoTranscriptFound:
            transcript = next(iter(transcript_list), None)
            if transcript is None:
                raise TranscriptUnavailable("No transcripts exist for this video.")
        raw = transcript.fetch().to_raw_data()
    except TranscriptUnavailable:
        raise
    except TranscriptsDisabled:
        raise TranscriptUnavailable("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise TranscriptUnavailable("No transcript found for this video.")
    except VideoUnavailable:
        raise TranscriptUnavailable("The video is unavailable (private, removed or region-locked).")
    except CouldNotRetrieveTranscript as exc:
        raise TranscriptUnavailable(f"Could not retrieve transcript: {_first_line(exc)}")
    except Exception as exc:  # noqa: BLE001
        raise TranscriptUnavailable(f"Unexpected transcript error: {_first_line(exc)}")

    segments: list[TranscriptSegment] = []
    for item in raw:
        text = clean_text(str(item.get("text", "")))
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                text=text,
                start=float(item.get("start", 0.0)),
                duration=float(item.get("duration", 0.0) or 0.0),
            )
        )
    if not segments:
        raise TranscriptUnavailable("The transcript is empty.")
    return segments


# ── Chunking ─────────────────────────────────────────────────────────────────
def chunk_segments(
    video_id: str,
    title: str,
    segments: list[TranscriptSegment],
    chunk_seconds: int = 60,
    overlap_seconds: int = 10,
    max_chars: int = 1500,
) -> list[Chunk]:
    """Group caption segments into ~chunk_seconds windows with a small overlap."""
    chunks: list[Chunk] = []
    n = len(segments)
    i = 0
    while i < n:
        start = segments[i].start
        j = i
        chars = 0
        while j < n:
            seg = segments[j]
            if j > i and (seg.start - start >= chunk_seconds or chars + len(seg.text) > max_chars):
                break
            chars += len(seg.text) + 1
            j += 1

        group = segments[i:j]
        last = group[-1]
        end = max(last.start + last.duration, last.start + 1.0)
        text = " ".join(s.text for s in group).strip()
        if len(text) >= 15 or not chunks:
            chunks.append(
                Chunk(
                    video_id=video_id,
                    title=title,
                    text=text,
                    start=round(start, 2),
                    end=round(end, 2),
                    chunk_index=len(chunks),
                    url=build_video_url(video_id, start),
                )
            )
        if j >= n:
            break

        k = j
        while k - 1 > i and segments[j].start - segments[k - 1].start <= overlap_seconds:
            k -= 1
        i = k
    return chunks
