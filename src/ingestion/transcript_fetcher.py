"""YouTube transcript fetcher supporting both v0.6.x and v1.x of youtube-transcript-api."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TranscriptResult:
    video_id: str
    text: str
    language: str
    is_generated: bool
    status: str  # "ok" | "disabled" | "not_found" | "unavailable" | "error"
    segments_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_snippet_text(snippet: Any) -> str:
    if isinstance(snippet, dict):
        return str(snippet.get("text", "")).strip()
    return str(getattr(snippet, "text", "")).strip()


def _join_transcript_segments(fetched: Any) -> tuple[str, int]:
    segments = list(fetched)
    parts = [_extract_snippet_text(seg).replace("\n", " ") for seg in segments]
    cleaned = " ".join(p for p in parts if p)
    return cleaned, len(segments)


class TranscriptFetcher:
    """Fetch and clean Indonesian/English YouTube transcripts with graceful fallbacks."""

    def __init__(self, preferred_languages: list[str] | None = None) -> None:
        self.preferred_languages = preferred_languages or ["id", "en"]

    def _list_transcripts(self, video_id: str) -> Any:
        # Kompatibel dengan youtube-transcript-api v1.x (instance method .list())
        # maupun v0.6.x (class method .list_transcripts())
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
            return YouTubeTranscriptApi.list_transcripts(video_id)
        api = YouTubeTranscriptApi()
        return api.list(video_id)

    def fetch(self, video_id: str) -> TranscriptResult:
        """Fetch the best available transcript for a video_id."""
        try:
            transcript_list = self._list_transcripts(video_id)

            chosen = None
            # 1. Prioritas transkrip manual sesuai urutan bahasa (id -> en)
            try:
                chosen = transcript_list.find_manually_created_transcript(self.preferred_languages)
            except NoTranscriptFound:
                chosen = None

            # 2. Fallback ke auto-generated transcript sesuai urutan bahasa
            if chosen is None:
                try:
                    chosen = transcript_list.find_generated_transcript(self.preferred_languages)
                except NoTranscriptFound:
                    chosen = None

            # 3. Fallback ke transkrip bahasa apa pun yang tersedia (dan translate ke 'id' bila perlu)
            if chosen is None:
                available = list(transcript_list)
                if available:
                    first = available[0]
                    if getattr(first, "is_translatable", False) and "id" in self.preferred_languages:
                        try:
                            chosen = first.translate("id")
                        except Exception:
                            chosen = first
                    else:
                        chosen = first

            if chosen is None:
                return TranscriptResult(
                    video_id=video_id,
                    text="",
                    language="",
                    is_generated=False,
                    status="not_found",
                    segments_count=0,
                )

            fetched = chosen.fetch()
            text, count = _join_transcript_segments(fetched)
            return TranscriptResult(
                video_id=video_id,
                text=text,
                language=getattr(chosen, "language_code", ""),
                is_generated=bool(getattr(chosen, "is_generated", False)),
                status="ok",
                segments_count=count,
            )

        except TranscriptsDisabled:
            logger.info("Transkrip dinonaktifkan pada video %s.", video_id)
            return TranscriptResult(
                video_id=video_id,
                text="",
                language="",
                is_generated=False,
                status="disabled",
            )
        except NoTranscriptFound:
            logger.info("Transkrip tidak ditemukan untuk video %s.", video_id)
            return TranscriptResult(
                video_id=video_id,
                text="",
                language="",
                is_generated=False,
                status="not_found",
            )
        except VideoUnavailable:
            logger.warning("Video %s tidak tersedia saat mengambil transkrip.", video_id)
            return TranscriptResult(
                video_id=video_id,
                text="",
                language="",
                is_generated=False,
                status="unavailable",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gagal mengambil transkrip video %s: %s", video_id, exc)
            return TranscriptResult(
                video_id=video_id,
                text="",
                language="",
                is_generated=False,
                status="error",
            )
