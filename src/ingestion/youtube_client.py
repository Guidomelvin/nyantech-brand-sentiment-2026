"""YouTube Data API v3 client for channel resolution, 2026 video listing, and full comment extraction."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.utils.logger import get_logger

logger = get_logger(__name__)

_ISO_DURATION_RE = re.compile(
    r"^P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)


def parse_iso8601_duration(duration_str: str | None) -> int:
    """Convert YouTube ISO-8601 duration (e.g. 'PT12M34S') into total seconds."""
    if not duration_str:
        return 0
    match = _ISO_DURATION_RE.match(duration_str)
    if not match:
        return 0
    days = int(match.group("days") or 0)
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def parse_youtube_timestamp(ts: str) -> datetime:
    """Parse RFC 3339 / ISO-8601 UTC timestamp from YouTube API."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)


def _is_retryable_http_error(exc: BaseException) -> bool:
    if not isinstance(exc, HttpError):
        return False
    status = getattr(exc.resp, "status", None)
    if status in {429, 500, 502, 503, 504}:
        return True
    if status == 403:
        content = exc.content.decode("utf-8", errors="ignore") if exc.content else ""
        # Retry rateLimitExceeded, do NOT retry quotaExceeded or commentsDisabled
        return "rateLimitExceeded" in content
    return False


@dataclass
class ChannelInfo:
    channel_id: str
    title: str
    handle: str
    uploads_playlist_id: str
    subscriber_count: int = 0
    video_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VideoRecord:
    video_id: str
    channel_id: str
    channel_title: str
    title: str
    description: str
    published_at_utc: str
    published_at_local: str
    tags: list[str] = field(default_factory=list)
    duration_seconds: int = 0
    view_count: int = 0
    like_count: int = 0
    comment_count: int = 0
    video_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CommentRecord:
    comment_id: str
    video_id: str
    parent_comment_id: str | None
    is_reply: bool
    author_display_name: str
    author_channel_id: str
    text: str
    like_count: int
    published_at_utc: str
    updated_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class YouTubeAPIClient:
    """Wrapper around YouTube Data API v3 optimized for quota efficiency and reliability."""

    def __init__(
        self,
        api_key: str,
        request_delay_seconds: float = 0.15,
        service: Any | None = None,
    ) -> None:
        if not api_key and service is None:
            raise ValueError(
                "YOUTUBE_API_KEY belum diisi. Setel pada file `.env` sebelum menjalankan Stage 1."
            )
        self.api_key = api_key
        self.request_delay_seconds = request_delay_seconds
        self.service = service or build("youtube", "v3", developerKey=api_key, cache_discovery=False)

    @retry(
        retry=retry_if_exception(_is_retryable_http_error),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _execute(self, request: Any) -> dict[str, Any]:
        response = request.execute()
        if self.request_delay_seconds > 0:
            time.sleep(self.request_delay_seconds)
        return response

    def resolve_channel(
        self,
        handle: str = "@NYANTECH",
        channel_id: str = "",
    ) -> ChannelInfo:
        """Resolve channel ID and uploads playlist ID from handle or explicit channel ID."""
        items: list[dict[str, Any]] = []

        if channel_id:
            resp = self._execute(
                self.service.channels().list(
                    part="id,snippet,contentDetails,statistics",
                    id=channel_id,
                )
            )
            items = resp.get("items", [])

        if not items and handle:
            normalized_handle = handle if handle.startswith("@") else f"@{handle}"
            resp = self._execute(
                self.service.channels().list(
                    part="id,snippet,contentDetails,statistics",
                    forHandle=normalized_handle,
                )
            )
            items = resp.get("items", [])

        if not items and handle:
            logger.warning(
                "Lookup via forHandle='%s' tidak ditemukan, fallback ke search.list...",
                handle,
            )
            search_resp = self._execute(
                self.service.search().list(
                    part="snippet",
                    q=handle.lstrip("@"),
                    type="channel",
                    maxResults=5,
                )
            )
            search_items = search_resp.get("items", [])
            if search_items:
                found_id = search_items[0]["snippet"]["channelId"]
                resp = self._execute(
                    self.service.channels().list(
                        part="id,snippet,contentDetails,statistics",
                        id=found_id,
                    )
                )
                items = resp.get("items", [])

        if not items:
            raise LookupError(
                f"Channel YouTube tidak ditemukan untuk handle='{handle}' / id='{channel_id}'."
            )

        ch = items[0]
        snippet = ch.get("snippet", {})
        stats = ch.get("statistics", {})
        uploads_id = ch["contentDetails"]["relatedPlaylists"]["uploads"]

        return ChannelInfo(
            channel_id=ch["id"],
            title=snippet.get("title", ""),
            handle=snippet.get("customUrl", handle),
            uploads_playlist_id=uploads_id,
            subscriber_count=int(stats.get("subscriberCount", 0) or 0),
            video_count=int(stats.get("videoCount", 0) or 0),
        )

    def fetch_videos_in_date_range(
        self,
        channel_info: ChannelInfo,
        start_date: str = "2026-01-01",
        end_date: str = "2026-12-31",
        timezone_name: str = "Asia/Jakarta",
        use_local_timezone: bool = True,
    ) -> list[VideoRecord]:
        """Fetch all videos uploaded between start_date and end_date (inclusive).

        Uses `playlistItems.list` on the channel's `uploads` playlist (1 quota unit/page)
        instead of `search.list` (100 quota units/page), then hydrates full video metadata
        via `videos.list` in batches of 50.
        """
        tz = ZoneInfo(timezone_name) if use_local_timezone else UTC
        start_dt = datetime.fromisoformat(start_date).replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=tz
        )
        end_dt = datetime.fromisoformat(end_date).replace(
            hour=23, minute=59, second=59, microsecond=999999, tzinfo=tz
        )

        candidate_ids: list[str] = []
        next_page_token: str | None = None
        consecutive_older_than_start = 0

        logger.info(
            "Mengambil daftar video dari playlist uploads '%s' (%s s.d. %s [%s])",
            channel_info.uploads_playlist_id,
            start_date,
            end_date,
            tz.key if isinstance(tz, ZoneInfo) else "UTC",
        )

        while True:
            resp = self._execute(
                self.service.playlistItems().list(
                    part="contentDetails,snippet",
                    playlistId=channel_info.uploads_playlist_id,
                    maxResults=50,
                    pageToken=next_page_token,
                )
            )

            items = resp.get("items", [])
            if not items:
                break

            for item in items:
                content = item.get("contentDetails", {})
                snippet = item.get("snippet", {})
                video_id = content.get("videoId") or snippet.get("resourceId", {}).get("videoId")
                published_raw = content.get("videoPublishedAt") or snippet.get("publishedAt")
                if not video_id or not published_raw:
                    continue

                pub_utc = parse_youtube_timestamp(published_raw)
                pub_local = pub_utc.astimezone(tz)

                if start_dt <= pub_local <= end_dt:
                    candidate_ids.append(video_id)
                    consecutive_older_than_start = 0
                elif pub_local < start_dt:
                    consecutive_older_than_start += 1
                else:
                    # Video rilis setelah end_date (misal 2027+), lanjutkan scan ke belakang
                    consecutive_older_than_start = 0

            next_page_token = resp.get("nextPageToken")
            # Safety buffer: hentikan pagination jika 25 video berurutan lebih lama dari start_date
            if not next_page_token or consecutive_older_than_start >= 25:
                break

        # Deduplikasi dengan mempertahankan urutan
        unique_ids = list(dict.fromkeys(candidate_ids))
        logger.info("Ditemukan %d video kandidat dalam rentang %s..%s.", len(unique_ids), start_date, end_date)

        videos: list[VideoRecord] = []
        for idx in range(0, len(unique_ids), 50):
            batch_ids = unique_ids[idx : idx + 50]
            vid_resp = self._execute(
                self.service.videos().list(
                    part="snippet,contentDetails,statistics",
                    id=",".join(batch_ids),
                )
            )
            for item in vid_resp.get("items", []):
                snippet = item.get("snippet", {})
                content_details = item.get("contentDetails", {})
                stats = item.get("statistics", {})
                published_raw = snippet.get("publishedAt", "")
                if not published_raw:
                    continue

                pub_utc = parse_youtube_timestamp(published_raw)
                pub_local = pub_utc.astimezone(ZoneInfo(timezone_name))

                # Verifikasi ulang batas tanggal menggunakan snippet.publishedAt resmi
                check_dt = pub_utc.astimezone(tz)
                if not (start_dt <= check_dt <= end_dt):
                    continue

                video_id = item["id"]
                videos.append(
                    VideoRecord(
                        video_id=video_id,
                        channel_id=snippet.get("channelId", channel_info.channel_id),
                        channel_title=snippet.get("channelTitle", channel_info.title),
                        title=snippet.get("title", ""),
                        description=snippet.get("description", ""),
                        published_at_utc=pub_utc.isoformat(),
                        published_at_local=pub_local.isoformat(),
                        tags=snippet.get("tags", []) or [],
                        duration_seconds=parse_iso8601_duration(content_details.get("duration")),
                        view_count=int(stats.get("viewCount", 0) or 0),
                        like_count=int(stats.get("likeCount", 0) or 0),
                        comment_count=int(stats.get("commentCount", 0) or 0),
                        video_url=f"https://www.youtube.com/watch?v={video_id}",
                    )
                )

        # Urutkan kronologis dari awal 2026 ke akhir 2026
        videos.sort(key=lambda v: v.published_at_utc)
        return videos

    def _fetch_remaining_replies(self, parent_comment_id: str, video_id: str) -> list[CommentRecord]:
        """Fetch all replies for a top-level comment thread when inline replies are truncated."""
        replies: list[CommentRecord] = []
        next_page_token: str | None = None

        while True:
            resp = self._execute(
                self.service.comments().list(
                    part="snippet",
                    parentId=parent_comment_id,
                    maxResults=100,
                    textFormat="plainText",
                    pageToken=next_page_token,
                )
            )
            for item in resp.get("items", []):
                snippet = item.get("snippet", {})
                replies.append(
                    CommentRecord(
                        comment_id=item["id"],
                        video_id=video_id,
                        parent_comment_id=parent_comment_id,
                        is_reply=True,
                        author_display_name=snippet.get("authorDisplayName", ""),
                        author_channel_id=snippet.get("authorChannelId", {}).get("value", ""),
                        text=snippet.get("textDisplay", "") or snippet.get("textOriginal", ""),
                        like_count=int(snippet.get("likeCount", 0) or 0),
                        published_at_utc=snippet.get("publishedAt", ""),
                        updated_at_utc=snippet.get("updatedAt", ""),
                    )
                )
            next_page_token = resp.get("nextPageToken")
            if not next_page_token:
                break

        return replies

    def fetch_all_comments(
        self,
        video_id: str,
        include_replies: bool = True,
        max_comments: int | None = None,
    ) -> list[CommentRecord]:
        """Fetch all top-level comments and (optionally) all replies for a video."""
        comments: list[CommentRecord] = []
        seen_ids: set[str] = set()
        next_page_token: str | None = None

        while True:
            try:
                resp = self._execute(
                    self.service.commentThreads().list(
                        part="snippet,replies",
                        videoId=video_id,
                        maxResults=100,
                        textFormat="plainText",
                        order="time",
                        pageToken=next_page_token,
                    )
                )
            except HttpError as exc:
                content = exc.content.decode("utf-8", errors="ignore") if exc.content else ""
                if getattr(exc.resp, "status", None) == 403 and "commentsDisabled" in content:
                    logger.warning("Komentar dinonaktifkan pada video %s.", video_id)
                    return comments
                raise

            for thread in resp.get("items", []):
                thread_snippet = thread.get("snippet", {})
                top = thread_snippet.get("topLevelComment", {})
                top_id = top.get("id")
                top_snippet = top.get("snippet", {})

                if top_id and top_id not in seen_ids:
                    seen_ids.add(top_id)
                    comments.append(
                        CommentRecord(
                            comment_id=top_id,
                            video_id=video_id,
                            parent_comment_id=None,
                            is_reply=False,
                            author_display_name=top_snippet.get("authorDisplayName", ""),
                            author_channel_id=top_snippet.get("authorChannelId", {}).get("value", ""),
                            text=top_snippet.get("textDisplay", "")
                            or top_snippet.get("textOriginal", ""),
                            like_count=int(top_snippet.get("likeCount", 0) or 0),
                            published_at_utc=top_snippet.get("publishedAt", ""),
                            updated_at_utc=top_snippet.get("updatedAt", ""),
                        )
                    )

                if max_comments is not None and len(comments) >= max_comments:
                    return comments[:max_comments]

                if include_replies and top_id:
                    total_reply_count = int(thread_snippet.get("totalReplyCount", 0) or 0)
                    inline_replies = thread.get("replies", {}).get("comments", []) or []

                    if total_reply_count > len(inline_replies):
                        reply_records = self._fetch_remaining_replies(top_id, video_id)
                    else:
                        reply_records = [
                            CommentRecord(
                                comment_id=rep["id"],
                                video_id=video_id,
                                parent_comment_id=top_id,
                                is_reply=True,
                                author_display_name=rep.get("snippet", {}).get("authorDisplayName", ""),
                                author_channel_id=rep.get("snippet", {})
                                .get("authorChannelId", {})
                                .get("value", ""),
                                text=rep.get("snippet", {}).get("textDisplay", "")
                                or rep.get("snippet", {}).get("textOriginal", ""),
                                like_count=int(rep.get("snippet", {}).get("likeCount", 0) or 0),
                                published_at_utc=rep.get("snippet", {}).get("publishedAt", ""),
                                updated_at_utc=rep.get("snippet", {}).get("updatedAt", ""),
                            )
                            for rep in inline_replies
                            if rep.get("id")
                        ]

                    for rep_record in reply_records:
                        if rep_record.comment_id not in seen_ids:
                            seen_ids.add(rep_record.comment_id)
                            comments.append(rep_record)
                            if max_comments is not None and len(comments) >= max_comments:
                                return comments[:max_comments]

            next_page_token = resp.get("nextPageToken")
            if not next_page_token:
                break

        return comments
