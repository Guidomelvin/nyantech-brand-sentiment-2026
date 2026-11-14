"""Unit & integration tests for Tahap 1 (YouTube scraping, 2026 filter, smartphone filter, transcripts)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.config import AppConfig, EnvSettings, PathsConfig
from src.ingestion.pipeline import run_stage1_ingestion
from src.ingestion.smartphone_filter import SmartphoneVideoFilter
from src.ingestion.transcript_fetcher import TranscriptFetcher, TranscriptResult
from src.ingestion.youtube_client import YouTubeAPIClient, parse_iso8601_duration


def _make_exec(payload: dict) -> MagicMock:
    req = MagicMock()
    req.execute.return_value = payload
    return req


def test_parse_iso8601_duration() -> None:
    assert parse_iso8601_duration("PT12M34S") == 754
    assert parse_iso8601_duration("PT1H2M3S") == 3723
    assert parse_iso8601_duration("PT45S") == 45
    assert parse_iso8601_duration("") == 0


def test_smartphone_filter_accepts_and_rejects_accurately() -> None:
    cfg = AppConfig(
        brands={
            "Samsung": ["samsung", "galaxy s", "galaxy a"],
            "Xiaomi": ["xiaomi", "redmi", "redmi note"],
            "ASUS": ["asus", "rog phone"],
            "Nothing": ["nothing phone", "nothing"],
            "Apple": ["apple", "iphone"],
        }
    )
    cfg.smartphone_filter.positive_keywords = [
        "smartphone",
        "hp",
        "review",
        "unboxing",
        "snapdragon",
        "rekomendasi hp",
    ]
    cfg.smartphone_filter.negative_keywords = ["laptop", "tablet", "tws", "smartwatch"]
    flt = SmartphoneVideoFilter(cfg.smartphone_filter, cfg.brands)

    assert flt.evaluate("Review Samsung Galaxy S26 Ultra Indonesia").is_smartphone is True
    assert flt.evaluate("HP 3 Jutaan Terbaik 2026! Redmi Note 15").is_smartphone is True
    assert flt.evaluate("Unboxing iPhone 17 Pro Max Resmi Indonesia").is_smartphone is True

    # Non-smartphone products from smartphone brands must be rejected
    assert flt.evaluate("Review Laptop ASUS ROG Zephyrus G16 2026").is_smartphone is False
    assert flt.evaluate("Unboxing Tablet Samsung Galaxy Tab S10").is_smartphone is False
    assert flt.evaluate("Review TWS Nothing Ear 2026").is_smartphone is False


def test_youtube_client_date_filter_and_full_comments() -> None:
    mock_service = MagicMock()

    mock_service.channels.return_value.list.return_value = _make_exec(
        {
            "items": [
                {
                    "id": "UC_NYANTECH_123",
                    "snippet": {"title": "NYANTECH", "customUrl": "@NYANTECH"},
                    "contentDetails": {"relatedPlaylists": {"uploads": "UU_NYANTECH_123"}},
                    "statistics": {"subscriberCount": "500000", "videoCount": "200"},
                }
            ]
        }
    )

    # Playlist berisi: 1 video 2027 (exclude), 2 video 2026 (include), 1 video 2025 (exclude)
    mock_service.playlistItems.return_value.list.return_value = _make_exec(
        {
            "items": [
                {
                    "contentDetails": {
                        "videoId": "vid_2027",
                        "videoPublishedAt": "2026-12-31T18:00:00Z",  # 2027-01-01 01:00 WIB
                    }
                },
                {
                    "contentDetails": {
                        "videoId": "vid_2026_phone",
                        "videoPublishedAt": "2026-06-15T10:00:00Z",
                    }
                },
                {
                    "contentDetails": {
                        "videoId": "vid_2026_jan1",
                        "videoPublishedAt": "2025-12-31T17:30:00Z",  # 2026-01-01 00:30 WIB
                    }
                },
                {
                    "contentDetails": {
                        "videoId": "vid_2025",
                        "videoPublishedAt": "2025-12-31T16:00:00Z",  # 2025-12-31 23:00 WIB
                    }
                },
            ]
        }
    )

    mock_service.videos.return_value.list.return_value = _make_exec(
        {
            "items": [
                {
                    "id": "vid_2026_phone",
                    "snippet": {
                        "channelId": "UC_NYANTECH_123",
                        "channelTitle": "NYANTECH",
                        "title": "Review Samsung Galaxy A57 5G Indonesia - Harga 5 Jutaan!",
                        "description": "Review lengkap HP Samsung terbaru harga Rp 5.499.000.",
                        "publishedAt": "2026-06-15T10:00:00Z",
                        "tags": ["samsung", "smartphone", "review"],
                    },
                    "contentDetails": {"duration": "PT14M20S"},
                    "statistics": {"viewCount": "120000", "likeCount": "4500", "commentCount": "3"},
                },
                {
                    "id": "vid_2026_jan1",
                    "snippet": {
                        "channelId": "UC_NYANTECH_123",
                        "channelTitle": "NYANTECH",
                        "title": "Review Laptop Gaming Terbaik Awal 2026",
                        "description": "Laptop RTX 5060",
                        "publishedAt": "2025-12-31T17:30:00Z",
                        "tags": ["laptop"],
                    },
                    "contentDetails": {"duration": "PT10M00S"},
                    "statistics": {"viewCount": "50000", "likeCount": "1000", "commentCount": "0"},
                },
            ]
        }
    )

    mock_service.commentThreads.return_value.list.return_value = _make_exec(
        {
            "items": [
                {
                    "snippet": {
                        "totalReplyCount": 2,
                        "topLevelComment": {
                            "id": "cmt_1",
                            "snippet": {
                                "authorDisplayName": "Budi",
                                "authorChannelId": {"value": "UC_BUDI"},
                                "textDisplay": "Kameranya gokil sih di harga 5 jutaan!",
                                "likeCount": 12,
                                "publishedAt": "2026-06-15T11:00:00Z",
                                "updatedAt": "2026-06-15T11:00:00Z",
                            },
                        },
                    },
                    # Inline replies hanya 1, padahal totalReplyCount=2 -> memicu comments.list
                    "replies": {"comments": [{"id": "rep_1", "snippet": {}}]},
                }
            ]
        }
    )

    mock_service.comments.return_value.list.return_value = _make_exec(
        {
            "items": [
                {
                    "id": "rep_1",
                    "snippet": {
                        "authorDisplayName": "Andi",
                        "authorChannelId": {"value": "UC_ANDI"},
                        "textDisplay": "Setuju bang, tapi exynosnya agak hangat",
                        "likeCount": 3,
                        "publishedAt": "2026-06-15T11:15:00Z",
                        "updatedAt": "2026-06-15T11:15:00Z",
                    },
                },
                {
                    "id": "rep_2",
                    "snippet": {
                        "authorDisplayName": "Sari",
                        "authorChannelId": {"value": "UC_SARI"},
                        "textDisplay": "Aftersales Samsung aman kok",
                        "likeCount": 5,
                        "publishedAt": "2026-06-15T12:00:00Z",
                        "updatedAt": "2026-06-15T12:00:00Z",
                    },
                },
            ]
        }
    )

    client = YouTubeAPIClient(api_key="dummy", request_delay_seconds=0.0, service=mock_service)
    ch = client.resolve_channel("@NYANTECH")
    assert ch.channel_id == "UC_NYANTECH_123"

    vids = client.fetch_videos_in_date_range(ch, "2026-01-01", "2026-12-31", "Asia/Jakarta", True)
    assert [v.video_id for v in vids] == ["vid_2026_jan1", "vid_2026_phone"]

    comments = client.fetch_all_comments("vid_2026_phone", include_replies=True)
    assert [c.comment_id for c in comments] == ["cmt_1", "rep_1", "rep_2"]
    assert comments[0].is_reply is False
    assert comments[1].is_reply is True
    assert comments[2].parent_comment_id == "cmt_1"


def test_stage1_pipeline_end_to_end(tmp_path: Path) -> None:
    # Reuse mocked client & transcript fetcher to verify persisted Parquet/JSON outputs
    mock_client = MagicMock()
    mock_client.resolve_channel.return_value = SimpleNamespace(
        channel_id="UC_NYANTECH_123",
        title="NYANTECH",
        subscriber_count=500000,
    )
    from src.ingestion.youtube_client import CommentRecord, VideoRecord

    mock_client.fetch_videos_in_date_range.return_value = [
        VideoRecord(
            video_id="vid_phone_1",
            channel_id="UC_NYANTECH_123",
            channel_title="NYANTECH",
            title="Review POCO X8 Pro Indonesia - HP 4 Jutaan Ngebut!",
            description="Harga resmi Rp 4.499.000",
            published_at_utc="2026-02-10T10:00:00+00:00",
            published_at_local="2026-02-10T17:00:00+07:00",
            tags=["poco", "smartphone"],
            duration_seconds=720,
            view_count=100000,
            like_count=5000,
            comment_count=1,
            video_url="https://www.youtube.com/watch?v=vid_phone_1",
        ),
        VideoRecord(
            video_id="vid_laptop_1",
            channel_id="UC_NYANTECH_123",
            channel_title="NYANTECH",
            title="Review Laptop Tipis Terbaik 2026",
            description="Laptop kerja",
            published_at_utc="2026-02-12T10:00:00+00:00",
            published_at_local="2026-02-12T17:00:00+07:00",
            tags=["laptop"],
            duration_seconds=600,
        ),
    ]
    mock_client.fetch_all_comments.return_value = [
        CommentRecord(
            comment_id="c1",
            video_id="vid_phone_1",
            parent_comment_id=None,
            is_reply=False,
            author_display_name="Netizen",
            author_channel_id="UC_1",
            text="Worth it banget buat gaming di 4 jutaan!",
            like_count=10,
            published_at_utc="2026-02-10T11:00:00Z",
            updated_at_utc="2026-02-10T11:00:00Z",
        )
    ]

    mock_fetcher = MagicMock(spec=TranscriptFetcher)
    mock_fetcher.fetch.return_value = TranscriptResult(
        video_id="vid_phone_1",
        text="Halo guys balik lagi di NYANTECH, kali ini kita bahas POCO X8 Pro seharga 4,5 juta.",
        language="id",
        is_generated=False,
        status="ok",
        segments_count=1,
    )

    cfg = AppConfig(
        brands={"POCO": ["poco", "poco x"]},
        paths=PathsConfig(
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
            outputs_dir=str(tmp_path / "outputs"),
        ),
    )
    cfg.smartphone_filter.positive_keywords = ["hp", "review", "smartphone"]
    cfg.smartphone_filter.negative_keywords = ["laptop"]

    summary = run_stage1_ingestion(
        app_config=cfg,
        env_settings=EnvSettings(YOUTUBE_API_KEY="dummy"),
        resume=False,
        yt_client=mock_client,
        transcript_fetcher=mock_fetcher,
    )

    assert summary["total_2026_videos"] == 2
    assert summary["smartphone_videos_count"] == 1
    assert summary["non_smartphone_videos_count"] == 1
    assert summary["total_comments_collected"] == 1
    assert Path(summary["videos_parquet"]).exists()
    assert Path(summary["comments_parquet"]).exists()
