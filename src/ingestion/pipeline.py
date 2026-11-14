"""Tahap 1 Pipeline Orchestrator: Scrape 2026 NYANTECH smartphone videos, transcripts, and comments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.config import AppConfig, EnvSettings, load_config
from src.ingestion.smartphone_filter import SmartphoneVideoFilter
from src.ingestion.transcript_fetcher import TranscriptFetcher
from src.ingestion.youtube_client import YouTubeAPIClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _save_records(records: list[dict[str, Any]], base_path_no_ext: Path) -> tuple[Path, Path]:
    """Save list of dict records to both Parquet and JSON (UTF-8)."""
    base_path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    parquet_path = base_path_no_ext.with_suffix(".parquet")
    json_path = base_path_no_ext.with_suffix(".json")

    df.to_parquet(parquet_path, index=False)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)

    return parquet_path, json_path


def run_stage1_ingestion(
    app_config: AppConfig | None = None,
    env_settings: EnvSettings | None = None,
    max_videos: int | None = None,
    resume: bool = True,
    yt_client: YouTubeAPIClient | None = None,
    transcript_fetcher: TranscriptFetcher | None = None,
) -> dict[str, Any]:
    """Execute Stage 1 end-to-end and persist raw datasets under data/raw/."""
    if app_config is None or env_settings is None:
        loaded_cfg, loaded_env = load_config()
        app_config = app_config or loaded_cfg
        env_settings = env_settings or loaded_env

    raw_dir = app_config.paths.resolve_raw_dir()
    client = yt_client or YouTubeAPIClient(
        api_key=env_settings.youtube_api_key,
        request_delay_seconds=app_config.scraping.request_delay_seconds,
    )
    fetcher = transcript_fetcher or TranscriptFetcher(
        preferred_languages=app_config.scraping.transcript_languages,
    )
    phone_filter = SmartphoneVideoFilter(
        filter_config=app_config.smartphone_filter,
        brand_aliases=app_config.brands,
    )

    # 1. Resolve Channel NYANTECH
    channel_info = client.resolve_channel(
        handle=app_config.channel.handle,
        channel_id=app_config.channel.channel_id,
    )
    logger.info(
        "Channel terdeteksi: %s (ID: %s | Subscribers: %d)",
        channel_info.title,
        channel_info.channel_id,
        channel_info.subscriber_count,
    )

    # 2. Ambil semua video rilis sepanjang 2026
    all_2026_videos = client.fetch_videos_in_date_range(
        channel_info=channel_info,
        start_date=app_config.date_filter.start_date,
        end_date=app_config.date_filter.end_date,
        timezone_name=app_config.channel.timezone,
        use_local_timezone=app_config.date_filter.use_local_timezone,
    )

    # 3. Filter khusus video yang membahas smartphone
    smartphone_videos: list[dict[str, Any]] = []
    rejected_videos: list[dict[str, Any]] = []

    for video in all_2026_videos:
        decision = phone_filter.evaluate(
            title=video.title,
            description=video.description,
            tags=video.tags,
            duration_seconds=video.duration_seconds,
        )
        row = video.to_dict()
        row["filter_score"] = decision.score
        row["filter_reasons"] = decision.reasons
        if decision.is_smartphone:
            smartphone_videos.append(row)
        else:
            rejected_videos.append(row)

    logger.info(
        "Filter smartphone: %d lolos, %d non-smartphone (total 2026: %d).",
        len(smartphone_videos),
        len(rejected_videos),
        len(all_2026_videos),
    )

    if rejected_videos:
        _save_records(rejected_videos, raw_dir / "videos_non_smartphone_2026")

    if max_videos is not None:
        smartphone_videos = smartphone_videos[:max_videos]

    # Checkpoint / Resume support agar kuota API aman jika proses sempat berhenti
    checkpoint_videos_path = raw_dir / "videos_2026.json"
    checkpoint_comments_path = raw_dir / "comments_2026.json"
    processed_video_map: dict[str, dict[str, Any]] = {}
    all_comments: list[dict[str, Any]] = []

    if resume and checkpoint_videos_path.exists() and checkpoint_comments_path.exists():
        with checkpoint_videos_path.open("r", encoding="utf-8") as fh:
            for existing_v in json.load(fh):
                processed_video_map[existing_v["video_id"]] = existing_v
        with checkpoint_comments_path.open("r", encoding="utf-8") as fh:
            all_comments = json.load(fh)
        logger.info(
            "Melanjutkan checkpoint: %d video & %d komentar sudah tersimpan.",
            len(processed_video_map),
            len(all_comments),
        )

    # 4. Ambil transkrip & seluruh komentar untuk tiap video smartphone 2026
    final_videos: list[dict[str, Any]] = []
    seen_comment_ids = {c["comment_id"] for c in all_comments}
    checkpoint_every = max(1, app_config.scraping.checkpoint_every_n_videos)

    for idx, video_row in enumerate(tqdm(smartphone_videos, desc="Tahap 1: Scraping Videos"), start=1):
        vid = video_row["video_id"]
        if vid in processed_video_map:
            final_videos.append(processed_video_map[vid])
            continue

        transcript_res = fetcher.fetch(vid)
        comments = client.fetch_all_comments(
            video_id=vid,
            include_replies=app_config.scraping.include_replies,
            max_comments=app_config.scraping.max_comments_per_video,
        )

        video_row["transcript_text"] = transcript_res.text
        video_row["transcript_language"] = transcript_res.language
        video_row["transcript_is_generated"] = transcript_res.is_generated
        video_row["transcript_status"] = transcript_res.status
        video_row["transcript_segments_count"] = transcript_res.segments_count
        video_row["scraped_comment_count"] = len(comments)
        final_videos.append(video_row)

        for comment in comments:
            c_dict = comment.to_dict()
            if c_dict["comment_id"] not in seen_comment_ids:
                seen_comment_ids.add(c_dict["comment_id"])
                all_comments.append(c_dict)

        if idx % checkpoint_every == 0:
            _save_records(final_videos, raw_dir / "videos_2026")
            _save_records(all_comments, raw_dir / "comments_2026")

    videos_parquet, videos_json = _save_records(final_videos, raw_dir / "videos_2026")
    comments_parquet, comments_json = _save_records(all_comments, raw_dir / "comments_2026")

    summary = {
        "channel_id": channel_info.channel_id,
        "channel_title": channel_info.title,
        "total_2026_videos": len(all_2026_videos),
        "smartphone_videos_count": len(final_videos),
        "non_smartphone_videos_count": len(rejected_videos),
        "total_comments_collected": len(all_comments),
        "videos_parquet": str(videos_parquet),
        "videos_json": str(videos_json),
        "comments_parquet": str(comments_parquet),
        "comments_json": str(comments_json),
    }
    logger.info("Tahap 1 selesai: %s", summary)
    return summary
