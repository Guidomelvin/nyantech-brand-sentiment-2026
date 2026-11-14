"""Unit & integration tests for Tahap 2 (Brand extraction, Price segments, Quarter grouping)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import AppConfig, EnvSettings, PathsConfig, PriceSegmentConfig
from src.extraction.brand_extractor import BrandExtractor
from src.extraction.pipeline import run_stage2_extraction
from src.extraction.price_extractor import PriceExtractor, map_price_to_segment
from src.extraction.quarter_grouper import assign_quarter


def test_brand_extractor_single_comparison_and_ambiguous_words() -> None:
    aliases = {
        "Samsung": ["samsung", "galaxy s", "galaxy a", "a57"],
        "Xiaomi": ["xiaomi", "redmi", "redmi note"],
        "Nothing": ["nothing", "nothing phone", "cmf phone"],
        "Apple": ["apple", "iphone"],
    }
    extractor = BrandExtractor(aliases)

    # Single brand
    res1 = extractor.extract(
        title="Review Samsung Galaxy A57 Indonesia",
        description="HP midrange terbaru dari Samsung.",
        transcript_text="There is nothing wrong with the camera quality.",
    )
    assert res1.primary_brand == "Samsung"
    assert res1.brands_mentioned == ["Samsung"]
    assert res1.is_comparison_video is False

    # Genuine Nothing Phone mention
    res2 = extractor.extract(
        title="Review Nothing Phone (3a) Indonesia - Desain Unik!",
        description="Unboxing Nothing Phone terbaru",
    )
    assert res2.primary_brand == "Nothing"
    assert res2.is_comparison_video is False

    # Comparison video
    res3 = extractor.extract(
        title="Samsung A57 vs Redmi Note 15 Pro - Mending Mana?",
        description="Komparasi HP 5 jutaan Samsung dan Xiaomi Redmi.",
    )
    assert res3.is_comparison_video is True
    assert set(res3.brands_mentioned) == {"Samsung", "Xiaomi"}


def test_price_extractor_indonesian_formats_and_segments() -> None:
    segments = [
        PriceSegmentConfig(label="1-3 juta", min_idr=0, max_idr=3_000_000),
        PriceSegmentConfig(label="3-6 juta", min_idr=3_000_001, max_idr=6_000_000),
        PriceSegmentConfig(label="6-10 juta", min_idr=6_000_001, max_idr=10_000_000),
        PriceSegmentConfig(label=">10 juta", min_idr=10_000_001, max_idr=None),
    ]
    extractor = PriceExtractor(segments=segments)

    r1 = extractor.extract(title="HP Infinix Hot 60 Harga Rp 2.199.000!")
    assert r1.price_segment == "1-3 juta"
    assert r1.estimated_price_idr == 2_199_000
    assert r1.price_source == "title"

    r2 = extractor.extract(
        title="Review Vivo V50 Indonesia",
        description="Baterai 6000 mAh, RAM 12GB, rilis tahun 2026. Harga resmi Rp 5.999.000.",
    )
    assert r2.price_segment == "3-6 juta"
    assert r2.estimated_price_idr == 5_999_000
    assert r2.price_source == "description"

    r3 = extractor.extract(
        title="Unboxing Xiaomi 16",
        description="Flagship compact",
        transcript_text="Di Indonesia HP ini dijual sekitar 8,5 juta rupiah.",
    )
    assert r3.price_segment == "6-10 juta"
    assert r3.price_source == "transcript"

    r4 = extractor.extract(title="iPhone 17 Pro Max Review - Harga 22 Jutaan!")
    assert r4.price_segment == ">10 juta"

    r5 = extractor.extract(title="Bahas Rumor Kamera HP Tahun Depan", description="Tanpa harga")
    assert r5.price_segment == "Unknown"

    assert map_price_to_segment(3_000_000, segments) == "1-3 juta"
    assert map_price_to_segment(3_000_001, segments) == "3-6 juta"
    assert map_price_to_segment(10_000_000, segments) == "6-10 juta"
    assert map_price_to_segment(10_000_001, segments) == ">10 juta"


def test_quarter_grouper_2026() -> None:
    assert assign_quarter("2026-01-15T10:00:00Z").quarter_label == "Q1 2026"
    assert assign_quarter("2026-04-01T00:00:00+07:00").quarter_label == "Q2 2026"
    assert assign_quarter("2026-08-20T12:00:00Z").quarter_label == "Q3 2026"
    assert assign_quarter("2026-12-31T16:00:00Z").quarter_label == "Q4 2026"


def test_stage2_pipeline_end_to_end(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "video_id": "v1",
                "title": "Review iQOO Z10 Indonesia - Raja HP 3 Jutaan?",
                "description": "Snapdragon baru, harga resmi Rp 3.499.000",
                "tags": ["iqoo", "vivo"],
                "published_at_utc": "2026-03-10T10:00:00Z",
                "transcript_text": "Performa iQOO Z10 kenceng banget di harga 3,5 juta.",
            },
            {
                "video_id": "v2",
                "title": "iPhone 17 Pro vs Samsung S26 Ultra - Raja Kamera 2026!",
                "description": "Flagship harga Rp 21.999.000 vs Rp 19.999.000",
                "tags": ["iphone", "samsung"],
                "published_at_utc": "2026-10-05T10:00:00Z",
                "transcript_text": "Kedua flagship diatas 10 juta ini bersaing ketat.",
            },
        ]
    ).to_parquet(raw_dir / "videos_2026.parquet", index=False)

    pd.DataFrame(
        [
            {"comment_id": "c1", "video_id": "v1", "text": "Mantap iQOO baterai badak!"},
            {"comment_id": "c2", "video_id": "v2", "text": "Samsung zoomnya juara"},
        ]
    ).to_parquet(raw_dir / "comments_2026.parquet", index=False)

    cfg = AppConfig(
        brands={
            "iQOO": ["iqoo"],
            "Apple": ["iphone", "apple"],
            "Samsung": ["samsung", "s26"],
        },
        price_segments=[
            PriceSegmentConfig(label="1-3 juta", min_idr=0, max_idr=3_000_000),
            PriceSegmentConfig(label="3-6 juta", min_idr=3_000_001, max_idr=6_000_000),
            PriceSegmentConfig(label="6-10 juta", min_idr=6_000_001, max_idr=10_000_000),
            PriceSegmentConfig(label=">10 juta", min_idr=10_000_001, max_idr=None),
        ],
        paths=PathsConfig(
            raw_dir=str(raw_dir),
            processed_dir=str(processed_dir),
            outputs_dir=str(tmp_path / "outputs"),
        ),
    )

    summary = run_stage2_extraction(
        app_config=cfg,
        env_settings=EnvSettings(),
        use_llm_for_ambiguous=False,
    )

    assert summary["videos_processed"] == 2
    assert summary["comments_enriched"] == 2
    assert summary["quarter_distribution"] == {"Q1 2026": 1, "Q4 2026": 1}

    videos_out = pd.read_parquet(summary["videos_enriched_parquet"])
    assert videos_out.loc[videos_out["video_id"] == "v1", "primary_brand"].iloc[0] == "iQOO"
    assert videos_out.loc[videos_out["video_id"] == "v1", "price_segment"].iloc[0] == "3-6 juta"
    assert videos_out.loc[videos_out["video_id"] == "v2", "price_segment"].iloc[0] == ">10 juta"
    assert bool(videos_out.loc[videos_out["video_id"] == "v2", "is_comparison_video"].iloc[0]) is True

    comments_out = pd.read_parquet(summary["comments_enriched_parquet"])
    assert "quarter_2026" in comments_out.columns
    assert "price_segment" in comments_out.columns
    assert "primary_brand" in comments_out.columns
