"""Tahap 2 Pipeline Orchestrator: Extract brands, price ranges, and 2026 quarters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import AppConfig, EnvSettings, load_config
from src.extraction.brand_extractor import BrandExtractor
from src.extraction.llm_refiner import LLMMetadataRefiner
from src.extraction.price_extractor import PriceExtractor
from src.extraction.quarter_grouper import assign_quarter
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _load_table(parquet_path: Path, json_path: Path) -> pd.DataFrame:
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if json_path.exists():
        return pd.read_json(json_path)
    raise FileNotFoundError(
        f"Dataset tidak ditemukan di {parquet_path} maupun {json_path}. "
        "Jalankan Tahap 1 terlebih dahulu."
    )


def _to_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if x is not None]
    return []


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def run_stage2_extraction(
    app_config: AppConfig | None = None,
    env_settings: EnvSettings | None = None,
    use_llm_for_ambiguous: bool = True,
) -> dict[str, Any]:
    """Execute Stage 2 enrichment on scraped 2026 videos and comments."""
    if app_config is None or env_settings is None:
        loaded_cfg, loaded_env = load_config()
        app_config = app_config or loaded_cfg
        env_settings = env_settings or loaded_env

    raw_dir = app_config.paths.resolve_raw_dir()
    processed_dir = app_config.paths.resolve_processed_dir()

    videos_df = _load_table(raw_dir / "videos_2026.parquet", raw_dir / "videos_2026.json")
    comments_df = _load_table(raw_dir / "comments_2026.parquet", raw_dir / "comments_2026.json")

    brand_extractor = BrandExtractor(brand_aliases=app_config.brands)
    price_extractor = PriceExtractor(segments=app_config.price_segments)
    llm_refiner = LLMMetadataRefiner(
        env_settings=env_settings,
        valid_brands=list(app_config.brands.keys()),
    )

    enriched_rows: list[dict[str, Any]] = []
    llm_refined_count = 0

    for row in videos_df.to_dict(orient="records"):
        title = str(row.get("title", "") or "")
        description = str(row.get("description", "") or "")
        tags = _to_str_list(row.get("tags"))
        transcript = str(row.get("transcript_text", "") or "")
        published_at = str(row.get("published_at_utc") or row.get("published_at_local") or "")

        brand_res = brand_extractor.extract(
            title=title,
            description=description,
            tags=tags,
            transcript_text=transcript,
        )
        price_res = price_extractor.extract(
            title=title,
            description=description,
            transcript_text=transcript,
        )
        quarter_info = assign_quarter(
            published_at=published_at,
            timezone_name=app_config.channel.timezone,
            use_local_timezone=app_config.date_filter.use_local_timezone,
        )

        # Optional LLM fallback untuk kasus brand atau harga belum terdeteksi secara deterministik
        extraction_method = "nlp_rules"
        if (
            use_llm_for_ambiguous
            and llm_refiner.enabled
            and (brand_res.primary_brand == "Unknown" or price_res.price_segment == "Unknown")
        ):
            refined = llm_refiner.refine(
                title=title,
                description=description,
                transcript_text=transcript,
            )
            if refined:
                extraction_method = "hybrid_nlp_llm"
                llm_refined_count += 1
                if brand_res.primary_brand == "Unknown" and refined["primary_brand"] != "Unknown":
                    brand_res.brands_mentioned = refined["brands_mentioned"]
                    brand_res.primary_brand = refined["primary_brand"]
                    brand_res.is_comparison_video = refined["is_comparison_video"]
                if price_res.price_segment == "Unknown" and refined["price_segment"] != "Unknown":
                    price_res.price_segment = refined["price_segment"]
                    price_res.estimated_price_idr = refined["estimated_price_idr"]
                    price_res.price_source = "llm"

        enriched_row = dict(row)
        enriched_row["tags"] = tags
        enriched_row.update(
            {
                "primary_brand": brand_res.primary_brand,
                "brands_mentioned": brand_res.brands_mentioned,
                "brands_count": len(brand_res.brands_mentioned),
                "is_comparison_video": brand_res.is_comparison_video,
                "brand_scores_json": json.dumps(brand_res.brand_scores, ensure_ascii=False),
                "price_segment": price_res.price_segment,
                "estimated_price_idr": price_res.estimated_price_idr,
                "price_source": price_res.price_source,
                "upload_year": quarter_info.upload_year,
                "upload_month": quarter_info.upload_month,
                "quarter_num": quarter_info.quarter_num,
                "quarter_2026": quarter_info.quarter_label,
                "upload_date_local": quarter_info.upload_date_local,
                "extraction_method": extraction_method,
            }
        )
        enriched_rows.append(enriched_row)

    enriched_videos_df = pd.DataFrame(enriched_rows)

    # Exploded table (1 baris per video-brand) untuk analisis komparasi multi-brand di Tahap 3 & 4
    exploded_df = enriched_videos_df.copy()
    if not exploded_df.empty:
        exploded_df["brand"] = exploded_df["brands_mentioned"].apply(
            lambda b_list: b_list if isinstance(b_list, list) and b_list else ["Unknown"]
        )
        exploded_df = exploded_df.explode("brand", ignore_index=True)

    # Join metadata video (brand, price segment, quarter) ke tabel komentar siap-pakai untuk Tahap 3
    join_cols = [
        "video_id",
        "title",
        "primary_brand",
        "brands_mentioned",
        "is_comparison_video",
        "price_segment",
        "estimated_price_idr",
        "quarter_2026",
        "quarter_num",
        "upload_month",
        "upload_date_local",
    ]
    if not comments_df.empty and not enriched_videos_df.empty:
        comments_enriched_df = comments_df.merge(
            enriched_videos_df[join_cols].rename(columns={"title": "video_title"}),
            on="video_id",
            how="left",
        )
    else:
        comments_enriched_df = comments_df.copy()

    # Simpan outputs Tahap 2
    videos_out_parquet = processed_dir / "videos_enriched_2026.parquet"
    videos_out_csv = processed_dir / "videos_enriched_2026.csv"
    videos_out_json = processed_dir / "videos_enriched_2026.json"
    exploded_out_parquet = processed_dir / "videos_brand_exploded_2026.parquet"
    comments_out_parquet = processed_dir / "comments_enriched_2026.parquet"

    enriched_videos_df.to_parquet(videos_out_parquet, index=False)
    enriched_videos_df.to_csv(videos_out_csv, index=False)
    with videos_out_json.open("w", encoding="utf-8") as fh:
        json.dump(enriched_rows, fh, ensure_ascii=False, indent=2, default=_json_default)

    exploded_df.to_parquet(exploded_out_parquet, index=False)
    comments_enriched_df.to_parquet(comments_out_parquet, index=False)

    brand_dist = (
        enriched_videos_df["primary_brand"].value_counts().to_dict()
        if not enriched_videos_df.empty
        else {}
    )
    price_dist = (
        enriched_videos_df["price_segment"].value_counts().to_dict()
        if not enriched_videos_df.empty
        else {}
    )
    quarter_dist = (
        enriched_videos_df["quarter_2026"].value_counts().sort_index().to_dict()
        if not enriched_videos_df.empty
        else {}
    )

    summary = {
        "videos_processed": len(enriched_videos_df),
        "comments_enriched": len(comments_enriched_df),
        "llm_refined_count": llm_refined_count,
        "brand_distribution": brand_dist,
        "price_segment_distribution": price_dist,
        "quarter_distribution": quarter_dist,
        "videos_enriched_parquet": str(videos_out_parquet),
        "comments_enriched_parquet": str(comments_out_parquet),
    }
    logger.info("Tahap 2 selesai: %s", summary)
    return summary
