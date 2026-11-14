"""CLI runner untuk Tahap 1: Data Scraping & Collection (Filter 2026)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Pastikan root project masuk sys.path saat dijalankan langsung via `python scripts/run_stage1_scrape.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.ingestion.pipeline import run_stage1_ingestion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tahap 1: Scrape video smartphone NYANTECH tahun 2026 beserta transkrip dan komentarnya."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path opsional ke file YAML config (default: config/settings.yaml).",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Batasi jumlah video untuk uji coba cepat / hemat kuota API.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Mulai scrape ulang dari awal tanpa memakai checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_config, env_settings = load_config(args.config)
    summary = run_stage1_ingestion(
        app_config=app_config,
        env_settings=env_settings,
        max_videos=args.max_videos,
        resume=not args.no_resume,
    )
    print("\n=== RINGKASAN TAHAP 1 ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
