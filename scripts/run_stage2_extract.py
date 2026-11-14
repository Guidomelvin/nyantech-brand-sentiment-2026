"""CLI runner untuk Tahap 2: Entity & Metadata Extraction (Brand, Harga, Kuartal 2026)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.extraction.pipeline import run_stage2_extraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tahap 2: Ekstraksi brand smartphone, segmen harga, dan kuartal 2026."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path opsional ke file YAML config (default: config/settings.yaml).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Nonaktifkan LLM refiner meskipun LLM_API_KEY tersedia di `.env`.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_config, env_settings = load_config(args.config)
    summary = run_stage2_extraction(
        app_config=app_config,
        env_settings=env_settings,
        use_llm_for_ambiguous=not args.no_llm,
    )
    print("\n=== RINGKASAN TAHAP 2 ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
