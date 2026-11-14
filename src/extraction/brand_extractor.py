"""Smartphone brand detection and normalization for Indonesian tech videos."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

# Alias yang merupakan kata umum bahasa Inggris sehingga butuh pencocokan case-sensitive
# atau konteks model smartphone yang jelas agar tidak memicu false positive di transkrip.
_AMBIGUOUS_LOWERCASE_ALIASES = {"nothing", "honor", "sharp"}


@dataclass
class BrandExtractionResult:
    brands_mentioned: list[str] = field(default_factory=list)
    primary_brand: str = "Unknown"
    is_comparison_video: bool = False
    brand_scores: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _count_alias_hits(text: str, alias: str) -> int:
    """Count occurrences of a brand/model alias in text with boundary safety."""
    alias_clean = alias.strip()
    if not alias_clean:
        return 0

    is_ambiguous = alias_clean.lower() in _AMBIGUOUS_LOWERCASE_ALIASES
    flags = 0 if is_ambiguous else re.IGNORECASE
    target_alias = alias_clean.capitalize() if is_ambiguous else alias_clean

    if " " in target_alias:
        pattern = rf"(?<![A-Za-z0-9]){re.escape(target_alias)}(?![A-Za-z0-9])"
    else:
        pattern = rf"(?<![A-Za-z0-9]){re.escape(target_alias)}(?![A-Za-z0-9])"

    return len(re.findall(pattern, text, flags=flags))


def _first_match_position(text: str, aliases: list[str]) -> int:
    """Return earliest character index of any alias in text, or large sentinel."""
    earliest = 10**9
    for alias in aliases:
        is_ambiguous = alias.strip().lower() in _AMBIGUOUS_LOWERCASE_ALIASES
        flags = 0 if is_ambiguous else re.IGNORECASE
        target = alias.strip().capitalize() if is_ambiguous else alias.strip()
        pattern = rf"(?<![A-Za-z0-9]){re.escape(target)}(?![A-Za-z0-9])"
        match = re.search(pattern, text, flags=flags)
        if match and match.start() < earliest:
            earliest = match.start()
    return earliest


class BrandExtractor:
    """Extract and rank smartphone brands from video title, tags, description, and transcript."""

    def __init__(self, brand_aliases: dict[str, list[str]]) -> None:
        self.brand_aliases = brand_aliases

    def extract(
        self,
        title: str,
        description: str = "",
        tags: list[str] | None = None,
        transcript_text: str = "",
    ) -> BrandExtractionResult:
        tags_text = " ".join(tags or [])
        scores: dict[str, int] = {}

        for brand, aliases in self.brand_aliases.items():
            title_hits = sum(_count_alias_hits(title, a) for a in aliases)
            tag_hits = sum(_count_alias_hits(tags_text, a) for a in aliases)
            desc_hits = sum(_count_alias_hits(description, a) for a in aliases)
            # Cap transcript hits per alias so repetitive narration doesn't drown out title/desc
            transcript_hits = min(
                sum(_count_alias_hits(transcript_text, a) for a in aliases),
                12,
            )

            total_score = (
                (min(title_hits, 3) * 10)
                + (min(tag_hits, 3) * 4)
                + (min(desc_hits, 4) * 3)
                + transcript_hits
            )

            if total_score > 0:
                scores[brand] = total_score

        if not scores:
            return BrandExtractionResult(
                brands_mentioned=[],
                primary_brand="Unknown",
                is_comparison_video=False,
                brand_scores={},
            )

        # Abaikan brand yang hanya muncul 1x di transkrip tanpa dukungan title/desc/tags
        # bila sudah ada brand kuat di title/description.
        has_strong_brand = any(score >= 6 for score in scores.values())
        filtered_scores = {
            brand: score
            for brand, score in scores.items()
            if (score >= 3 or not has_strong_brand)
        }
        if not filtered_scores:
            filtered_scores = scores

        # Urutkan berdasarkan skor tertinggi; tie-breaker: kemunculan paling awal di judul
        ranked = sorted(
            filtered_scores.items(),
            key=lambda kv: (
                -kv[1],
                _first_match_position(title, self.brand_aliases.get(kv[0], [kv[0]])),
                kv[0],
            ),
        )

        brands_mentioned = [brand for brand, _ in ranked]
        primary_brand = brands_mentioned[0]

        # Deteksi apakah video merupakan komparasi / roundup beberapa brand
        title_brands = [
            b
            for b in brands_mentioned
            if sum(_count_alias_hits(title, a) for a in self.brand_aliases.get(b, [b])) > 0
        ]
        is_comparison = len(title_brands) >= 2 or (
            len(brands_mentioned) >= 2
            and any(
                kw in title.lower()
                for kw in ("vs", "komparasi", "banding", "rekomendasi", "terbaik", "pilihan", "perbandingan")
            )
        )

        return BrandExtractionResult(
            brands_mentioned=brands_mentioned,
            primary_brand=primary_brand,
            is_comparison_video=is_comparison,
            brand_scores=dict(ranked),
        )
