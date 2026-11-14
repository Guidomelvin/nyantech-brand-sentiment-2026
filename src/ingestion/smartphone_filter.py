"""Smartphone video classifier/filter for NYANTECH YouTube videos."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.config import SmartphoneFilterConfig

# Kata kunci spesifik non-smartphone pada judul yang kuat menandakan produk bukan HP
_STRONG_NON_PHONE_TITLE_TERMS = (
    "laptop",
    "notebook",
    "ultrabook",
    "macbook",
    "ipad",
    "tablet",
    "galaxy tab",
    "xiaomi pad",
    "redmi pad",
    "huawei matepad",
    "smartwatch",
    "galaxy watch",
    "apple watch",
    "huawei watch",
    "tws",
    "earbuds",
    "earphone",
    "headphone",
    "airpods",
    "galaxy buds",
    "nothing ear",
    "rakit pc",
    "pc gaming",
    "monitor",
    "keyboard",
    "mouse",
    "powerbank",
    "gan charger",
    "mirrorless",
    "playstation",
    "ps5",
    "nintendo switch",
    "rog ally",
    "steam deck",
)

# Sinyal kuat khusus smartphone di judul (mengalahkan kata ambigu)
_STRONG_PHONE_TITLE_TERMS = (
    "smartphone",
    "hp ",
    " hp",
    "hape",
    "handphone",
    "ponsel",
    "iphone",
    "galaxy s",
    "galaxy a",
    "galaxy z",
    "galaxy m",
    "redmi note",
    "poco ",
    "rog phone",
    "zenfone",
    "pixel ",
    "nothing phone",
    "cmf phone",
    "infinix",
    "tecno",
    "itel",
    "iqoo",
    "realme",
    "reno",
    "find x",
    "find n",
    "vivo x",
    "vivo v",
    "xperia",
    "flagship",
    "midrange",
)


@dataclass
class FilterDecision:
    is_smartphone: bool
    score: int
    reasons: list[str] = field(default_factory=list)


def _contains_term(text_lower: str, term: str) -> bool:
    term_clean = term.strip().lower()
    if not term_clean:
        return False
    # Gunakan word boundary untuk token pendek/alfanumerik agar tidak false match substring
    if " " in term_clean:
        return term_clean in text_lower
    pattern = rf"(?<![a-z0-9]){re.escape(term_clean)}(?![a-z0-9])"
    return bool(re.search(pattern, text_lower))


class SmartphoneVideoFilter:
    """Rule-based multi-signal filter to keep only smartphone-related videos."""

    def __init__(
        self,
        filter_config: SmartphoneFilterConfig,
        brand_aliases: dict[str, list[str]],
    ) -> None:
        self.config = filter_config
        self.brand_aliases = brand_aliases

    def evaluate(
        self,
        title: str,
        description: str = "",
        tags: list[str] | None = None,
        duration_seconds: int = 0,
    ) -> FilterDecision:
        """Score a video on whether its topic is smartphone reviews/discussions."""
        if self.config.exclude_shorts and 0 < duration_seconds <= 60:
            return FilterDecision(
                is_smartphone=False,
                score=-10,
                reasons=["excluded_youtube_short"],
            )

        title_lower = f" {title.lower()} "
        desc_lower = f" {description.lower()} "
        tags_lower = f" {' '.join(tags or []).lower()} "

        score = 0
        reasons: list[str] = []

        # 1. Cek apakah judul secara eksplisit membahas produk non-HP (laptop/tablet/TWS/watch/PC)
        has_strong_non_phone_title = any(
            term in title_lower for term in _STRONG_NON_PHONE_TITLE_TERMS
        )
        has_strong_phone_title = any(
            term in title_lower for term in _STRONG_PHONE_TITLE_TERMS
        )

        if has_strong_non_phone_title and not has_strong_phone_title:
            return FilterDecision(
                is_smartphone=False,
                score=-5,
                reasons=["non_smartphone_product_in_title"],
            )

        # 2. Brand & model signals
        matched_brands: set[str] = set()
        for brand, aliases in self.brand_aliases.items():
            for alias in aliases:
                if _contains_term(title_lower, alias):
                    if brand not in matched_brands:
                        score += 3
                        matched_brands.add(brand)
                        reasons.append(f"title_brand:{brand}")
                elif _contains_term(tags_lower, alias):
                    if brand not in matched_brands:
                        score += 2
                        matched_brands.add(brand)
                        reasons.append(f"tag_brand:{brand}")
                elif _contains_term(desc_lower, alias):
                    if brand not in matched_brands:
                        score += 1
                        matched_brands.add(brand)
                        reasons.append(f"desc_brand:{brand}")

        # 3. Positive smartphone domain keywords
        for kw in self.config.positive_keywords:
            if _contains_term(title_lower, kw):
                score += 2
                reasons.append(f"title_kw:{kw}")
            elif _contains_term(tags_lower, kw) or _contains_term(desc_lower, kw):
                score += 1
                reasons.append(f"meta_kw:{kw}")

        # 4. Negative hardware keywords (selain yang sudah lolos strong phone title)
        for neg_kw in self.config.negative_keywords:
            if _contains_term(title_lower, neg_kw):
                score -= 3
                reasons.append(f"title_neg:{neg_kw}")

        is_smartphone = score >= self.config.min_confidence_score
        return FilterDecision(
            is_smartphone=is_smartphone,
            score=score,
            reasons=reasons[:12],
        )
