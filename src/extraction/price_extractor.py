"""Indonesian smartphone price parser and price-segment classifier."""

from __future__ import annotations

import re
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

from src.config import PriceSegmentConfig

# Minimum & maksimum harga smartphone wajar dalam IDR (untuk menyingkirkan aksesoris/typo)
_MIN_PHONE_PRICE_IDR = 750_000
_MAX_PHONE_PRICE_IDR = 60_000_000

# Pola range jutaan: e.g. "1-3 juta", "3 - 6 jutaan", "2 sampai 4 jt"
_JUTA_RANGE_RE = re.compile(
    r"(?:rp\.?\s*)?(\d{1,2}(?:[.,]\d{1,2})?)\s*(?:-|s\.?d\.?|sampai|hingga|to)\s*"
    r"(\d{1,2}(?:[.,]\d{1,2})?)\s*(?:juta(?:an)?|jt(?:an)?)\b",
    flags=re.IGNORECASE,
)

# Pola angka tunggal jutaan: e.g. "Rp 4,5 juta", "5 jutaan", "12.9 jt", "16jt"
_JUTA_SINGLE_RE = re.compile(
    r"(?:(?:rp\.?|idr|harga|mulai|sekitar|kisaran|dibanderol|di\s*bawah|under|di\s*atas|budget|kelas)\s*)?"
    r"(\d{1,2}(?:[.,]\d{1,2})?)\s*(?:juta(?:an)?|jt(?:an)?)\b",
    flags=re.IGNORECASE,
)

# Pola nominal penuh IDR: e.g. "Rp 2.999.000", "Rp14.499.000", "IDR 5,499,000"
_FULL_IDR_RE = re.compile(
    r"(?:rp\.?|idr)\s*(\d{1,2}(?:[.,]\d{3}){2})\b",
    flags=re.IGNORECASE,
)

# Pola jutaan tanpa 'Rp' jika formatnya ribuan lengkap & ada kata harga/dibanderol
_CONTEXT_FULL_IDR_RE = re.compile(
    r"(?:harga(?:nya)?|dibanderol|dijual|resmi|preorder|po)\s*(?:di|mulai|sekitar)?\s*"
    r"(?:rp\.?\s*)?(\d{1,2}[.,]\d{3}[.,]\d{3})\b",
    flags=re.IGNORECASE,
)


@dataclass
class PriceExtractionResult:
    price_segment: str = "Unknown"
    estimated_price_idr: int | None = None
    extracted_prices_idr: list[int] = field(default_factory=list)
    price_source: str = "none"  # "title" | "description" | "transcript" | "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_decimal_juta(raw_num: str) -> int:
    normalized = raw_num.replace(",", ".")
    return int(round(float(normalized) * 1_000_000))


def _parse_full_idr(raw_num: str) -> int:
    digits = re.sub(r"[.,]", "", raw_num)
    return int(digits)


def extract_idr_candidates(text: str) -> list[int]:
    """Extract plausible smartphone prices (in IDR) from Indonesian text."""
    if not text:
        return []

    candidates: list[int] = []

    # 1. Range jutaan (ambil midpoint untuk estimasi, sekaligus simpan kedua ujung jika valid)
    for match in _JUTA_RANGE_RE.finditer(text):
        low = _parse_decimal_juta(match.group(1))
        high = _parse_decimal_juta(match.group(2))
        if low <= high and _MIN_PHONE_PRICE_IDR <= high <= _MAX_PHONE_PRICE_IDR:
            mid = (low + high) // 2
            if _MIN_PHONE_PRICE_IDR <= mid <= _MAX_PHONE_PRICE_IDR:
                candidates.append(mid)

    # 2. Nominal penuh Rp X.XXX.XXX
    for regex in (_FULL_IDR_RE, _CONTEXT_FULL_IDR_RE):
        for match in regex.finditer(text):
            val = _parse_full_idr(match.group(1))
            if _MIN_PHONE_PRICE_IDR <= val <= _MAX_PHONE_PRICE_IDR:
                candidates.append(val)

    # 3. Nominal tunggal jutaan (hindari mengulang teks yang sudah tertangkap range jutaan)
    text_without_ranges = _JUTA_RANGE_RE.sub(" ", text)
    for match in _JUTA_SINGLE_RE.finditer(text_without_ranges):
        val = _parse_decimal_juta(match.group(1))
        # Jika muncul bentuk "5 jutaan", tambahkan sedikit offset (+300rb) agar tidak bias ke batas bawah?
        # Lebih aman pakai nilai literal kecuali tepat di batas segmen (misal "3 jutaan" biasanya 3-3.9 jt -> 3.3 jt)
        snippet = match.group(0).lower()
        if "jutaan" in snippet or "jtan" in snippet:
            val += 300_000
        elif "di bawah" in snippet or "dibawah" in snippet or "under" in snippet:
            val = max(_MIN_PHONE_PRICE_IDR, val - 100_000)
        elif "di atas" in snippet:
            val += 500_000

        if _MIN_PHONE_PRICE_IDR <= val <= _MAX_PHONE_PRICE_IDR:
            candidates.append(val)

    return candidates


def map_price_to_segment(
    price_idr: int | None,
    segments: list[PriceSegmentConfig] | None = None,
) -> str:
    """Map an IDR amount into one of the configured segments."""
    if price_idr is None:
        return "Unknown"

    if segments:
        for seg in segments:
            if seg.max_idr is None and price_idr >= seg.min_idr:
                return seg.label
            if seg.max_idr is not None and seg.min_idr <= price_idr <= seg.max_idr:
                return seg.label

    # Fallback default sesuai spesifikasi project
    if price_idr <= 3_000_000:
        return "1-3 juta"
    if price_idr <= 6_000_000:
        return "3-6 juta"
    if price_idr <= 10_000_000:
        return "6-10 juta"
    return ">10 juta"


class PriceExtractor:
    """Hierarchical price segment extractor (Title -> Description -> Transcript)."""

    def __init__(self, segments: list[PriceSegmentConfig] | None = None) -> None:
        self.segments = segments or []

    def extract(
        self,
        title: str,
        description: str = "",
        transcript_text: str = "",
    ) -> PriceExtractionResult:
        for source_name, text in (
            ("title", title),
            ("description", description),
            ("transcript", transcript_text),
        ):
            candidates = extract_idr_candidates(text)
            if candidates:
                # Untuk transkrip, ambil median agar outlier (misal sebut harga kompetitor) lebih stabil;
                # untuk title, ambil median (umumnya 1 harga utama).
                estimated = int(round(statistics.median(candidates)))
                segment = map_price_to_segment(estimated, self.segments)
                return PriceExtractionResult(
                    price_segment=segment,
                    estimated_price_idr=estimated,
                    extracted_prices_idr=candidates[:10],
                    price_source=source_name,
                )

        return PriceExtractionResult(
            price_segment="Unknown",
            estimated_price_idr=None,
            extracted_prices_idr=[],
            price_source="none",
        )
