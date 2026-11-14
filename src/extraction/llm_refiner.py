"""Optional LLM-assisted entity and price refinement for ambiguous videos."""

from __future__ import annotations

import json
from typing import Any

from src.config import EnvSettings
from src.extraction.price_extractor import map_price_to_segment
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """Kamu adalah asisten NLP ahli pasar smartphone Indonesia.
Tugasmu mengekstrak metadata dari judul, deskripsi, dan cuplikan transkrip video YouTube tech (NYANTECH).
Kembalikan HANYA JSON valid dengan format persis berikut:
{
  "brands_mentioned": ["Samsung", ...],
  "primary_brand": "Samsung",
  "is_comparison_video": false,
  "estimated_price_idr": 5499000,
  "price_segment": "3-6 juta"
}
Aturan:
- Pilihan `price_segment` HARUS salah satu dari: "1-3 juta", "3-6 juta", "6-10 juta", ">10 juta", "Unknown".
- Jika video tidak menyebut harga, isi `estimated_price_idr`: null dan `price_segment`: "Unknown".
- Nama brand harus dinormalisasi rapi (misal: Samsung, Apple, Xiaomi, POCO, OPPO, Vivo, iQOO, Realme, Nothing, Infinix, Tecno, Itel, ASUS, Google, Huawei, Honor, Motorola, ZTE, Sony, Sharp).
"""


class LLMMetadataRefiner:
    """OpenAI-compatible LLM refiner for ambiguous brand/price cases."""

    def __init__(self, env_settings: EnvSettings, valid_brands: list[str]) -> None:
        self.env = env_settings
        self.valid_brands = valid_brands
        self.enabled = bool(
            env_settings.llm_api_key
            and env_settings.llm_provider.lower() not in {"", "none", "off", "false"}
        )
        self._client: Any | None = None

        if self.enabled:
            from openai import OpenAI

            kwargs: dict[str, Any] = {"api_key": env_settings.llm_api_key}
            if env_settings.llm_base_url:
                kwargs["base_url"] = env_settings.llm_base_url
            self._client = OpenAI(**kwargs)

    def refine(
        self,
        title: str,
        description: str = "",
        transcript_text: str = "",
    ) -> dict[str, Any] | None:
        """Query LLM to refine brand & price extraction; returns None if disabled or on failure."""
        if not self.enabled or self._client is None:
            return None

        user_payload = (
            f"Judul: {title}\n\n"
            f"Deskripsi:\n{description[:1500]}\n\n"
            f"Cuplikan Transkrip:\n{transcript_text[:2500]}"
        )

        try:
            response = self._client.chat.completions.create(
                model=self.env.llm_model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_payload},
                ],
            )
            raw_content = response.choices[0].message.content or "{}"
            parsed = json.loads(raw_content)

            est_price = parsed.get("estimated_price_idr")
            if isinstance(est_price, (int, float)) and est_price > 0:
                est_price = int(est_price)
                segment = map_price_to_segment(est_price)
            else:
                est_price = None
                segment = parsed.get("price_segment", "Unknown")
                if segment not in {"1-3 juta", "3-6 juta", "6-10 juta", ">10 juta", "Unknown"}:
                    segment = "Unknown"

            brands = [str(b).strip() for b in parsed.get("brands_mentioned", []) if str(b).strip()]
            primary = str(parsed.get("primary_brand") or (brands[0] if brands else "Unknown")).strip()
            if primary != "Unknown" and primary not in brands:
                brands.insert(0, primary)

            return {
                "brands_mentioned": brands,
                "primary_brand": primary,
                "is_comparison_video": bool(parsed.get("is_comparison_video", len(brands) >= 2)),
                "estimated_price_idr": est_price,
                "price_segment": segment,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM refinement gagal untuk judul '%s': %s", title, exc)
            return None
