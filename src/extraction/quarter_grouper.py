"""Quarter grouping utilities for 2026 YouTube video release dates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo


@dataclass
class QuarterInfo:
    upload_year: int
    upload_month: int
    quarter_num: int
    quarter_label: str  # e.g. "Q1 2026"
    upload_date_local: str  # YYYY-MM-DD

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assign_quarter(
    published_at: str,
    timezone_name: str = "Asia/Jakarta",
    use_local_timezone: bool = True,
) -> QuarterInfo:
    """Group an ISO-8601 timestamp into Q1..Q4 <YEAR> (e.g. 'Q1 2026')."""
    dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    if use_local_timezone:
        dt = dt.astimezone(ZoneInfo(timezone_name))

    quarter_num = (dt.month - 1) // 3 + 1
    return QuarterInfo(
        upload_year=dt.year,
        upload_month=dt.month,
        quarter_num=quarter_num,
        quarter_label=f"Q{quarter_num} {dt.year}",
        upload_date_local=dt.date().isoformat(),
    )
