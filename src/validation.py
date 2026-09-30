"""Field-level parsing/validation helpers and loader reports.

All ``parse_*`` functions are pure: they return ``None`` when a value is missing or
cannot be interpreted, and the loader decides whether that is an anomaly or an
invalid record.
"""

from __future__ import annotations

import ipaddress
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

# Zeek ASCII-log placeholders that may leak into JSON exports.
MISSING_STRINGS = frozenset({"", "-", "(empty)"})
UNKNOWN_SRC = "__unknown_src__"


def clean_str(value: Any) -> Optional[str]:
    """Return a stripped string, or ``None`` for missing/unsupported values."""
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return None if s in MISSING_STRINGS else s
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value) if math.isfinite(value) else None
    return None


def parse_timestamp(value: Any) -> Optional[float]:
    """Parse epoch seconds (number / numeric string) or ISO-8601 into float seconds."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v if math.isfinite(v) and v >= 0 else None
    if isinstance(value, str):
        s = value.strip()
        if s in MISSING_STRINGS:
            return None
        try:
            v = float(s)
            return v if math.isfinite(v) and v >= 0 else None
        except ValueError:
            pass
        try:
            iso = s[:-1] + "+00:00" if s[-1] in ("Z", "z") else s
            dt = datetime.fromisoformat(iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            return None
    return None


def parse_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() else None
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def parse_port(value: Any) -> Optional[int]:
    port = parse_int(value)
    return port if port is not None and 0 <= port <= 65535 else None


def parse_ip(value: Any) -> Optional[str]:
    """Return the canonical IP string, or ``None`` if missing/not an IP address."""
    s = clean_str(value)
    if s is None:
        return None
    try:
        return str(ipaddress.ip_address(s))
    except ValueError:
        return None


def parse_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("t", "true", "1", "yes"):
            return True
        if s in ("f", "false", "0", "no"):
            return False
    return None


def parse_answers(value: Any) -> Optional[list[str]]:
    """Normalise Zeek ``answers`` to a list of strings.

    * ``None`` (field absent / null)          -> ``None``  (unknown)
    * ``[]`` or ``""`` or ``"-"``              -> ``[]``    (present but empty)
    * list of values / comma separated string -> list of non-empty strings
    * unsupported type                        -> ``None``
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        items = [clean_str(v) for v in value]
        return [i for i in items if i is not None]
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip() not in MISSING_STRINGS]
    return None


def parse_ttls(value: Any) -> Optional[list[float]]:
    """Normalise Zeek ``TTLs`` to a list of non-negative finite floats (same rules as answers)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return [v] if math.isfinite(v) and v >= 0 else []
    if isinstance(value, str):
        items: list[Any] = [p for p in value.split(",")]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        return None
    out: list[float] = []
    for item in items:
        if isinstance(item, bool) or item is None:
            continue
        try:
            v = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v) and v >= 0:
            out.append(v)
    return out


@dataclass
class ValidationReport:
    """Per-file loading statistics."""

    source_name: str
    path: str
    label: int
    detected_format: str = "unknown"
    n_records_seen: int = 0
    n_valid: int = 0
    n_invalid: int = 0
    invalid_reasons: Counter = field(default_factory=Counter)
    anomalies: Counter = field(default_factory=Counter)
    missing_counts: Counter = field(default_factory=Counter)

    def add_invalid(self, reason: str) -> None:
        self.n_records_seen += 1
        self.n_invalid += 1
        self.invalid_reasons[reason] += 1

    def add_valid(self) -> None:
        self.n_records_seen += 1
        self.n_valid += 1

    @property
    def invalid_fraction(self) -> float:
        return self.n_invalid / self.n_records_seen if self.n_records_seen else 0.0

    def field_presence(self) -> dict[str, float]:
        """Fraction of valid records in which each field has a usable value."""
        if self.n_valid == 0:
            return {}
        return {k: 1.0 - v / self.n_valid for k, v in self.missing_counts.items()}

    def to_dict(self) -> dict:
        return {
            "source_name": self.source_name,
            "path": self.path,
            "label": self.label,
            "detected_format": self.detected_format,
            "n_records_seen": self.n_records_seen,
            "n_valid": self.n_valid,
            "n_invalid": self.n_invalid,
            "invalid_fraction": self.invalid_fraction,
            "invalid_reasons": dict(self.invalid_reasons),
            "anomalies_coerced_to_missing": dict(self.anomalies),
            "missing_counts_among_valid": dict(self.missing_counts),
            "field_presence_fraction": self.field_presence(),
        }


def check_events_dataframe(df: pd.DataFrame) -> None:
    """Hard consistency checks on the merged event table. Raises ``ValueError``."""
    required = ["event_id", "label", "timestamp", "src_ip", "query"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Event table is missing columns: {missing}")
    if df.empty:
        raise ValueError("No valid DNS records were loaded.")
    if df["event_id"].duplicated().any():
        raise ValueError("Duplicate event_id values detected; event IDs must be unique.")
    if not set(df["label"].unique()) <= {0, 1}:
        raise ValueError("Labels must be 0 (benign) or 1 (tunnel).")
    if not df["timestamp"].map(math.isfinite).all():
        raise ValueError("Non-finite timestamps present after validation.")
