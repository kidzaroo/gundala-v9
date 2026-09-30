"""Load Zeek ``dns.log`` JSON exports (NDJSON or JSON array) into one event table.

Labels come from the *source file* (benign=0, tunnel=1) and are attached only as
metadata/target; they are never read by feature code.

``event_id`` is ``evt_`` + first 16 hex chars of ``sha1("<source_name>:<record position>")``.
It is unique, deterministic (same files -> same IDs, so split manifests are reusable) and,
unlike a readable ``benign_000123`` id, it does not correlate with the label when it is
used as a stable tie-breaker for equal timestamps.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import pandas as pd

from .domain_utils import normalize_query
from .validation import (
    UNKNOWN_SRC,
    ValidationReport,
    check_events_dataframe,
    clean_str,
    parse_answers,
    parse_bool,
    parse_int,
    parse_ip,
    parse_port,
    parse_timestamp,
    parse_ttls,
)

logger = logging.getLogger(__name__)

EVENT_COLUMNS = [
    "event_id", "label", "source_file", "source_line", "timestamp", "src_ip", "dst_ip",
    "src_port", "dst_port", "proto", "query", "qtype", "qtype_name", "rcode", "rcode_name",
    "answers", "ttls", "rejected", "aa", "tc", "rd", "ra",
]
LABEL_NAMES = {0: "benign", 1: "tunnel"}

QTYPE_NAMES = {
    1: "A", 2: "NS", 5: "CNAME", 6: "SOA", 10: "NULL", 12: "PTR", 13: "HINFO", 15: "MX",
    16: "TXT", 17: "RP", 18: "AFSDB", 24: "SIG", 25: "KEY", 28: "AAAA", 33: "SRV", 35: "NAPTR",
    39: "DNAME", 41: "OPT", 43: "DS", 46: "RRSIG", 47: "NSEC", 48: "DNSKEY", 50: "NSEC3",
    51: "NSEC3PARAM", 52: "TLSA", 64: "SVCB", 65: "HTTPS", 99: "SPF", 250: "TSIG", 251: "IXFR",
    252: "AXFR", 255: "ANY", 257: "CAA",
}
RCODE_NAMES = {
    0: "NOERROR", 1: "FORMERR", 2: "SERVFAIL", 3: "NXDOMAIN", 4: "NOTIMP", 5: "REFUSED",
    6: "YXDOMAIN", 7: "YXRRSET", 8: "NXRRSET", 9: "NOTAUTH", 10: "NOTZONE",
}


def make_event_id(source_name: str, position: int) -> str:
    digest = hashlib.sha1(f"{source_name}:{position}".encode("utf-8")).hexdigest()
    return "evt_" + digest[:16]


def detect_format(path: Path, encoding: str = "utf-8") -> str:
    """Return ``"json_array"`` if the first non-blank character is ``[``, else ``"jsonl"``."""
    with path.open("r", encoding=encoding, errors="replace") as fh:
        while True:
            chunk = fh.read(4096)
            if not chunk:
                return "jsonl"
            stripped = chunk.lstrip("\ufeff \t\r\n")
            if stripped:
                return "json_array" if stripped[0] == "[" else "jsonl"


def iter_json_records(
    path: Path, fmt: str, encoding: str = "utf-8"
) -> Iterator[tuple[int, Optional[dict], Optional[str]]]:
    """Yield ``(position, record, error)``; ``record`` is ``None`` when ``error`` is set.

    ``position`` is the 1-based line number (JSONL) or element index (JSON array).
    """
    if fmt == "json_array":
        with path.open("r", encoding=encoding, errors="replace") as fh:
            text = fh.read().lstrip("\ufeff")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: not a valid JSON array ({exc}). "
                             "If this is NDJSON set input.format=jsonl.") from exc
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a JSON array at top level, got {type(data).__name__}.")
        for i, item in enumerate(data, start=1):
            if isinstance(item, dict):
                yield i, item, None
            else:
                yield i, None, "not_an_object"
        return

    with path.open("r", encoding=encoding, errors="replace") as fh:
        for i, line in enumerate(fh, start=1):
            text = line.strip().lstrip("\ufeff")
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                yield i, None, "invalid_json"
                continue
            if isinstance(obj, dict):
                yield i, obj, None
            else:
                yield i, None, "not_an_object"


def lookup_field(record: Mapping[str, Any], candidates: Sequence[str]) -> tuple[bool, Any]:
    """Find the first candidate name in ``record``.

    A dotted name is tried as a flat key (``"id.orig_h"``) first and then as a nested
    path (``record["id"]["orig_h"]``). Returns ``(found, value)``.
    """
    for name in candidates:
        if name in record:
            return True, record[name]
        if "." in name:
            cur: Any = record
            ok = True
            for part in name.split("."):
                if isinstance(cur, Mapping) and part in cur:
                    cur = cur[part]
                else:
                    ok = False
                    break
            if ok:
                return True, cur
    return False, None


def normalize_record(
    raw: Mapping[str, Any],
    field_mapping: Mapping[str, Sequence[str]],
    required_fields: Sequence[str],
    anomalies: Optional[dict] = None,
) -> tuple[Optional[dict], Optional[str]]:
    """Convert one raw Zeek record into canonical values.

    Returns ``(values, None)`` for a valid record, or ``(None, reason)`` if invalid.
    ``anomalies`` (a Counter-like dict) receives ``"unparsable:<field>"`` counts for
    present-but-unusable values that were coerced to missing.
    """
    anomalies = anomalies if anomalies is not None else {}

    def get(name: str) -> Any:
        found, value = lookup_field(raw, field_mapping.get(name, ()))
        return value if found else None

    def note(name: str) -> None:
        anomalies[f"unparsable:{name}"] = anomalies.get(f"unparsable:{name}", 0) + 1

    out: dict[str, Any] = {}

    raw_ts = get("timestamp")
    ts = parse_timestamp(raw_ts)
    if raw_ts is not None and ts is None:
        return None, "invalid_timestamp"
    out["timestamp"] = ts

    raw_q = get("query")
    if raw_q is not None and not isinstance(raw_q, str):
        return None, "invalid_type:query"
    out["query"] = normalize_query(raw_q)

    for name, fn in (("src_ip", parse_ip), ("dst_ip", parse_ip)):
        raw_val = get(name)
        val = fn(raw_val)
        if raw_val is not None and clean_str(raw_val) is not None and val is None:
            note(name)
        out[name] = val
    for name in ("src_port", "dst_port"):
        raw_val = get(name)
        val = parse_port(raw_val)
        if raw_val is not None and val is None:
            note(name)
        out[name] = val

    proto = clean_str(get("proto"))
    out["proto"] = proto.lower() if proto else None

    raw_qtype = get("qtype")
    qtype = parse_int(raw_qtype)
    if raw_qtype is not None and qtype is None:
        note("qtype")
    qtype_name = clean_str(get("qtype_name"))
    qtype_name = qtype_name.upper() if qtype_name else None
    if qtype_name is None and qtype is not None:
        qtype_name = QTYPE_NAMES.get(qtype, f"TYPE{qtype}")
    out["qtype"], out["qtype_name"] = qtype, qtype_name

    raw_rcode = get("rcode")
    rcode = parse_int(raw_rcode)
    if raw_rcode is not None and rcode is None:
        note("rcode")
    rcode_name = clean_str(get("rcode_name"))
    rcode_name = rcode_name.upper() if rcode_name else None
    if rcode_name is None and rcode is not None:
        rcode_name = RCODE_NAMES.get(rcode, f"RCODE{rcode}")
    out["rcode"], out["rcode_name"] = rcode, rcode_name

    raw_answers = get("answers")
    out["answers"] = parse_answers(raw_answers)
    if raw_answers is not None and out["answers"] is None:
        note("answers")
    raw_ttls = get("ttls")
    out["ttls"] = parse_ttls(raw_ttls)
    if raw_ttls is not None and out["ttls"] is None:
        note("ttls")

    for name in ("rejected", "aa", "tc", "rd", "ra"):
        raw_val = get(name)
        val = parse_bool(raw_val)
        if raw_val is not None and val is None:
            note(name)
        out[name] = val

    for name in required_fields:
        value = out.get(name)
        if value is None or (isinstance(value, str) and value == ""):
            return None, f"missing_required:{name}"
    return out, None


def load_dns_file(
    path: str | Path,
    label: int,
    source_name: str,
    field_mapping: Mapping[str, Sequence[str]],
    fmt: str = "auto",
    encoding: str = "utf-8",
    required_fields: Sequence[str] = ("timestamp",),
) -> tuple[pd.DataFrame, ValidationReport]:
    """Load one file. ``label`` is assigned from the source file, never from content."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    detected = detect_format(path, encoding) if fmt == "auto" else fmt
    report = ValidationReport(source_name=source_name, path=str(path), label=label, detected_format=detected)
    anomalies: dict[str, int] = {}
    rows: list[dict] = []

    for position, record, error in iter_json_records(path, detected, encoding):
        if record is None:
            report.add_invalid(error or "invalid_record")
            continue
        values, reason = normalize_record(record, field_mapping, required_fields, anomalies)
        if values is None:
            report.add_invalid(reason or "invalid_record")
            continue
        report.add_valid()
        for name, value in values.items():
            if value is None or (name == "query" and value == ""):
                report.missing_counts[name] += 1
        values["event_id"] = make_event_id(source_name, position)
        values["label"] = int(label)
        values["source_file"] = path.name
        values["source_line"] = position
        if values["src_ip"] is None:
            values["src_ip"] = UNKNOWN_SRC
        rows.append(values)

    report.anomalies.update(anomalies)
    df = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    df["timestamp"] = df["timestamp"].astype("float64")
    df["label"] = df["label"].astype("int64")
    logger.info(
        "%s: format=%s records=%d valid=%d invalid=%d", source_name, detected,
        report.n_records_seen, report.n_valid, report.n_invalid,
    )
    return df, report


def class_distribution(df: pd.DataFrame) -> dict:
    """Counts per class and the *actual* benign:tunnel ratio (never forced to 7:1)."""
    n0 = int((df["label"] == 0).sum())
    n1 = int((df["label"] == 1).sum())
    ratio = (n0 / n1) if n1 else None
    return {
        "benign": n0,
        "tunnel": n1,
        "total": n0 + n1,
        "ratio_benign_to_tunnel": ratio,
        "ratio_text": f"{ratio:.3f}:1" if ratio is not None else "undefined (no tunnel events)",
    }


def field_availability(df: pd.DataFrame) -> dict[str, bool]:
    """Dataset-level availability: does a field have at least one usable value?"""
    avail: dict[str, bool] = {}
    for col in ("dst_ip", "src_port", "dst_port", "proto", "qtype_name", "rcode_name",
                "answers", "ttls", "rejected", "aa", "tc", "rd", "ra"):
        avail[col] = bool(df[col].notna().any()) if col in df.columns else False
    avail["query"] = bool((df["query"] != "").any())
    avail["src_ip"] = bool((df["src_ip"] != UNKNOWN_SRC).any())
    return avail


def load_dataset(cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, dict, dict]:
    """Load benign + tunnel files according to ``cfg``.

    Returns ``(events, reports, availability)``. The class ratio is whatever the files
    contain; no resampling happens here.
    """
    inp = cfg["input"]
    fm = cfg["field_mapping"]
    req = cfg["validation"]["required_fields"]
    max_invalid = float(cfg["validation"]["max_invalid_fraction"])
    if not inp.get("benign_path") or not inp.get("tunnel_path"):
        raise ValueError("Both input.benign_path and input.tunnel_path must be provided.")

    frames, reports = [], {}
    for name, label, key in (("benign", 0, "benign_path"), ("tunnel", 1, "tunnel_path")):
        df, rep = load_dns_file(
            inp[key], label, name, fm,
            fmt=inp["format"], encoding=inp["encoding"], required_fields=req,
        )
        if rep.n_records_seen and rep.invalid_fraction > max_invalid:
            raise ValueError(
                f"{inp[key]}: {rep.invalid_fraction:.1%} of records are invalid "
                f"(> max_invalid_fraction={max_invalid}). Reasons: {dict(rep.invalid_reasons)}. "
                "Check input.format and field_mapping."
            )
        frames.append(df)
        reports[name] = rep
    events = pd.concat(frames, ignore_index=True)
    check_events_dataframe(events)
    return events, reports, field_availability(events)
