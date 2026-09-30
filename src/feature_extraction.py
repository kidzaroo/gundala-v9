"""Stateless per-event feature extraction.

Everything here depends on a single DNS event only (no statistics are learned from
data, no label is read), so it is safe to run before or after the train/test split.

Mathematical definitions (q = normalised query string, n = |q| characters incl. dots,
s = subdomain string, labels = q.split(".")):

* Shannon entropy   H(x) = - sum_{c in set(x)} p(c) * log2 p(c),  p(c) = count_x(c) / |x|,
  ``H("") = 0``. Computed over *characters* (dots included) of the query (``query_entropy``)
  and of the subdomain string (``subdomain_entropy``).
* Ratios            ratio_X = count_X / n   for X in {digit, letter, hyphen};
  ``unique_char_ratio = |set(q)| / n``. Every ratio is 0 when n = 0.
  Digits are ``0-9`` and letters ``a-z`` (query is lower-cased).
* ``mean_label_length = sum(len(l) for l in labels) / len(labels)`` (0 when no label).
* ``longest_digit_run`` / ``longest_alpha_run``: length of the longest maximal run of
  ``[0-9]`` / ``[a-z]`` characters in q.
* TTL statistics are over the TTLs list of the event (``NaN`` if the list is empty or the
  field does not exist in the dataset; imputed later with training statistics).

Missing-data conventions
------------------------
* ``answers``/``TTLs`` field never present in the dataset  -> count/TTL features are NaN.
* Field present in the dataset but unset for this event (Zeek omits unset optional
  fields)                                                -> 0 answers / 0 TTLs.
* ``rcode``/``qtype``/``proto`` missing                    -> category ``"MISSING"``.
* DNS flags / ``rejected`` missing                         -> NaN (imputed: most frequent).
* ``is_nxdomain`` = 1 iff rcode name == ``NXDOMAIN`` (0 when rcode is missing).

Metadata columns (``src_ip``, ``dst_ip``, raw ``query``, ``sld``, ``timestamp``,
``event_id``, ``source_file``, ``label``) are *never* classifier inputs.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from .domain_utils import parse_domain

LEXICAL_CONTINUOUS = [
    "query_length", "num_labels", "subdomain_length", "subdomain_num_labels",
    "max_label_length", "mean_label_length", "query_entropy", "subdomain_entropy",
    "digit_count", "digit_ratio", "letter_count", "letter_ratio", "hyphen_count",
    "hyphen_ratio", "unique_char_count", "unique_char_ratio", "longest_digit_run",
    "longest_alpha_run",
]
LEXICAL_BINARY = ["is_ip_query", "is_reverse_dns", "is_single_label", "is_valid_domain"]
RESPONSE_CONTINUOUS = ["num_answers", "num_ttls", "ttl_min", "ttl_max", "ttl_mean"]
RESPONSE_BINARY = ["is_nxdomain", "flag_aa", "flag_tc", "flag_rd", "flag_ra", "rejected_flag"]
CATEGORICAL = ["qtype_cat", "rcode_cat", "proto_cat"]

EVENT_CONTINUOUS = LEXICAL_CONTINUOUS + RESPONSE_CONTINUOUS
EVENT_BINARY = LEXICAL_BINARY + RESPONSE_BINARY
EVENT_CATEGORICAL = CATEGORICAL

GROUPING_META_COLUMNS = ["sld", "subdomain", "domain_kind"]

# Columns that must never be used directly as classifier features.
FORBIDDEN_FEATURE_COLUMNS = frozenset({
    "src_ip", "dst_ip", "query", "sld", "subdomain", "domain_kind", "timestamp", "event_id",
    "source_file", "source_line", "label",
})

_DIGIT_RUN = re.compile(r"[0-9]+")
_ALPHA_RUN = re.compile(r"[a-z]+")


def shannon_entropy(text: str) -> float:
    """``H = -sum p(c) log2 p(c)`` over characters of ``text``; 0 for the empty string."""
    n = len(text)
    if n == 0:
        return 0.0
    h = 0.0
    for count in Counter(text).values():
        p = count / n
        h -= p * math.log2(p)
    return h + 0.0  # avoid -0.0


def _ratio(count: float, total: int) -> float:
    return count / total if total > 0 else 0.0


def compute_lexical_features(query: str) -> dict:
    """Lexical features + grouping metadata for one query string (handles empty/invalid)."""
    parts = parse_domain(query)
    q = parts.normalized
    sub = parts.subdomain
    n = len(q)
    labels = q.split(".") if q else []
    sub_labels = sub.split(".") if sub else []
    label_lens = [len(label) for label in labels]

    digits = sum(1 for c in q if "0" <= c <= "9")
    letters = sum(1 for c in q if "a" <= c <= "z")
    hyphens = q.count("-")
    uniq = len(set(q))
    digit_runs = [len(m) for m in _DIGIT_RUN.findall(q)]
    alpha_runs = [len(m) for m in _ALPHA_RUN.findall(q)]

    return {
        "sld": parts.registrable,
        "subdomain": sub,
        "domain_kind": parts.kind,
        "query_length": n,
        "num_labels": len(labels),
        "subdomain_length": len(sub),
        "subdomain_num_labels": len(sub_labels),
        "max_label_length": max(label_lens) if label_lens else 0,
        "mean_label_length": (sum(label_lens) / len(label_lens)) if label_lens else 0.0,
        "query_entropy": shannon_entropy(q),
        "subdomain_entropy": shannon_entropy(sub),
        "digit_count": digits,
        "digit_ratio": _ratio(digits, n),
        "letter_count": letters,
        "letter_ratio": _ratio(letters, n),
        "hyphen_count": hyphens,
        "hyphen_ratio": _ratio(hyphens, n),
        "unique_char_count": uniq,
        "unique_char_ratio": _ratio(uniq, n),
        "longest_digit_run": max(digit_runs) if digit_runs else 0,
        "longest_alpha_run": max(alpha_runs) if alpha_runs else 0,
        "is_ip_query": int(parts.kind == "ip"),
        "is_reverse_dns": int(parts.kind == "reverse_dns"),
        "is_single_label": int(parts.kind == "single_label"),
        "is_valid_domain": int(parts.is_valid),
    }


def get_event_feature_groups(include_categorical: bool = True) -> tuple[list[str], list[str], list[str]]:
    """Return ``(continuous, binary, categorical)`` feature names in fixed order."""
    return (
        list(EVENT_CONTINUOUS),
        list(EVENT_BINARY),
        list(EVENT_CATEGORICAL) if include_categorical else [],
    )


def _resolve_availability(df: pd.DataFrame, availability: Optional[Mapping[str, bool]]) -> dict[str, bool]:
    inferred = {
        col: bool(df[col].notna().any()) if col in df.columns else False
        for col in ("answers", "ttls", "aa", "tc", "rd", "ra", "rejected")
    }
    if availability:
        for key in inferred:
            if key in availability:
                inferred[key] = bool(availability[key])
    return inferred


def _flag_series(series: pd.Series, available: bool) -> pd.Series:
    if not available:
        return pd.Series(np.nan, index=series.index, dtype="float64")
    return series.map(lambda v: float(bool(v)) if v is not None and not pd.isna(v) else np.nan).astype("float64")


def _list_stat(values, fn) -> float:
    if isinstance(values, (list, tuple)) and len(values) > 0:
        return float(fn(values))
    return float("nan")


def extract_event_features(
    df: pd.DataFrame, field_availability: Optional[Mapping[str, bool]] = None
) -> pd.DataFrame:
    """Compute per-event features (+ ``sld``/``subdomain``/``domain_kind`` metadata).

    Returns a DataFrame with the same index as ``df``. ``field_availability`` (dataset
    level, from the loader) tells whether ``answers``/``TTLs``/flags exist at all; if
    omitted it is inferred from ``df``.
    """
    avail = _resolve_availability(df, field_availability)
    columns = GROUPING_META_COLUMNS + EVENT_CONTINUOUS + EVENT_BINARY + EVENT_CATEGORICAL
    if len(df) == 0:
        return pd.DataFrame(columns=columns)

    queries = df["query"].fillna("").astype(str).to_numpy()
    unique_q = pd.unique(queries)
    lex = pd.DataFrame([compute_lexical_features(q) for q in unique_q], index=unique_q)
    lex = lex.reindex(queries)
    lex.index = df.index
    feat = lex.copy()

    feat["qtype_cat"] = df["qtype_name"].fillna("MISSING").astype(str)
    feat["rcode_cat"] = df["rcode_name"].fillna("MISSING").astype(str)
    feat["proto_cat"] = df["proto"].fillna("MISSING").astype(str)
    feat["is_nxdomain"] = (feat["rcode_cat"] == "NXDOMAIN").astype(int)

    nan = float("nan")
    if avail["answers"]:
        feat["num_answers"] = df["answers"].map(
            lambda v: float(len(v)) if isinstance(v, (list, tuple)) else 0.0
        )
    else:
        feat["num_answers"] = nan
    if avail["ttls"]:
        feat["num_ttls"] = df["ttls"].map(lambda v: float(len(v)) if isinstance(v, (list, tuple)) else 0.0)
        feat["ttl_min"] = df["ttls"].map(lambda v: _list_stat(v, min))
        feat["ttl_max"] = df["ttls"].map(lambda v: _list_stat(v, max))
        feat["ttl_mean"] = df["ttls"].map(lambda v: _list_stat(v, lambda x: sum(x) / len(x)))
    else:
        for col in ("num_ttls", "ttl_min", "ttl_max", "ttl_mean"):
            feat[col] = nan

    feat["flag_aa"] = _flag_series(df["aa"], avail["aa"])
    feat["flag_tc"] = _flag_series(df["tc"], avail["tc"])
    feat["flag_rd"] = _flag_series(df["rd"], avail["rd"])
    feat["flag_ra"] = _flag_series(df["ra"], avail["ra"])
    feat["rejected_flag"] = _flag_series(df["rejected"], avail["rejected"])

    feat = feat[columns]
    for col in EVENT_CONTINUOUS + EVENT_BINARY:
        feat[col] = feat[col].astype("float64")
    return feat


def add_event_features(
    df: pd.DataFrame, field_availability: Optional[Mapping[str, bool]] = None
) -> pd.DataFrame:
    """Return ``df`` with per-event feature columns appended (metadata kept intact)."""
    feats = extract_event_features(df, field_availability)
    return pd.concat([df, feats], axis=1)
