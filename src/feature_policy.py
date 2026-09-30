"""Feature-set selection and data-artefact audits.

Why this module exists
----------------------
Near-perfect scores on a two-file dataset (benign.json / tunnel.json) can come from
*recording artefacts* instead of tunneling behaviour: a field that exists in only one
file, TTL/flag/query-type conventions of the capture setup, or simply identical feature
vectors on both sides of a random split. This module provides

* :func:`resolve_feature_selection`: ``feature_set`` (``all`` | ``query_only``), explicit
  ``features.exclude`` and automatic exclusion of artefact fields, with a record of every
  dropped feature and why (ablation-friendly);
* :func:`schema_artifact_audit`: compares per-class *field presence* on the TRAINING subset
  only (no test data, no model fitting) and flags fields whose presence differs drastically;
* :func:`leakage_audit`: single-feature AUC (training data), category purity, duplicate
  feature vectors / query / SLD overlap between train and test, and human-readable warnings.

The audits are diagnostics: they never feed information back into model fitting, except
that the configured ``schema_artifact_policy='exclude'`` removes flagged *fields' features*
from both subsets (decided from training data only).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .feature_extraction import (
    EVENT_BINARY,
    EVENT_CATEGORICAL,
    EVENT_CONTINUOUS,
    FORBIDDEN_FEATURE_COLUMNS,
)
from .window_features import WINDOW_FEATURE_COLUMNS

logger = logging.getLogger(__name__)

FEATURE_SETS = ("all", "query_only")
ARTIFACT_POLICIES = ("exclude", "warn", "off")

# Raw log field -> features (per-event and window) that are derived from it.
FIELD_TO_FEATURES: dict[str, list[str]] = {
    "answers": ["num_answers", "win_answers_mean"],
    "ttls": ["num_ttls", "ttl_min", "ttl_max", "ttl_mean", "win_ttl_mean"],
    "aa": ["flag_aa"],
    "tc": ["flag_tc"],
    "rd": ["flag_rd"],
    "ra": ["flag_ra"],
    "rejected": ["rejected_flag"],
    "rcode_name": ["rcode_cat", "is_nxdomain", "win_nxdomain_count", "win_nxdomain_ratio"],
    "qtype_name": ["qtype_cat", "win_unique_qtypes"],
    "proto": ["proto_cat"],
    "dst_ip": ["win_unique_dst_ips"],
}
AUDIT_FIELDS = tuple(FIELD_TO_FEATURES)
RESPONSE_PROTOCOL_FEATURES = frozenset(f for fs in FIELD_TO_FEATURES.values() for f in fs)
ALL_FEATURE_NAMES = frozenset(
    EVENT_CONTINUOUS + EVENT_BINARY + EVENT_CATEGORICAL + WINDOW_FEATURE_COLUMNS
)


@dataclass
class FeatureSelection:
    """Final feature lists of one configuration (+ what was dropped and why)."""

    continuous: list[str]            # event continuous + window columns (sliding approach)
    binary: list[str]
    categorical: list[str]
    event_columns: list[str]         # per-event columns only
    window_columns: list[str]
    dropped: dict[str, str] = field(default_factory=dict)

    @property
    def all_columns(self) -> list[str]:
        return self.continuous + self.binary + self.categorical


def resolve_feature_selection(
    approach: str,
    *,
    include_categorical: bool = True,
    feature_set: str = "all",
    exclude: Iterable[str] = (),
    excluded_fields: Iterable[str] = (),
) -> FeatureSelection:
    """Apply ``include_categorical``, ``feature_set``, ``exclude`` and artefact exclusions."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Invalid features.feature_set={feature_set!r}. Allowed: {list(FEATURE_SETS)}")
    exclude = list(exclude or [])
    unknown = sorted(set(exclude) - ALL_FEATURE_NAMES)
    if unknown:
        raise ValueError(f"features.exclude contains unknown feature names: {unknown}. "
                         f"Valid names: {sorted(ALL_FEATURE_NAMES)}")

    dropped: dict[str, str] = {}

    def drop(name: str, reason: str) -> None:
        dropped.setdefault(name, reason)

    if not include_categorical:
        for n in EVENT_CATEGORICAL:
            drop(n, "features.include_categorical=false")
    if feature_set == "query_only":
        for n in sorted(RESPONSE_PROTOCOL_FEATURES):
            drop(n, "features.feature_set=query_only")
    for n in exclude:
        drop(n, "features.exclude")
    for fld in excluded_fields:
        for n in FIELD_TO_FEATURES.get(fld, []):
            drop(n, f"schema_artifact:{fld}")

    window_candidates = list(WINDOW_FEATURE_COLUMNS) if approach == "sliding_window" else []
    candidates = set(EVENT_CONTINUOUS + EVENT_BINARY + EVENT_CATEGORICAL) | set(window_candidates)

    def keep(names: Sequence[str]) -> list[str]:
        return [n for n in names if n not in dropped]

    cont, binary, cat = keep(EVENT_CONTINUOUS), keep(EVENT_BINARY), keep(EVENT_CATEGORICAL)
    wcols = keep(window_candidates)
    if not (cont + binary + cat + wcols):
        raise ValueError("No features left after applying the feature policy.")
    overlap = FORBIDDEN_FEATURE_COLUMNS & set(cont + binary + cat + wcols)
    if overlap:  # defensive: metadata must never become a feature
        raise AssertionError(f"Forbidden metadata used as features: {sorted(overlap)}")
    return FeatureSelection(
        continuous=cont + wcols, binary=binary, categorical=cat,
        event_columns=cont + binary + cat, window_columns=wcols,
        dropped={k: v for k, v in dropped.items() if k in candidates},
    )


# ----------------------------------------------------------------------------- audits
def schema_artifact_audit(train_raw: pd.DataFrame, threshold: float = 0.5) -> dict:
    """Per-class presence of each raw field, computed on the TRAINING subset only.

    A field is *flagged* when ``|presence(tunnel) - presence(benign)| >= threshold``: the
    mere existence of the field then reveals the source file (capture setup), not the
    behaviour. "Present" means a usable, non-null value (an empty answers list counts as
    present; an omitted field does not).
    """
    fields: dict[str, dict] = {}
    flagged: list[str] = []
    labels = train_raw["label"]
    for fld in AUDIT_FIELDS:
        if fld not in train_raw.columns:
            continue
        presence = train_raw[fld].notna().groupby(labels).mean()
        p0 = float(presence.get(0, np.nan))
        p1 = float(presence.get(1, np.nan))
        diff = abs(p1 - p0)
        is_flagged = bool(diff >= threshold)
        if is_flagged:
            flagged.append(fld)
        fields[fld] = {
            "presence_benign": p0, "presence_tunnel": p1, "abs_difference": diff, "flagged": is_flagged,
            "features_affected": FIELD_TO_FEATURES[fld],
        }
    return {
        "basis": "training subset only",
        "threshold": float(threshold),
        "fields": fields,
        "flagged_fields": flagged,
    }


def _vector_hash(df: pd.DataFrame, continuous: Sequence[str], binary: Sequence[str],
                 categorical: Sequence[str]) -> pd.Series:
    parts = []
    numeric = list(continuous) + list(binary)
    if numeric:
        parts.append(df[numeric].astype("float64").fillna(-1.0).round(6))
    if categorical:
        parts.append(df[list(categorical)].astype(str))
    return pd.util.hash_pandas_object(pd.concat(parts, axis=1), index=False)


def leakage_audit(
    train_ev: pd.DataFrame,
    test_ev: pd.DataFrame,
    continuous: Sequence[str],
    binary: Sequence[str],
    categorical: Sequence[str],
    *,
    auc_flag: float = 0.98,
    purity_flag: float = 0.99,
    duplicate_flag: float = 0.5,
    overlap_flag: float = 0.9,
) -> dict:
    """Diagnostics explaining suspiciously high scores (reporting only).

    * single-feature AUC and category purity are computed on TRAINING data only;
    * duplicate-vector / query / SLD overlap compare test rows with training rows and are
      used for reporting, never for fitting or selecting anything.
    """
    names = {0: "benign", 1: "tunnel"}
    warnings: list[str] = []
    y = train_ev["label"].to_numpy()

    aucs: dict[str, float] = {}
    for c in list(continuous) + list(binary):
        x = train_ev[c].astype("float64")
        if not x.notna().any():
            aucs[c] = 0.5
            continue
        x = x.fillna(x.median())
        if x.nunique() < 2:
            aucs[c] = 0.5
            continue
        a = float(roc_auc_score(y, x))
        aucs[c] = max(a, 1.0 - a)
    top = dict(sorted(aucs.items(), key=lambda kv: kv[1], reverse=True)[:10])
    separators = {c: a for c, a in top.items() if a >= auc_flag}
    if separators:
        listed = ", ".join(f"'{c}' ({a:.4f})" for c, a in separators.items())
        warnings.append(
            f"{len(separators)} feature(s) each separate the classes almost perfectly on their own "
            f"(single-feature AUC on training data): {listed}. Check whether this reflects tunneling "
            "behaviour or a recording artefact; compare with features.feature_set=query_only / "
            "features.exclude and with split.strategy=group_sld or chronological."
        )

    purity: dict[str, dict] = {}
    for c in categorical:
        ct = pd.crosstab(train_ev[c], train_ev["label"])
        total = int(ct.to_numpy().sum())
        pur = float(ct.max(axis=1).sum() / total) if total else float("nan")
        share = (ct[1] / ct.sum(axis=1)) if 1 in ct.columns else pd.Series(dtype=float)
        purity[c] = {
            "purity": pur, "n_categories": int(len(ct)),
            "tunnel_share_by_category": {str(k): float(v) for k, v in share.head(15).items()},
        }
        if pur >= purity_flag:
            warnings.append(f"Categorical feature '{c}' is almost a perfect class indicator "
                            f"(purity={pur:.4f} on training data).")

    duplicates: dict = {}
    cols = list(continuous) + list(binary) + list(categorical)
    if cols:
        h_train = _vector_hash(train_ev, continuous, binary, categorical)
        h_test = _vector_hash(test_ev, continuous, binary, categorical)
        train_set = set(h_train.tolist())
        in_train = h_test.isin(train_set)
        frac = in_train.groupby(test_ev["label"].to_numpy()).mean()
        duplicates["test_rows_with_identical_vector_in_train"] = {
            names[int(k)]: float(v) for k, v in frac.items()
        }
        both = pd.DataFrame({
            "h": pd.concat([h_train, h_test], ignore_index=True),
            "y": np.concatenate([train_ev["label"].to_numpy(), test_ev["label"].to_numpy()]),
        })
        nun = both.groupby("h")["y"].nunique()
        duplicates["feature_vectors_with_both_labels"] = int((nun > 1).sum())
        duplicates["unique_vector_fraction_train"] = float(h_train.nunique() / max(len(h_train), 1))
        for lbl, v in duplicates["test_rows_with_identical_vector_in_train"].items():
            if v >= duplicate_flag:
                warnings.append(
                    f"{v:.0%} of test '{lbl}' events have a feature vector identical to a training event: "
                    "memorisation can inflate scores (see split.strategy=group_sld)."
                )
        if duplicates["feature_vectors_with_both_labels"]:
            warnings.append(
                f"{duplicates['feature_vectors_with_both_labels']} feature vectors occur with BOTH labels "
                "(label noise or non-separable events)."
            )

    overlap: dict = {}
    for col, key in (("query", "query_overlap"), ("sld", "sld_overlap")):
        seen = set(train_ev[col].tolist())
        frac = test_ev[col].isin(seen).groupby(test_ev["label"].to_numpy()).mean()
        overlap[key] = {names[int(k)]: float(v) for k, v in frac.items()}
    for lbl, v in overlap["sld_overlap"].items():
        if v >= overlap_flag:
            warnings.append(
                f"{v:.0%} of test '{lbl}' events belong to an SLD that also appears in training: "
                "the test set does not measure generalisation to unseen domains."
            )

    for w in warnings:
        logger.warning("LEAKAGE AUDIT: %s", w)
    return {
        "top_single_feature_auc_train": top,
        "categorical_purity_train": purity,
        "duplicates": duplicates,
        "overlap_test_in_train": overlap,
        "warnings": warnings,
        "thresholds": {"single_feature_auc": auc_flag, "category_purity": purity_flag,
                       "duplicate_fraction": duplicate_flag, "sld_overlap": overlap_flag},
    }
