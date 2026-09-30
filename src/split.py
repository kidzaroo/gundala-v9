"""De-duplication, train/test splitting and split manifests.

All of this happens BEFORE any augmentation, balancing, fitted encoding/scaling,
imputation or feature selection. The split is computed once and reused by every
classifier and every window size.

``group_sld`` keeps every event of a registrable domain (SLD) on ONE side of the split, so the
test set measures generalisation to *unseen domains*. Because the window grouping key is
``(src_ip, SLD)``, no window group is ever cut in two: window features of each subset are
identical to those computed on the full stream (no thinning of history).

Limitations of ``stratified_random``
------------------------------------
Random stratified splitting assumes events are i.i.d. DNS traffic is not: queries of the
same client/domain are temporally correlated and domains repeat (a tunneling domain, or a
popular benign domain, appears on both sides of the split). Test events then have "close
relatives" in training, which typically inflates scores compared with deployment on
future traffic. It also removes events from the time series, so per-subset window
history is incomplete (see README). Use ``chronological`` to measure temporal
generalisation, and never claim stratified-random results simulate live streaming.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data_loader import class_distribution
from .domain_utils import get_registrable_domain

logger = logging.getLogger(__name__)

DEDUP_COLUMNS = [
    "timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "proto", "query", "qtype_name",
    "rcode_name", "answers_key", "ttls_key", "rejected", "aa", "tc", "rd", "ra",
]


def _list_key(value) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(list(value), sort_keys=False)


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove records with identical content *and* label (first occurrence kept).

    "Identical" = all canonical Zeek fields equal (timestamp, endpoints, query, types,
    answers, TTLs, flags). Content that appears with *both* labels is NOT removed (it
    would silently pick a label) but is counted as ``label_conflict_events``.
    """
    n_before = len(df)
    tmp = df.copy()
    tmp["answers_key"] = tmp["answers"].map(_list_key)
    tmp["ttls_key"] = tmp["ttls"].map(_list_key)
    keep_cols = DEDUP_COLUMNS + ["label"]
    dup_mask = tmp.duplicated(subset=keep_cols, keep="first")
    removed_by_label = df.loc[dup_mask, "label"].value_counts().to_dict()
    deduped = df.loc[~dup_mask].reset_index(drop=True)
    tmp = tmp.loc[~dup_mask]
    conflicts = int(tmp.duplicated(subset=DEDUP_COLUMNS, keep=False).sum())
    report = {
        "n_before": n_before,
        "n_duplicates_removed": int(dup_mask.sum()),
        "removed_benign": int(removed_by_label.get(0, 0)),
        "removed_tunnel": int(removed_by_label.get(1, 0)),
        "n_after": len(deduped),
        "label_conflict_events": conflicts,
        "definition": "identical canonical fields + identical label; first occurrence kept",
    }
    logger.info("Dedup: removed %d identical records (%d -> %d); label-conflict events: %d",
                report["n_duplicates_removed"], n_before, len(deduped), conflicts)
    return deduped, report


def _check_both_classes(train: pd.DataFrame, test: pd.DataFrame, strategy: str) -> None:
    for name, part in (("train", train), ("test", test)):
        present = {int(v) for v in part["label"].unique()}
        if present != {0, 1}:
            raise ValueError(
                f"The {name} subset has classes {sorted(present)} after '{strategy}' split; both "
                "classes are required. If benign.json and tunnel.json were captured in different "
                "time ranges, use split.chronological_mode=per_class or a stratified_random split. "
                "For group_sld the dataset needs several distinct SLDs per class."
            )


def _group_sld_test_mask(df: pd.DataFrame, test_size: float, random_seed: int) -> np.ndarray:
    """Boolean mask of test events: whole SLD groups are assigned to the test subset.

    Groups are visited in a seeded random order and added to the test set while neither class
    would exceed ``test_size`` of its events (greedy, per-class target). If a class would get no
    test event at all (few/huge domains), the smallest group containing that class is moved to
    test (a warning is logged because the test fraction of that class then exceeds the target).
    """
    slds = df["query"].map(get_registrable_domain)
    counts = pd.crosstab(slds, df["label"]).reindex(columns=[0, 1], fill_value=0)
    names = counts.index.to_numpy()
    c = {0: counts[0].to_numpy(), 1: counts[1].to_numpy()}
    target = {lbl: test_size * float(c[lbl].sum()) for lbl in (0, 1)}
    got = {0: 0, 1: 0}
    chosen = np.zeros(len(names), dtype=bool)
    rng = np.random.default_rng(random_seed)
    for i in rng.permutation(len(names)):
        if got[0] + c[0][i] <= target[0] and got[1] + c[1][i] <= target[1]:
            chosen[i] = True
            got[0] += int(c[0][i])
            got[1] += int(c[1][i])
    for lbl in (0, 1):
        if got[lbl] == 0:
            candidates = np.where((c[lbl] > 0) & ~chosen)[0]
            if len(candidates):
                pick = candidates[np.argmin(c[lbl][candidates])]
                chosen[pick] = True
                got[0] += int(c[0][pick])
                got[1] += int(c[1][pick])
                logger.warning(
                    "group_sld: class %d would have no test events; moved SLD '%s' (%d events) to test "
                    "(class test fraction exceeds %.0f%%).", lbl, names[pick], int(c[lbl][pick]), 100 * test_size)
    return slds.isin(set(names[chosen])).to_numpy()


def _chronological_order(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["timestamp", "event_id"], kind="mergesort")


def split_events(
    df: pd.DataFrame,
    strategy: str = "stratified_random",
    test_size: float = 0.30,
    random_seed: int = 42,
    chronological_mode: str = "global",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(train, test)`` event tables (indices reset, original row order kept)."""
    if strategy == "stratified_random":
        positions = np.arange(len(df))
        train_pos, test_pos = train_test_split(
            positions, test_size=test_size, random_state=random_seed, shuffle=True,
            stratify=df["label"].to_numpy(),
        )
        train = df.iloc[np.sort(train_pos)].reset_index(drop=True)
        test = df.iloc[np.sort(test_pos)].reset_index(drop=True)
    elif strategy == "group_sld":
        mask = _group_sld_test_mask(df, test_size, random_seed)
        train = df.loc[~mask].reset_index(drop=True)
        test = df.loc[mask].reset_index(drop=True)
    elif strategy == "chronological":
        if chronological_mode == "global":
            ordered = _chronological_order(df)
            n_train = int(np.floor(len(ordered) * (1.0 - test_size)))
            train = ordered.iloc[:n_train].reset_index(drop=True)
            test = ordered.iloc[n_train:].reset_index(drop=True)
        elif chronological_mode == "per_class":
            trains, tests = [], []
            for _, part in df.groupby("label"):
                ordered = _chronological_order(part)
                n_train = int(np.floor(len(ordered) * (1.0 - test_size)))
                trains.append(ordered.iloc[:n_train])
                tests.append(ordered.iloc[n_train:])
            train = _chronological_order(pd.concat(trains)).reset_index(drop=True)
            test = _chronological_order(pd.concat(tests)).reset_index(drop=True)
        else:
            raise ValueError(f"Unknown chronological_mode {chronological_mode!r}")
    else:
        raise ValueError(f"Unknown split strategy {strategy!r}")
    _check_both_classes(train, test, strategy)
    return train, test


def build_manifest(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Manifest with event IDs and subset (plus tracking columns; none are features)."""
    cols = ["event_id", "label", "source_file", "source_line", "timestamp"]
    m = pd.concat(
        [train[cols].assign(subset="train"), test[cols].assign(subset="test")], ignore_index=True
    )
    return m[["event_id", "subset", "label", "source_file", "source_line", "timestamp"]]


def apply_manifest(df: pd.DataFrame, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-create a previous split from a manifest (by ``event_id``)."""
    subset_by_id = dict(zip(manifest["event_id"], manifest["subset"]))
    subsets = df["event_id"].map(subset_by_id)
    n_missing = int(subsets.isna().sum())
    if n_missing:
        raise ValueError(
            f"{n_missing} events are not present in the split manifest; the manifest does not "
            "match the loaded data."
        )
    bad = set(subsets.unique()) - {"train", "test"}
    if bad:
        raise ValueError(f"Manifest contains unknown subset values: {sorted(bad)}")
    train = df.loc[subsets == "train"].reset_index(drop=True)
    test = df.loc[subsets == "test"].reset_index(drop=True)
    _check_both_classes(train, test, "manifest")
    return train, test


def _subset_summary(part: pd.DataFrame) -> dict:
    summary = class_distribution(part)
    summary["time_min"] = float(part["timestamp"].min()) if len(part) else None
    summary["time_max"] = float(part["timestamp"].max()) if len(part) else None
    return summary


def summarize_split(train: pd.DataFrame, test: pd.DataFrame, strategy: str, test_size: float,
                    random_seed: int) -> dict:
    n = len(train) + len(test)
    return {
        "strategy": strategy,
        "requested_test_size": test_size,
        "actual_test_fraction": len(test) / n if n else None,
        "random_seed": random_seed,
        "train": _subset_summary(train),
        "test": _subset_summary(test),
        "overlap_event_ids": int(len(set(train["event_id"]) & set(test["event_id"]))),
    }
