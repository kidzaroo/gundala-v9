"""Preprocessing ColumnTransformers and imbalance samplers.

Pipeline layout (imbalanced-learn ``Pipeline``; every fitted step sees TRAIN data only,
and samplers run only inside ``fit``)::

    prep     : impute (+ scale)  ->  columns ordered [continuous | binary | categorical(ordinal)]
    sampler  : none | RandomOverSampler | SMOTE | SMOTENC     (training data only)
    expand   : one-hot of the categorical columns (unseen test categories -> all zeros)
    clf      : classifier

Why ordinal first, one-hot after the sampler?  SMOTENC needs categorical columns to be
single integer-coded columns; one-hotting afterwards keeps every synthetic row a valid
category. Unseen categories at test time are coded ``-1`` by the ordinal encoder and then
ignored (all-zero) by the one-hot encoder.

SMOTE policy: plain SMOTE interpolates and therefore cannot represent nominal categorical
features. ``smote`` raises an informative error if nominal categorical columns are active
(set ``features.include_categorical: false`` or use ``smotenc``). Binary 0/1 flag columns
are treated as numeric by ``smote`` (synthetic values can be fractional) and as
categorical by ``smotenc``.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Optional, Sequence

import numpy as np
from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)


class _RecordMixin:
    """Record class counts before/after resampling (for reporting)."""

    def fit_resample(self, X, y, **params):  # type: ignore[override]
        self.class_counts_before_ = {int(k): int(v) for k, v in Counter(np.asarray(y).tolist()).items()}
        X_res, y_res = super().fit_resample(X, y, **params)  # type: ignore[misc]
        self.class_counts_after_ = {int(k): int(v) for k, v in Counter(np.asarray(y_res).tolist()).items()}
        return X_res, y_res


class RecordingRandomOverSampler(_RecordMixin, RandomOverSampler):
    pass


class RecordingSMOTE(_RecordMixin, SMOTE):
    pass


class RecordingSMOTENC(_RecordMixin, SMOTENC):
    pass


def make_preprocessor(
    continuous: Sequence[str],
    binary: Sequence[str],
    categorical: Sequence[str],
    scale: bool,
) -> ColumnTransformer:
    """Impute (+ optionally scale) and ordinal-encode. Fit on TRAIN only."""
    transformers: list = []
    if continuous:
        steps: list = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
        if scale:
            steps.append(("scale", StandardScaler()))
        transformers.append(("cont", SkPipeline(steps), list(continuous)))
    if binary:
        transformers.append((
            "bin",
            SkPipeline([("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True))]),
            list(binary),
        ))
    if categorical:
        transformers.append((
            "cat",
            SkPipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
                ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype="float64")),
            ]),
            list(categorical),
        ))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False, sparse_threshold=0.0)


def make_expander(n_continuous: int, n_binary: int, n_categorical: int) -> Any:
    """One-hot the categorical block (placed last by ``make_preprocessor``)."""
    if n_categorical == 0:
        return "passthrough"
    start = n_continuous + n_binary
    return ColumnTransformer(
        [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(range(start, start + n_categorical)))],
        remainder="passthrough",
        verbose_feature_names_out=False,
        sparse_threshold=0.0,
    )


def _safe_k_neighbors(k: int, n_minority: int) -> int:
    k_eff = min(int(k), int(n_minority) - 1)
    if k_eff < 1:
        raise ValueError(
            f"Not enough minority samples ({n_minority}) for SMOTE/SMOTENC (need at least 2). "
            "Use balancing.method=random_oversampling or none."
        )
    if k_eff != k:
        logger.warning("SMOTE k_neighbors reduced from %d to %d (minority class has %d samples).",
                       k, k_eff, n_minority)
    return k_eff


def build_sampler(
    method: str,
    sampling_strategy: Any,
    random_seed: int,
    *,
    k_neighbors: int = 5,
    n_continuous: int = 0,
    n_binary: int = 0,
    n_categorical: int = 0,
    n_minority: Optional[int] = None,
) -> Any:
    """Return an imblearn sampler (or ``"passthrough"``) for the given method."""
    if method == "none":
        return "passthrough"
    if method == "random_oversampling":
        return RecordingRandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_seed)
    if method not in ("smote", "smotenc"):
        raise ValueError(f"Unknown balancing method {method!r}")
    if n_minority is None:
        raise ValueError("n_minority is required to configure SMOTE safely.")
    k = _safe_k_neighbors(k_neighbors, n_minority)
    if method == "smote":
        if n_categorical > 0:
            raise ValueError(
                "balancing.method='smote' cannot handle nominal categorical features "
                "(qtype/rcode/proto): interpolating category codes is meaningless. Either set "
                "features.include_categorical=false to use SMOTE on numeric features only, or use "
                "balancing.method='smotenc'."
            )
        return RecordingSMOTE(sampling_strategy=sampling_strategy, k_neighbors=k, random_state=random_seed)
    cat_idx = list(range(n_continuous, n_continuous + n_binary + n_categorical))
    if not cat_idx:
        raise ValueError("balancing.method='smotenc' needs categorical/binary features; use 'smote' instead.")
    return RecordingSMOTENC(
        categorical_features=cat_idx, sampling_strategy=sampling_strategy, k_neighbors=k,
        random_state=random_seed,
    )


def get_final_feature_names(
    pipeline, continuous: Sequence[str], binary: Sequence[str], categorical: Sequence[str]
) -> list[str]:
    """Feature names seen by the classifier, in column order (after one-hot expansion)."""
    base = list(continuous) + list(binary)
    if not categorical:
        return base
    prep = pipeline.named_steps["prep"]
    encoder = prep.named_transformers_["cat"].named_steps["encode"]
    expand = pipeline.named_steps["expand"]
    onehot = expand.named_transformers_["onehot"]
    onehot_names: list[str] = []
    for col, labels, codes in zip(categorical, encoder.categories_, onehot.categories_):
        for code in codes:
            idx = int(code)
            label = labels[idx] if 0 <= idx < len(labels) else "UNKNOWN"
            onehot_names.append(f"{col}={label}")
    return onehot_names + base
