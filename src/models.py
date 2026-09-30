"""Classifier factories and the full imbalanced-learn pipeline builder.

Positive class is always ``1`` (DNS tunneling). Every estimator gets an explicit
``random_state``. Class weighting and oversampling are never combined silently: see
:func:`resolve_class_weight_policy`.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Sequence

from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from .preprocessing import build_sampler, make_expander, make_preprocessor

logger = logging.getLogger(__name__)

MODEL_LABELS = {
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "mlp": "MLP",
}


def resolve_class_weight_policy(balancing_method: str, class_weight: str, allow_both: bool) -> str:
    """Return a human-readable policy string or raise if weighting+resampling is implicit."""
    if class_weight == "none":
        return "none"
    if balancing_method != "none" and not allow_both:
        raise ValueError(
            f"class_weight='{class_weight}' combined with balancing.method='{balancing_method}' would "
            "double-compensate for imbalance. Choose one, or set "
            "balancing.allow_class_weight_with_resampling=true to do it deliberately."
        )
    if balancing_method != "none":
        logger.warning("class_weight=%s AND %s are both active (explicitly allowed).", class_weight, balancing_method)
        return f"{class_weight}+{balancing_method} (explicitly allowed)"
    return class_weight


def build_classifier(
    name: str,
    params: Mapping[str, Any],
    random_seed: int,
    n_jobs: int = 1,
    class_weight: str = "none",
    n_neg: Optional[int] = None,
    n_pos: Optional[int] = None,
):
    """Create an unfitted classifier. ``params`` come from ``models.params.<name>``."""
    p = dict(params or {})
    if name == "random_forest":
        return RandomForestClassifier(
            random_state=random_seed, n_jobs=n_jobs,
            class_weight="balanced" if class_weight == "balanced" else None, **p,
        )
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:  # pragma: no cover
            raise ImportError("lightgbm is not installed: pip install lightgbm") from exc
        return LGBMClassifier(
            random_state=random_seed, n_jobs=n_jobs, verbose=-1, deterministic=True,
            force_row_wise=True, class_weight="balanced" if class_weight == "balanced" else None, **p,
        )
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover
            raise ImportError("xgboost is not installed: pip install xgboost") from exc
        extra: dict[str, Any] = {}
        if class_weight == "balanced":
            if not n_pos:
                raise ValueError("n_pos required for class_weight='balanced' with XGBoost.")
            extra["scale_pos_weight"] = float(n_neg) / float(n_pos)
        return XGBClassifier(
            random_state=random_seed, n_jobs=n_jobs, tree_method="hist", eval_metric="logloss",
            objective="binary:logistic", **extra, **p,
        )
    if name == "mlp":
        if class_weight == "balanced":
            logger.warning("MLPClassifier has no class_weight support; 'balanced' is ignored for mlp.")
        p["hidden_layer_sizes"] = tuple(p.get("hidden_layer_sizes", (128, 64)))
        return MLPClassifier(random_state=random_seed, **p)
    raise ValueError(f"Unknown model {name!r}")


def build_pipeline(
    model_name: str,
    *,
    continuous: Sequence[str],
    binary: Sequence[str],
    categorical: Sequence[str],
    balancing: Mapping[str, Any],
    model_params: Mapping[str, Any],
    random_seed: int,
    n_jobs: int,
    train_class_counts: Mapping[int, int],
    cv_folds_for_sampler: Optional[int] = None,
) -> Pipeline:
    """Build ``prep -> sampler -> expand -> clf``.

    ``train_class_counts`` are the class counts of the TRAINING set only; they are used
    to size SMOTE ``k_neighbors`` and XGBoost ``scale_pos_weight``.
    ``cv_folds_for_sampler`` shrinks the minority count estimate when the pipeline will be
    cross-validated (each training fold has fewer minority samples).
    """
    method = balancing["method"]
    class_weight = balancing.get("class_weight", "none")
    resolve_class_weight_policy(method, class_weight, bool(balancing.get("allow_class_weight_with_resampling", False)))

    n_min = int(min(train_class_counts.values()))
    if cv_folds_for_sampler:
        n_min = max(int(n_min * (cv_folds_for_sampler - 1) / cv_folds_for_sampler), 1)

    scale = model_name == "mlp" or method in ("smote", "smotenc")
    prep = make_preprocessor(continuous, binary, categorical, scale=scale)
    sampler = build_sampler(
        method, balancing["sampling_strategy"], random_seed,
        k_neighbors=int(balancing.get("smote_k_neighbors", 5)),
        n_continuous=len(continuous), n_binary=len(binary), n_categorical=len(categorical),
        n_minority=n_min,
    )
    expand = make_expander(len(continuous), len(binary), len(categorical))
    clf = build_classifier(
        model_name, model_params, random_seed, n_jobs=n_jobs, class_weight=class_weight,
        n_neg=int(train_class_counts.get(0, 0)), n_pos=int(train_class_counts.get(1, 0)),
    )
    return Pipeline([("prep", prep), ("sampler", sampler), ("expand", expand), ("clf", clf)])
