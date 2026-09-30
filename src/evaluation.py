"""Metrics, threshold selection (on validation data from TRAIN only) and plots."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split  # noqa: E402

logger = logging.getLogger(__name__)

TARGET_NAMES = ["benign", "dns_tunneling"]


def positive_class_proba(classifier, X) -> np.ndarray:
    """Probability of class 1 (DNS tunneling), located via ``classes_`` (never assumed)."""
    classes = list(classifier.classes_)
    if 1 not in classes:
        raise ValueError(f"Classifier was not trained with positive class 1 (classes_={classes}).")
    return classifier.predict_proba(X)[:, classes.index(1)]


def compute_metrics(y_true: Sequence[int], proba: Sequence[float], threshold: float = 0.5) -> dict:
    """All scalar metrics. Positive class = 1; ROC-AUC/AP use probabilities, not labels."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    roc_auc, roc_note = float("nan"), ""
    if n_pos == 0 or n_neg == 0:
        roc_note = "undefined: test set contains only one class"
    else:
        roc_auc = float(roc_auc_score(y_true, proba))
    ap, ap_note = float("nan"), ""
    if n_pos == 0:
        ap_note = "undefined: no positive samples in test set"
    else:
        ap = float(average_precision_score(y_true, proba))

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": roc_auc,
        "roc_auc_note": roc_note,
        "average_precision": ap,
        "average_precision_note": ap_note,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "y_pred": y_pred,
    }


def make_classification_report(y_true, y_pred) -> tuple[str, dict]:
    kwargs = dict(labels=[0, 1], target_names=TARGET_NAMES, zero_division=0)
    return (
        classification_report(y_true, y_pred, **kwargs),
        classification_report(y_true, y_pred, output_dict=True, **kwargs),
    )


def select_threshold_on_validation(
    make_pipeline: Callable[[], object],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    validation_size: float,
    random_seed: int,
) -> tuple[float, dict]:
    """Pick the F1-maximising threshold using a validation split carved out of TRAIN.

    A fresh pipeline is fit on the remaining train part (resampling applies there only);
    the validation part is never resampled and the test set is never touched.
    """
    idx = np.arange(len(y_train))
    fit_idx, val_idx = train_test_split(
        idx, test_size=validation_size, random_state=random_seed, stratify=y_train, shuffle=True
    )
    pipe = make_pipeline()
    pipe.fit(X_train.iloc[fit_idx], y_train[fit_idx])  # type: ignore[attr-defined]
    proba = positive_class_proba(pipe.named_steps["clf"], _transform_before_clf(pipe, X_train.iloc[val_idx]))  # type: ignore[attr-defined]
    prec, rec, thr = precision_recall_curve(y_train[val_idx], proba)
    if len(thr) == 0:
        return 0.5, {"note": "degenerate validation set; fell back to 0.5"}
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    best = int(np.argmax(f1))
    return float(thr[best]), {"validation_f1": float(f1[best]), "n_validation": int(len(val_idx))}


def _transform_before_clf(pipeline, X):
    """Apply every fitted step except the sampler and the final classifier."""
    Xt = X
    for name, step in pipeline.steps[:-1]:
        if name == "sampler" or step is None or step == "passthrough":
            continue
        Xt = step.transform(Xt)
    return Xt


transform_before_classifier = _transform_before_clf


# ----------------------------------------------------------------------------- plots
def plot_confusion_matrix(tn: int, fp: int, fn: int, tp: int, path: Path, title: str) -> None:
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred benign", "pred tunnel"])
    ax.set_yticks([0, 1], labels=["true benign", "true tunnel"])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="black")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_roc_curve(y_true, proba, path: Path, title: str) -> bool:
    if len(np.unique(y_true)) < 2:
        return False
    fpr, tpr, _ = roc_curve(y_true, proba, pos_label=1)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, proba):.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def plot_pr_curve(y_true, proba, path: Path, title: str) -> bool:
    if int((np.asarray(y_true) == 1).sum()) == 0:
        return False
    prec, rec, _ = precision_recall_curve(y_true, proba, pos_label=1)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(rec, prec, label=f"AP={average_precision_score(y_true, proba):.4f}")
    ax.axhline(float(np.mean(y_true)), color="k", linestyle="--", linewidth=0.8, label="prevalence")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def save_feature_importance(
    classifier, feature_names: Sequence[str], out_dir: Path, top_k: int = 20, plot: bool = True
) -> Optional[pd.DataFrame]:
    """Save ``feature_importance.csv`` (+png) for models exposing ``feature_importances_``."""
    importances = getattr(classifier, "feature_importances_", None)
    if importances is None:
        return None
    importances = np.asarray(importances, dtype=float)
    if len(importances) != len(feature_names):
        logger.warning("Importance length %d != #features %d; skipped.", len(importances), len(feature_names))
        return None
    df = (
        pd.DataFrame({"feature": list(feature_names), "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    df.to_csv(out_dir / "feature_importance.csv", index=False)
    if plot:
        top = df.head(top_k).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6, max(3, 0.28 * len(top) + 1)))
        ax.barh(top["feature"], top["importance"])
        ax.set_xlabel("Importance (model-specific; not comparable across model types)")
        fig.tight_layout()
        fig.savefig(out_dir / "feature_importance.png", dpi=130)
        plt.close(fig)
    return df


COMPARISON_METRICS = [
    ("f1", "F1 (tunnel = positive)"),
    ("recall", "Recall (tunnel)"),
    ("precision", "Precision (tunnel)"),
    ("false_positive_rate", "False positive rate"),
    ("roc_auc", "ROC-AUC"),
    ("average_precision", "Average precision (PR-AUC)"),
    ("inference_ms_per_event", "Inference ms / event (preprocess + predict)"),
]


def plot_metric_comparison(results: pd.DataFrame, out_dir: Path) -> list[Path]:
    """One grouped bar chart per metric: x = baseline / window size, bars = classifier."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if results.empty:
        return paths
    configs = (
        results[["approach", "window_seconds"]].drop_duplicates()
        .sort_values(["window_seconds"]).itertuples(index=False)
    )
    config_list = [(a, int(w)) for a, w in configs]
    labels = ["baseline" if a == "baseline" else f"{w}s" for a, w in config_list]
    models = list(dict.fromkeys(results["model"]))
    width = 0.8 / max(len(models), 1)
    for metric, title in COMPARISON_METRICS:
        if metric not in results.columns:
            continue
        fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 3, 4))
        for m_i, model in enumerate(models):
            vals = []
            for a, w in config_list:
                row = results[(results["model"] == model) & (results["approach"] == a) & (results["window_seconds"] == w)]
                vals.append(float(row[metric].iloc[0]) if len(row) else np.nan)
            ax.bar(np.arange(len(labels)) + m_i * width, vals, width, label=model)
        ax.set_xticks(np.arange(len(labels)) + width * (len(models) - 1) / 2, labels=labels)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"compare_{metric}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)
    return paths


def dump_json(obj, path: Path) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=default, allow_nan=True)
