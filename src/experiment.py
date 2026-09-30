"""Experiment orchestration: load -> dedupe -> split -> features -> train/evaluate -> save.

Order of operations (leakage-safe):

1. load files, label from source file, validation report
2. de-duplicate identical records
3. split train/test ONCE (manifest saved, reused by every model and window)
4. stateless per-event features for train and test separately
5. window features: train windows from TRAIN events only, test windows from TEST events
   only (``split_isolated``); optional ``train_carryover`` for chronological splits
6. per model/config: imblearn Pipeline (impute/scale/encode -> resample -> classifier)
   fit on TRAIN only; evaluate on the untouched test set
"""

from __future__ import annotations

import logging
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from .config import validate_config
from .data_loader import class_distribution, load_dataset
from .domain_utils import configure_domain_extraction, suffix_list_info
from .evaluation import (
    compute_metrics,
    dump_json,
    make_classification_report,
    plot_confusion_matrix,
    plot_metric_comparison,
    plot_pr_curve,
    plot_roc_curve,
    positive_class_proba,
    save_feature_importance,
    select_threshold_on_validation,
    transform_before_classifier,
)
from .feature_extraction import (
    FORBIDDEN_FEATURE_COLUMNS,
    add_event_features,
    get_event_feature_groups,
)
from .models import build_pipeline, resolve_class_weight_policy
from .preprocessing import get_final_feature_names
from .split import (
    apply_manifest,
    build_manifest,
    deduplicate,
    split_events,
    summarize_split,
)
from .window_features import (
    WINDOW_FEATURE_COLUMNS,
    WINDOW_INPUT_COLUMNS,
    compute_window_features,
    compute_window_features_with_history,
)

logger = logging.getLogger(__name__)

RESULT_COLUMNS = [
    "experiment_id", "model", "approach", "window_seconds", "split_strategy",
    "window_history_policy", "balancing_method", "sampling_strategy", "random_seed",
    "n_train_original", "n_train_resampled", "n_test", "n_features", "accuracy", "precision",
    "recall", "f1", "roc_auc", "average_precision", "false_positive_rate", "tn", "fp", "fn", "tp",
    "feature_extraction_seconds", "training_seconds", "inference_seconds", "inference_ms_per_event",
    # extras
    "n_features_transformed", "threshold", "threshold_mode", "class_weight_policy",
    "preprocessing_seconds", "model_inference_seconds", "feature_extraction_test_seconds",
    "batch_end_to_end_ms_per_event", "tuning_enabled", "tuning_seconds", "roc_auc_note",
    "train_neg_before", "train_pos_before", "train_neg_after", "train_pos_after",
    "data_origin", "experiment_dir",
]


@dataclass
class EventTables:
    train: pd.DataFrame
    test: pd.DataFrame
    fe_train_seconds: float
    fe_test_seconds: float


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory {path} is not empty. Choose a new --output or pass --overwrite "
                "(experiments are never overwritten silently)."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _environment_info() -> dict:
    packages = ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "lightgbm", "xgboost",
                "tldextract", "PyYAML", "matplotlib", "joblib"]
    versions = {}
    for p in packages:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = "not installed"
    return {"python": sys.version, "platform": platform.platform(), "packages": versions}


def _window_task(window: float, train_in: pd.DataFrame, test_in: pd.DataFrame, policy: str):
    """Compute train/test window features for one window size (runs in a worker)."""
    t0 = time.perf_counter()
    train_w = compute_window_features(train_in, window)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    if policy == "train_carryover":
        test_w = compute_window_features_with_history(test_in, train_in, window)
    else:
        test_w = compute_window_features(test_in, window)
    test_s = time.perf_counter() - t0
    return window, train_w, test_w, train_s, test_s


def _experiment_id(approach: str, window: int, model: str) -> str:
    return f"{approach}_w{int(window):02d}_{model}"


def _feature_groups_for(approach: str, include_categorical: bool) -> tuple[list[str], list[str], list[str]]:
    cont, binary, cat = get_event_feature_groups(include_categorical)
    if approach == "sliding_window":
        cont = cont + WINDOW_FEATURE_COLUMNS
    overlap = FORBIDDEN_FEATURE_COLUMNS & set(cont + binary + cat)
    if overlap:  # defensive: metadata must never become a feature
        raise AssertionError(f"Forbidden metadata used as features: {sorted(overlap)}")
    return cont, binary, cat


def run_single_experiment(
    *,
    cfg: dict,
    model_name: str,
    approach: str,
    window_seconds: int,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    test_event_ids: np.ndarray,
    groups: tuple[list[str], list[str], list[str]],
    fe_seconds: float,
    fe_test_seconds: float,
    experiment_dir: Path,
    history_policy: str,
    n_jobs: int,
) -> dict:
    """Fit on TRAIN, evaluate on TEST, save all artefacts of one configuration."""
    seed = cfg["random_seed"]
    bal = cfg["balancing"]
    tun = cfg["tuning"]
    thr_cfg = cfg["threshold"]
    continuous, binary, categorical = groups
    experiment_id = experiment_dir.name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    train_counts = {int(k): int(v) for k, v in pd.Series(y_train).value_counts().items()}
    policy_text = resolve_class_weight_policy(
        bal["method"], bal["class_weight"], bool(bal["allow_class_weight_with_resampling"])
    )
    model_params = cfg["models"]["params"].get(model_name, {})
    cv_folds = int(tun["cv_folds"]) if tun["enabled"] else None

    def make_base():
        return build_pipeline(
            model_name, continuous=continuous, binary=binary, categorical=categorical,
            balancing=bal, model_params=model_params, random_seed=seed, n_jobs=n_jobs,
            train_class_counts=train_counts, cv_folds_for_sampler=cv_folds,
        )

    base = make_base()

    # -- optional hyperparameter tuning: CV on TRAIN only, resampling inside each fold
    tuning_seconds, best_params = 0.0, None
    if tun["enabled"]:
        grid = tun["param_grids"].get(model_name)
        if grid:
            n_iter = min(int(tun["n_iter"]), int(np.prod([len(v) for v in grid.values()])))
            search = RandomizedSearchCV(
                base, param_distributions=grid, n_iter=n_iter, scoring=tun["scoring"],
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed),
                random_state=seed, n_jobs=1, refit=False, error_score="raise",
            )
            t0 = time.perf_counter()
            search.fit(X_train, y_train)
            tuning_seconds = time.perf_counter() - t0
            best_params = search.best_params_
            base.set_params(**best_params)
        else:
            logger.warning("Tuning enabled but no param_grid for %s; using defaults.", model_name)

    # -- threshold: fixed, or chosen on a validation split of TRAIN (never on test)
    threshold_info: dict = {}
    if thr_cfg["mode"] == "validation_f1":
        threshold, threshold_info = select_threshold_on_validation(
            lambda: clone(base), X_train, y_train, float(thr_cfg["validation_size"]), seed
        )
    else:
        threshold = float(thr_cfg["value"])

    # -- final fit on the whole training set
    pipeline = clone(base)
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    training_seconds = time.perf_counter() - t0

    # -- inference timing: preprocessing and model prediction measured separately
    t0 = time.perf_counter()
    Xt = transform_before_classifier(pipeline, X_test)
    preprocessing_seconds = time.perf_counter() - t0
    clf = pipeline.named_steps["clf"]
    t0 = time.perf_counter()
    proba = positive_class_proba(clf, Xt)
    model_inference_seconds = time.perf_counter() - t0
    if len(proba) != len(y_test):
        raise AssertionError("Number of predictions differs from number of test events.")
    inference_seconds = preprocessing_seconds + model_inference_seconds

    metrics = compute_metrics(y_test, proba, threshold)
    y_pred = metrics.pop("y_pred")
    report_text, report_dict = make_classification_report(y_test, y_pred)

    sampler = pipeline.named_steps["sampler"]
    before = getattr(sampler, "class_counts_before_", train_counts)
    after = getattr(sampler, "class_counts_after_", train_counts)
    feature_names = get_final_feature_names(pipeline, continuous, binary, categorical)
    scaled = model_name == "mlp" or bal["method"] in ("smote", "smotenc")
    all_missing = [c for c in continuous + binary + categorical if X_train[c].isna().all()]

    # -- artefacts
    title = f"{model_name} | {approach} W={window_seconds}s"
    (experiment_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    pd.DataFrame(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        index=["true_benign", "true_tunnel"], columns=["pred_benign", "pred_tunnel"],
    ).to_csv(experiment_dir / "confusion_matrix.csv")
    pd.DataFrame({
        "event_id": test_event_ids, "y_true": y_test, "proba_tunnel": proba, "y_pred": y_pred,
    }).to_csv(experiment_dir / "predictions.csv", index=False)
    if cfg["outputs"]["save_plots"]:
        plot_confusion_matrix(metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"],
                              experiment_dir / "confusion_matrix.png", title)
        plot_roc_curve(y_test, proba, experiment_dir / "roc_curve.png", title)
        plot_pr_curve(y_test, proba, experiment_dir / "precision_recall_curve.png", title)
    save_feature_importance(clf, feature_names, experiment_dir, int(cfg["outputs"]["top_k_importance"]),
                            plot=bool(cfg["outputs"]["save_plots"]))
    dump_json({
        "classifier": clf.get_params(deep=False),
        "sampler": sampler.get_params(deep=False) if hasattr(sampler, "get_params") else "none",
        "preprocessing": {
            "continuous": "median imputation" + (" + standard scaling" if scaled else ""),
            "binary": "most-frequent imputation",
            "categorical": "constant 'MISSING' imputation + ordinal (unknown=-1) -> one-hot (unknown ignored)",
        },
        "class_weight_policy": policy_text,
        "tuning_best_params": best_params,
        "threshold": threshold, "threshold_selection": threshold_info,
    }, experiment_dir / "model_params.json")
    dump_json({
        "experiment_id": experiment_id, "approach": approach, "window_seconds": window_seconds,
        "window_history_policy": history_policy,
        "input_features": {"continuous": continuous, "binary": binary, "categorical": categorical},
        "input_feature_order": continuous + binary + categorical,
        "transformed_feature_names": feature_names,
        "features_entirely_missing_in_train": all_missing,
        "metadata_never_used_as_features": sorted(FORBIDDEN_FEATURE_COLUMNS),
        "note": "Window features require per-(src_ip, SLD) event history at inference; that state "
                "is NOT stored in the model artifact.",
    }, experiment_dir / "feature_schema.json")
    dump_json({**metrics, "classification_report": report_dict,
               "train_class_counts_before_resampling": before,
               "train_class_counts_after_resampling": after,
               "training_seconds": training_seconds, "preprocessing_seconds": preprocessing_seconds,
               "model_inference_seconds": model_inference_seconds,
               "inference_seconds": inference_seconds,
               "feature_extraction_seconds": fe_seconds}, experiment_dir / "metrics.json")
    if cfg["outputs"]["save_models"]:
        joblib.dump(pipeline, experiment_dir / "model_pipeline.joblib")

    n_test = len(y_test)
    row = {
        "experiment_id": experiment_id, "model": model_name, "approach": approach,
        "window_seconds": window_seconds, "split_strategy": cfg["split"]["strategy"],
        "window_history_policy": history_policy if approach == "sliding_window" else "not_applicable",
        "balancing_method": bal["method"], "sampling_strategy": str(bal["sampling_strategy"]),
        "random_seed": seed, "n_train_original": int(len(y_train)),
        "n_train_resampled": int(sum(after.values())), "n_test": n_test,
        "n_features": len(continuous) + len(binary) + len(categorical),
        "accuracy": metrics["accuracy"], "precision": metrics["precision"], "recall": metrics["recall"],
        "f1": metrics["f1"], "roc_auc": metrics["roc_auc"],
        "average_precision": metrics["average_precision"],
        "false_positive_rate": metrics["false_positive_rate"],
        "tn": metrics["tn"], "fp": metrics["fp"], "fn": metrics["fn"], "tp": metrics["tp"],
        "feature_extraction_seconds": fe_seconds, "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "inference_ms_per_event": inference_seconds * 1000.0 / n_test,
        "n_features_transformed": len(feature_names), "threshold": threshold,
        "threshold_mode": thr_cfg["mode"], "class_weight_policy": policy_text,
        "preprocessing_seconds": preprocessing_seconds,
        "model_inference_seconds": model_inference_seconds,
        "feature_extraction_test_seconds": fe_test_seconds,
        "batch_end_to_end_ms_per_event": (fe_test_seconds + inference_seconds) * 1000.0 / n_test,
        "tuning_enabled": bool(tun["enabled"]), "tuning_seconds": tuning_seconds,
        "roc_auc_note": metrics["roc_auc_note"],
        "train_neg_before": before.get(0, 0), "train_pos_before": before.get(1, 0),
        "train_neg_after": after.get(0, 0), "train_pos_after": after.get(1, 0),
        "data_origin": cfg["data_origin"], "experiment_dir": str(experiment_dir),
    }
    return row


def build_event_tables(train_raw: pd.DataFrame, test_raw: pd.DataFrame, availability: dict) -> EventTables:
    """Per-event features for train and test, timed separately (stateless, no fitting)."""
    t0 = time.perf_counter()
    train = add_event_features(train_raw, availability)
    fe_train = time.perf_counter() - t0
    t0 = time.perf_counter()
    test = add_event_features(test_raw, availability)
    fe_test = time.perf_counter() - t0
    return EventTables(train, test, fe_train, fe_test)


def _write_run_notes(cfg: dict, out: Path) -> None:
    lines = [
        "# Run notes", "",
        f"* data_origin: `{cfg['data_origin']}`",
        f"* split strategy: `{cfg['split']['strategy']}` (test_size={cfg['split']['test_size']}, seed={cfg['random_seed']})",
        f"* window history policy: `{cfg['windows']['history_policy']}`",
        f"* balancing: `{cfg['balancing']['method']}` (sampling_strategy={cfg['balancing']['sampling_strategy']})",
        f"* threshold mode: `{cfg['threshold']['mode']}`", "",
    ]
    if cfg["data_origin"] != "unspecified":
        lines += [f"> **Data origin = {cfg['data_origin']}.** If this is synthetic/smoke-test data, "
                  "the numbers are examples of the pipeline running, NOT research results.", ""]
    lines += [
        "Caveats: results come from ONE split and ONE seed; configurations must not be ranked by test "
        "score and then reported as final without an independent evaluation set. Do not rank by "
        "accuracy alone (imbalanced data). Stratified-random results do not simulate live streaming.",
        "Window features need per-(src_ip, SLD) history at inference time that is NOT part of the "
        "saved model artifact. Timings: `inference_seconds` = preprocessing + model predict_proba on "
        "the test matrix (excludes feature extraction); it is not streaming latency.",
    ]
    (out / "RUN_NOTES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiments(cfg: dict, overwrite: bool = False) -> pd.DataFrame:
    """Run all configured experiments and return the results table."""
    cfg = validate_config(cfg)
    out = Path(cfg["output_dir"])
    _prepare_output_dir(out, overwrite)
    seed = cfg["random_seed"]
    n_jobs = cfg["runtime"]["n_jobs"]
    history_policy = cfg["windows"]["history_policy"]

    # fail early on incompatible balancing / class-weight settings
    resolve_class_weight_policy(cfg["balancing"]["method"], cfg["balancing"]["class_weight"],
                                bool(cfg["balancing"]["allow_class_weight_with_resampling"]))
    configure_domain_extraction(cfg["domain"]["include_private_suffixes"], cfg["domain"]["local_suffixes"])
    with (out / "config_used.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
    dump_json(_environment_info(), out / "environment.json")

    # 1-2. load + dedupe ------------------------------------------------------
    events, reports, availability = load_dataset(cfg)
    loaded_dist = class_distribution(events)
    logger.info("Loaded %d valid records; class distribution: %s", len(events), loaded_dist)
    dedup_report = {"enabled": False}
    if cfg["dedupe"]["enabled"]:
        events, dedup_report = deduplicate(events)
        dedup_report["enabled"] = True
    deduped_dist = class_distribution(events)

    # 3. split (once) ---------------------------------------------------------
    manifest_path = cfg["split"]["manifest_path"]
    if manifest_path:
        train_raw, test_raw = apply_manifest(events, pd.read_csv(manifest_path))
    else:
        train_raw, test_raw = split_events(
            events, cfg["split"]["strategy"], cfg["split"]["test_size"], seed,
            cfg["split"]["chronological_mode"],
        )
    manifest = build_manifest(train_raw, test_raw)
    manifest.to_csv(out / "split_manifest.csv", index=False)
    split_summary = summarize_split(train_raw, test_raw, cfg["split"]["strategy"],
                                    cfg["split"]["test_size"], seed)
    logger.info("Split: train=%s test=%s", split_summary["train"], split_summary["test"])

    # 4. per-event features ---------------------------------------------------
    tables = build_event_tables(train_raw, test_raw, availability)
    y_train = tables.train["label"].to_numpy()
    y_test = tables.test["label"].to_numpy()
    test_ids = tables.test["event_id"].to_numpy()

    kinds = pd.concat([tables.train["domain_kind"], tables.test["domain_kind"]]).value_counts().to_dict()
    n_groups = int(pd.concat([tables.train, tables.test]).groupby(["src_ip", "sld"]).ngroups)
    dump_json({
        "data_origin": cfg["data_origin"],
        "warning": ("SYNTHETIC/SMOKE-TEST DATA: metrics are not research results."
                    if "synthetic" in str(cfg["data_origin"]).lower() else None),
        "files": {k: v.to_dict() for k, v in reports.items()},
        "class_distribution_loaded": loaded_dist,
        "class_ratio_note": "Actual ratio is reported; data is never rebalanced to 7:1 automatically.",
        "field_availability": availability,
        "deduplication": dedup_report,
        "class_distribution_after_dedup": deduped_dist,
        "split": split_summary,
        "domain_kind_counts": kinds,
        "n_src_ip_sld_groups": n_groups,
        "suffix_list": suffix_list_info(),
        "split_limitations": "Stratified random split ignores temporal dependence and repeated domains; "
                             "see README. Window features are computed separately on train and test.",
    }, out / "dataset_summary.json")
    _write_run_notes(cfg, out)

    # 5. window features ------------------------------------------------------
    windows = list(cfg["windows"]["sizes"])
    window_data: dict[int, dict] = {}
    if windows and cfg["models"]["enabled"]:
        train_in = tables.train[WINDOW_INPUT_COLUMNS]
        test_in = tables.test[WINDOW_INPUT_COLUMNS]
        logger.info("Computing window features for W=%s (policy=%s, n_jobs=%s)", windows, history_policy, n_jobs)
        outputs = Parallel(n_jobs=n_jobs)(
            delayed(_window_task)(w, train_in, test_in, history_policy) for w in windows
        )
        for w, train_w, test_w, s_train, s_test in outputs:
            window_data[w] = {"train": train_w, "test": test_w, "seconds_train": s_train, "seconds_test": s_test}

    # 6. train + evaluate -----------------------------------------------------
    configs: list[tuple[str, int]] = []
    if cfg["run_baseline"]:
        configs.append(("baseline", 0))
    configs += [("sliding_window", w) for w in windows]

    include_cat = bool(cfg["features"]["include_categorical"])
    rows: list[dict] = []
    for approach, w in configs:
        groups = _feature_groups_for(approach, include_cat)
        ev_cont, ev_bin, ev_cat = get_event_feature_groups(include_cat)
        event_cols = ev_cont + ev_bin + ev_cat
        X_train = tables.train[event_cols]
        X_test = tables.test[event_cols]
        fe_seconds = tables.fe_train_seconds + tables.fe_test_seconds
        fe_test_seconds = tables.fe_test_seconds
        if approach == "sliding_window":
            wd = window_data[w]
            X_train = pd.concat([X_train, wd["train"]], axis=1)
            X_test = pd.concat([X_test, wd["test"]], axis=1)
            fe_seconds += wd["seconds_train"] + wd["seconds_test"]
            fe_test_seconds += wd["seconds_test"]
        for model_name in cfg["models"]["enabled"]:
            exp_dir = out / "experiments" / _experiment_id(approach, w, model_name)
            logger.info("Running %s", exp_dir.name)
            row = run_single_experiment(
                cfg=cfg, model_name=model_name, approach=approach, window_seconds=w,
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                test_event_ids=test_ids, groups=groups, fe_seconds=fe_seconds,
                fe_test_seconds=fe_test_seconds, experiment_dir=exp_dir,
                history_policy=history_policy, n_jobs=n_jobs,
            )
            rows.append(row)
            logger.info("  f1=%.4f recall=%.4f fpr=%.4f roc_auc=%s", row["f1"], row["recall"],
                        row["false_positive_rate"], f"{row['roc_auc']:.4f}" if row["roc_auc"] == row["roc_auc"] else "n/a")

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    results.to_csv(out / "results.csv", index=False)
    if cfg["outputs"]["save_plots"]:
        plot_metric_comparison(results, out / "figures")
    logger.info("Done. Results: %s", out / "results.csv")
    return results
