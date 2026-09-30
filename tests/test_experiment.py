"""End-to-end smoke tests on SYNTHETIC data (not research results)."""

import copy

import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_CONFIG, validate_config
from src.experiment import RESULT_COLUMNS, run_experiments

REQUIRED_COLUMNS = [
    "experiment_id", "model", "approach", "window_seconds", "split_strategy", "window_history_policy",
    "balancing_method", "sampling_strategy", "random_seed", "n_train_original", "n_train_resampled",
    "n_test", "n_features", "accuracy", "precision", "recall", "f1", "roc_auc", "average_precision",
    "false_positive_rate", "tn", "fp", "fn", "tp", "feature_extraction_seconds", "training_seconds",
    "inference_seconds", "inference_ms_per_event",
]


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory, synthetic_files):
    root = tmp_path_factory.mktemp("runs")
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["input"].update(benign_path=str(synthetic_files[0]), tunnel_path=str(synthetic_files[1]))
    cfg["data_origin"] = "synthetic_smoke_test"
    cfg["windows"]["sizes"] = [5, 60]
    cfg["outputs"]["save_models"] = True
    cfg["outputs"]["save_plots"] = False
    p = cfg["models"]["params"]
    p["random_forest"]["n_estimators"] = 20
    p["lightgbm"]["n_estimators"] = 20
    p["xgboost"]["n_estimators"] = 20
    p["mlp"].update({"hidden_layer_sizes": [16], "max_iter": 30})
    outs = []
    for name in ("run_a", "run_b"):
        c = copy.deepcopy(cfg)
        c["output_dir"] = str(root / name)
        outs.append((c, run_experiments(c)))
    return outs


def test_24_style_grid_and_columns(two_runs):
    cfg, res = two_runs[0]
    # 4 classifiers x (baseline + 2 windows) in this reduced run
    assert len(res) == 4 * 3
    assert set(REQUIRED_COLUMNS) <= set(res.columns) and list(res.columns) == RESULT_COLUMNS
    base = res[res.approach == "baseline"]
    assert (base.window_seconds == 0).all() and len(base) == 4
    sw = res[res.approach == "sliding_window"]
    assert sorted(sw.window_seconds.unique()) == [5, 60]
    assert set(res.model) == {"random_forest", "lightgbm", "xgboost", "mlp"}
    assert (res.split_strategy == "stratified_random").all()
    assert (res.balancing_method == "random_oversampling").all()
    assert res.experiment_id.is_unique


def test_full_grid_size_is_24_for_default_windows():
    cfg = validate_config(copy.deepcopy(DEFAULT_CONFIG))
    n = len(cfg["models"]["enabled"]) * (int(cfg["run_baseline"]) + len(cfg["windows"]["sizes"]))
    assert n == 24


def test_same_split_for_all_models_and_windows(two_runs):
    cfg, res = two_runs[0]
    assert res.n_test.nunique() == 1 and res.n_train_original.nunique() == 1
    from pathlib import Path

    out = Path(cfg["output_dir"])
    manifest = pd.read_csv(out / "split_manifest.csv")
    test_ids = set(manifest.loc[manifest.subset == "test", "event_id"])
    for exp in res.experiment_id:
        preds = pd.read_csv(out / "experiments" / exp / "predictions.csv")
        assert set(preds["event_id"]) == test_ids       # same test events everywhere
        assert len(preds) == res.n_test.iloc[0]          # one prediction per test event
        assert preds["event_id"].is_unique


def test_test_set_not_resampled_and_training_is_balanced(two_runs):
    _, res = two_runs[0]
    assert (res.tn + res.fp + res.fn + res.tp == res.n_test).all()
    assert (res.n_train_resampled > res.n_train_original).all()
    assert (res.train_neg_after == res.train_pos_after).all()
    assert (res.train_pos_before < res.train_neg_before).all()


def test_reproducible_with_same_seed(two_runs):
    (cfg_a, a), (cfg_b, b) = two_runs
    from pathlib import Path

    pd.testing.assert_frame_equal(
        pd.read_csv(Path(cfg_a["output_dir"]) / "split_manifest.csv"),
        pd.read_csv(Path(cfg_b["output_dir"]) / "split_manifest.csv"),
    )
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision",
                   "tn", "fp", "fn", "tp", "n_train_resampled"]
    pd.testing.assert_frame_equal(a[["experiment_id"] + metric_cols], b[["experiment_id"] + metric_cols])
    pa = pd.read_csv(Path(cfg_a["output_dir"]) / "experiments" / "baseline_w00_random_forest" / "predictions.csv")
    pb = pd.read_csv(Path(cfg_b["output_dir"]) / "experiments" / "baseline_w00_random_forest" / "predictions.csv")
    np.testing.assert_allclose(pa["proba_tunnel"], pb["proba_tunnel"])


def test_artifacts_are_written(two_runs):
    from pathlib import Path

    cfg, res = two_runs[0]
    out = Path(cfg["output_dir"])
    for name in ("results.csv", "dataset_summary.json", "split_manifest.csv", "config_used.yaml",
                 "environment.json", "RUN_NOTES.md"):
        assert (out / name).is_file(), name
    exp = out / "experiments" / "sliding_window_w05_random_forest"
    for name in ("model_params.json", "feature_schema.json", "classification_report.txt",
                 "confusion_matrix.csv", "metrics.json", "model_pipeline.joblib", "feature_importance.csv"):
        assert (exp / name).is_file(), name
    assert not (out / "experiments" / "baseline_w00_mlp" / "feature_importance.csv").exists()
    import json

    summary = json.loads((out / "dataset_summary.json").read_text())
    assert summary["class_distribution_loaded"]["benign"] == 700
    assert summary["class_distribution_loaded"]["ratio_benign_to_tunnel"] == 7.0
    assert "synthetic" in summary["warning"].lower()
    assert summary["suffix_list"]["library"] == "tldextract"


def test_existing_output_dir_is_not_overwritten(two_runs):
    cfg, _ = two_runs[0]
    with pytest.raises(FileExistsError):
        run_experiments(copy.deepcopy(cfg))


def test_window_features_differ_from_baseline_feature_count(two_runs):
    _, res = two_runs[0]
    base = res[(res.approach == "baseline")].n_features.iloc[0]
    win = res[(res.approach == "sliding_window")].n_features.iloc[0]
    assert win == base + 21
