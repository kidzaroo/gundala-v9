"""Feature policy, artefact audits and the group_sld split."""

import copy
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_CONFIG, validate_config
from src.data_loader import load_dataset
from src.domain_utils import get_registrable_domain
from src.experiment import run_experiments
from src.feature_policy import (
    RESPONSE_PROTOCOL_FEATURES,
    leakage_audit,
    resolve_feature_selection,
    schema_artifact_audit,
)
from src.split import split_events
from src.synthetic_data import generate_records


# ------------------------------------------------------------------ feature selection
def test_query_only_drops_response_and_protocol_features():
    sel = resolve_feature_selection("sliding_window", feature_set="query_only")
    assert not set(sel.all_columns) & RESPONSE_PROTOCOL_FEATURES
    for keep in ("query_entropy", "query_length", "win_count", "win_iat_mean", "win_unique_queries"):
        assert keep in sel.all_columns
    assert sel.dropped["num_ttls"] == "features.feature_set=query_only"
    assert sel.categorical == []


def test_baseline_selection_has_no_window_columns_and_reports_only_relevant_drops():
    sel = resolve_feature_selection("baseline", feature_set="query_only")
    assert sel.window_columns == [] and not any(c.startswith("win_") for c in sel.all_columns)
    assert not any(k.startswith("win_") for k in sel.dropped)


def test_default_selection_keeps_everything():
    base = resolve_feature_selection("baseline")
    win = resolve_feature_selection("sliding_window")
    assert len(win.all_columns) == len(base.all_columns) + 21 and not base.dropped


def test_user_exclude_and_unknown_names():
    sel = resolve_feature_selection("sliding_window", exclude=["num_ttls", "win_ttl_mean"])
    assert "num_ttls" not in sel.all_columns and "win_ttl_mean" not in sel.all_columns
    assert sel.dropped["num_ttls"] == "features.exclude"
    with pytest.raises(ValueError, match="unknown feature names"):
        resolve_feature_selection("baseline", exclude=["not_a_feature"])


def test_artifact_fields_remove_derived_features_with_reason():
    sel = resolve_feature_selection("sliding_window", excluded_fields=["ttls", "rcode_name"])
    for name in ("num_ttls", "ttl_min", "ttl_max", "ttl_mean", "win_ttl_mean",
                 "rcode_cat", "is_nxdomain", "win_nxdomain_count", "win_nxdomain_ratio"):
        assert name not in sel.all_columns
        assert sel.dropped[name].startswith("schema_artifact:")
    assert "num_answers" in sel.all_columns


def test_include_categorical_false_and_empty_selection():
    assert resolve_feature_selection("baseline", include_categorical=False).categorical == []
    from src.feature_extraction import EVENT_BINARY, EVENT_CONTINUOUS

    with pytest.raises(ValueError, match="No features left"):
        resolve_feature_selection("baseline", feature_set="query_only",
                                  exclude=EVENT_CONTINUOUS + EVENT_BINARY)


def test_config_validation_of_new_options():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["split"]["strategy"] = "group_sld"
    assert validate_config(cfg)["split"]["strategy"] == "group_sld"
    for key, bad in (("feature_set", "nope"), ("schema_artifact_policy", "sometimes"),
                     ("schema_artifact_threshold", 0)):
        broken = copy.deepcopy(DEFAULT_CONFIG)
        broken["features"][key] = bad
        with pytest.raises(ValueError):
            validate_config(broken)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["split"]["strategy"] = "group_sld"
    cfg["windows"]["history_policy"] = "train_carryover"
    with pytest.raises(ValueError, match="chronological"):
        validate_config(cfg)


def test_cli_accepts_new_flags():
    import main

    args = main.build_parser().parse_args([
        "--feature-set", "query_only", "--exclude-features", "num_ttls", "qtype_cat",
        "--chronological-mode", "per_class", "--split-strategy", "group_sld",
        "--schema-artifact-policy", "warn",
    ])
    assert args.exclude_features == ["num_ttls", "qtype_cat"] and args.split_strategy == "group_sld"
    assert args.chronological_mode == "per_class" and args.schema_artifact_policy == "warn"


# ------------------------------------------------------------------ schema artefact audit
def _events_from(base_config):
    df, _, _ = load_dataset(base_config)
    return df


def test_schema_artifact_audit_flags_field_present_in_one_class_only(base_config):
    df = _events_from(base_config)
    clean = schema_artifact_audit(df, 0.5)
    assert clean["flagged_fields"] == [] and clean["basis"] == "training subset only"
    df.loc[df["label"] == 0, ["answers", "ttls"]] = None  # benign file lacks answers/TTLs
    audit = schema_artifact_audit(df, 0.5)
    assert {"answers", "ttls"} <= set(audit["flagged_fields"])
    assert audit["fields"]["answers"]["presence_benign"] == 0.0
    assert audit["fields"]["answers"]["presence_tunnel"] > 0.5
    assert "num_ttls" in audit["fields"]["ttls"]["features_affected"]


# ------------------------------------------------------------------ leakage audit
def _frame(n, rng, leak=False):
    y = rng.integers(0, 2, n)
    return pd.DataFrame({
        "event_id": [f"e{rng.integers(10**9)}_{i}" for i in range(n)],
        "label": y,
        "query": [f"q{rng.integers(0, 10**6)}.site{rng.integers(0, 10**6)}.com" for _ in range(n)],
        "sld": [f"site{rng.integers(0, 10**6)}.com" for _ in range(n)],
        "c0": (y * 10.0 + rng.normal(size=n) * 1e-3) if leak else rng.normal(size=n),
        "b0": rng.integers(0, 2, n).astype(float),
        "cat0": rng.choice(["A", "B", "C"], n),
    })


def test_leakage_audit_flags_single_feature_separator():
    rng = np.random.default_rng(1)
    rep = leakage_audit(_frame(400, rng, leak=True), _frame(150, rng, leak=True), ["c0"], ["b0"], ["cat0"])
    assert rep["top_single_feature_auc_train"]["c0"] > 0.99
    assert any("'c0'" in w for w in rep["warnings"])


def test_leakage_audit_clean_data_has_no_warnings():
    rng = np.random.default_rng(2)
    rep = leakage_audit(_frame(600, rng), _frame(200, rng), ["c0"], ["b0"], ["cat0"])
    assert rep["warnings"] == []
    assert set(rep["overlap_test_in_train"]["sld_overlap"]) <= {"benign", "tunnel"}


def test_leakage_audit_detects_identical_vectors_and_sld_overlap():
    rng = np.random.default_rng(3)
    train = _frame(300, rng)
    test = train.sample(100, random_state=0).copy()
    test["event_id"] = [f"t{i}" for i in range(len(test))]
    rep = leakage_audit(train, test, ["c0"], ["b0"], ["cat0"])
    frac = rep["duplicates"]["test_rows_with_identical_vector_in_train"]
    assert all(v == 1.0 for v in frac.values())
    assert any("identical to a training event" in w for w in rep["warnings"])
    assert any("SLD that also appears in training" in w for w in rep["warnings"])
    assert rep["overlap_test_in_train"]["query_overlap"]["benign"] == 1.0


def test_leakage_audit_flags_vectors_with_both_labels():
    rng = np.random.default_rng(4)
    train = _frame(100, rng)
    test = train.iloc[:20].copy()
    test["label"] = 1 - test["label"]
    rep = leakage_audit(train, test, ["c0"], ["b0"], ["cat0"])
    assert rep["duplicates"]["feature_vectors_with_both_labels"] >= 1


# ------------------------------------------------------------------ group_sld split
def test_group_sld_split_keeps_each_sld_and_window_group_on_one_side(base_config):
    df = _events_from(base_config)
    tr, te = split_events(df, "group_sld", 0.3, 42)
    sld = lambda d: set(d["query"].map(get_registrable_domain))  # noqa: E731
    assert not sld(tr) & sld(te)
    assert set(tr["label"]) == {0, 1} and set(te["label"]) == {0, 1}
    pairs = lambda d: set(zip(d["src_ip"], d["query"].map(get_registrable_domain)))  # noqa: E731
    assert not pairs(tr) & pairs(te)  # window groups (src_ip, SLD) are never cut in two
    assert len(tr) + len(te) == len(df) and not set(tr["event_id"]) & set(te["event_id"])
    tr2, te2 = split_events(df, "group_sld", 0.3, 42)
    assert te["event_id"].tolist() == te2["event_id"].tolist()


def test_group_sld_moves_smallest_group_when_class_would_get_no_test_events(caplog):
    rows = []
    for d in range(10):
        rows += [{"event_id": f"b{d}_{i}", "label": 0, "query": f"x{i}.benign{d}.com",
                  "timestamp": float(i), "src_ip": "10.0.0.1"} for i in range(50)]
    for d in range(2):
        rows += [{"event_id": f"t{d}_{i}", "label": 1, "query": f"x{i}.tunnel{d}.net",
                  "timestamp": float(i), "src_ip": "10.0.0.2"} for i in range(40)]
    df = pd.DataFrame(rows)
    with caplog.at_level(logging.WARNING):
        tr, te = split_events(df, "group_sld", 0.3, 7)
    assert (te["label"] == 1).sum() == 40 and (tr["label"] == 1).sum() == 40
    assert "would have no test events" in caplog.text


def test_group_sld_with_single_tunnel_domain_fails_informatively():
    rows = [{"event_id": f"b{d}_{i}", "label": 0, "query": f"x{i}.benign{d}.com", "timestamp": 0.0,
             "src_ip": "a"} for d in range(5) for i in range(20)]
    rows += [{"event_id": f"t{i}", "label": 1, "query": f"x{i}.only.net", "timestamp": 0.0, "src_ip": "b"}
             for i in range(30)]
    with pytest.raises(ValueError, match="distinct SLDs"):
        split_events(pd.DataFrame(rows), "group_sld", 0.3, 1)


# ------------------------------------------------------------------ integration
def _small_cfg(base_config, out, **features):
    cfg = copy.deepcopy(base_config)
    cfg["output_dir"] = str(out)
    cfg["models"]["enabled"] = ["random_forest"]
    cfg["windows"]["sizes"] = [5]
    cfg["features"].update(features)
    return cfg


def test_query_only_run_reports_policy_and_audit(base_config, tmp_path):
    res = run_experiments(_small_cfg(base_config, tmp_path / "q", feature_set="query_only"))
    full = run_experiments(_small_cfg(base_config, tmp_path / "f"))
    assert (res.n_features < full.n_features).all()
    schema = json.loads((tmp_path / "q/experiments/baseline_w00_random_forest/feature_schema.json").read_text())
    assert schema["features_dropped_by_policy"]["num_ttls"] == "features.feature_set=query_only"
    assert "categorical" in schema["input_features"] and schema["input_features"]["categorical"] == []
    summary = json.loads((tmp_path / "q/dataset_summary.json").read_text())
    assert {"leakage_audit", "schema_artifact_audit", "feature_policy"} <= set(summary)
    assert "top_single_feature_auc_train" in summary["leakage_audit"]
    assert "feature_set: `query_only`" in (tmp_path / "q/RUN_NOTES.md").read_text()


def test_schema_artifact_is_excluded_by_default_and_kept_when_off(base_config, tmp_path):
    benign, tunnel = generate_records(400, 60, seed=11)
    for rec in benign:  # benign capture lacks answers/TTLs entirely
        rec.pop("answers", None)
        rec.pop("TTLs", None)
    (tmp_path / "benign.json").write_text("\n".join(json.dumps(r) for r in benign))
    (tmp_path / "tunnel.json").write_text(json.dumps(tunnel))
    cfg = _small_cfg(base_config, tmp_path / "ex")
    cfg["input"].update(benign_path=str(tmp_path / "benign.json"), tunnel_path=str(tmp_path / "tunnel.json"))
    excluded = run_experiments(cfg)
    summary = json.loads((tmp_path / "ex/dataset_summary.json").read_text())
    audit = summary["schema_artifact_audit"]
    assert {"answers", "ttls"} <= set(audit["flagged_fields"]) and audit["policy"] == "exclude"
    schema = json.loads((tmp_path / "ex/experiments/baseline_w00_random_forest/feature_schema.json").read_text())
    assert schema["features_dropped_by_policy"]["num_answers"] == "schema_artifact:answers"
    assert "num_ttls" not in schema["input_feature_order"]

    cfg_off = _small_cfg(base_config, tmp_path / "off", schema_artifact_policy="warn")
    cfg_off["input"].update(benign_path=str(tmp_path / "benign.json"), tunnel_path=str(tmp_path / "tunnel.json"))
    kept = run_experiments(cfg_off)
    assert (kept.n_features > excluded.n_features).all()
    kept_schema = json.loads((tmp_path / "off/experiments/baseline_w00_random_forest/feature_schema.json").read_text())
    assert "num_ttls" in kept_schema["input_feature_order"]


def test_group_sld_end_to_end_reports_no_sld_overlap(base_config, tmp_path):
    cfg = _small_cfg(base_config, tmp_path / "g")
    cfg["split"]["strategy"] = "group_sld"
    res = run_experiments(cfg)
    assert (res.split_strategy == "group_sld").all() and res.n_test.nunique() == 1
    summary = json.loads((Path(cfg["output_dir"]) / "dataset_summary.json").read_text())
    overlap = summary["leakage_audit"]["overlap_test_in_train"]["sld_overlap"]
    assert all(v == 0.0 for v in overlap.values())
    assert summary["split"]["strategy"] == "group_sld"
