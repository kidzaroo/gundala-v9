"""Leakage-related guarantees: split, window isolation, resampling and preprocessing."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_dataset
from src.evaluation import select_threshold_on_validation, transform_before_classifier
from src.feature_extraction import add_event_features
from src.models import build_pipeline, resolve_class_weight_policy
from src.split import apply_manifest, build_manifest, deduplicate, split_events, summarize_split
from src.window_features import (
    WINDOW_INPUT_COLUMNS,
    compute_window_features,
    compute_window_features_with_history,
)

from conftest import make_window_events

CONT, BIN, CAT = ["c0", "c1"], ["b0"], ["cat0"]


@pytest.fixture
def events(base_config):
    df, _, avail = load_dataset(base_config)
    return df, avail


# ------------------------------------------------------------------ split
def test_stratified_split_is_disjoint_stratified_and_reproducible(events):
    df, _ = events
    tr1, te1 = split_events(df, "stratified_random", 0.3, 42)
    tr2, te2 = split_events(df, "stratified_random", 0.3, 42)
    tr3, _ = split_events(df, "stratified_random", 0.3, 43)
    assert tr1["event_id"].tolist() == tr2["event_id"].tolist()
    assert te1["event_id"].tolist() == te2["event_id"].tolist()
    assert tr1["event_id"].tolist() != tr3["event_id"].tolist()
    assert not set(tr1["event_id"]) & set(te1["event_id"])
    assert len(tr1) + len(te1) == len(df)
    assert len(te1) == pytest.approx(0.3 * len(df), abs=1)
    for part in (tr1, te1):  # both classes, same class ratio as the full data (stratified)
        assert part["label"].mean() == pytest.approx(df["label"].mean(), abs=0.01)
    s = summarize_split(tr1, te1, "stratified_random", 0.3, 42)
    assert s["overlap_event_ids"] == 0 and s["train"]["total"] == len(tr1)


def test_chronological_split_orders_by_time(events):
    df, _ = events
    interleaved = pd.DataFrame({
        "event_id": [f"e{i}" for i in range(200)], "label": [i % 5 == 0 for i in range(200)],
        "timestamp": [float(i) for i in range(200)],
    })
    interleaved["label"] = interleaved["label"].astype(int)
    it, ie = split_events(interleaved, "chronological", 0.3, 42)
    assert it["timestamp"].max() < ie["timestamp"].min() and len(ie) == 60
    trp, tep = split_events(df, "chronological", 0.3, 42, chronological_mode="per_class")
    for lbl in (0, 1):
        assert trp.loc[trp.label == lbl, "timestamp"].max() <= tep.loc[tep.label == lbl, "timestamp"].min()


def test_chronological_split_with_disjoint_class_periods_fails_informatively():
    rows = [{"event_id": f"b{i}", "label": 0, "timestamp": float(i), "source_file": "b", "source_line": i}
            for i in range(100)]
    rows += [{"event_id": f"t{i}", "label": 1, "timestamp": 1000.0 + i, "source_file": "t", "source_line": i}
             for i in range(20)]
    with pytest.raises(ValueError, match="both"):
        split_events(pd.DataFrame(rows), "chronological", 0.3, 42)


def test_manifest_roundtrip_reproduces_split(events):
    df, _ = events
    tr, te = split_events(df, "stratified_random", 0.3, 42)
    manifest = build_manifest(tr, te)
    assert set(manifest["subset"]) == {"train", "test"} and manifest["event_id"].is_unique
    tr2, te2 = apply_manifest(df.sample(frac=1, random_state=1), manifest)
    assert set(tr2["event_id"]) == set(tr["event_id"]) and set(te2["event_id"]) == set(te["event_id"])
    with pytest.raises(ValueError):
        apply_manifest(df, manifest.iloc[:-5])


def test_dedup_removes_identical_records_and_reports(events):
    df, _ = events
    dup = pd.concat([df, df.iloc[:10].assign(event_id=[f"dup{i}" for i in range(10)])], ignore_index=True)
    out, rep = deduplicate(dup)
    assert rep["n_duplicates_removed"] == 10 and len(out) == len(df)
    flipped = df.iloc[:3].assign(event_id=["x1", "x2", "x3"], label=1 - df.iloc[:3]["label"])
    _, rep2 = deduplicate(pd.concat([df, flipped], ignore_index=True))
    assert rep2["n_duplicates_removed"] == 0 and rep2["label_conflict_events"] == 6


# ------------------------------------------------------------------ window isolation
def _events_with_features(events, n=None):
    df, avail = events
    tr, te = split_events(df, "stratified_random", 0.3, 42)
    return add_event_features(tr, avail), add_event_features(te, avail)


def test_train_windows_do_not_read_test_events_and_vice_versa(events, monkeypatch):
    import src.experiment as experiment

    tr, te = _events_with_features(events)
    w = 30
    train_ids, test_ids = set(tr["event_id"]), set(te["event_id"])
    seen = []
    real = experiment.compute_window_features

    def spy(frame, window):
        seen.append(set(frame["event_id"]))
        return real(frame, window)

    monkeypatch.setattr(experiment, "compute_window_features", spy)
    _, train_w, test_w, _, _ = experiment._window_task(
        w, tr[WINDOW_INPUT_COLUMNS], te[WINDOW_INPUT_COLUMNS], "split_isolated"
    )
    # every aggregation call saw events of exactly one subset
    assert len(seen) == 2 and seen[0] == train_ids and seen[1] == test_ids
    assert train_w.index.equals(tr.index) and test_w.index.equals(te.index)

    # sanity: mixing the subsets WOULD change test features (what isolation avoids)
    mixed = real(pd.concat([tr[WINDOW_INPUT_COLUMNS], te[WINDOW_INPUT_COLUMNS]], ignore_index=True), w)
    assert mixed.iloc[len(tr):]["win_count"].sum() > test_w["win_count"].sum()


def test_window_features_ignore_label_and_other_metadata():
    ev = make_window_events([{"ts": t} for t in (0, 1, 2, 3)])
    a = compute_window_features(ev.assign(label=[0, 1, 0, 1]), 5)
    b = compute_window_features(ev.assign(label=[1, 0, 1, 0]), 5)
    pd.testing.assert_frame_equal(a, b)
    assert "label" not in WINDOW_INPUT_COLUMNS and "source_file" not in WINDOW_INPUT_COLUMNS


def test_carryover_uses_only_earlier_train_history():
    train = make_window_events(
        [{"ts": t, "id": f"tr{t}"} for t in (6, 7, 8, 9)] + [{"ts": 12, "id": "tr12"}]
    )
    test = make_window_events([{"ts": 10, "id": "te10"}])
    isolated = compute_window_features(test, 5)
    carry = compute_window_features_with_history(test, train, 5)
    assert isolated["win_count"].iloc[0] == 1
    # (5, 10] contains train events at 6,7,8,9 plus the test event; the train event at t=12 is future
    assert carry["win_count"].iloc[0] == 5
    assert carry.index.equals(test.index)


# ------------------------------------------------------------------ resampling / preprocessing
def _pipe(name="random_forest", method="random_oversampling", cat=CAT, **bal):
    balancing = {"method": method, "sampling_strategy": "auto", "smote_k_neighbors": 5,
                 "class_weight": "none", "allow_class_weight_with_resampling": False, **bal}
    return build_pipeline(
        name, continuous=CONT, binary=BIN, categorical=cat, balancing=balancing,
        model_params={"n_estimators": 10} if name == "random_forest" else {},
        random_seed=42, n_jobs=1, train_class_counts={0: 270, 1: 30},
    )


def test_oversampling_touches_training_only_and_not_test(toy_matrix):
    X, y = toy_matrix
    X_train, y_train, X_test = X.iloc[:200], y[:200], X.iloc[200:].copy()
    before = X_test.copy(deep=True)
    pipe = _pipe()
    pipe.fit(X_train, y_train)
    sampler = pipe.named_steps["sampler"]
    assert sampler.class_counts_before_ == {0: int((y_train == 0).sum()), 1: int((y_train == 1).sum())}
    assert sampler.class_counts_after_[0] == sampler.class_counts_after_[1] == sampler.class_counts_before_[0]
    proba = pipe.predict_proba(X_test)
    assert len(proba) == len(X_test)  # no resampling at predict time
    pd.testing.assert_frame_equal(X_test, before)


def test_preprocessing_is_fit_on_training_data_only(toy_matrix):
    X, y = toy_matrix
    X_train, y_train = X.iloc[:200].copy(), y[:200]
    X_test = X.iloc[200:].copy()
    X_test["c0"] = 1e6   # extreme values that must not influence fitted statistics
    X_test["c1"] = np.nan
    pipe = _pipe(name="mlp", method="none")
    pipe.named_steps["clf"].set_params(hidden_layer_sizes=(8,), max_iter=20)
    pipe.fit(X_train, y_train)
    cont = pipe.named_steps["prep"].named_transformers_["cont"]
    np.testing.assert_allclose(cont.named_steps["impute"].statistics_, X_train[CONT].median().to_numpy())
    np.testing.assert_allclose(cont.named_steps["scale"].mean_[0], X_train["c0"].mean())
    Xt = transform_before_classifier(pipe, X_test)
    assert np.isfinite(Xt).all()  # test NaNs imputed with TRAIN median


def test_unseen_test_category_is_handled(toy_matrix):
    X, y = toy_matrix
    pipe = _pipe()
    pipe.fit(X.iloc[:200], y[:200])
    X_test = X.iloc[200:].copy()
    X_test["cat0"] = "NEVER_SEEN"
    proba = pipe.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2) and np.isfinite(proba).all()


def test_smote_with_categoricals_raises_informative_error():
    with pytest.raises(ValueError, match="smotenc"):
        _pipe(method="smote")


def test_smote_numeric_only_and_smotenc_work(toy_matrix):
    X, y = toy_matrix
    p = _pipe(method="smote", cat=[])
    p.fit(X.iloc[:200][CONT + BIN], y[:200])
    assert p.named_steps["sampler"].class_counts_after_[1] == p.named_steps["sampler"].class_counts_after_[0]
    q = _pipe(method="smotenc")
    q.fit(X.iloc[:200], y[:200])
    assert q.named_steps["sampler"].class_counts_after_[1] == q.named_steps["sampler"].class_counts_after_[0]
    assert len(q.predict_proba(X.iloc[200:])) == 100


def test_smote_k_is_reduced_safely_and_fails_when_impossible():
    kw = dict(continuous=CONT, binary=BIN, categorical=[], model_params={"n_estimators": 5}, random_seed=1,
              n_jobs=1)
    bal = {"method": "smote", "sampling_strategy": "auto", "smote_k_neighbors": 5, "class_weight": "none"}
    pipe = build_pipeline("random_forest", balancing=bal, train_class_counts={0: 100, 1: 3}, **kw)
    assert pipe.named_steps["sampler"].k_neighbors == 2
    with pytest.raises(ValueError, match="minority"):
        build_pipeline("random_forest", balancing=bal, train_class_counts={0: 100, 1: 1}, **kw)


def test_sampling_strategy_is_configurable(toy_matrix):
    X, y = toy_matrix
    pipe = _pipe(sampling_strategy=0.5)
    pipe.fit(X, y)
    after = pipe.named_steps["sampler"].class_counts_after_
    assert after[1] / after[0] == pytest.approx(0.5, abs=0.01)


def test_class_weight_and_oversampling_are_never_combined_silently():
    with pytest.raises(ValueError, match="double-compensate"):
        resolve_class_weight_policy("random_oversampling", "balanced", False)
    assert "explicitly allowed" in resolve_class_weight_policy("random_oversampling", "balanced", True)
    assert resolve_class_weight_policy("none", "balanced", False) == "balanced"
    with pytest.raises(ValueError):
        _pipe(class_weight="balanced")


def test_threshold_selection_uses_training_data_only(toy_matrix):
    X, y = toy_matrix
    thr, info = select_threshold_on_validation(lambda: _pipe(), X, y, 0.25, 42)
    assert 0.0 < thr <= 1.0 and info["n_validation"] == len(y) // 4
