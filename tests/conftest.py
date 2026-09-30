"""Shared fixtures/helpers for the test-suite."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_CONFIG  # noqa: E402
from src.feature_extraction import shannon_entropy  # noqa: E402
from src.synthetic_data import write_dataset  # noqa: E402


def make_window_events(rows: list[dict]) -> pd.DataFrame:
    """Build a window-feature input table from minimal row dicts.

    Row keys: ``ts`` (required), ``src`` (default ``10.0.0.1``), ``sld`` (default
    ``example.com``), ``sub`` (subdomain, default ``""``), ``dst``, ``qtype``, ``nx``,
    ``ans``, ``ttl``, ``id`` (event id; default ``e%04d`` by row order).
    """
    recs = []
    for i, r in enumerate(rows):
        sld = r.get("sld", "example.com")
        sub = r.get("sub", "")
        query = f"{sub}.{sld}" if sub else sld
        recs.append({
            "event_id": r.get("id", f"e{i:04d}"),
            "timestamp": float(r["ts"]),
            "src_ip": r.get("src", "10.0.0.1"),
            "sld": sld,
            "query": query,
            "subdomain": sub,
            "dst_ip": r.get("dst", "10.0.0.53"),
            "qtype_cat": r.get("qtype", "A"),
            "query_length": len(query),
            "query_entropy": shannon_entropy(query),
            "subdomain_length": len(sub),
            "digit_ratio": sum(c.isdigit() for c in query) / len(query),
            "is_nxdomain": int(r.get("nx", 0)),
            "num_answers": float(r.get("ans", 1.0)),
            "ttl_mean": float(r.get("ttl", 60.0)),
        })
    return pd.DataFrame(recs)


@pytest.fixture
def window_events():
    return make_window_events


@pytest.fixture(scope="session")
def synthetic_files(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    benign, tunnel = write_dataset(out, n_benign=700, n_tunnel=100, seed=7)
    return benign, tunnel


@pytest.fixture
def base_config(synthetic_files):
    """Small, fast configuration for integration tests."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["input"]["benign_path"] = str(synthetic_files[0])
    cfg["input"]["tunnel_path"] = str(synthetic_files[1])
    cfg["data_origin"] = "synthetic_smoke_test"
    cfg["windows"]["sizes"] = [5, 60]
    cfg["outputs"]["save_models"] = False
    cfg["outputs"]["save_plots"] = False
    params = cfg["models"]["params"]
    params["random_forest"]["n_estimators"] = 20
    params["lightgbm"]["n_estimators"] = 20
    params["xgboost"]["n_estimators"] = 20
    params["mlp"].update({"hidden_layer_sizes": [16], "max_iter": 30})
    return cfg


@pytest.fixture
def toy_matrix():
    """Small mixed-type training matrix (continuous, binary, categorical)."""
    rng = np.random.default_rng(0)
    n = 300
    y = rng.permutation(np.r_[np.zeros(270, dtype=int), np.ones(30, dtype=int)])
    df = pd.DataFrame({
        "c0": rng.normal(size=n) + 2.0 * y,
        "c1": rng.normal(size=n),
        "b0": (rng.random(n) < 0.3 + 0.4 * y).astype(float),
        "cat0": np.where(rng.random(n) < 0.5 + 0.3 * y, "A", "B"),
    })
    df.loc[5, "c1"] = np.nan
    return df, y
