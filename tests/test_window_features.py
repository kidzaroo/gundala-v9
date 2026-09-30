import math

import numpy as np
import pandas as pd
import pytest

from src.window_features import (
    WINDOW_FEATURE_COLUMNS,
    StreamingWindowFeatureExtractor,
    compute_window_features,
)

from conftest import make_window_events


def col(feats, name):
    return feats[name].tolist()


def test_output_schema_and_one_row_per_event(window_events):
    ev = window_events([{"ts": t} for t in (0, 1, 2)])
    feats = compute_window_features(ev, 5)
    assert list(feats.columns) == WINDOW_FEATURE_COLUMNS
    assert len(feats) == len(ev) and feats.index.equals(ev.index)


def test_lower_boundary_excluded_upper_included(window_events):
    # W=5, events at 0, 4.999, 5, 9.999, 10 in one group: window is (t-5, t]
    ev = window_events([{"ts": t} for t in (0, 4.999, 5, 9.999, 10)])
    feats = compute_window_features(ev, 5)
    assert col(feats, "win_count") == [1, 2, 2, 2, 2]


def test_event_exactly_at_t_minus_w_is_excluded(window_events):
    ev = window_events([{"ts": 0}, {"ts": 5}])
    assert col(compute_window_features(ev, 5), "win_count") == [1, 1]
    assert col(compute_window_features(ev, 5.000001), "win_count") == [1, 2]


def test_future_events_never_contribute(window_events):
    rows = [{"ts": t, "sub": f"s{i}"} for i, t in enumerate((0, 1, 2, 3, 4, 5, 6))]
    full = compute_window_features(window_events(rows), 10)
    for k in range(1, len(rows) + 1):
        prefix = compute_window_features(window_events(rows[:k]), 10)
        pd.testing.assert_frame_equal(prefix, full.iloc[:k])


def test_same_timestamp_uses_stable_event_id_order(window_events):
    rows = [{"ts": 1.0, "id": "c"}, {"ts": 1.0, "id": "a"}, {"ts": 1.0, "id": "b"}]
    feats = compute_window_features(window_events(rows), 5)
    # processing order is a, b, c -> counts 1, 2, 3 for events a, b, c
    by_id = dict(zip(["c", "a", "b"], col(feats, "win_count")))
    assert by_id == {"a": 1, "b": 2, "c": 3}


def test_input_row_order_does_not_matter(window_events):
    rows = [{"ts": t, "id": f"e{i}", "sub": f"s{i % 3}"} for i, t in enumerate((0, 0.5, 0.5, 1, 2, 7, 7.2))]
    a = compute_window_features(window_events(rows), 5).assign(id=[r["id"] for r in rows]).set_index("id")
    shuffled = [rows[i] for i in (4, 0, 6, 2, 5, 1, 3)]
    b = compute_window_features(window_events(shuffled), 5).assign(id=[r["id"] for r in shuffled]).set_index("id")
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index())


def test_groups_are_separated_by_src_ip_and_sld(window_events):
    rows = [
        {"ts": 0, "src": "A", "sld": "x.com"},
        {"ts": 1, "src": "B", "sld": "x.com"},   # other client, same SLD
        {"ts": 2, "src": "A", "sld": "y.com"},   # same client, other SLD
        {"ts": 3, "src": "A", "sld": "x.com"},
    ]
    feats = compute_window_features(window_events(rows), 60)
    assert col(feats, "win_count") == [1, 1, 1, 2]


def test_single_event_defaults(window_events):
    feats = compute_window_features(window_events([{"ts": 100}]), 10)
    row = feats.iloc[0]
    assert row["win_count"] == 1 and row["win_query_rate"] == pytest.approx(0.1)
    assert row["win_query_len_std"] == 0 and row["win_entropy_std"] == 0 and row["win_iat_std"] == 0
    assert row["win_iat_mean"] == 10  # documented fill: W
    assert row["win_unique_query_ratio"] == 1 and row["win_nxdomain_ratio"] == 0


def test_iat_mean_and_std(window_events):
    feats = compute_window_features(window_events([{"ts": 0}, {"ts": 1}, {"ts": 3}]), 10)
    last = feats.iloc[-1]
    assert last["win_iat_mean"] == pytest.approx(1.5)
    assert last["win_iat_std"] == pytest.approx(0.5)
    assert feats.iloc[1]["win_iat_mean"] == pytest.approx(1.0)


def test_iat_only_counts_events_inside_window(window_events):
    feats = compute_window_features(window_events([{"ts": 0}, {"ts": 1}, {"ts": 20}, {"ts": 21}]), 5)
    assert feats.iloc[2]["win_iat_mean"] == 5  # alone in window -> fill W
    assert feats.iloc[3]["win_iat_mean"] == pytest.approx(1.0)


def test_unique_counts_and_nxdomain_and_max(window_events):
    rows = [
        {"ts": 0, "sub": "aaa", "nx": 1, "dst": "1.1.1.1", "qtype": "A"},
        {"ts": 1, "sub": "aaa", "nx": 0, "dst": "1.1.1.1", "qtype": "TXT"},
        {"ts": 2, "sub": "bbbbbb", "nx": 1, "dst": "2.2.2.2", "qtype": "A"},
    ]
    last = compute_window_features(window_events(rows), 10).iloc[-1]
    assert last["win_count"] == 3
    assert last["win_unique_queries"] == 2 and last["win_unique_query_ratio"] == pytest.approx(2 / 3)
    assert last["win_unique_subdomains"] == 2 and last["win_unique_subdomain_ratio"] == pytest.approx(2 / 3)
    assert last["win_nxdomain_count"] == 2 and last["win_nxdomain_ratio"] == pytest.approx(2 / 3)
    assert last["win_unique_qtypes"] == 2 and last["win_unique_dst_ips"] == 2
    assert last["win_query_len_max"] == len("bbbbbb.example.com")
    assert last["win_subdomain_len_mean"] == pytest.approx((3 + 3 + 6) / 3)


def test_max_and_uniques_shrink_when_events_leave_window(window_events):
    rows = [{"ts": 0, "sub": "x" * 20}, {"ts": 10, "sub": "y"}]
    feats = compute_window_features(window_events(rows), 5)
    assert feats.iloc[1]["win_query_len_max"] == len("y.example.com")
    assert feats.iloc[1]["win_unique_subdomains"] == 1


def test_nan_answers_and_ttl_are_ignored_in_means(window_events):
    ev = window_events([{"ts": 0, "ans": 2, "ttl": 10}, {"ts": 1, "ans": 4, "ttl": 30}])
    ev.loc[1, "ttl_mean"] = np.nan
    last = compute_window_features(ev, 10).iloc[-1]
    assert last["win_answers_mean"] == 3 and last["win_ttl_mean"] == 10
    ev["ttl_mean"] = np.nan
    assert math.isnan(compute_window_features(ev, 10).iloc[-1]["win_ttl_mean"])


def _naive_reference(ev: pd.DataFrame, window: float) -> pd.DataFrame:
    """O(n^2) reference implementation used only to validate the efficient one."""
    w_us = int(round(window * 1e6))
    ts = np.rint(ev["timestamp"].to_numpy() * 1e6).astype(np.int64)
    out = {}
    order = sorted(range(len(ev)), key=lambda i: (ev["src_ip"][i], ev["sld"][i], ts[i], ev["event_id"][i]))
    for pos, i in enumerate(order):
        members = []
        for j in order[:pos + 1]:
            if (ev["src_ip"][j], ev["sld"][j]) == (ev["src_ip"][i], ev["sld"][i]) and ts[j] > ts[i] - w_us:
                members.append(j)
        qlen = np.array([ev["query_length"][j] for j in members], dtype=float)
        gaps = np.diff(np.array([ts[j] for j in members], dtype=float)) / 1e6
        out[i] = [
            len(members), len(members) / window, len({ev["query"][j] for j in members}),
            qlen.mean(), qlen.std(), qlen.max(),
            gaps.mean() if len(gaps) else window, gaps.std() if len(gaps) else 0.0,
        ]
    cols = ["win_count", "win_query_rate", "win_unique_queries", "win_query_len_mean",
            "win_query_len_std", "win_query_len_max", "win_iat_mean", "win_iat_std"]
    return pd.DataFrame([out[i] for i in range(len(ev))], columns=cols)


@pytest.mark.parametrize("window", [5, 15, 60])
def test_matches_naive_reference_with_ties_and_groups(window):
    rng = np.random.default_rng(3)
    n = 400
    rows = [{
        "ts": round(float(rng.uniform(0, 200)), 1),  # rounding -> many timestamp ties
        "src": f"10.0.0.{rng.integers(1, 4)}", "sld": f"d{rng.integers(1, 3)}.com",
        "sub": "".join(rng.choice(list("abcdef"), size=int(rng.integers(0, 12)))),
    } for _ in range(n)]
    ev = make_window_events(rows)
    fast = compute_window_features(ev, window)
    ref = _naive_reference(ev, window)
    for c in ref.columns:
        np.testing.assert_allclose(fast[c].to_numpy(), ref[c].to_numpy(), rtol=1e-9, atol=1e-9, err_msg=c)


def test_streaming_extractor_equals_batch(window_events):
    rng = np.random.default_rng(5)
    rows = sorted(
        [{"ts": round(float(rng.uniform(0, 100)), 2), "src": f"h{rng.integers(0, 2)}",
          "sld": f"s{rng.integers(0, 2)}.com", "sub": "x" * int(rng.integers(0, 6)),
          "nx": int(rng.integers(0, 2))} for _ in range(150)],
        key=lambda r: r["ts"],
    )
    ev = window_events(rows)
    batch = compute_window_features(ev, 10)
    stream = StreamingWindowFeatureExtractor(10)
    order = ev.sort_values(["timestamp", "event_id"]).index
    for idx in order:
        r = ev.loc[idx]
        got = stream.update(
            timestamp=r["timestamp"], src_ip=r["src_ip"], sld=r["sld"], query=r["query"],
            subdomain=r["subdomain"], dst_ip=r["dst_ip"], qtype_cat=r["qtype_cat"],
            query_length=r["query_length"], query_entropy=r["query_entropy"],
            subdomain_length=r["subdomain_length"], digit_ratio=r["digit_ratio"],
            is_nxdomain=r["is_nxdomain"], num_answers=r["num_answers"], ttl_mean=r["ttl_mean"],
        )
        np.testing.assert_allclose(
            [got[c] for c in WINDOW_FEATURE_COLUMNS], batch.loc[idx].to_numpy(), rtol=1e-9, atol=1e-9
        )
    assert len(stream) > 0
    stream.prune(1e9)
    assert len(stream) == 0


def test_streaming_rejects_out_of_order_events():
    s = StreamingWindowFeatureExtractor(5)
    kw = dict(src_ip="a", sld="x.com", query="x.com", subdomain="", dst_ip=None, qtype_cat="A",
              query_length=5, query_entropy=1.0, subdomain_length=0, digit_ratio=0.0, is_nxdomain=0,
              num_answers=1.0, ttl_mean=1.0)
    s.update(timestamp=10.0, **kw)
    with pytest.raises(ValueError):
        s.update(timestamp=9.0, **kw)


def test_uses_real_timestamps_not_row_counts(window_events):
    dense = compute_window_features(window_events([{"ts": i * 0.01} for i in range(50)]), 1)
    sparse = compute_window_features(window_events([{"ts": i * 10.0} for i in range(50)]), 1)
    assert dense["win_count"].iloc[-1] == 50
    assert sparse["win_count"].max() == 1


def test_invalid_window_and_empty_input(window_events):
    with pytest.raises(ValueError):
        compute_window_features(window_events([{"ts": 0}]), 0)
    empty = compute_window_features(window_events([{"ts": 0}]).iloc[0:0], 5)
    assert empty.empty and list(empty.columns) == WINDOW_FEATURE_COLUMNS
