"""Trailing time-based sliding-window features per ``(src_ip, SLD)`` group.

Window definition
-----------------
For an event at time ``t`` and window ``W`` seconds the window is the half-open interval
``(t - W, t]``:

* the event at exactly ``t - W`` is **excluded** (lower bound is open),
* the current event is **included** (upper bound is closed),
* events in the future are never used (a window only looks backwards).

Events are processed in a *stable total order* ``(timestamp, event_id)`` inside each group.
For events sharing the same timestamp, only those that come *earlier* in this order (plus
the event itself) contribute; later ones do not. Consequently two events with an identical
timestamp can get different window features. Timestamps are converted to integer
microseconds (Zeek resolution) so the ``t - W`` boundary comparison is exact.

The prediction unit stays one DNS event: one feature row per event, label untouched.
Only the columns in :data:`WINDOW_INPUT_COLUMNS` are read, so the label (and any other
metadata) cannot leak into window features by construction.

Feature definitions (n = number of events in the window, including the current one)
---------------------------------------------------------------------------------
win_count                 n
win_query_rate            n / W                         (queries per second)
win_unique_queries        number of distinct normalised query strings
win_unique_query_ratio    win_unique_queries / n
win_unique_subdomains     number of distinct non-empty subdomain strings
win_unique_subdomain_ratio win_unique_subdomains / n
win_query_len_mean/std/max  mean, population std (ddof=0), max of query length
win_entropy_mean/std      mean / population std of per-event query entropy
win_subdomain_len_mean    mean subdomain length
win_digit_ratio_mean      mean of per-event digit ratio
win_nxdomain_count/ratio  number / fraction (count/n) of NXDOMAIN events
win_unique_qtypes         number of distinct query types
win_unique_dst_ips        number of distinct non-null destination IPs
win_answers_mean          mean num_answers over events where it is defined (NaN if none)
win_ttl_mean              mean of per-event mean TTL over events where defined (NaN if none)
win_iat_mean/std          mean / population std of inter-arrival times (seconds) between
                          *consecutive events of the same group inside the window*

Fill conventions: std with fewer than 2 values is 0.0; with n = 1 there is no
inter-arrival time, so ``win_iat_mean = W`` (the previous event, if any, is at least W
seconds earlier) and ``win_iat_std = 0``. NaN means (answers/TTL) are imputed downstream
with training-set statistics.

Complexity: one pass over events, amortised O(1) per event (deque + reference-counted
dicts + monotonic deque for the max); no scan over the whole dataset per event.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

WINDOW_FEATURE_COLUMNS = [
    "win_count", "win_query_rate", "win_unique_queries", "win_unique_query_ratio",
    "win_unique_subdomains", "win_unique_subdomain_ratio", "win_query_len_mean",
    "win_query_len_std", "win_query_len_max", "win_entropy_mean", "win_entropy_std",
    "win_subdomain_len_mean", "win_digit_ratio_mean", "win_nxdomain_count",
    "win_nxdomain_ratio", "win_unique_qtypes", "win_unique_dst_ips", "win_answers_mean",
    "win_ttl_mean", "win_iat_mean", "win_iat_std",
]

WINDOW_INPUT_COLUMNS = [
    "event_id", "timestamp", "src_ip", "sld", "query", "subdomain", "dst_ip", "qtype_cat",
    "query_length", "query_entropy", "subdomain_length", "digit_ratio", "is_nxdomain",
    "num_answers", "ttl_mean",
]

_US = 1_000_000
_VAR_EPS = 1e-12


def _inc(d: dict, key) -> None:
    d[key] = d.get(key, 0) + 1


def _dec(d: dict, key) -> None:
    c = d[key] - 1
    if c:
        d[key] = c
    else:
        del d[key]


def _std_int(s: int, ss: int, n: int) -> float:
    """Exact population std from integer sums (non-negative numerator by construction)."""
    if n < 2:
        return 0.0
    return math.sqrt((ss * n - s * s) / (n * n))


def _std_float(s: float, ss: float, n: int) -> float:
    if n < 2:
        return 0.0
    mean = s / n
    var = ss / n - mean * mean
    return math.sqrt(var) if var > _VAR_EPS else 0.0


class GroupWindow:
    """Incremental trailing-window state for ONE ``(src_ip, SLD)`` group."""

    __slots__ = (
        "window_seconds", "window_us", "events", "maxq", "iats", "q_cnt", "sub_cnt", "dst_cnt",
        "qt_cnt", "n", "s_qlen", "ss_qlen", "s_ent", "ss_ent", "s_sub", "s_dr", "nx", "ans_s",
        "ans_n", "ttl_s", "ttl_n", "iat_s", "iat_ss", "last_ts_us",
    )

    def __init__(self, window_seconds: float):
        self.window_seconds = float(window_seconds)
        self.window_us = int(round(window_seconds * _US))
        self.events: deque = deque()
        self.maxq: deque = deque()
        self.iats: deque = deque()
        self.q_cnt: dict = {}
        self.sub_cnt: dict = {}
        self.dst_cnt: dict = {}
        self.qt_cnt: dict = {}
        self.n = 0
        self.s_qlen = self.ss_qlen = 0
        self.s_ent = self.ss_ent = 0.0
        self.s_sub = 0
        self.s_dr = 0.0
        self.nx = 0
        self.ans_s = 0.0
        self.ans_n = 0
        self.ttl_s = 0.0
        self.ttl_n = 0
        self.iat_s = self.iat_ss = 0
        self.last_ts_us: Optional[int] = None

    def _evict_one(self) -> None:
        ts, query, sub, dst, qtype, qlen, ent, sublen, dr, nx, ans, ttl = self.events.popleft()
        self.n -= 1
        self.s_qlen -= qlen
        self.ss_qlen -= qlen * qlen
        self.s_ent -= ent
        self.ss_ent -= ent * ent
        self.s_sub -= sublen
        self.s_dr -= dr
        self.nx -= nx
        _dec(self.q_cnt, query)
        if sub:
            _dec(self.sub_cnt, sub)
        if dst is not None:
            _dec(self.dst_cnt, dst)
        _dec(self.qt_cnt, qtype)
        if ans == ans:  # not NaN
            self.ans_s -= ans
            self.ans_n -= 1
        if ttl == ttl:
            self.ttl_s -= ttl
            self.ttl_n -= 1
        if self.events:  # drop the inter-arrival gap between the evicted event and its successor
            d = self.iats.popleft()
            self.iat_s -= d
            self.iat_ss -= d * d

    def add(self, ts_us: int, query: str, sub: str, dst, qtype: str, qlen: int, ent: float,
            sublen: int, dr: float, nx: int, ans: float, ttl: float) -> tuple:
        """Add the current event and return its window feature tuple (order of
        :data:`WINDOW_FEATURE_COLUMNS`). Events must arrive in non-decreasing time order."""
        if self.last_ts_us is not None and ts_us < self.last_ts_us:
            raise ValueError("Events of a group must be added in non-decreasing timestamp order.")
        self.last_ts_us = ts_us
        cutoff = ts_us - self.window_us  # window is (cutoff, ts_us]
        ev = self.events
        while ev and ev[0][0] <= cutoff:
            self._evict_one()
        while self.maxq and self.maxq[0][0] <= cutoff:
            self.maxq.popleft()

        if ev:
            gap = ts_us - ev[-1][0]
            self.iats.append(gap)
            self.iat_s += gap
            self.iat_ss += gap * gap
        ev.append((ts_us, query, sub, dst, qtype, qlen, ent, sublen, dr, nx, ans, ttl))
        while self.maxq and self.maxq[-1][1] <= qlen:
            self.maxq.pop()
        self.maxq.append((ts_us, qlen))

        self.n += 1
        self.s_qlen += qlen
        self.ss_qlen += qlen * qlen
        self.s_ent += ent
        self.ss_ent += ent * ent
        self.s_sub += sublen
        self.s_dr += dr
        self.nx += nx
        _inc(self.q_cnt, query)
        if sub:
            _inc(self.sub_cnt, sub)
        if dst is not None:
            _inc(self.dst_cnt, dst)
        _inc(self.qt_cnt, qtype)
        if ans == ans:
            self.ans_s += ans
            self.ans_n += 1
        if ttl == ttl:
            self.ttl_s += ttl
            self.ttl_n += 1

        n = self.n
        n_gaps = len(self.iats)
        if n_gaps:
            iat_mean = self.iat_s / n_gaps / _US
            iat_std = _std_int(self.iat_s, self.iat_ss, n_gaps) / _US
        else:
            iat_mean = self.window_seconds
            iat_std = 0.0
        return (
            float(n),
            n / self.window_seconds,
            float(len(self.q_cnt)),
            len(self.q_cnt) / n,
            float(len(self.sub_cnt)),
            len(self.sub_cnt) / n,
            self.s_qlen / n,
            _std_int(self.s_qlen, self.ss_qlen, n),
            float(self.maxq[0][1]),
            self.s_ent / n,
            _std_float(self.s_ent, self.ss_ent, n),
            self.s_sub / n,
            self.s_dr / n,
            float(self.nx),
            self.nx / n,
            float(len(self.qt_cnt)),
            float(len(self.dst_cnt)),
            (self.ans_s / self.ans_n) if self.ans_n else float("nan"),
            (self.ttl_s / self.ttl_n) if self.ttl_n else float("nan"),
            iat_mean,
            iat_std,
        )


def _to_us(timestamps: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(timestamps, dtype="float64") * _US).astype(np.int64)


def compute_window_features(events: pd.DataFrame, window_seconds: float) -> pd.DataFrame:
    """Window features for every event of ``events`` using ONLY ``events``.

    Returns a DataFrame indexed like ``events`` with :data:`WINDOW_FEATURE_COLUMNS`.
    """
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive.")
    missing = [c for c in WINDOW_INPUT_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"compute_window_features: missing columns {missing}")
    n = len(events)
    if n == 0:
        return pd.DataFrame(columns=WINDOW_FEATURE_COLUMNS, index=events.index, dtype="float64")

    data = events[WINDOW_INPUT_COLUMNS]  # label and other metadata are deliberately not read
    ts_us = _to_us(data["timestamp"].to_numpy())
    gcode = data.groupby(["src_ip", "sld"], sort=False, dropna=False).ngroup().to_numpy()
    eids = np.asarray(data["event_id"].astype(str).to_numpy(), dtype=str)
    order = np.lexsort((eids, ts_us, gcode))  # primary: group, then time, then event_id

    g_l = gcode.tolist()
    ts_l = ts_us.tolist()
    q_l = data["query"].tolist()
    sub_l = data["subdomain"].tolist()
    dst_l = [None if (v is None or (isinstance(v, float) and v != v)) else v for v in data["dst_ip"].tolist()]
    qt_l = data["qtype_cat"].tolist()
    qlen_l = data["query_length"].astype("int64").tolist()
    ent_l = data["query_entropy"].astype("float64").tolist()
    sublen_l = data["subdomain_length"].astype("int64").tolist()
    dr_l = data["digit_ratio"].astype("float64").tolist()
    nx_l = data["is_nxdomain"].astype("int64").tolist()
    ans_l = data["num_answers"].astype("float64").tolist()
    ttl_l = data["ttl_mean"].astype("float64").tolist()

    out = np.empty((n, len(WINDOW_FEATURE_COLUMNS)), dtype="float64")
    prev_g = -1
    win: Optional[GroupWindow] = None
    for pos in order.tolist():
        g = g_l[pos]
        if g != prev_g:
            win = GroupWindow(window_seconds)
            prev_g = g
        out[pos, :] = win.add(  # type: ignore[union-attr]
            ts_l[pos], q_l[pos], sub_l[pos], dst_l[pos], qt_l[pos], qlen_l[pos], ent_l[pos],
            sublen_l[pos], dr_l[pos], nx_l[pos], ans_l[pos], ttl_l[pos],
        )
    return pd.DataFrame(out, columns=WINDOW_FEATURE_COLUMNS, index=events.index)


def compute_window_features_with_history(
    target: pd.DataFrame, history: pd.DataFrame, window_seconds: float
) -> pd.DataFrame:
    """Window features of ``target`` events whose windows may also contain ``history`` events.

    Used only by the separate *train carry-over* experiment. Because a window only looks
    backwards, a history event contributes to a target event only if it occurred at or
    before it (same-timestamp ties follow the stable ``(timestamp, event_id)`` order).
    Labels are not read.
    """
    combined = pd.concat(
        [history[WINDOW_INPUT_COLUMNS], target[WINDOW_INPUT_COLUMNS]], ignore_index=True
    )
    feats = compute_window_features(combined, window_seconds)
    result = feats.iloc[len(history):].copy()
    result.index = target.index
    return result


class StreamingWindowFeatureExtractor:
    """Online version of :func:`compute_window_features` (same code path, per-group state).

    The fitted model artifact does NOT contain this state: at inference time the
    deployment must keep the last ``W`` seconds of events per ``(src_ip, SLD)`` (this
    object) in addition to the saved model/preprocessing pipeline.
    """

    def __init__(self, window_seconds: float):
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")
        self.window_seconds = float(window_seconds)
        self._groups: dict[tuple, GroupWindow] = {}

    def update(self, *, timestamp: float, src_ip: str, sld: str, query: str, subdomain: str,
               dst_ip: Optional[str], qtype_cat: str, query_length: int, query_entropy: float,
               subdomain_length: int, digit_ratio: float, is_nxdomain: int,
               num_answers: float, ttl_mean: float) -> dict:
        key = (src_ip, sld)
        win = self._groups.get(key)
        if win is None:
            win = self._groups[key] = GroupWindow(self.window_seconds)
        values = win.add(
            int(round(timestamp * _US)), query, subdomain, dst_ip, qtype_cat, int(query_length),
            float(query_entropy), int(subdomain_length), float(digit_ratio), int(is_nxdomain),
            float(num_answers), float(ttl_mean),
        )
        return dict(zip(WINDOW_FEATURE_COLUMNS, values))

    def prune(self, now_timestamp: float) -> int:
        """Drop groups whose newest event is outside ``(now - W, now]``; returns #dropped."""
        cutoff = int(round(now_timestamp * _US)) - int(round(self.window_seconds * _US))
        stale = [k for k, w in self._groups.items() if w.last_ts_us is not None and w.last_ts_us <= cutoff]
        for k in stale:
            del self._groups[k]
        return len(stale)

    def __len__(self) -> int:
        return len(self._groups)
