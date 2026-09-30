import math

import numpy as np
import pandas as pd
import pytest

from src.data_loader import EVENT_COLUMNS
from src.feature_extraction import (
    EVENT_BINARY,
    EVENT_CATEGORICAL,
    EVENT_CONTINUOUS,
    FORBIDDEN_FEATURE_COLUMNS,
    add_event_features,
    compute_lexical_features,
    extract_event_features,
    get_event_feature_groups,
    shannon_entropy,
)
from src.validation import parse_answers, parse_ttls


def _events(rows):
    df = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    return df


def _row(**kw):
    base = dict.fromkeys(EVENT_COLUMNS)
    base.update(event_id="e", label=0, source_file="f", source_line=1, timestamp=1.0,
                src_ip="10.0.0.1", query="")
    base.update(kw)
    return base


def test_entropy_values():
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("aaaa") == 0.0
    assert shannon_entropy("aabb") == pytest.approx(1.0)
    assert shannon_entropy("abcd") == pytest.approx(2.0)
    p = [0.5, 0.25, 0.25]
    assert shannon_entropy("aabc") == pytest.approx(-sum(x * math.log2(x) for x in p))


def test_lexical_feature_definitions():
    f = compute_lexical_features("a1b2.example.com")
    assert f["query_length"] == 16
    assert f["num_labels"] == 3
    assert f["subdomain_length"] == 4 and f["subdomain_num_labels"] == 1
    assert f["max_label_length"] == 7
    assert f["mean_label_length"] == pytest.approx(14 / 3)
    assert f["digit_count"] == 2 and f["digit_ratio"] == pytest.approx(2 / 16)
    assert f["letter_count"] == 12 and f["letter_ratio"] == pytest.approx(12 / 16)
    assert f["hyphen_count"] == 0 and f["hyphen_ratio"] == 0.0
    assert f["unique_char_count"] == 12 and f["unique_char_ratio"] == pytest.approx(12 / 16)
    assert f["longest_digit_run"] == 1 and f["longest_alpha_run"] == 7
    assert f["query_entropy"] == pytest.approx(shannon_entropy("a1b2.example.com"))
    assert f["subdomain_entropy"] == pytest.approx(shannon_entropy("a1b2"))
    assert f["sld"] == "example.com" and f["subdomain"] == "a1b2"
    g = compute_lexical_features("x-y-12345.test.co.id")
    assert g["hyphen_count"] == 2 and g["longest_digit_run"] == 5


def test_empty_query_is_handled():
    f = compute_lexical_features("")
    assert f["query_length"] == 0 and f["num_labels"] == 0
    assert f["query_entropy"] == 0.0 and f["digit_ratio"] == 0.0 and f["unique_char_ratio"] == 0.0
    assert f["is_valid_domain"] == 0 and f["domain_kind"] == "empty"
    assert compute_lexical_features(None)["query_length"] == 0  # type: ignore[arg-type]


def test_ip_and_reverse_queries():
    assert compute_lexical_features("8.8.8.8")["is_ip_query"] == 1
    r = compute_lexical_features("4.3.2.1.in-addr.arpa")
    assert r["is_reverse_dns"] == 1 and r["is_ip_query"] == 0
    assert compute_lexical_features("wpad")["is_single_label"] == 1


def test_missing_fields_do_not_fail():
    df = _events([_row(query="a.example.com"), _row(query="b.example.com", event_id="e2")])
    feats = extract_event_features(df)
    assert len(feats) == 2
    assert (feats["qtype_cat"] == "MISSING").all() and (feats["proto_cat"] == "MISSING").all()
    assert feats["num_answers"].isna().all()  # answers field unavailable in dataset
    assert feats["ttl_mean"].isna().all()
    assert feats["flag_aa"].isna().all()
    assert (feats["is_nxdomain"] == 0).all()


def test_answers_and_ttls_present_but_empty_or_odd_format():
    df = _events([
        _row(event_id="a", query="a.example.com", answers=["1.1.1.1", "2.2.2.2"], ttls=[10.0, 30.0],
             rcode_name="NOERROR", qtype_name="A"),
        _row(event_id="b", query="b.example.com", answers=None, ttls=None, rcode_name="NXDOMAIN"),
        _row(event_id="c", query="c.example.com", answers=[], ttls=[]),
    ])
    f = extract_event_features(df)
    assert f.loc[0, ["num_answers", "num_ttls", "ttl_min", "ttl_max", "ttl_mean"]].tolist() == [2, 2, 10, 30, 20]
    assert f.loc[1, "num_answers"] == 0 and f.loc[1, "num_ttls"] == 0
    assert math.isnan(f.loc[1, "ttl_mean"]) and math.isnan(f.loc[2, "ttl_max"])
    assert f.loc[1, "is_nxdomain"] == 1 and f.loc[0, "is_nxdomain"] == 0


def test_parse_answers_and_ttls_formats():
    assert parse_answers(None) is None
    assert parse_answers([]) == []
    assert parse_answers("") == []
    assert parse_answers("-") == []
    assert parse_answers("1.1.1.1, 2.2.2.2") == ["1.1.1.1", "2.2.2.2"]
    assert parse_answers(["a", None, "", "-"]) == ["a"]
    assert parse_answers({"x": 1}) is None
    assert parse_ttls(None) is None
    assert parse_ttls("10, 20.5") == [10.0, 20.5]
    assert parse_ttls([1, "2", "x", None, -5, float("nan")]) == [1.0, 2.0]
    assert parse_ttls(300) == [300.0]


def test_feature_groups_exclude_metadata():
    cont, binary, cat = get_event_feature_groups(True)
    all_features = set(cont + binary + cat)
    assert not (all_features & FORBIDDEN_FEATURE_COLUMNS)
    assert len(all_features) == len(cont + binary + cat)  # no duplicates
    assert get_event_feature_groups(False)[2] == []
    assert cont == EVENT_CONTINUOUS and binary == EVENT_BINARY and cat == EVENT_CATEGORICAL


def test_add_event_features_keeps_metadata_and_index():
    df = _events([_row(query="a.example.com", label=1)])
    out = add_event_features(df)
    assert out.loc[0, "label"] == 1 and out.loc[0, "sld"] == "example.com"
    assert list(out.index) == [0]
    empty = extract_event_features(df.iloc[0:0])
    assert len(empty) == 0


def test_lexical_features_do_not_depend_on_label():
    a = add_event_features(_events([_row(query="xk3j9d8f.example.com", label=0)]))
    b = add_event_features(_events([_row(query="xk3j9d8f.example.com", label=1)]))
    cols = EVENT_CONTINUOUS + EVENT_BINARY
    np.testing.assert_allclose(a[cols].to_numpy(dtype=float), b[cols].to_numpy(dtype=float), equal_nan=True)
