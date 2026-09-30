import copy
import json

import pandas as pd
import pytest

from src.config import DEFAULT_CONFIG
from src.data_loader import (
    class_distribution,
    detect_format,
    load_dataset,
    load_dns_file,
    lookup_field,
    make_event_id,
)
from src.synthetic_data import write_dataset
from src.validation import UNKNOWN_SRC

FM = DEFAULT_CONFIG["field_mapping"]


def _rec(ts=1.0, query="www.Example.com.", **kw):
    r = {"ts": ts, "id.orig_h": "10.0.0.1", "id.resp_h": "10.0.0.53", "id.orig_p": 4000,
         "id.resp_p": 53, "proto": "UDP", "query": query, "qtype": 1, "rcode": 0,
         "AA": False, "TC": False, "RD": True, "RA": True, "rejected": False,
         "answers": ["1.2.3.4"], "TTLs": [60.0]}
    r.update(kw)
    return r


def _write_jsonl(path, records, extra_lines=()):
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
        for line in extra_lines:
            fh.write(line + "\n")


def test_jsonl_and_json_array_both_load(tmp_path):
    recs = [_rec(ts=1.0), _rec(ts=2.0)]
    p1, p2 = tmp_path / "a.json", tmp_path / "b.json"
    _write_jsonl(p1, recs)
    p2.write_text(json.dumps(recs))
    assert detect_format(p1) == "jsonl" and detect_format(p2) == "json_array"
    d1, r1 = load_dns_file(p1, 0, "benign", FM)
    d2, r2 = load_dns_file(p2, 0, "benign", FM)
    assert r1.detected_format == "jsonl" and r2.detected_format == "json_array"
    assert len(d1) == len(d2) == 2
    pd.testing.assert_frame_equal(d1.drop(columns="source_file"), d2.drop(columns="source_file"))


def test_array_with_leading_whitespace_and_explicit_format(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("\n  " + json.dumps([_rec()]))
    assert detect_format(p) == "json_array"
    df, rep = load_dns_file(p, 1, "tunnel", FM, fmt="json_array")
    assert len(df) == 1 and rep.n_invalid == 0


def test_label_comes_from_source_file_and_metadata_is_kept(tmp_path):
    p = tmp_path / "t.json"
    _write_jsonl(p, [_rec(ts=5.0, label=0), _rec(ts=6.0)])  # a "label" key inside data is ignored
    df, _ = load_dns_file(p, 1, "tunnel", FM)
    assert (df["label"] == 1).all()
    assert (df["source_file"] == "t.json").all()
    assert df["source_line"].tolist() == [1, 2]
    assert df["event_id"].is_unique
    assert df["event_id"].iloc[0] == make_event_id("tunnel", 1)


def test_normalisation(tmp_path):
    p = tmp_path / "n.json"
    _write_jsonl(p, [_rec(ts="1700000000.5", query="WWW.Example.COM.")])
    df, _ = load_dns_file(p, 0, "benign", FM)
    row = df.iloc[0]
    assert row["timestamp"] == pytest.approx(1700000000.5)
    assert row["query"] == "www.example.com"
    assert row["src_ip"] == "10.0.0.1" and row["dst_ip"] == "10.0.0.53"
    assert row["proto"] == "udp"
    assert row["qtype_name"] == "A" and row["rcode_name"] == "NOERROR"  # from numeric codes
    assert row["ttls"] == [60.0] and row["answers"] == ["1.2.3.4"]


def test_nested_zeek_id_object_is_supported():
    found, val = lookup_field({"id": {"orig_h": "1.2.3.4"}}, ["id.orig_h"])
    assert found and val == "1.2.3.4"
    assert lookup_field({"id.orig_h": "9.9.9.9", "id": {"orig_h": "1.1.1.1"}}, ["id.orig_h"])[1] == "9.9.9.9"
    assert lookup_field({}, ["id.orig_h"]) == (False, None)


def test_invalid_records_are_counted_not_fatal(tmp_path):
    p = tmp_path / "bad.json"
    _write_jsonl(
        p, [_rec(ts=1.0)],
        extra_lines=[
            '{"ts": 1.0, "query": ',                 # truncated JSON
            "",                                      # blank line: ignored
            "[1, 2, 3]",                             # not an object
            json.dumps(_rec(ts="not-a-time")),       # invalid timestamp
            json.dumps({k: v for k, v in _rec().items() if k != "ts"}),   # missing required ts
            json.dumps(_rec(query=["a"])),           # wrong type
            json.dumps(_rec(ts=2.0)),                # valid
        ],
    )
    df, rep = load_dns_file(p, 0, "benign", FM)
    assert rep.n_valid == 2 and len(df) == 2
    assert rep.n_invalid == 5 and rep.n_records_seen == 7
    assert rep.invalid_reasons["invalid_json"] == 1
    assert rep.invalid_reasons["not_an_object"] == 1
    assert rep.invalid_reasons["invalid_timestamp"] == 1
    assert rep.invalid_reasons["missing_required:timestamp"] == 1
    assert rep.invalid_reasons["invalid_type:query"] == 1


def test_missing_optional_fields_and_empty_query_are_explicit(tmp_path):
    p = tmp_path / "m.json"
    _write_jsonl(p, [{"ts": 1.0}, {"ts": 2.0, "query": "", "answers": "-", "TTLs": None},
                     {"ts": 3.0, "id.orig_h": "not-an-ip", "query": "a.com"}])
    df, rep = load_dns_file(p, 0, "benign", FM)
    assert len(df) == 3 and rep.n_invalid == 0
    assert (df["query"].iloc[:2] == "").all()
    assert df["src_ip"].tolist() == [UNKNOWN_SRC] * 3
    assert rep.missing_counts["src_ip"] == 3 and rep.missing_counts["query"] == 2
    assert rep.anomalies["unparsable:src_ip"] == 1
    assert df["answers"].iloc[1] == []  # "-" = present but empty
    assert df["ttls"].iloc[1] is None


def test_custom_field_mapping(tmp_path):
    p = tmp_path / "c.json"
    _write_jsonl(p, [{"time": 5, "client": "10.1.1.1", "qname": "a.b.com"}])
    fm = copy.deepcopy(FM)
    fm.update({"timestamp": ["time"], "src_ip": ["client"], "query": ["qname"]})
    df, _ = load_dns_file(p, 0, "benign", fm)
    assert df.iloc[0]["timestamp"] == 5 and df.iloc[0]["src_ip"] == "10.1.1.1" and df.iloc[0]["query"] == "a.b.com"


def test_load_dataset_reports_actual_ratio_and_does_not_rebalance(tmp_path):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    b, t = tmp_path / "benign.json", tmp_path / "tunnel.json"
    _write_jsonl(b, [_rec(ts=float(i)) for i in range(30)])
    t.write_text(json.dumps([_rec(ts=float(i), query=f"x{i}.t.com") for i in range(10)]))
    cfg["input"].update(benign_path=str(b), tunnel_path=str(t))
    df, reports, avail = load_dataset(cfg)
    dist = class_distribution(df)
    assert dist["benign"] == 30 and dist["tunnel"] == 10 and dist["ratio_benign_to_tunnel"] == 3.0
    assert set(reports) == {"benign", "tunnel"} and avail["answers"] is True
    assert df["event_id"].is_unique


def test_too_many_invalid_records_raises(tmp_path):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    b, t = tmp_path / "benign.json", tmp_path / "tunnel.json"
    _write_jsonl(b, [], extra_lines=["garbage"] * 10)
    _write_jsonl(t, [_rec()])
    cfg["input"].update(benign_path=str(b), tunnel_path=str(t))
    with pytest.raises(ValueError, match="invalid"):
        load_dataset(cfg)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dns_file(tmp_path / "nope.json", 0, "benign", FM)


def test_synthetic_noise_files_load(tmp_path):
    b, t = write_dataset(tmp_path, n_benign=50, n_tunnel=20, seed=1, add_noise=True)
    db, rb = load_dns_file(b, 0, "benign", FM)
    dt, rt = load_dns_file(t, 1, "tunnel", FM)
    assert rb.n_invalid >= 2 and rt.n_invalid >= 2
    assert (db["query"] == "").sum() >= 1  # empty query kept and flagged
    assert len(db) + rb.n_invalid == rb.n_records_seen
