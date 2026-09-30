"""Synthetic Zeek-style DNS logs for SMOKE TESTS ONLY.

The generated traffic is simplistic and trivially separable. Any metric obtained from it
is an example of the pipeline running, NOT a research result.

Usage::

    python -m src.synthetic_data --out data/synthetic --n-benign 7000 --n-tunnel 1000
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np

_BENIGN_DOMAINS = [
    "www.google.com", "mail.google.com", "www.youtube.com", "api.github.com", "github.com",
    "www.wikipedia.org", "id.wikipedia.org", "www.detik.com", "m.kompas.com", "cdn.example.co.id",
    "portal.kampus.ac.id", "www.bbc.co.uk", "static.xx.fbcdn.net", "clients1.google.com",
    "update.microsoft.com", "time.windows.com", "ocsp.digicert.com", "www.tokopedia.com",
    "images.bukalapak.com", "www.cloudflare.com", "api.twitter.com", "login.live.com",
    "fonts.gstatic.com", "www.amazon.com", "s3.amazonaws.com", "mirror.ubuntu.com",
]
_TUNNEL_DOMAINS = ["t.tunnel-example.net", "dns.c2-lab.co.id", "x.exfil-test.org"]
_BENIGN_QTYPES = [(1, "A"), (28, "AAAA"), (15, "MX"), (16, "TXT"), (12, "PTR")]
_BENIGN_QTYPE_P = [0.55, 0.35, 0.03, 0.04, 0.03]
_TUNNEL_QTYPES = [(16, "TXT"), (10, "NULL"), (5, "CNAME"), (1, "A")]
_TUNNEL_QTYPE_P = [0.55, 0.2, 0.15, 0.1]


def _record(ts: float, src: str, dst: str, sport: int, query: str, qtype, rcode: int,
            answers: Optional[list], ttls: Optional[list]) -> dict:
    rec = {
        "ts": round(ts, 6),
        "uid": "C" + hashlib.sha1(f"{ts}|{query}".encode()).hexdigest()[:12],
        "id.orig_h": src, "id.orig_p": sport, "id.resp_h": dst, "id.resp_p": 53,
        "proto": "udp", "trans_id": int(ts * 1000) % 65536, "query": query,
        "qclass": 1, "qclass_name": "C_INTERNET", "qtype": qtype[0], "qtype_name": qtype[1],
        "rcode": rcode, "rcode_name": "NXDOMAIN" if rcode == 3 else "NOERROR",
        "AA": False, "TC": False, "RD": True, "RA": True, "Z": 0, "rejected": False,
    }
    if answers is not None:
        rec["answers"] = answers
        rec["TTLs"] = ttls
    return rec


def _benign_records(n: int, rng: np.random.Generator, t0: float, duration: float) -> list[dict]:
    clients = [f"10.0.0.{i}" for i in range(2, 22)]
    resolvers = ["10.0.0.1", "8.8.8.8"]
    ranks = np.arange(1, len(_BENIGN_DOMAINS) + 1)
    zipf_p = (1 / ranks) / (1 / ranks).sum()
    out = []
    for _ in range(n):
        ts = t0 + rng.uniform(0, duration)
        name = _BENIGN_DOMAINS[rng.choice(len(_BENIGN_DOMAINS), p=zipf_p)]
        if rng.random() < 0.08:  # CDN-like random-ish label
            name = f"{''.join(rng.choice(list('abcdef0123456789'), size=8))}.{name.split('.', 1)[-1]}"
        qtype = _BENIGN_QTYPES[rng.choice(len(_BENIGN_QTYPES), p=_BENIGN_QTYPE_P)]
        nx = rng.random() < 0.03
        answers = None if nx else [f"93.184.{rng.integers(0, 255)}.{rng.integers(1, 255)}"]
        ttls = None if nx else [float(rng.choice([30, 60, 300, 600, 3600]))]
        src = clients[rng.integers(0, len(clients))]
        out.append(_record(ts, src, resolvers[rng.integers(0, 2)], int(rng.integers(1024, 65535)),
                           name, qtype, 3 if nx else 0, answers, ttls))
    return out


def _tunnel_records(n: int, rng: np.random.Generator, t0: float, duration: float) -> list[dict]:
    clients = ["10.0.0.5", "10.0.0.9", "10.0.0.17"]
    out: list[dict] = []
    while len(out) < n:
        start = t0 + rng.uniform(0, duration)
        src = clients[rng.integers(0, len(clients))]
        domain = _TUNNEL_DOMAINS[rng.integers(0, len(_TUNNEL_DOMAINS))]
        burst = int(rng.integers(20, 60))
        ts = start
        for _ in range(min(burst, n - len(out))):
            ts += float(rng.exponential(0.15))
            raw = rng.integers(0, 256, size=int(rng.integers(20, 34)), dtype=np.uint8).tobytes()
            payload = base64.b32encode(raw).decode().lower().rstrip("=")
            label = payload if len(payload) <= 60 else payload[:60]
            sub = label if len(label) <= 40 else f"{label[:40]}.{label[40:]}"
            qtype = _TUNNEL_QTYPES[rng.choice(len(_TUNNEL_QTYPES), p=_TUNNEL_QTYPE_P)]
            nx = rng.random() < 0.15
            answers = None if nx else ["\"" + payload[:16] + "\""]
            ttls = None if nx else [0.0]
            out.append(_record(ts, src, "10.0.0.1", int(rng.integers(1024, 65535)),
                               f"{sub}.{domain}", qtype, 3 if nx else 0, answers, ttls))
    return out


def generate_records(n_benign: int = 7000, n_tunnel: int = 1000, seed: int = 42,
                     t0: float = 1_700_000_000.0, duration: float = 1800.0) -> tuple[list[dict], list[dict]]:
    """Return ``(benign_records, tunnel_records)`` as Zeek-like dicts sorted by ``ts``."""
    rng = np.random.default_rng(seed)
    benign = sorted(_benign_records(n_benign, rng, t0, duration), key=lambda r: r["ts"])
    tunnel = sorted(_tunnel_records(n_tunnel, rng, t0, duration), key=lambda r: r["ts"])
    return benign, tunnel


def write_dataset(out_dir: str | Path, n_benign: int = 7000, n_tunnel: int = 1000, seed: int = 42,
                  benign_format: str = "jsonl", tunnel_format: str = "json_array",
                  add_noise: bool = False) -> tuple[Path, Path]:
    """Write ``benign.json`` / ``tunnel.json`` (JSONL and JSON array respectively by default)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    benign, tunnel = generate_records(n_benign, n_tunnel, seed)
    benign_path, tunnel_path = out / "benign.json", out / "tunnel.json"
    _write(benign_path, benign, benign_format, add_noise)
    _write(tunnel_path, tunnel, tunnel_format, add_noise)
    (out / "DATA_ORIGIN.txt").write_text(
        "SYNTHETIC smoke-test data generated by src/synthetic_data.py.\n"
        "Metrics computed on it are NOT research results.\n", encoding="utf-8")
    return benign_path, tunnel_path


def _noise_records(base: dict) -> list:
    """Edge cases: empty query, missing answers/TTLs, missing ts, non-object, exact duplicate."""
    empty_q = dict(base, query="", ts=base["ts"] + 0.5)
    no_answers = {k: v for k, v in base.items() if k not in ("answers", "TTLs")}
    no_answers["ts"] = base["ts"] + 0.25
    no_ts = {k: v for k, v in base.items() if k != "ts"}
    return [empty_q, no_answers, no_ts, 12345, dict(base)]


def _write(path: Path, records: list[dict], fmt: str, add_noise: bool) -> None:
    extra = _noise_records(records[0]) if add_noise else []
    if fmt == "json_array":
        with path.open("w", encoding="utf-8") as fh:
            json.dump(records + extra, fh)
        return
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
        if add_noise:
            fh.write('{"ts": 1.0, "query": \n')  # invalid JSON line
            fh.write("\n")                       # blank line (ignored)
            for rec in extra:
                fh.write(json.dumps(rec) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SYNTHETIC Zeek-style DNS logs (smoke test only).")
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--n-benign", type=int, default=7000)
    parser.add_argument("--n-tunnel", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--add-noise", action="store_true", help="append invalid/edge-case records")
    args = parser.parse_args()
    b, t = write_dataset(args.out, args.n_benign, args.n_tunnel, args.seed, add_noise=args.add_noise)
    print(f"Wrote {b} (JSON Lines) and {t} (JSON array). SYNTHETIC DATA - not research results.")


if __name__ == "__main__":
    main()
