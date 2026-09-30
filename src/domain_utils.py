"""Domain normalisation and registrable-domain (eTLD+1) / subdomain extraction.

Definition of SLD in this project
---------------------------------
"SLD" means the *registrable domain* (eTLD+1) according to the Public Suffix List
(PSL): ``a.b.example.com -> example.com`` and ``a.example.co.id -> example.co.id``.

Offline / reproducibility
-------------------------
``tldextract`` is configured with ``suffix_list_urls=()`` and ``cache_dir=None`` so it
NEVER touches the network and never reads a per-user cache; it uses the PSL snapshot
bundled inside the installed ``tldextract`` wheel. The snapshot identity (package
version + SHA-256 of the snapshot file) is reported by :func:`suffix_list_info` and is
written into ``dataset_summary.json``. Pin the ``tldextract`` version to reproduce runs.

Special cases (``kind`` values)
-------------------------------
``registrable``   normal name under a known public suffix.
``public_suffix`` the query *is* a public suffix (``com``, ``co.id``): SLD = query itself.
``single_label``  one label (``wpad``, ``localhost``): SLD = the label itself.
``local``         suffix in the configurable local list (``.local``, ``.lan``, ...):
                  SLD = one label + local suffix.
``unlisted_suffix`` last label unknown to the PSL: SLD = last two labels.
``ip``            query is an IPv4/IPv6 literal: SLD = the literal itself (one group per IP).
``reverse_dns``   ``*.in-addr.arpa`` / ``*.ip6.arpa``: SLD = the two (IPv4) / four (IPv6)
                  labels nearest the suffix + suffix, i.e. the /16 network.
``invalid``       syntax violation (empty label, label > 63, name > 253, characters outside
                  ``[a-z0-9_*-]``). Characters are replaced by ``_`` and empty labels are
                  dropped, then the PSL rules are applied to the salvaged name, so invalid
                  queries are *not* lumped into one giant group; only if nothing is left is
                  the reserved SLD ``__invalid__`` used.
``empty``         missing/empty query: reserved SLD ``__empty__``. Because grouping is on
                  ``(src_ip, SLD)``, these form one group per source IP.
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
from functools import lru_cache
from importlib import metadata, resources
from typing import NamedTuple, Optional

EMPTY_SLD = "__empty__"
INVALID_SLD = "__invalid__"

DEFAULT_LOCAL_SUFFIXES = (
    "local", "lan", "home", "internal", "localdomain", "corp", "intranet", "home.arpa",
)

_VALID_CHARS = re.compile(r"^[a-z0-9_*-]+$")
_INVALID_CHAR = re.compile(r"[^a-z0-9_*-]")

_state: dict = {
    "include_private": False,
    "local_suffixes": tuple(DEFAULT_LOCAL_SUFFIXES),
    "extractor": None,
}


class DomainParts(NamedTuple):
    normalized: str
    subdomain: str
    registrable: str
    kind: str

    @property
    def is_valid(self) -> bool:
        return self.kind not in ("empty", "invalid")


def configure_domain_extraction(
    include_private_suffixes: bool = False,
    local_suffixes: Optional[list[str] | tuple[str, ...]] = None,
) -> None:
    """(Re)configure extraction. Clears caches so results stay consistent."""
    _state["include_private"] = bool(include_private_suffixes)
    if local_suffixes is not None:
        _state["local_suffixes"] = tuple(s.strip().lower().strip(".") for s in local_suffixes if s.strip())
    _state["extractor"] = None
    parse_domain.cache_clear()


def _get_extractor():
    if _state["extractor"] is None:
        import tldextract

        _state["extractor"] = tldextract.TLDExtract(
            suffix_list_urls=(),  # never fetch the PSL at runtime
            cache_dir=None,       # no on-disk cache -> only the bundled snapshot
            fallback_to_snapshot=True,
            include_psl_private_domains=_state["include_private"],
        )
    return _state["extractor"]


def suffix_list_info() -> dict:
    """Describe the PSL source used (for documentation/reproducibility)."""
    info: dict = {
        "library": "tldextract",
        "library_version": "unknown",
        "source": "snapshot bundled with the installed tldextract package (no network)",
        "include_private_suffixes": _state["include_private"],
        "snapshot_sha256": "unknown",
    }
    try:
        info["library_version"] = metadata.version("tldextract")
    except metadata.PackageNotFoundError:
        pass
    try:
        data = resources.files("tldextract").joinpath(".tld_set_snapshot").read_bytes()
        info["snapshot_sha256"] = hashlib.sha256(data).hexdigest()
    except Exception:  # pragma: no cover - depends on packaging details
        pass
    return info


def normalize_query(query: object) -> str:
    """Lowercase, strip whitespace and trailing dot(s). ``None``/non-str -> ``""``."""
    if not isinstance(query, str):
        return ""
    q = query.strip().lower().rstrip(".")
    return q


def is_ip_literal(q: str) -> bool:
    if not q or not (":" in q or q[0].isdigit()):
        return False
    try:
        ipaddress.ip_address(q)
        return True
    except ValueError:
        return False


def is_valid_domain_syntax(q: str) -> bool:
    """RFC-1035 style length limits plus a restricted character set (see module doc)."""
    if not q or len(q) > 253:
        return False
    for label in q.split("."):
        if not label or len(label) > 63 or not _VALID_CHARS.match(label):
            return False
    return True


def _reverse_suffix(q: str) -> Optional[tuple[str, int]]:
    if q == "in-addr.arpa" or q == "ip6.arpa":
        return None
    if q.endswith(".in-addr.arpa"):
        return "in-addr.arpa", 2
    if q.endswith(".ip6.arpa"):
        return "ip6.arpa", 4
    return None


def _split_with_psl(q: str) -> tuple[str, str, str]:
    """Return ``(subdomain, registrable, kind)`` for a syntactically usable name."""
    labels = q.split(".")
    if len(labels) == 1:
        return "", q, "single_label"
    for suffix in _state["local_suffixes"]:
        if q == suffix:
            return "", q, "public_suffix"
        if q.endswith("." + suffix):
            n_suffix = suffix.count(".") + 1
            reg = ".".join(labels[-(n_suffix + 1):])
            sub = ".".join(labels[: -(n_suffix + 1)])
            return sub, reg, "local"
    res = _get_extractor()(q)
    if res.suffix:
        if res.domain:
            return res.subdomain, f"{res.domain}.{res.suffix}", "registrable"
        return "", q, "public_suffix"
    return ".".join(labels[:-2]), ".".join(labels[-2:]), "unlisted_suffix"


@lru_cache(maxsize=500_000)
def parse_domain(query: str) -> DomainParts:
    """Normalise ``query`` and split it into subdomain / registrable domain / kind."""
    q = normalize_query(query)
    if not q:
        return DomainParts("", "", EMPTY_SLD, "empty")
    if is_ip_literal(q):
        return DomainParts(q, "", q, "ip")
    rev = _reverse_suffix(q)
    if rev is not None:
        suffix, keep = rev
        before = q[: -(len(suffix) + 1)].split(".")
        sld_labels = before[-keep:]
        sub = ".".join(before[:-keep]) if len(before) > keep else ""
        return DomainParts(q, sub, ".".join(sld_labels + [suffix]), "reverse_dns")
    if is_valid_domain_syntax(q):
        sub, reg, kind = _split_with_psl(q)
        return DomainParts(q, sub, reg, kind)

    salvaged = ".".join(_INVALID_CHAR.sub("_", label) for label in q.split(".") if label)
    if not salvaged:
        return DomainParts(q, "", INVALID_SLD, "invalid")
    sub, reg, _ = _split_with_psl(salvaged)
    return DomainParts(q, sub, reg, "invalid")


def get_registrable_domain(query: object) -> str:
    """Registrable domain (eTLD+1) used as the SLD grouping key."""
    return parse_domain(query if isinstance(query, str) else "").registrable


def get_subdomain(query: object) -> str:
    """Everything left of the registrable domain (``""`` if none)."""
    return parse_domain(query if isinstance(query, str) else "").subdomain


def get_domain_kind(query: object) -> str:
    return parse_domain(query if isinstance(query, str) else "").kind
