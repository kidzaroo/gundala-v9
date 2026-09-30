import socket

import pytest

from src.domain_utils import (
    EMPTY_SLD,
    INVALID_SLD,
    configure_domain_extraction,
    get_domain_kind,
    get_registrable_domain,
    get_subdomain,
    normalize_query,
    parse_domain,
    suffix_list_info,
)


@pytest.fixture(autouse=True)
def _default_config():
    configure_domain_extraction(False)
    yield
    configure_domain_extraction(False)


def test_multilevel_public_suffix():
    assert get_registrable_domain("a.example.co.id") == "example.co.id"
    assert get_registrable_domain("a.b.example.com") == "example.com"
    assert get_registrable_domain("www.bbc.co.uk") == "bbc.co.uk"
    # never "last two labels"
    assert get_registrable_domain("x.y.example.co.id") != "co.id"


def test_subdomain_extraction_is_separate_function():
    assert get_subdomain("a.b.example.com") == "a.b"
    assert get_subdomain("example.com") == ""
    assert get_subdomain("a.example.co.id") == "a"


def test_normalization_lowercase_and_trailing_dot():
    assert normalize_query("WWW.Example.COM.") == "www.example.com"
    assert get_registrable_domain("WWW.Example.COM.") == "example.com"
    assert normalize_query(None) == ""
    assert normalize_query(".") == ""


def test_empty_and_missing_query():
    for q in ("", None, ".", "   "):
        assert get_registrable_domain(q) == EMPTY_SLD
        assert get_domain_kind(q) == "empty"


def test_single_label_ip_reverse_local():
    assert get_domain_kind("wpad") == "single_label"
    assert get_registrable_domain("wpad") == "wpad"
    assert get_domain_kind("192.168.1.10") == "ip"
    assert get_registrable_domain("192.168.1.10") == "192.168.1.10"
    assert get_domain_kind("2001:db8::1") == "ip"
    p = parse_domain("4.3.2.1.in-addr.arpa")
    assert (p.kind, p.registrable, p.subdomain) == ("reverse_dns", "2.1.in-addr.arpa", "4.3")
    p = parse_domain("printer.office.local")
    assert (p.kind, p.registrable, p.subdomain) == ("local", "office.local", "printer")
    assert parse_domain("x.y.home.arpa").registrable == "y.home.arpa"
    assert get_domain_kind("co.id") == "public_suffix"
    assert get_domain_kind("foo.bar.notatld") == "unlisted_suffix"


def test_invalid_queries_are_not_lumped_together():
    a = parse_domain("a..example.com")
    b = parse_domain("bad name.example.org")
    c = parse_domain("x" * 70 + ".foo.net")
    assert a.kind == b.kind == c.kind == "invalid"
    assert len({a.registrable, b.registrable, c.registrable}) == 3
    assert a.registrable == "example.com"
    assert b.registrable == "example.org" and b.subdomain == "bad_name"
    assert parse_domain("...").registrable == EMPTY_SLD  # normalises to empty
    leading = parse_domain("..a")
    assert leading.kind == "invalid" and leading.registrable == "a"


def test_control_characters_are_salvaged_not_crashing():
    p = parse_domain("\x00")
    assert p.kind == "invalid"
    assert p.registrable not in (EMPTY_SLD, INVALID_SLD)
    assert not p.is_valid


def test_works_offline(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    configure_domain_extraction(False)  # rebuild extractor under the patch
    assert get_registrable_domain("a.example.co.id") == "example.co.id"


def test_suffix_list_info_documents_source():
    info = suffix_list_info()
    assert info["library"] == "tldextract"
    assert "network" in info["source"]
    assert info["library_version"] != ""


def test_private_suffix_option_changes_grouping():
    assert get_registrable_domain("user.github.io") == "github.io"
    configure_domain_extraction(True)
    assert get_registrable_domain("user.github.io") == "user.github.io"
