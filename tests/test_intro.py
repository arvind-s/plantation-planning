import xml.etree.ElementTree as ET

from plantation import intro


def test_banner_is_valid_svg_with_no_external_references():
    svg = intro.banner_svg()
    root = ET.fromstring(svg)                          # well-formed XML
    assert root.tag.endswith("svg")
    assert root.attrib["viewBox"] == "0 0 1200 220"
    assert "http://" not in svg.replace("http://www.w3.org/2000/svg", "")
    assert "https://" not in svg and "href" not in svg   # the app makes no network calls


def test_banner_is_deterministic_and_colourful():
    assert intro.banner_svg() == intro.banner_svg()
    colours = {c.lower() for c in __import__("re").findall(r"#[0-9a-fA-F]{6}", intro.banner_svg())}
    assert len(colours) >= 10                          # sky, sun, clouds, rain, hills, trees


def test_title_is_plantation_specific():
    assert "Where to Plant Next" in intro.TITLE
    assert intro.SUBTITLE == "Decadal precip assessment (JJASOND)"


def test_about_covers_what_how_and_limits():
    t = intro.ABOUT_MD
    for needle in ("What it measures", "How to use it", "How the number is made",
                   "Reading the results", "Limits", "GSMaP", "IMD", "800", "2016", "2025"):
        assert needle in t, needle
    assert "suitability" in t                          # says it is not a suitability score
