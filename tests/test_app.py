import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box
from streamlit.testing.v1 import AppTest

from plantation import figures, metrics


def _write_fixture(d):
    g = gpd.GeoDataFrame(
        {"key": ["KA|A", "TN|B"], "state": ["KA", "TN"], "district": ["A", "B"]},
        geometry=[box(76, 12, 77, 13), box(78, 10, 79, 11)], crs=4326)
    g.to_file(d / "districts.geojson", driver="GeoJSON")
    rows = []
    for st, di, m in (("KA", "A", 900.0), ("TN", "B", 700.0)):
        for per, add in (("JJASOND", 0), ("JANDEC", 150)):
            rows.append({"state": st, "district": di, "period": per, "y2025": m + add,
                         "mean_mm": m + add, "n_years": 1, "n_cells": 3,
                         "missing_days": 0, "low_cells": False})
    df = pd.DataFrame(rows)
    for s in ("gsmap", "imd"):
        df.to_parquet(d / f"district_season_{s}.parquet", index=False)


@pytest.fixture
def app(tmp_path, monkeypatch):
    _write_fixture(tmp_path)
    monkeypatch.setenv("PLANTATION_DATA_DIR", str(tmp_path))
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    return at


def test_title_and_subtitle(app):
    assert not app.exception
    assert app.title[0].value == "plantation_planning"
    assert "Decadal precip assessment" in app.caption[0].value


def test_default_threshold_800_counts_one_shortlisted(app):
    assert app.sidebar.number_input(key="threshold").value == 800
    shortlisted = [m for m in app.metric if m.label == "Shortlisted"][0]
    assert shortlisted.value.startswith("1")


def test_threshold_editable(app):
    app.sidebar.number_input(key="threshold").set_value(600).run()
    shortlisted = [m for m in app.metric if m.label == "Shortlisted"][0]
    assert shortlisted.value.startswith("2")


def test_period_filter_default_junDec_and_switch(app):
    assert app.sidebar.radio(key="period").value == "Jun-Dec"
    app.sidebar.radio(key="period").set_value("Jan-Dec").run()
    assert not app.exception


def test_empty_filter_shows_message_not_error(app):
    app.sidebar.multiselect(key="states").set_value(["KA"]).run()
    app.sidebar.text_input(key="query").set_value("zzz").run()
    assert not app.exception
    assert any("No districts" in i.value for i in app.info)


def test_choropleth_has_two_traces():
    df = pd.DataFrame({"state": ["KA", "TN"], "district": ["A", "B"], "key": ["KA|A", "TN|B"],
                       "mean_mm": [900.0, 700.0], "n_cells": [3, 1], "low_cells": [False, True],
                       "missing_days": [0, 2]})
    view = metrics.shortlist(df.assign(period="JJASOND"), "JJASOND", 800)
    geo = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"key": "KA|A"},
         "geometry": {"type": "Polygon", "coordinates": [[[76, 12], [77, 12], [77, 13], [76, 12]]]}},
        {"type": "Feature", "properties": {"key": "TN|B"},
         "geometry": {"type": "Polygon", "coordinates": [[[78, 10], [79, 10], [79, 11], [78, 10]]]}}]}
    fig = figures.choropleth(geo, view, 800)
    assert len(fig.data) == 2
    t = figures.display_table(view)
    assert list(t["Shortlisted"]) == ["Yes", "No"]
    assert "few grid cells" in t.loc[1, "Notes"] and "data gaps" in t.loc[1, "Notes"]


def test_shortlisted_color_range_brackets_threshold():
    geo = {"type": "FeatureCollection", "features": []}
    for means in ([900.0, 1200.0], [800.0], [900.0, 900.0]):
        df = pd.DataFrame({"state": "KA", "district": [f"D{i}" for i in range(len(means))],
                           "mean_mm": means, "n_cells": 3, "low_cells": False, "missing_days": 0})
        df["key"] = df["state"] + "|" + df["district"]
        view = metrics.shortlist(df.assign(period="JJASOND"), "JJASOND", 800)
        fig = figures.choropleth(geo, view, 800)
        tr = [t for t in fig.data if t.showscale is not False][0]
        assert tr.zmin < 800 < tr.zmax
        assert tr.zmin != tr.zmax
