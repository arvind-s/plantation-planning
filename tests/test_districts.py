import geopandas as gpd
import pytest
from shapely.geometry import Polygon, box

from plantation import districts


def _gdf(rows, crs=4326):
    return gpd.GeoDataFrame(
        {"District": [r[0] for r in rows], "State": [r[1] for r in rows]},
        geometry=[r[2] for r in rows], crs=crs)


def test_same_district_name_in_two_states_kept_separate(tmp_path):
    p = tmp_path / "d.shp"
    _gdf([("Aska", "OD", box(84, 19, 85, 20)),
          ("Aska", "KA", box(75, 13, 76, 14))]).to_file(p)
    g = districts.load_districts(str(p))
    assert len(g) == 2
    assert set(g["key"]) == {"OD|Aska", "KA|Aska"}
    assert list(g["district_id"]) == [0, 1]
    assert list(g["state"]) == ["KA", "OD"]          # ordered by state


def test_duplicate_state_district_pair_rejected(tmp_path):
    p = tmp_path / "d.shp"
    _gdf([("A", "KA", box(0, 0, 1, 1)), ("A", "KA", box(2, 2, 3, 3))]).to_file(p)
    with pytest.raises(ValueError, match="not unique"):
        districts.load_districts(str(p))


def test_invalid_geometry_repaired_and_reprojected(tmp_path):
    bow = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])    # self-intersecting
    assert not bow.is_valid
    p = tmp_path / "d.shp"
    _gdf([("Bow", "KA", bow)]).to_crs(3857).to_file(p)
    g = districts.load_districts(str(p))
    assert g.crs.to_epsg() == 4326
    assert g.geometry.is_valid.all()
    assert not g.geometry.is_empty.any()


def test_write_geojson_keeps_key(tmp_path):
    p = tmp_path / "d.shp"
    _gdf([("A", "KA", box(0, 0, 1, 1))]).to_file(p)
    g = districts.load_districts(str(p))
    out = tmp_path / "d.geojson"
    districts.write_geojson(g, str(out))
    back = gpd.read_file(out)
    assert list(back["key"]) == ["KA|A"]


def test_bbox_pads():
    g = _gdf([("A", "KA", box(0, 0, 1, 1))])
    assert districts.bbox(g, pad=0.5) == (-0.5, -0.5, 1.5, 1.5)
