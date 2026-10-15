import hashlib
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

import build_dataset as bd
from plantation import districts

LATS = np.array([1.5, 0.5])
LONS = np.array([0.5, 1.5])


def _gdf():
    g = gpd.GeoDataFrame({"District": ["A", "B"], "State": ["KA", "KA"]},
                         geometry=[box(0, 0, 1, 2), box(1, 0, 2, 2)], crs=4326)
    return g


def _loader(year):
    days = 366 if year % 4 == 0 else 365
    stack = np.zeros((days, 2, 2))
    stack[:, :, 0] = 2.0          # district A column: 2 mm/day
    stack[:, :, 1] = 4.0          # district B column: 4 mm/day
    return stack, None, LATS, LONS, 0


def _prep(tmp_path):
    p = tmp_path / "d.shp"
    _gdf().to_file(p)
    return districts.load_districts(str(p))


def test_build_source_totals_and_mean(tmp_path):
    gdf = _prep(tmp_path)
    df = bd.build_source(_loader, gdf, [2023, 2024], res=1.0)
    jj = df[df.period == "JJASOND"].set_index("district")
    assert jj.loc["A", "y2023"] == 2.0 * 214
    assert jj.loc["B", "mean_mm"] == 4.0 * 214
    jd = df[df.period == "JANDEC"].set_index("district")
    assert jd.loc["A", "y2024"] == 2.0 * 366           # leap year
    assert jd.loc["A", "mean_mm"] == 2.0 * (365 + 366) / 2
    assert len(df) == 4 and not df["mean_mm"].isna().any()


def test_build_is_deterministic(tmp_path):
    gdf = _prep(tmp_path)
    paths = []
    for i in range(2):
        df = bd.build_source(_loader, gdf, [2023, 2024], res=1.0)
        p = tmp_path / f"o{i}.parquet"
        df.to_parquet(p, index=False)
        paths.append(p)
    h = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    assert h[0] == h[1]


def test_imd_style_loader_is_aligned_to_calendar(tmp_path):
    gdf = _prep(tmp_path)

    def imd_like(year):
        t = pd.date_range(f"{year}-06-01", f"{year}-12-31", freq="D")   # only Jun-Dec present
        st = np.full((len(t), 2, 2), 3.0)
        return st, t, LATS, LONS, 0
    df = bd.build_source(imd_like, gdf, [2023], res=1.0)
    jd = df[df.period == "JANDEC"].iloc[0]
    assert jd.missing_days == 151                      # Jan 1 - May 31 absent
    assert jd.y2023 == 3.0 * 214                       # skipped, not zero-filled


def test_shifted_grid_in_later_year_rejected(tmp_path):
    gdf = _prep(tmp_path)

    def shifted(year):
        stack, t, lats, lons, m = _loader(year)
        return stack, t, (lats if year == 2023 else lats + 0.5), lons, m
    with pytest.raises(ValueError, match="grid"):
        bd.build_source(shifted, gdf, [2023, 2024], res=1.0)


def test_manifest_merges_sources_and_hashes_components(tmp_path):
    shp = tmp_path / "x.shp"
    for ext in (".shp", ".dbf", ".shx", ".prj"):
        (tmp_path / f"x{ext}").write_bytes(ext.encode())
    out = tmp_path / "out"
    out.mkdir()
    pq = out / "district_season_gsmap.parquet"
    pq.write_bytes(b"g")
    bd.write_manifest(str(out), str(shp), [2016], 3, "gsmap", 2, 0.1, str(pq))
    pq2 = out / "district_season_imd.parquet"
    pq2.write_bytes(b"i")
    bd.write_manifest(str(out), str(shp), [2016], 3, "imd", 2, 0.25, str(pq2))
    m = json.loads((out / "manifest.json").read_text())
    assert set(m["sources"]) == {"gsmap", "imd"}
    assert "GSMaP_Gauge v6 daily_G 00Z-23Z" in m["sources"]["gsmap"]["product"]
    assert "IMD Pune" in m["sources"]["imd"]["product"]
    assert m["sources"]["imd"]["imdlib_version"]
    assert set(m["input_shp_sha256"]) == {"x.shp", "x.dbf", "x.shx", "x.prj"}
    first = (out / "manifest.json").read_text()
    bd.write_manifest(str(out), str(shp), [2016], 3, "imd", 2, 0.25, str(pq2))
    assert (out / "manifest.json").read_text() == first
