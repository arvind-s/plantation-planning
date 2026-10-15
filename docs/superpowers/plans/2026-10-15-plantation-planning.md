# plantation_planning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit app, "plantation_planning" (subtitle "Decadal precip assessment (JJASOND)"), showing each of 156 districts' 10-year mean cumulative rainfall (mm) from GSMaP or IMD, with an editable threshold (default 800 mm) and a Jun-Dec (default) / Jan-Dec period filter.

**Architecture:** An offline build script turns daily gridded rainfall (2016-2025, full year) into per-district annual totals using area-weighted cell means, and writes small committed parquet files. The Streamlit app only reads those files (no network calls). GSMaP and IMD readers are copied from `~/Documents/Agroforestry/rainfall_app/rainfall/` so the repo deploys on its own.

**Tech Stack:** Python, geopandas/shapely, numpy/scipy.sparse, pandas/pyarrow, imdlib (build only), Streamlit, Plotly, pytest.

**Spec:** `docs/superpowers/specs/2026-10-15-plantation-planning-design.md`

## Global Constraints

- Years are 2016-2025 inclusive (`YEARS = list(range(2016, 2026))`).
- Periods are exactly two, keyed `JJASOND` (Jun 1-Dec 31, default) and `JANDEC` (Jan 1-Dec 31).
- Threshold is a sidebar number input, default 800 mm, editable; a district is shortlisted when `mean_mm >= threshold` (equal is included).
- Rows are keyed by (`state`, `district`), never by district name alone.
- Per-district daily value is the area-weighted mean over valid grid cells; missing days are skipped (not zero-filled) and counted in `missing_days`.
- GSMaP: daily mm = value (mm/hr) x 24; filename version tag resolved by listing the month directory. IMD: mm/day directly, nodata -999.
- Output must be deterministic: same input gives the same bytes; sorted rows, fixed column order, no timestamps. Reruns skip cached days.
- The app makes no network calls and has no suitability score.
- Own git repo at `plantation_planning/`; GitHub target `arvind-s/plantation-planning`. Do not push without asking the user. Never push the `/Users/mipl` home repo.
- Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- Run Python as `python` (base anaconda; geopandas, scipy, imdlib, streamlit, plotly, pyarrow are installed). Run Python that reads downloaded data with `-I`.

## Review Focus

Failure modes the spec implies; each has a test in the owning task.

1. Filter leaves zero districts (e.g. a state and search that do not intersect): app must show a message, not crash. (Task 6)
2. Threshold exactly equal to a district's mean: included. (Task 3)
3. Transient FTP error must not be silently treated as a missing day; only a genuinely absent file is a missing day. (Task 4)
4. District whose overlapping IMD cells are all sea/nodata: falls back to the nearest valid cell, and is flagged `low_cells`. (Task 2)
5. Same district name in two states: kept as two separate rows. (Task 1)

## File Structure

```
plantation_planning/
  app.py
  build_dataset.py
  plantation/__init__.py
  plantation/districts.py     load/repair/reproject shapefile; write simplified geojson
  plantation/weights.py       area-overlap cell weights (sparse) per source grid
  plantation/metrics.py       daily district means, yearly totals, table, shortlist
  plantation/gsmap.py         JAXA FTP reader (copied) + load_stack
  plantation/imd.py           imdlib reader (copied) + year_stack
  plantation/figures.py       choropleth + display table
  tests/ (one file per module)
  data/out/                   committed outputs
  data/raw_cache/             day cache, gitignored
  requirements.txt  requirements-build.txt  .gitignore  README.md
```

Note: the spec lists cell weights under `districts.py`; this plan puts them in `weights.py` to keep each file to one responsibility.

---

### Task 1: Project scaffold and district loader

**Files:**
- Create: `plantation/__init__.py`, `plantation/districts.py`, `tests/test_districts.py`, `requirements.txt`, `requirements-build.txt`, `.gitignore`, `pytest.ini`

**Interfaces:**
- Produces: `districts.load_districts(shp_path: str) -> GeoDataFrame` with columns `district_id` (int, 0..N-1, ordered by state then district), `state`, `district`, `key` (`"state|district"`), `geometry` (valid, EPSG:4326). Raises `ValueError` if (`state`, `district`) is not unique or any geometry is empty.
- Produces: `districts.write_geojson(gdf, path, tolerance=0.005)` writing features with properties `key`, `state`, `district`.
- Produces: `districts.bbox(gdf, pad=0.5) -> (minx, miny, maxx, maxy)`.

- [ ] **Step 1: Scaffold files**

`.gitignore`:
```
__pycache__/
.pytest_cache/
data/raw_cache/
.ipynb_checkpoints/
```
`pytest.ini`:
```
[pytest]
testpaths = tests
```
`requirements.txt`:
```
streamlit>=1.40
pandas>=2.0
numpy>=1.24
pyarrow>=14.0
plotly>=5.24
```
`requirements-build.txt`:
```
-r requirements.txt
geopandas>=0.14
shapely>=2.0
scipy>=1.10
imdlib>=0.1.2
```
`plantation/__init__.py`: empty file. Run `mkdir -p tests data/out data/raw_cache`.

- [ ] **Step 2: Write the failing tests**

`tests/test_districts.py`:
```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_districts.py -v`
Expected: FAIL (`ImportError`/`cannot import name 'districts'`).

- [ ] **Step 4: Implement**

`plantation/districts.py`:
```python
"""Load the 5-state district shapefile: repair, reproject, key, simplify."""
from __future__ import annotations

import geopandas as gpd
from shapely import make_valid


def load_districts(shp_path: str) -> gpd.GeoDataFrame:
    g = gpd.read_file(shp_path)
    for col in ("District", "State"):
        if col not in g.columns:
            raise ValueError(f"column '{col}' not in shapefile: {list(g.columns)}")
    if g.crs is None:
        raise ValueError("shapefile has no CRS")
    g = g.to_crs(4326)
    g["district"] = g["District"].astype(str).str.strip()
    g["state"] = g["State"].astype(str).str.strip()
    if g.duplicated(["state", "district"]).any():
        raise ValueError("(state, district) is not unique in the shapefile")
    g["geometry"] = g.geometry.apply(lambda geom: make_valid(geom) if geom is not None else geom)
    if g.geometry.isna().any() or g.geometry.is_empty.any():
        raise ValueError("district(s) with empty geometry")
    g = g.sort_values(["state", "district"], kind="stable").reset_index(drop=True)
    g["district_id"] = g.index.astype(int)
    g["key"] = g["state"] + "|" + g["district"]
    return g[["district_id", "state", "district", "key", "geometry"]]


def write_geojson(gdf: gpd.GeoDataFrame, path: str, tolerance: float = 0.005) -> None:
    out = gdf[["key", "state", "district", "geometry"]].copy()
    out["geometry"] = out.geometry.simplify(tolerance, preserve_topology=True)
    out.to_file(path, driver="GeoJSON")


def bbox(gdf: gpd.GeoDataFrame, pad: float = 0.5) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = gdf.total_bounds
    return (minx - pad, miny - pad, maxx + pad, maxy + pad)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_districts.py -v`
Expected: 5 passed.

- [ ] **Step 6: Check against the real shapefile**

Run (from `plantation_planning/`): `PYTHONPATH=. python -c "from plantation import districts as d; g=d.load_districts('../input/5_states_all_districts.shp'); print(len(g), g['state'].value_counts().to_dict(), g.geometry.is_valid.all(), d.bbox(g))"`
Expected: `156`, counts `AP 26, KA 30, OD 30, TN 37, TS 33`, `True`, and a bbox covering roughly lon 74-88, lat 8-23.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: scaffold project and district loader

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Cell weights

**Files:**
- Create: `plantation/weights.py`, `tests/test_weights.py`

**Interfaces:**
- Produces: `weights.build_weights(geoms, lats, lons, res, valid_mask=None) -> (W, n_cells)`
  - `geoms`: sequence of shapely polygons in lon/lat degrees (index = district order).
  - `lats`, `lons`: 1-D arrays of cell-centre coordinates (any order). `res`: cell size in degrees.
  - `valid_mask`: optional bool array `(len(lats), len(lons))`; False cells are ignored.
  - `W`: `scipy.sparse.csr_matrix` shape `(D, nlat*nlon)`, each row sums to 1; column index = `row*nlon + col`.
  - `n_cells`: int array length D, number of cells with weight (1 for a fallback district).

- [ ] **Step 1: Write the failing tests**

`tests/test_weights.py`:
```python
import numpy as np
from shapely.geometry import box

from plantation import weights

LATS = np.array([2.5, 1.5, 0.5])        # descending, like GSMaP
LONS = np.array([0.5, 1.5, 2.5])
RES = 1.0


def _dense(W):
    return W.toarray().reshape(W.shape[0], 3, 3)


def test_weights_proportional_to_overlap_and_sum_to_one():
    # covers cell (lat .5,lon .5) fully and half of (lat .5, lon 1.5)
    g = box(0, 0, 1.5, 1)
    W, n = weights.build_weights([g], LATS, LONS, RES)
    d = _dense(W)[0]
    assert np.isclose(d[2, 0], 2 / 3) and np.isclose(d[2, 1], 1 / 3)
    assert np.isclose(W.sum(), 1.0)
    assert n[0] == 2


def test_district_inside_one_cell_gets_that_cell():
    W, n = weights.build_weights([box(1.2, 1.2, 1.4, 1.4)], LATS, LONS, RES)
    assert np.isclose(_dense(W)[0][1, 1], 1.0) and n[0] == 1


def test_invalid_cells_ignored_and_renormalised():
    valid = np.ones((3, 3), bool)
    valid[2, 1] = False                    # drop the half-covered cell
    W, n = weights.build_weights([box(0, 0, 1.5, 1)], LATS, LONS, RES, valid)
    assert np.isclose(_dense(W)[0][2, 0], 1.0) and n[0] == 1


def test_all_overlapping_cells_invalid_falls_back_to_nearest_valid():
    valid = np.zeros((3, 3), bool)
    valid[0, 2] = True                     # only far corner (lat 2.5, lon 2.5) valid
    W, n = weights.build_weights([box(0.1, 0.1, 0.9, 0.9)], LATS, LONS, RES, valid)
    assert np.isclose(_dense(W)[0][0, 2], 1.0)
    assert n[0] == 1


def test_matrix_shape():
    W, _ = weights.build_weights([box(0, 0, 1, 1), box(2, 2, 3, 3)], LATS, LONS, RES)
    assert W.shape == (2, 9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_weights.py -v`
Expected: FAIL (`cannot import name 'weights'`).

- [ ] **Step 3: Implement**

`plantation/weights.py`:
```python
"""Area-overlap weights mapping grid cells to district polygons."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from shapely import box


def _nearest_valid(geom, lats, lons, valid_mask):
    p = geom.representative_point()
    rr, cc = np.meshgrid(np.arange(len(lats)), np.arange(len(lons)), indexing="ij")
    d2 = (lats[rr] - p.y) ** 2 + (lons[cc] - p.x) ** 2
    if valid_mask is not None:
        d2 = np.where(valid_mask, d2, np.inf)
    r, c = np.unravel_index(np.argmin(d2), d2.shape)
    return int(r), int(c)


def build_weights(geoms, lats, lons, res, valid_mask=None):
    lats = np.asarray(lats, dtype="float64")
    lons = np.asarray(lons, dtype="float64")
    nlat, nlon = len(lats), len(lons)
    half = res / 2.0
    rows, cols, vals = [], [], []
    n_cells = np.zeros(len(geoms), dtype=int)
    for d, geom in enumerate(geoms):
        minx, miny, maxx, maxy = geom.bounds
        ri = np.where((lats + half >= miny) & (lats - half <= maxy))[0]
        ci = np.where((lons + half >= minx) & (lons - half <= maxx))[0]
        found = []
        for r in ri:
            for c in ci:
                if valid_mask is not None and not valid_mask[r, c]:
                    continue
                cell = box(lons[c] - half, lats[r] - half, lons[c] + half, lats[r] + half)
                a = geom.intersection(cell).area
                if a > 0:
                    found.append((int(r), int(c), a))
        if not found:
            r, c = _nearest_valid(geom, lats, lons, valid_mask)
            found = [(r, c, 1.0)]
        total = sum(a for _, _, a in found)
        for r, c, a in found:
            rows.append(d)
            cols.append(r * nlon + c)
            vals.append(a / total)
        n_cells[d] = len(found)
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(len(geoms), nlat * nlon))
    return W, n_cells
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_weights.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: area-overlap cell weights with nearest-valid fallback

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Metrics (daily means, yearly totals, table, shortlist)

**Files:**
- Create: `plantation/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Consumes: `W` from `weights.build_weights`.
- Produces (all in `metrics`):
  - `PERIODS = {"JJASOND": range(6, 13), "JANDEC": range(1, 13)}`
  - `district_daily(stack, W) -> np.ndarray (T, D)`: `stack` is `(T, nlat, nlon)` mm/day with NaN for missing. Weights renormalised over valid cells; NaN where a district has no valid cell that day.
  - `align_days(stack, times, year) -> np.ndarray`: places `stack` (T, ...) on the full `Jan 1..Dec 31` calendar of `year` (365 or 366 days), NaN for absent days.
  - `yearly_totals(daily, year) -> DataFrame` columns `district_id, period, year, total_mm, missing_days`. `daily` is `(n_days_in_year, D)` ndarray aligned to the full calendar.
  - `build_table(totals, meta, years) -> DataFrame` columns in this order: `state, district, period, y{year}..., mean_mm, n_years, n_cells, missing_days, low_cells`, sorted by `state, district, period`, floats rounded to 2 dp. `meta` has `district_id, state, district, n_cells`; `totals` is the concatenation of `yearly_totals` over years.
  - `shortlist(df, period, threshold, states=None, query="") -> DataFrame` filtered rows plus bool column `above`, sorted by `mean_mm` descending (stable).

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:
```python
import numpy as np
import pandas as pd
from scipy import sparse

from plantation import metrics


def _W():
    # 2 districts over a 1x2 grid: d0 = cell0, d1 = 50/50 of both cells
    return sparse.csr_matrix(np.array([[1.0, 0.0], [0.5, 0.5]]))


def test_district_daily_weighted_mean():
    stack = np.array([[[10.0, 20.0]]])            # T=1, 1x2 grid
    out = metrics.district_daily(stack, _W())
    assert np.allclose(out, [[10.0, 15.0]])


def test_district_daily_renormalises_over_valid_cells():
    stack = np.array([[[np.nan, 20.0]]])
    out = metrics.district_daily(stack, _W())
    assert np.isnan(out[0, 0])                    # d0 has no valid cell
    assert np.isclose(out[0, 1], 20.0)            # d1 uses only the valid cell


def test_align_days_leap_and_missing():
    times = pd.to_datetime(["2024-01-01", "2024-12-31"])
    st = np.ones((2, 1, 1))
    full = metrics.align_days(st, times, 2024)
    assert full.shape[0] == 366                   # leap year
    assert np.isfinite(full[0]).all() and np.isfinite(full[-1]).all()
    assert np.isnan(full[1:-1]).all()


def test_yearly_totals_windows_leap_year():
    n = 366
    daily = np.ones((n, 1))                       # 1 mm every day, 1 district
    t = metrics.yearly_totals(daily, 2024)
    jj = t[t.period == "JJASOND"].iloc[0]
    jd = t[t.period == "JANDEC"].iloc[0]
    assert jd.total_mm == 366                     # Feb 29 included
    assert jj.total_mm == 30 + 31 + 31 + 30 + 31 + 30 + 31 == 214
    assert jd.missing_days == 0


def test_yearly_totals_skips_missing_and_counts():
    daily = np.ones((365, 1))
    daily[200:205, 0] = np.nan                    # 5 missing days in Jul
    t = metrics.yearly_totals(daily, 2023)
    jj = t[t.period == "JJASOND"].iloc[0]
    assert jj.missing_days == 5
    assert jj.total_mm == 214 - 5
    all_nan = metrics.yearly_totals(np.full((365, 1), np.nan), 2023)
    assert all_nan["total_mm"].isna().all()


def _table():
    meta = pd.DataFrame({"district_id": [0, 1], "state": ["KA", "KA"],
                         "district": ["B", "A"], "n_cells": [1, 4]})
    rows = []
    for y, (a, b) in {2024: (900.0, 700.0), 2025: (1100.0, 800.0)}.items():
        for did, v in ((0, a), (1, b)):
            for per in ("JJASOND", "JANDEC"):
                rows.append((did, per, y, v if per == "JJASOND" else v + 100, 0))
    totals = pd.DataFrame(rows, columns=["district_id", "period", "year", "total_mm", "missing_days"])
    return metrics.build_table(totals, meta, [2024, 2025])


def test_build_table_columns_mean_and_order():
    df = _table()
    assert list(df.columns) == ["state", "district", "period", "y2024", "y2025",
                                "mean_mm", "n_years", "n_cells", "missing_days", "low_cells"]
    assert list(df["district"]) == ["A", "A", "B", "B"]       # sorted
    b = df[(df.district == "B") & (df.period == "JJASOND")].iloc[0]
    assert b.mean_mm == 1000.0 and b.n_years == 2 and bool(b.low_cells) is True
    a = df[(df.district == "A") & (df.period == "JANDEC")].iloc[0]
    assert a.mean_mm == 850.0 and bool(a.low_cells) is False


def test_shortlist_threshold_boundary_is_inclusive():
    df = _table()
    s = metrics.shortlist(df, "JJASOND", threshold=750.0)     # A mean = exactly 750
    assert dict(zip(s.district, s.above)) == {"B": True, "A": True}
    s2 = metrics.shortlist(df, "JJASOND", threshold=750.01)
    assert dict(zip(s2.district, s2.above)) == {"B": True, "A": False}
    assert list(s.district) == ["B", "A"]                      # mean desc


def test_shortlist_filters_and_can_be_empty():
    df = _table()
    assert metrics.shortlist(df, "JJASOND", 800, states=["TN"]).empty
    assert len(metrics.shortlist(df, "JJASOND", 800, query="a")) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL (`cannot import name 'metrics'`).

- [ ] **Step 3: Implement**

`plantation/metrics.py`:
```python
"""Daily district means, per-year window totals, the output table, shortlist."""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS = {"JJASOND": range(6, 13), "JANDEC": range(1, 13)}


def district_daily(stack: np.ndarray, W) -> np.ndarray:
    """(T, nlat, nlon) mm/day (NaN = missing) -> (T, D) area-weighted means."""
    X = stack.reshape(stack.shape[0], -1)
    valid = np.isfinite(X)
    num = W @ np.where(valid, X, 0.0).T
    den = W @ valid.astype("float64").T
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 1e-12, num / den, np.nan)
    return out.T


def align_days(stack: np.ndarray, times, year: int) -> np.ndarray:
    full = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    out = np.full((len(full),) + stack.shape[1:], np.nan)
    pos = full.get_indexer(pd.DatetimeIndex(times).normalize())
    keep = pos >= 0
    out[pos[keep]] = stack[keep]
    return out


def yearly_totals(daily: np.ndarray, year: int) -> pd.DataFrame:
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    if daily.shape[0] != len(days):
        raise ValueError(f"{year}: expected {len(days)} days, got {daily.shape[0]}")
    rows = []
    for period, months in PERIODS.items():
        sel = np.isin(days.month, list(months))
        block = daily[sel]
        valid = np.isfinite(block)
        total = np.where(valid.any(axis=0), np.nansum(block, axis=0), np.nan)
        missing = (~valid).sum(axis=0)
        for d in range(daily.shape[1]):
            rows.append((d, period, year, float(total[d]), int(missing[d])))
    return pd.DataFrame(rows, columns=["district_id", "period", "year", "total_mm", "missing_days"])


def build_table(totals: pd.DataFrame, meta: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    base = meta.set_index("district_id")[["state", "district", "n_cells"]]
    parts = []
    for period, g in totals.groupby("period"):
        wide = g.pivot(index="district_id", columns="year", values="total_mm").reindex(columns=years)
        miss = g.groupby("district_id")["missing_days"].sum()
        df = base.join(wide)
        df["period"] = period
        ycols = []
        for y in years:
            df = df.rename(columns={y: f"y{y}"})
            ycols.append(f"y{y}")
        df["mean_mm"] = df[ycols].mean(axis=1, skipna=True)
        df["n_years"] = df[ycols].notna().sum(axis=1).astype(int)
        df["missing_days"] = miss.reindex(df.index).astype(int)
        df["low_cells"] = df["n_cells"] < 2
        parts.append(df)
    out = pd.concat(parts)
    cols = ["state", "district", "period"] + [f"y{y}" for y in years] + [
        "mean_mm", "n_years", "n_cells", "missing_days", "low_cells"]
    out = out[cols].sort_values(["state", "district", "period"], kind="stable").reset_index(drop=True)
    num = [f"y{y}" for y in years] + ["mean_mm"]
    out[num] = out[num].round(2)
    return out


def shortlist(df: pd.DataFrame, period: str, threshold: float,
              states: list[str] | None = None, query: str = "") -> pd.DataFrame:
    d = df[df["period"] == period]
    if states:
        d = d[d["state"].isin(states)]
    if query:
        d = d[d["district"].str.contains(query, case=False, regex=False)]
    d = d.assign(above=d["mean_mm"] >= threshold)
    return d.sort_values("mean_mm", ascending=False, kind="stable").reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: district daily means, yearly totals, table, shortlist

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Source readers (GSMaP and IMD)

**Files:**
- Create: `plantation/gsmap.py`, `plantation/imd.py`, `tests/test_sources.py`

**Interfaces:**
- Produces: `gsmap.load_stack(year, bbox, cache_dir, ftp_factory=None, retries=3) -> (stack, lats, lons, missing_days)`
  - `stack`: `(T, nlat, nlon)` float64 mm/day over the window for every day of `year`, in calendar order, NaN for days absent at source.
  - `lats`, `lons`: 1-D window axes. `missing_days`: int, count of days whose file does not exist at source.
  - `ftp_factory`: zero-arg callable returning a logged-in `ftplib.FTP`; default connects to JAXA. Only genuinely absent files (`FileNotFoundError`) count as missing; any other FTP error is retried `retries` times (reconnecting) and then re-raised.
- Produces: `imd.year_stack(year, bbox, cache_dir) -> (stack, lats, lons)` with `stack` shape `(T, nlat, nlon)` mm/day, nodata NaN, covering the whole year (aligned later by `metrics.align_days`), and `times` available via `imd.year_times(da)`. To keep one return shape, this function returns `(stack, times, lats, lons)`.
- Constants consumed later: `gsmap.RES = 0.1`, `imd.RES = 0.25`.

- [ ] **Step 1: Copy the existing readers**

Run:
```bash
cp ~/Documents/Agroforestry/rainfall_app/rainfall/gsmap.py plantation/gsmap.py
cp ~/Documents/Agroforestry/rainfall_app/rainfall/imd.py plantation/imd.py
```

- [ ] **Step 2: Write the failing tests**

`tests/test_sources.py`:
```python
import ftplib
from datetime import date

import numpy as np
import pytest

from plantation import gsmap

BBOX = (77.0, 12.0, 78.0, 13.0)


def test_parse_grid_converts_mm_hr_to_mm_day_and_masks_missing():
    a = np.full((gsmap.NLAT, gsmap.NLON), 0.5, dtype="<f4")
    a[0, 0] = -999.9
    g = gsmap.parse_grid(a.tobytes())
    assert np.isnan(g[0, 0])
    assert np.isclose(g[1, 1], 12.0)               # 0.5 mm/hr x 24


def test_window_slices_cover_bbox():
    rs, cs = gsmap.window_slices(BBOX)
    lat, lon = gsmap.lats()[rs], gsmap.lons()[cs]
    assert lat.min() <= 12.0 and lat.max() >= 13.0
    assert lon.min() <= 77.0 and lon.max() >= 78.0


def _seed_cache(cache, days, value=1.0):
    rs, cs = gsmap.window_slices(BBOX)
    data = np.full((rs.stop - rs.start, cs.stop - cs.start), value)
    for d in days:
        np.savez_compressed(gsmap._cache_path(str(cache), d, BBOX), data=data,
                            lats=gsmap.lats()[rs], lons=gsmap.lons()[cs])


def test_load_stack_uses_cache_without_network(tmp_path):
    year_days = [d.date() for d in __import__("pandas").date_range("2023-01-01", "2023-12-31")]
    _seed_cache(tmp_path, year_days)

    def boom():
        raise AssertionError("network must not be touched on full cache hit")
    stack, lats, lons, missing = gsmap.load_stack(2023, BBOX, str(tmp_path), ftp_factory=boom)
    assert stack.shape[0] == 365 and missing == 0
    assert np.allclose(stack, 1.0)


class _FakeFTP:
    """Month listing is empty -> every uncached day is a genuinely absent file."""
    def nlst(self, path):
        raise ftplib.error_perm("550 no such dir")
    def quit(self):
        pass


def test_absent_file_counts_as_missing_not_error(tmp_path):
    days = [d.date() for d in __import__("pandas").date_range("2023-01-01", "2023-12-31")]
    _seed_cache(tmp_path, days[:-1])                       # all but Dec 31 cached
    gsmap._DIR_CACHE.clear()
    stack, _, _, missing = gsmap.load_stack(2023, BBOX, str(tmp_path), ftp_factory=lambda: _FakeFTP())
    assert missing == 1
    assert np.isnan(stack[-1]).all() and np.isfinite(stack[0]).all()


class _FlakyFTP:
    def nlst(self, path):
        raise ftplib.error_temp("421 service not available")
    def quit(self):
        pass


def test_transient_ftp_error_is_raised_not_counted_missing(tmp_path):
    days = [d.date() for d in __import__("pandas").date_range("2023-01-01", "2023-12-31")]
    _seed_cache(tmp_path, days[:-1])
    gsmap._DIR_CACHE.clear()
    with pytest.raises(ftplib.error_temp):
        gsmap.load_stack(2023, BBOX, str(tmp_path), ftp_factory=lambda: _FlakyFTP(), retries=2)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_sources.py -v`
Expected: parse/slice tests PASS (copied code); the three `load_stack` tests FAIL (`has no attribute 'load_stack'`).

- [ ] **Step 4: Implement `gsmap.load_stack`**

Append to `plantation/gsmap.py` (add `import pandas as pd` to the imports at the top):
```python
def _new_ftp() -> ftplib.FTP:
    ftp = ftplib.FTP(FTP_HOST, timeout=120)
    ftp.login(FTP_USER, FTP_PASS)
    return ftp


def load_stack(year: int, bbox: tuple, cache_dir: str, ftp_factory=None, retries: int = 3):
    """Every day of ``year`` over ``bbox``: (stack[T,nlat,nlon], lats, lons, missing_days).

    Days whose file does not exist at JAXA are NaN and counted as missing. Any
    other FTP failure is retried (reconnecting) and then raised, so a flaky
    connection can never silently under-count rainfall.
    """
    factory = ftp_factory or _new_ftp
    rs, cs = window_slices(bbox)
    wl, wo = lats()[rs], lons()[cs]
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    stack = np.full((len(days), len(wl), len(wo)), np.nan)
    missing = 0
    ftp = None
    try:
        for i, ts in enumerate(days):
            d = ts.date()
            if os.path.exists(_cache_path(cache_dir, d, bbox)):
                stack[i] = fetch_window(d, bbox, cache_dir)["data"]
                continue
            for attempt in range(retries):
                try:
                    if ftp is None:
                        ftp = factory()
                    stack[i] = fetch_window(d, bbox, cache_dir, ftp=ftp)["data"]
                    break
                except FileNotFoundError:
                    missing += 1
                    break
                except (ftplib.all_errors, OSError):
                    try:
                        if ftp is not None:
                            ftp.quit()
                    except Exception:  # noqa: BLE001 -- connection already dead
                        pass
                    ftp = None
                    if attempt == retries - 1:
                        raise
    finally:
        if ftp is not None:
            try:
                ftp.quit()
            except Exception:  # noqa: BLE001
                pass
    return stack, wl, wo, missing
```
Also change `resolve_remote_file` so a failed month listing other than a missing directory is not swallowed: replace the `except ftplib.error_perm:` block with
```python
        except ftplib.error_perm as e:
            if not str(e).startswith("550"):
                raise
            _DIR_CACHE[month] = []
```
Note: `ftplib.error_temp` (421) in the flaky test is a subclass of `ftplib.Error`, which `ftplib.all_errors` covers, so it is retried and then raised.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_sources.py -v`
Expected: 5 passed.

- [ ] **Step 6: Adapt `imd.py`**

In `plantation/imd.py`: add `RES = 0.25` below `FILL_BELOW`, delete `realtime_day`, `_rt_cache_path` and `SEASON_MONTHS` (not needed), and replace `season_window`, `window_axes`, `season_times` with:
```python
def year_stack(year: int, bbox: tuple[float, float, float, float], cache_dir: str):
    """Whole-year IMD rainfall (mm/day) over ``bbox``: (stack[T,nlat,nlon], times, lats, lons).

    Nodata (-999) becomes NaN. Uses the local cache if present, otherwise downloads.
    """
    try:
        d = imdlib.open_data("rain", year, year, "yearwise", file_dir=cache_dir)
    except Exception:  # noqa: BLE001 -- not cached yet
        d = imdlib.get_data("rain", year, year, fn_format="yearwise", file_dir=cache_dir)
    da = d.get_xarray()["rain"]
    da = da.where(da > FILL_BELOW)
    minx, miny, maxx, maxy = bbox
    da = da.sel(lat=slice(miny, maxy), lon=slice(minx, maxx))
    stack = np.asarray(da.values, dtype="float64")
    times = pd.to_datetime(da["time"].values)
    lats = np.asarray(da["lat"].values, dtype="float64")
    lons = np.asarray(da["lon"].values, dtype="float64")
    return stack, times, lats, lons
```
Remove the now-unused `os`, `socket`/`time` imports only if unused (keep `socket.setdefaulttimeout(60)`). Verify import works: `python -c "import plantation.imd"`.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: GSMaP and IMD readers; load_stack distinguishes absent files from FTP errors

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Build script

**Files:**
- Create: `build_dataset.py`, `tests/test_build.py`

**Interfaces:**
- Consumes: `districts.load_districts`, `districts.write_geojson`, `districts.bbox`, `weights.build_weights`, `metrics.district_daily/align_days/yearly_totals/build_table`, `gsmap.load_stack`, `gsmap.RES`, `imd.year_stack`, `imd.RES`.
- Produces: `build_dataset.build_source(loader, gdf, years, res) -> DataFrame` where `loader(year) -> (stack, times_or_None, lats, lons, missing_days)`; `times_or_None` is a DatetimeIndex when the stack is not already a full calendar (IMD), else None (GSMaP, already aligned). Returns the table from `metrics.build_table`. A year's source-reported `missing_days` is added to each district's `missing_days` only if it is not already reflected in NaN data (the table's `missing_days` comes from NaN counts, so `loader` missing counts are logged, not added).
- CLI: `python build_dataset.py --shp ../input/5_states_all_districts.shp --source {gsmap,imd,both} [--years 2016 ... 2025] [--out data/out] [--cache data/raw_cache]`.
- Writes: `{out}/districts.geojson`, `{out}/district_season_{source}.parquet`, `{out}/manifest.json`.

- [ ] **Step 1: Write the failing tests**

`tests/test_build.py`:
```python
import hashlib

import geopandas as gpd
import numpy as np
import pandas as pd
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_build.py -v`
Expected: FAIL (`No module named 'build_dataset'`).

- [ ] **Step 3: Implement**

`build_dataset.py`:
```python
"""Build per-district rainfall tables for plantation_planning.

    python build_dataset.py --shp ../input/5_states_all_districts.shp --source both

Reads daily gridded rainfall (GSMaP via JAXA FTP, IMD via imdlib) for the last
10 full years, computes area-weighted district means, sums per year for the
Jun-Dec and Jan-Dec windows, and writes the files the Streamlit app reads.
Reruns reuse the day cache and produce identical bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from plantation import districts, gsmap, imd, metrics, weights

YEARS = list(range(2016, 2026))


def build_source(loader, gdf, years, res):
    """loader(year) -> (stack, times|None, lats, lons, missing_days)."""
    geoms = list(gdf.geometry)
    W = n_cells = None
    totals = []
    for year in years:
        stack, times, lats, lons, missing = loader(year)
        if W is None:
            valid = np.isfinite(stack).any(axis=0)
            W, n_cells = weights.build_weights(geoms, lats, lons, res, valid_mask=valid)
        if times is not None:
            stack = metrics.align_days(stack, times, year)
        daily = metrics.district_daily(stack, W)
        totals.append(metrics.yearly_totals(daily, year))
        print(f"  {year}: source-reported missing days = {missing}", flush=True)
    meta = gdf[["district_id", "state", "district"]].assign(n_cells=n_cells)
    return metrics.build_table(pd.concat(totals, ignore_index=True), meta, list(years))


def _loader_for(source, bbox, cache):
    if source == "gsmap":
        c = os.path.join(cache, "gsmap")

        def gsmap_loader(y):
            stack, lats, lons, missing = gsmap.load_stack(y, bbox, c)
            return stack, None, lats, lons, missing
        return gsmap_loader, gsmap.RES
    c = os.path.join(cache, "imd")
    def imd_loader(y):
        stack, times, lats, lons = imd.year_stack(y, bbox, c)
        return stack, times, lats, lons, 0
    return imd_loader, imd.RES


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shp", default="../input/5_states_all_districts.shp")
    ap.add_argument("--source", choices=["gsmap", "imd", "both"], default="both")
    ap.add_argument("--years", type=int, nargs="+", default=YEARS)
    ap.add_argument("--out", default="data/out")
    ap.add_argument("--cache", default="data/raw_cache")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    gdf = districts.load_districts(a.shp)
    districts.write_geojson(gdf, os.path.join(a.out, "districts.geojson"))
    bbox = districts.bbox(gdf)
    manifest = {"input_shp_sha256": _sha256(a.shp), "years": list(a.years),
                "n_districts": int(len(gdf)), "sources": {}}
    for source in (["gsmap", "imd"] if a.source == "both" else [a.source]):
        print(f"[{source}]", flush=True)
        loader, res = _loader_for(source, bbox, a.cache)
        df = build_source(loader, gdf, a.years, res)
        path = os.path.join(a.out, f"district_season_{source}.parquet")
        df.to_parquet(path, index=False, engine="pyarrow")
        manifest["sources"][source] = {"rows": int(len(df)), "res_deg": res,
                                       "sha256": _sha256(path)}
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_build.py -v`
Expected: 3 passed.

- [ ] **Step 5: Smoke-build IMD for one year against the real shapefile**

Run: `python build_dataset.py --source imd --years 2024 --out /tmp/pp_smoke` (use the scratchpad dir instead of `/tmp` if available), then:
```bash
python -I -c "import pandas as pd; df=pd.read_parquet('<out>/district_season_imd.parquet'); print(len(df)); print(df[df.period=='JJASOND'].sort_values('mean_mm').iloc[[0,-1]][['state','district','y2024','n_cells']]); print(df.mean_mm.isna().sum(), (df.n_cells<2).sum())"
```
Expected: 312 rows (156 districts x 2 periods), 0 NaN means, plausible Jun-Dec totals (dry rain-shadow districts a few hundred mm, west-coast/Odisha-hills districts 2000+ mm). This downloads one IMD year (small). Investigate any NaN before continuing.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: build script producing per-district parquet + manifest

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Streamlit app

**Files:**
- Create: `plantation/figures.py`, `app.py`, `tests/test_app.py`

**Interfaces:**
- Consumes: `metrics.shortlist`, parquet columns from Task 3, `districts.geojson` with property `key`.
- Produces: `figures.choropleth(geojson: dict, view: DataFrame, threshold: float) -> go.Figure` (`view` is `shortlist` output with `key` column added); `figures.display_table(view) -> DataFrame` with columns `State, District, 10-yr mean (mm), Shortlisted, Grid cells, Notes`.
- App reads `PLANTATION_DATA_DIR` (default `data/out` next to `app.py`).

- [ ] **Step 1: Write the failing tests**

`tests/test_app.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_app.py -v`
Expected: FAIL (`cannot import name 'figures'`).

- [ ] **Step 3: Implement `figures.py`**

`plantation/figures.py`:
```python
"""Plotly map and display table for the app."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

GREY = "#d9d9d9"


def choropleth(geojson: dict, view: pd.DataFrame, threshold: float) -> go.Figure:
    fig = go.Figure()
    common = dict(geojson=geojson, featureidkey="properties.key",
                  marker_line_width=0.4, marker_line_color="white")
    hover = "%{customdata[0]}, %{customdata[1]}<br>%{customdata[2]:.0f} mm<extra>%{customdata[3]}</extra>"
    for flag, label in ((False, "below threshold"), (True, "shortlisted")):
        part = view[view["above"] == flag]
        if part.empty:
            continue
        cd = pd.DataFrame({"d": part["district"], "s": part["state"],
                           "m": part["mean_mm"], "l": label}).values
        if flag:
            fig.add_trace(go.Choroplethmap(
                locations=part["key"], z=part["mean_mm"], colorscale="Blues",
                zmin=threshold, zmax=max(float(part["mean_mm"].max()), threshold + 1),
                colorbar=dict(title="10-yr mean (mm)"), customdata=cd,
                hovertemplate=hover, **common))
        else:
            fig.add_trace(go.Choroplethmap(
                locations=part["key"], z=[0] * len(part), showscale=False,
                colorscale=[[0, GREY], [1, GREY]], customdata=cd,
                hovertemplate=hover, **common))
    fig.update_layout(map_style="carto-positron", map_center={"lat": 16.0, "lon": 80.0},
                      map_zoom=4.2, margin=dict(l=0, r=0, t=0, b=0), height=560)
    return fig


def display_table(view: pd.DataFrame) -> pd.DataFrame:
    def notes(r):
        n = []
        if r["low_cells"]:
            n.append("few grid cells")
        if r["missing_days"] > 0:
            n.append("data gaps")
        return ", ".join(n)
    return pd.DataFrame({
        "State": view["state"], "District": view["district"],
        "10-yr mean (mm)": view["mean_mm"],
        "Shortlisted": view["above"].map({True: "Yes", False: "No"}),
        "Grid cells": view["n_cells"],
        "Notes": view.apply(notes, axis=1)}).reset_index(drop=True)
```

- [ ] **Step 4: Implement `app.py`**

`app.py`:
```python
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from plantation import figures, metrics

DATA = Path(os.environ.get("PLANTATION_DATA_DIR", Path(__file__).parent / "data" / "out"))
PERIOD_KEYS = {"Jun-Dec": "JJASOND", "Jan-Dec": "JANDEC"}
SOURCES = {"GSMaP (JAXA, 0.1°)": "gsmap", "IMD (gauge, 0.25°)": "imd"}

st.set_page_config(page_title="plantation_planning", layout="wide")
st.title("plantation_planning")
st.caption("Decadal precip assessment (JJASOND)")


@st.cache_data
def load(source: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA / f"district_season_{source}.parquet")
    df["key"] = df["state"] + "|" + df["district"]
    return df


@st.cache_data
def load_geojson() -> dict:
    return json.loads((DATA / "districts.geojson").read_text())


with st.sidebar:
    src_label = st.radio("Rainfall source", list(SOURCES), key="source")
    per_label = st.radio("Period", list(PERIOD_KEYS), key="period", horizontal=True)
    threshold = st.number_input("Threshold (mm)", min_value=0, max_value=6000,
                                value=800, step=50, key="threshold",
                                help="Districts with a 10-year mean at or above this are shortlisted.")
    df = load(SOURCES[src_label])
    states = st.multiselect("State", sorted(df["state"].unique()), key="states")
    query = st.text_input("Search district", key="query")

view = metrics.shortlist(df, PERIOD_KEYS[per_label], float(threshold), states or None, query)
if view.empty:
    st.info("No districts match the current filters.")
    st.stop()

n_above = int(view["above"].sum())
c1, c2, c3 = st.columns(3)
c1.metric("Shortlisted", f"{n_above} of {len(view)}", f"{100 * n_above / len(view):.0f}%", delta_color="off")
top, low = view.iloc[0], view.iloc[-1]
c2.metric("Highest 10-yr mean", f"{top['mean_mm']:.0f} mm", top["district"], delta_color="off")
c3.metric("Lowest 10-yr mean", f"{low['mean_mm']:.0f} mm", low["district"], delta_color="off")

tab_map, tab_table = st.tabs(["Map", "Ranked table"])
with tab_map:
    st.plotly_chart(figures.choropleth(load_geojson(), view, float(threshold)), use_container_width=True)
with tab_table:
    table = figures.display_table(view)
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", table.to_csv(index=False).encode(),
                       file_name="plantation_planning_shortlist.csv", mime="text/csv")

years = sorted(c for c in df.columns if c.startswith("y") and c[1:].isdigit())
st.caption(f"Mean of annual totals, {years[0][1:]}-{years[-1][1:]}, from {src_label}. "
           "Each district value is an area-weighted mean over the grid cells inside it.")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_app.py -v`
Expected: 6 passed. If `AppTest` cannot find a widget by key, check the `key=` on the matching widget in `app.py`; the metric label `"Shortlisted"` must match exactly.

- [ ] **Step 6: Visual check with a real smoke build**

Run `python build_dataset.py --source imd --years 2024 --out data/out` (one cheap IMD year; GSMaP file is absent so the app's GSMaP option will error, which is expected until Task 7). Copy `district_season_imd.parquet` to `district_season_gsmap.parquet` temporarily, run `streamlit run app.py`, and check the map renders, the threshold change re-colours districts, and the table downloads. Delete the temporary GSMaP copy afterwards. Do not commit smoke data (`git status` should be clean of `data/out` changes before the next commit, or reset them).

- [ ] **Step 7: Commit**

```bash
git add plantation/figures.py app.py tests/test_app.py && git commit -m "feat: Streamlit app with threshold, period filter, map and ranked table

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Full build, README, verification

**Files:**
- Create: `README.md`; Modify: `data/out/*` (generated)

- [ ] **Step 1: Pre-flight for the long GSMaP build**

Run: `python -I -c "import ftplib; f=ftplib.FTP('hokusai.eorc.jaxa.jp', timeout=60); f.login('rainmap','Niskur+1404'); print(len(f.nlst('/standard/v6/daily_G/00Z-23Z/201606'))); f.quit()"`
Expected: about 30 filenames. Also check free disk: `df -h .` needs at least 2 GB (only small windows are cached, roughly 3,650 files x ~30-60 KB).

- [ ] **Step 2: Run the full build in the background**

Run: `python build_dataset.py --source both 2>&1 | tee data/raw_cache/build.log` as a background command (expected several hours; IMD finishes first or second depending on order, GSMaP is FTP-bound). Per the user's autonomous-run preference, run it end to end without stopping for approval, and report on completion. If it stops on a persistent FTP error it can be re-run and resumes from the day cache.

- [ ] **Step 3: Verify the output**

Run:
```bash
python -I -c "
import pandas as pd
for s in ('gsmap','imd'):
    df = pd.read_parquet(f'data/out/district_season_{s}.parquet')
    print(s, len(df), 'NaN means:', df.mean_mm.isna().sum(), 'n_years<10:', (df.n_years<10).sum(),
          'low_cells:', df[df.period=='JJASOND'].low_cells.sum(), 'gap districts:', (df.missing_days>0).sum())
a = pd.read_parquet('data/out/district_season_gsmap.parquet'); b = pd.read_parquet('data/out/district_season_imd.parquet')
m = a[a.period=='JJASOND'].merge(b[b.period=='JJASOND'], on=['state','district'], suffixes=('_g','_i'))
print('corr', m.mean_mm_g.corr(m.mean_mm_i).round(3), 'median abs diff mm', (m.mean_mm_g-m.mean_mm_i).abs().median().round(0))
print('shortlisted at 800:', (a[a.period=='JJASOND'].mean_mm>=800).sum(), (b[b.period=='JJASOND'].mean_mm>=800).sum())
"
```
Expected: 312 rows per source, 0 NaN means, `n_years<10` count 0. GSMaP and IMD district means should correlate strongly (corr above about 0.9). Report any larger disagreement, and a hand check of one district (e.g. sum IMD cells for one known district independently) before accepting.

- [ ] **Step 4: Re-run the build to confirm idempotence**

Run the build a second time (it reads the day cache; expect minutes). Then `git status data/out`: the parquet and manifest must be unchanged (no diff).

- [ ] **Step 5: README**

`README.md` covering: what the app is, how to run (`streamlit run app.py`), how to rebuild (`python build_dataset.py --shp <path> --source both`), data sources (JAXA GSMaP v6 daily gauge, IMD Pune 0.25°), method (area-weighted district mean, years 2016-2025, Jun-Dec vs Jan-Dec), caveats from the spec (GSMaP reanalysis/operational tags, IMD coarse for small districts, `Notes` column), and that the district shapefile is not committed (the simplified `districts.geojson` is).

- [ ] **Step 6: Run the whole test suite and commit**

Run: `python -m pytest -v`
Expected: all pass (about 27 tests).
```bash
git add -A && git commit -m "data: full 2016-2025 build for GSMaP and IMD; add README

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 7: Ask before publishing**

Ask the user to confirm creating `arvind-s/plantation-planning` and pushing. Pushing is outward-facing: the repo contains only code and small derived data (no shapefile, no credentials; note the JAXA guest password is a public guest login already in the copied reader). Only after the user says yes: `gh repo create arvind-s/plantation-planning --public --source=. --push` (or `git remote add origin https://github.com/arvind-s/plantation-planning.git && git push -u origin main`, which needs a PAT with `repo` scope; raise `http.postBuffer` if the push fails). Then offer Streamlit Cloud deployment.
