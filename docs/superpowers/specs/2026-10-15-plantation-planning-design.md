# plantation_planning: design spec

Subtitle: **Decadal precip assessment (JJASOND)**
Date: 2026-10-15

## Purpose

First-pass shortlist of districts for next year's plantation. For each of 156
districts (AP, KA, OD, TN, TS) the app shows the **10-year mean cumulative
rainfall (mm)** and flags districts at or above a user-editable threshold
(default 800 mm). Field/planning staff use it to narrow down districts before
looking at finer-scale data (see the Rainfall Window Explorer for block level).

## Decisions (agreed with the user)

- Output per district is the 10-year mean only. Per-year totals are kept in
  the build cache because the mean is derived from them; the UI shows the mean.
- Threshold is a sidebar number input, default 800 mm, editable. A district is
  shortlisted when its 10-year mean >= threshold.
- The build covers the **full year** (Jan 1-Dec 31) for every year. The app has
  a period filter with two values: **Jun-Dec** (default, JJASOND) and
  **Jan-Dec** (full year). Both come from the same daily data, so switching is
  instant and needs no extra build.
- Two rainfall sources with a toggle: JAXA GSMaP (0.1 deg) and IMD gridded
  rain (0.25 deg), same as the Rainfall Window Explorer.
- Years: 2016-2025 (the last 10 complete years; 2026 is still in progress).
- No suitability score. Data-first, consistent with the earlier app.
- Separate project, own git repo, pushed to GitHub as
  `arvind-s/plantation-planning` (not into `rainfall-window-explorer`).
  Push is confirmed with the user at push time.

## Input

`../input/5_states_all_districts.shp`: 156 districts, columns `District`,
`State`, `geometry`. CRS is a Lambert Conformal Conic projection; reproject to
EPSG:4326. Some geometries are invalid; repair with `make_valid` before use.
State counts: AP 26, KA 30, OD 30, TN 37, TS 33.

## Architecture

```
plantation_planning/
  app.py                    Streamlit app (reads data/out only, no downloads)
  build_dataset.py          CLI: builds data/out parquet + geojson
  plantation/
    districts.py            load, repair, reproject, simplify
    weights.py              area-overlap cell weights per source grid
    gsmap.py                JAXA FTP reader (copied from rainfall_app)
    imd.py                  imdlib loader (copied from rainfall_app)
    metrics.py              period windowing, annual totals, 10-yr mean
    figures.py              choropleth + ranked table helpers
  data/out/                 committed outputs the app reads
  tests/
  requirements.txt          runtime only
  requirements-build.txt    adds imdlib
```

The GSMaP and IMD code is copied, not imported, so the repo deploys on its own.

## Data flow

1. `districts.py` loads the shapefile, repairs and reprojects it, and writes a
   simplified `districts.geojson` for the map (geometry for computation stays
   full resolution).
2. For each source, the build reads daily rainfall for 2016-2025, restricted
   to the South India bounding box of the districts, and caches each day as a
   small `.npz`. The GSMaP files are global (3600x1200 float32, about 17 MB);
   only the window is stored. Reruns skip cached days.
   - GSMaP: daily total mm = value (mm/hr) x 24. Filename version tag varies
     by year, so resolve it by listing the month directory (existing behaviour).
   - IMD: `imdlib` yearly files, mm/day directly, nodata -999.
3. Per district, per day, the mean over grid cells weighted by the area of
   overlap between each cell and the district. Weights are computed once per
   source grid. Missing days are skipped, not zero-filled, and counted.
4. `metrics.py` sums days in each window (Jan 1-Dec 31 and Jun 1-Dec 31) per
   year, then averages the 10 years.
5. Output `district_season_{gsmap,imd}.parquet`, one row per district x
   period, columns: `district`, `state`, `period` (`JJASOND` or `JANDEC`), per-year `y2016..y2025`,
   `mean_mm`, `n_cells`, `missing_days`.

## Edge cases

- **Small districts on the IMD grid:** report `n_cells`; flag districts with
  fewer than 2 cells in the table. A district containing no cell centre falls
  back to the nearest valid cell (same approach as the existing coastal-block
  handling).
- **Sea cells:** IMD nodata cells are excluded from weights; weights are
  renormalised over valid cells.
- **Known source gaps:** days missing at the source are counted in
  `missing_days`. A year with a gap is totalled over its valid days, and the
  table flags districts where any year has gaps.
- **Duplicate district names across states:** key rows by (`state`,
  `district`), never by name alone.

## App

- Header "plantation_planning", subtitle "Decadal precip assessment (JJASOND)".
- Sidebar: source toggle (GSMaP / IMD), period filter (Jun-Dec default / Jan-Dec), threshold
  (mm, default 800, editable), state multiselect, district search.
- KPI strip: districts shortlisted (count and % of districts in view), highest
  and lowest 10-year mean.
- Choropleth of the 10-year mean; districts below the threshold are visually
  de-emphasised, districts at or above it are emphasised.
- Ranked table: state, district, 10-year mean (mm), above/below, `n_cells`
  flag. Sortable, with CSV download of the current filter.
- The data files are small and committed, so the app makes no network calls.

## Determinism

Same input produces the same bytes: sorted keys, fixed column order, no
timestamps in the parquet. Reruns are idempotent and read the day cache. A
`manifest.json` records inputs, year range, source versions and row counts.

## Testing

- Cell weights: a district made of known cells gives the expected weighted mean.
- Windowing: Jun-Dec and Jan-Dec sums on synthetic daily data, including
  missing days and leap year.
- Threshold filter: boundary case (mean exactly equal to threshold is included).
- Build sanity: output has 156 rows per source and period, no NaN means, and
  one hand-checked district matches an independent computation.

## Risks

- **GSMaP volume:** the full year for 10 years is about 3,650 daily global
  files (about 62 GB over FTP, likely 6+ hours). The existing `rainfall_app`
  cache holds only Sep-Dec 2021-2025 (about 600 days), so only a small part is
  reusable.
- **Grid mismatch:** IMD at 0.25 deg is coarse for small districts; the
  `n_cells` flag makes this visible.
- **Source disagreement:** GSMaP and IMD will differ for some districts, which
  can move a district across the threshold. The toggle makes this visible; the
  spec does not pick a winner.
- **Two GSMaP product generations:** earlier years use a reanalysis tag and
  recent ones the operational tag, so a small step change across years is
  possible and is not corrected.

## Out of scope

Suitability scoring, block-level analysis, forecasting, other states.
