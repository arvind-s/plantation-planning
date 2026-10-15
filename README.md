# plantation_planning

**Decadal precip assessment (JJASOND)** — a first-pass district shortlist for next year's plantation.

For each of 156 districts in Andhra Pradesh, Karnataka, Odisha, Tamil Nadu and Telangana, the app shows the
**10-year mean cumulative rainfall (mm)** for 2016–2025 and flags districts at or above a threshold
(default **800 mm**, editable in the sidebar).

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app reads only the committed files in `data/out/` and makes no network calls.

Sidebar controls:

- **Rainfall source** — IMD (gauge-based grid, 0.25°, the default) or GSMaP (JAXA satellite, 0.1°).
- **Period** — Jun–Dec (default, JJASOND) or Jan–Dec (full year).
- **Threshold (mm)** — districts with a 10-year mean at or above it are shortlisted (equal counts).
- State filter and district search.

Outputs: KPI strip, choropleth map (shortlisted districts in blue, the rest grey), ranked table with CSV download.

## Method

1. Daily rainfall (mm/day) for every day of 2016–2025 from each source.
2. Per district and day, the mean over the grid cells inside the district, weighted by each cell's overlap
   area with the district polygon (renormalised over cells that have data).
3. Sum the days in each window (Jun 1–Dec 31, Jan 1–Dec 31) per year, then average the 10 yearly totals.

Missing days are skipped, not zero-filled, and counted in `missing_days`.

## Rebuild the data

```bash
pip install -r requirements-build.txt
python build_dataset.py --shp ../input/5_states_all_districts.shp --source both
```

Writes `data/out/districts.geojson`, `district_season_{gsmap,imd}.parquet` and `manifest.json`.
Reruns reuse the day cache in `data/raw_cache/` (gitignored) and produce identical bytes.
The first GSMaP build downloads about 3,650 daily global files from JAXA's FTP server
(roughly 60 GB transferred, several hours); only the small South-India window of each is kept.
The district shapefile itself is not committed; the simplified `districts.geojson` is.

## Data sources

- **GSMaP** — JAXA GSMaP_Gauge daily, v6 standard, `hokusai.eorc.jaxa.jp` (public guest login).
  Value is mm/hr, daily total = value × 24.
- **IMD** — IMD Pune 0.25° gauge-based daily rainfall via the `imdlib` package.

## Caveats

- GSMaP filenames carry a version tag; the filename version tag varies by year (resolved at download).
  A small step change across years is possible and is not corrected.
- GSMaP has a known source gap: JAXA has no file for 2025-10-10. That day is excluded from the totals and
  counted in `missing_days` (shown as "1 missing day" in the Notes column for every GSMaP row).
- IMD at 0.25° is coarse for small districts. The table's **Notes** column flags districts with fewer than
  2 grid cells ("few grid cells") and districts with any missing days ("N missing days").
- GSMaP and IMD will disagree for some districts, which can move a district across the threshold.
  Switch the source to see both; the app does not pick a winner.
- Attribution: GSMaP provided by EORC, JAXA. IMD Pune gridded rainfall, Pai et al. 2014.
- This is a screening tool. There is no suitability score.

## Tests

Tests import geopandas, so install the build requirements first:

```bash
pip install -r requirements.txt -r requirements-build.txt
python -m pytest
```
