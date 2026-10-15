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
        if W is not None and not (np.shape(lats) == np.shape(first[0]) and np.shape(lons) == np.shape(first[1])
                                  and np.allclose(lats, first[0]) and np.allclose(lons, first[1])):
            raise ValueError(f"grid for {year} differs from {years[0]}; weights cannot be reused")
        if W is None:
            first = (lats, lons)
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
        os.makedirs(c, exist_ok=True)

        def gsmap_loader(y):
            stack, lats, lons, missing = gsmap.load_stack(y, bbox, c)
            return stack, None, lats, lons, missing
        return gsmap_loader, gsmap.RES
    c = os.path.join(cache, "imd")
    os.makedirs(c, exist_ok=True)   # imdlib uses os.mkdir (non-recursive)

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


PRODUCTS = {"gsmap": "GSMaP_Gauge v6 daily_G 00Z-23Z",
            "imd": "IMD Pune 0.25 deg gridded rainfall"}
SHP_EXTS = (".shp", ".dbf", ".shx", ".prj", ".cpg")


def write_manifest(out_dir, shp, years, n_districts, source, rows, res, parquet_path):
    """Update manifest.json: keep other sources' entries, refresh this one. Deterministic."""
    mpath = os.path.join(out_dir, "manifest.json")
    manifest = {}
    if os.path.exists(mpath):
        with open(mpath) as f:
            manifest = json.load(f)
    base = os.path.splitext(shp)[0]
    manifest["input_shp_sha256"] = {os.path.basename(base + e): _sha256(base + e)
                                    for e in SHP_EXTS if os.path.exists(base + e)}
    manifest["years"] = list(years)
    manifest["n_districts"] = int(n_districts)
    entry = {"rows": int(rows), "res_deg": res, "sha256": _sha256(parquet_path),
             "product": PRODUCTS[source]}
    if source == "imd":
        entry["imdlib_version"] = _imdlib_version()
    manifest.setdefault("sources", {})[source] = entry
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def _imdlib_version():
    import importlib.metadata
    return importlib.metadata.version("imdlib")


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
    for source in (["gsmap", "imd"] if a.source == "both" else [a.source]):
        print(f"[{source}]", flush=True)
        loader, res = _loader_for(source, bbox, a.cache)
        df = build_source(loader, gdf, a.years, res)
        path = os.path.join(a.out, f"district_season_{source}.parquet")
        df.to_parquet(path, index=False, engine="pyarrow")
        write_manifest(a.out, a.shp, a.years, len(gdf), source, len(df), res, path)


if __name__ == "__main__":
    main()
