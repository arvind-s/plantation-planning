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
