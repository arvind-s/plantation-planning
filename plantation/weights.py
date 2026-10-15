"""Area-overlap weights mapping grid cells to district polygons."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from shapely import box


def _nearest_valid(geom, lats, lons, valid_mask, res):
    if valid_mask is not None and not np.any(valid_mask):
        raise ValueError("no valid cells in grid")
    p = geom.representative_point()
    rr, cc = np.meshgrid(np.arange(len(lats)), np.arange(len(lons)), indexing="ij")
    d2 = (lats[rr] - p.y) ** 2 + (lons[cc] - p.x) ** 2
    if valid_mask is not None:
        d2 = np.where(valid_mask, d2, np.inf)
    r, c = np.unravel_index(np.argmin(d2), d2.shape)
    dist = float(np.sqrt(d2[r, c]))
    if dist > 3 * res:
        raise ValueError(
            f"nearest valid cell is {dist:.3f} degrees from district representative "
            f"point, exceeding 3*res = {3 * res:.3f}"
        )
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
            r, c = _nearest_valid(geom, lats, lons, valid_mask, res)
            found = [(r, c, 1.0)]
        total = sum(a for _, _, a in found)
        for r, c, a in found:
            rows.append(d)
            cols.append(r * nlon + c)
            vals.append(a / total)
        n_cells[d] = len(found)
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(len(geoms), nlat * nlon))
    return W, n_cells
