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
