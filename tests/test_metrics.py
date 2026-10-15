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
