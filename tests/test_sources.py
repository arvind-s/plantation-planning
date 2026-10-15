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
