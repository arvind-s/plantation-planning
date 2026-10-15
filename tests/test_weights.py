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


def test_all_false_mask_raises():
    valid = np.zeros((3, 3), bool)
    try:
        weights.build_weights([box(0.1, 0.1, 0.9, 0.9)], LATS, LONS, RES, valid)
    except ValueError as e:
        assert "no valid cells" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_district_far_outside_grid_raises():
    try:
        weights.build_weights([box(50, 50, 51, 51)], LATS, LONS, RES)
    except ValueError as e:
        assert "degrees" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_ascending_lats_same_weights():
    lats_asc = LATS[::-1].copy()           # [0.5, 1.5, 2.5]
    W, n = weights.build_weights([box(0, 0, 1.5, 1)], lats_asc, LONS, RES)
    d = _dense(W)[0]
    assert np.isclose(d[0, 0], 2 / 3) and np.isclose(d[0, 1], 1 / 3)
    assert np.isclose(W.sum(), 1.0) and n[0] == 2
