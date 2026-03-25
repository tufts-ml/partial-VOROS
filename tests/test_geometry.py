"""Test that pvoros._geometry matches src/utils_voros exactly.

Covers all 5 geometric polygon cases and verifies numerical equivalence
to 1e-10 tolerance for reduced_area across a grid of (h, k, alpha, kappa, r).
"""

import os
import sys
import math
import numpy as np
import pytest

# Make src/ importable
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_SRC = os.path.join(_ROOT, 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import utils_voros as _src
from pvoros import _geometry as _new


# ---- Helpers ----

def polygon_area_shoelace(pts):
    """Shoelace area (absolute value)."""
    if not pts:
        return 0.0
    n = len(pts)
    a = 0.0
    for i in range(n):
        x, y = pts[i]
        xn, yn = pts[(i + 1) % n]
        a += x * yn - y * xn
    return abs(a) / 2.0


# ---- Geometry primitives ----

def test_area_matches():
    """_geometry.area matches utils_voros.area on several polygons."""
    polygons = [
        [(0, 0), (1, 0), (1, 1), (0, 1)],          # unit square
        [(0, 0), (0.5, 1), (1, 0)],                  # triangle
        [(0.1, 0.2), (0.4, 0.9), (0.8, 0.7), (0.6, 0.1)],  # quad
    ]
    for poly in polygons:
        assert math.isclose(_new.area(poly), _src.area(poly), rel_tol=1e-12, abs_tol=1e-14)


def test_clip_polygon_matches():
    """_clip_polygon_with_halfplane matches between new and src."""
    poly = [(0, 0), (1, 0), (1, 1), (0, 1)]
    # clip by x <= 0.5
    new_clipped = _new._clip_polygon_with_halfplane(poly, 1, 0, 0.5)
    src_clipped = _src._clip_polygon_with_halfplane(poly, 1, 0, 0.5)
    assert len(new_clipped) == len(src_clipped)
    for (nx, ny), (sx, sy) in zip(sorted(new_clipped), sorted(src_clipped)):
        assert abs(nx - sx) < 1e-12
        assert abs(ny - sy) < 1e-12


def test_intersect_halfplanes_matches():
    """_intersect_halfplanes matches."""
    halfplanes = [(1, 0, 0.7), (-1, 0, 0.0), (0, 1, 0.8)]
    new_poly = _new._intersect_halfplanes(halfplanes)
    src_poly = _src._intersect_halfplanes(halfplanes)
    assert len(new_poly) == len(src_poly)
    na = polygon_area_shoelace(new_poly)
    sa = polygon_area_shoelace(src_poly)
    assert abs(na - sa) < 1e-11


# ---- Total feasible region: 5 cases ----

@pytest.mark.parametrize("label,P,N,alpha,kappa,expected_verts", [
    ("case1", 100, 900, 0.2, 400, 4),   # precision & capacity -> quadrilateral
    ("case2", 100, 900, 0.2,  60, 3),   # capacity very tight -> triangle
    ("case3", 100, 900, 0.05, 5000, 4), # capacity inactive, shallow precision -> quad
    ("case4", 100, 900, 0.7,  5000, 3), # capacity inactive, steep precision -> triangle
    ("case5", 100, 900, 0.05,  970, 5), # both active, intersection outside square -> pentagon
])
def test_total_region_polygon_matches_src(label, P, N, alpha, kappa, expected_verts):
    new_poly = _new.compute_total_region_polygon(P, N, alpha, kappa)
    src_poly = _src.compute_total_region_polygon(P, N, alpha, kappa)

    assert len(new_poly) == len(src_poly), f"{label}: vertex count mismatch"
    assert len(new_poly) == expected_verts, f"{label}: expected {expected_verts} vertices"

    new_area = polygon_area_shoelace(new_poly)
    src_area = polygon_area_shoelace(src_poly)
    assert abs(new_area - src_area) < 1e-11, f"{label}: area mismatch {new_area} vs {src_area}"

    new_ta, _ = _new.total_region_area(P, N, alpha, kappa)
    src_ta, _ = _src.total_region_area(P, N, alpha, kappa)
    assert abs(new_ta - src_ta) < 1e-11


# ---- reduced_area grid: new vs src.reduced_area_tested ----

# Grid of parameters covering all 5 cases
_GRID_CASES = [
    # (P, N, alpha, kappa, h, k, r)
    (100, 900, 0.2,   400, 0.10, 0.60, 1.0),   # case1
    (100, 900, 0.2,    60, 0.04, 0.20, 1.0),   # case2
    (100, 900, 0.05, 5000, 0.10, 0.60, 1.0),   # case3
    (100, 900, 0.7,  5000, 0.03, 0.80, 1.0),   # case4
    (100, 900, 0.05,  970, 0.10, 0.60, 1.0),   # case5
    # varied cost ratios
    (100, 900, 0.2,   400, 0.10, 0.60, 0.1),
    (100, 900, 0.2,   400, 0.10, 0.60, 5.0),
    (100, 900, 0.2,   400, 0.05, 0.40, 2.0),
    (100, 900, 0.2,   400, 0.20, 0.80, 0.5),
    # different class balance
    (50,  950, 0.1,   300, 0.02, 0.50, 1.5),
    (200, 800, 0.3,   500, 0.15, 0.70, 0.8),
    # near-boundary cases
    (100, 900, 0.2,   400, 0.00, 0.00, 1.0),
    (100, 900, 0.2,   400, 0.30, 0.95, 1.0),
]


@pytest.mark.parametrize("P,N,alpha,kappa,h,k,r", _GRID_CASES)
def test_reduced_area_matches_src(P, N, alpha, kappa, h, k, r):
    new_val = _new.reduced_area(h, k, kappa, alpha, P, N, r)
    src_val = _src.reduced_area_tested(h, k, kappa, alpha, P, N, r)
    assert abs(new_val - src_val) < 1e-10, (
        f"Mismatch at P={P},N={N},a={alpha},k={kappa},h={h},k={k},r={r}: "
        f"new={new_val:.15g} src={src_val:.15g}"
    )
    assert 0.0 <= new_val <= 1.0


def test_reduced_area_return_total_area():
    val, ta = _new.reduced_area(0.1, 0.6, 400, 0.2, 100, 900, 1.0, return_total_area=True)
    src_val, src_ta = _src.reduced_area_tested(0.1, 0.6, 400, 0.2, 100, 900, 1.0, return_total_area=True)
    assert abs(val - src_val) < 1e-10
    assert abs(ta - src_ta) < 1e-10


def test_reduced_area_return_details():
    val, details = _new.reduced_area(0.1, 0.6, 400, 0.2, 100, 900, 1.0, return_details=True)
    assert 'total_polygon' in details
    assert 'iso_polygon' in details
    assert 'iso_line' in details
    assert 't' in details
    assert 0.0 <= val <= 1.0
    assert details['total_polygon']
    assert details['iso_polygon']


# ---- keep_model ----

@pytest.mark.parametrize("fpr,tpr,alpha,kappa,N,P,expected", [
    (0.1, 0.6, 0.2, 400, 900, 100, True),    # inside feasible region
    (0.5, 0.1, 0.2, 400, 900, 100, False),   # below precision line
    (0.4, 0.9, 0.2,  60, 900, 100, False),   # above capacity line
])
def test_keep_model_matches_src(fpr, tpr, alpha, kappa, N, P, expected):
    new_result = _new.keep_model(fpr, tpr, alpha, kappa, N, P)
    src_result = _src.keep_model(fpr, tpr, alpha, kappa, N, P)
    assert new_result == src_result
    assert new_result == expected


# ---- ratio_to_t / t_to_ratio ----

@pytest.mark.parametrize("r,P,N", [
    (0.0, 100, 100),
    (1.0, 100, 100),
    (0.5, 200, 800),
    (2.0,  50, 950),
])
def test_ratio_to_t_matches_src(r, P, N):
    assert abs(_new.ratio_to_t(r, P, N) - _src.ratio_to_t(r, P, N)) < 1e-14


@pytest.mark.parametrize("t,P,N", [
    (0.0, 100, 100),
    (0.5, 100, 100),
    (0.3, 200, 800),
    (0.9,  50, 950),
])
def test_t_to_ratio_matches_src(t, P, N):
    assert abs(_new.t_to_ratio(t, P, N) - _src.t_to_ratio(t, P, N)) < 1e-14


# ---- voros function: new vs src ----

def test_voros_matches_src():
    """voros() in _geometry matches utils_voros.voros() for a realistic input."""
    rng = np.random.default_rng(42)
    fprs = np.sort(rng.uniform(0, 1, 20))
    tprs = np.sort(rng.uniform(0, 1, 20))
    P, N = 100, 900
    kappa = 400.0
    alpha = 0.2

    new_v = _new.voros(fprs, tprs, kappa, alpha, P, N, 0.1, 1.0, n_points=200)
    src_v = _src.voros(fprs, tprs, kappa, alpha, P, N, 0.1, 1.0, n_points=200)
    assert abs(new_v - src_v) < 1e-10
    assert 0.0 <= new_v <= 1.0


def test_voros_return_best_thresholds_matches_src():
    rng = np.random.default_rng(7)
    fprs = np.sort(rng.uniform(0, 0.5, 15))
    tprs = 0.3 + 0.5 * np.sort(rng.uniform(0, 1, 15))
    thresholds = np.linspace(0.9, 0.1, 15)
    P, N = 100, 900
    kappa = 400.0
    alpha = 0.2

    new_v, new_ts, new_thr = _new.voros(
        fprs, tprs, kappa, alpha, P, N, 0.1, 1.0, n_points=50,
        return_best_thresholds=True, thresholds=thresholds,
    )
    src_v, src_ts, src_thr = _src.voros(
        fprs, tprs, kappa, alpha, P, N, 0.1, 1.0, n_points=50,
        return_best_thresholds=True, thresholds=thresholds,
    )
    assert abs(new_v - src_v) < 1e-10
    np.testing.assert_allclose(new_ts, src_ts, atol=1e-13)
    np.testing.assert_allclose(new_thr, src_thr, atol=1e-13)
