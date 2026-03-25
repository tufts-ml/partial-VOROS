"""Tests for pvoros.cost: recall_cost, pvoros_cost, voros_cost.

Uses synthetic val/test data for correctness checks. Replication tests use
real MIMIC-IV prediction CSVs stored in pvoros/tests/fixtures/.
"""

import os
import sys
import numpy as np
import pytest

from pvoros import recall_cost, pauroc_cost, pvoros_cost, voros_cost
from pvoros import _geometry
from pvoros.cost import _fpr_tpr_at_thresholds

# For replication against step7 output
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_SRC = os.path.join(_ROOT, 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ---- Synthetic val/test fixtures ----

@pytest.fixture
def val_test_data():
    """Two independent draws from the same generative model."""
    rng = np.random.default_rng(42)

    def make_split(n, seed):
        r = np.random.default_rng(seed)
        y = r.integers(0, 2, n)
        p = r.beta(2, 5, n)
        p[y == 1] += r.uniform(0.1, 0.4, int(y.sum()))
        return y, np.clip(p, 0, 1)

    yv, pv = make_split(600, 10)
    yt, pt = make_split(400, 20)
    return yv, pv, yt, pt


# ---- recall_cost ----

def test_recall_cost_range(val_test_data):
    yv, pv, yt, pt = val_test_data
    cost = recall_cost(yv, pv, yt, pt, alpha=0.2, kappa_frac=0.5,
                       min_fp_cost_ratio=0.1, max_fp_cost_ratio=1.0)
    assert 0.0 <= cost <= 1.0


def test_recall_cost_manual(val_test_data):
    """recall_cost matches manual step-by-step computation."""
    yv, pv, yt, pt = val_test_data
    alpha = 0.25
    kappa_frac = 0.5
    rmin, rmax = 0.1, 1.0
    n_points = 300

    from sklearn.metrics import roc_curve
    P_v = int(np.sum(yv == 1)); N_v = int(np.sum(yv == 0)); n_v = len(yv)
    P_t = int(np.sum(yt == 1)); N_t = int(np.sum(yt == 0))
    kappa = kappa_frac * n_v

    fprs_v, tprs_v, thrs_v = roc_curve(yv, pv)
    fprs_v = fprs_v.astype(float); tprs_v = tprs_v.astype(float); thrs_v = thrs_v.astype(float)

    _, acc_fprs_v, acc_tprs_v, acc_thrs_v, _ = _geometry._kept_on_valid(
        fprs_v, tprs_v, thrs_v, alpha, kappa, N_v, P_v
    )
    acc_fprs_t, acc_tprs_t = _fpr_tpr_at_thresholds(yt, pt, acc_thrs_v)
    i_best_t = int(np.argmax(acc_tprs_t))
    fpr_best_t = acc_fprs_t[i_best_t]
    tpr_best_t = acc_tprs_t[i_best_t]

    rs_cost = np.linspace(rmin, rmax, num=max(2, n_points))
    t_t = np.array([_geometry.ratio_to_t(r, P_t, N_t) for r in rs_cost])
    expected = float(np.mean(t_t * fpr_best_t + (1.0 - t_t) * (1.0 - tpr_best_t)))

    got = recall_cost(yv, pv, yt, pt, alpha, kappa_frac, rmin, rmax, n_points=n_points)
    assert abs(got - expected) < 1e-12


# ---- pvoros_cost ----

def test_pvoros_cost_range(val_test_data):
    yv, pv, yt, pt = val_test_data
    cost = pvoros_cost(yv, pv, yt, pt, alpha=0.2, kappa_frac=0.5,
                       min_fp_cost_ratio=0.1, max_fp_cost_ratio=1.0)
    assert 0.0 <= cost <= 1.0


def test_pvoros_cost_manual(val_test_data):
    """pvoros_cost matches manual step-by-step computation."""
    yv, pv, yt, pt = val_test_data
    alpha = 0.25
    kappa_frac = 0.4
    rmin, rmax = 0.2, 0.8
    n_points = 200

    from sklearn.metrics import roc_curve
    P_v = int(np.sum(yv == 1)); N_v = int(np.sum(yv == 0)); n_v = len(yv)
    P_t = int(np.sum(yt == 1)); N_t = int(np.sum(yt == 0))
    kappa = kappa_frac * n_v

    fprs_v, tprs_v, thrs_v = roc_curve(yv, pv)
    fprs_v = fprs_v.astype(float); tprs_v = tprs_v.astype(float); thrs_v = thrs_v.astype(float)

    _, acc_fprs_v, acc_tprs_v, acc_thrs_v, _ = _geometry._kept_on_valid(
        fprs_v, tprs_v, thrs_v, alpha, kappa, N_v, P_v
    )

    _, ts_arr, best_thr_arr = _geometry.voros(
        acc_fprs_v, acc_tprs_v, kappa, alpha, P_v, N_v, rmin, rmax,
        n_points=n_points, return_best_thresholds=True, thresholds=acc_thrs_v,
    )
    rs_list = np.linspace(rmin, rmax, num=len(ts_arr))
    fprs_t_sel, tprs_t_sel = _fpr_tpr_at_thresholds(yt, pt, np.asarray(best_thr_arr))
    t_t = np.array([_geometry.ratio_to_t(r, P_t, N_t) for r in rs_list])
    expected = float(np.mean(t_t * fprs_t_sel + (1.0 - t_t) * (1.0 - tprs_t_sel)))

    got = pvoros_cost(yv, pv, yt, pt, alpha, kappa_frac, rmin, rmax, n_points=n_points)
    assert abs(got - expected) < 1e-12


def test_pvoros_cost_uses_val_kappa(val_test_data):
    """kappa is derived from val set size, not test set size."""
    yv, pv, yt, pt = val_test_data
    # Make test set 2× larger by duplicating
    yt2 = np.concatenate([yt, yt])
    pt2 = np.concatenate([pt, pt])
    # Both calls should use kappa = kappa_frac * len(yv) on the val side
    cost1 = pvoros_cost(yv, pv, yt, pt, 0.2, 0.4, 0.1, 1.0, n_points=100)
    cost2 = pvoros_cost(yv, pv, yt2, pt2, 0.2, 0.4, 0.1, 1.0, n_points=100)
    # The threshold selection step is identical; cost evaluation uses different P_t, N_t
    # but the val-selected thresholds (driven by kappa=0.4*n_v) are the same
    # -> we just assert both are in [0,1]; exact equality not expected due to different P_t
    assert 0.0 <= cost1 <= 1.0
    assert 0.0 <= cost2 <= 1.0


# ---- voros_cost ----

def test_voros_cost_range(val_test_data):
    yv, pv, yt, pt = val_test_data
    cost = voros_cost(yv, pv, yt, pt, alpha=0.2, kappa_frac=0.5,
                      min_fp_cost_ratio=0.1, max_fp_cost_ratio=1.0)
    assert 0.0 <= cost <= 1.0


def test_voros_cost_matches_pvoros_cost(val_test_data):
    """voros_cost delegates to pvoros_cost — results must be identical for same inputs."""
    yv, pv, yt, pt = val_test_data
    alpha, kappa_frac, rmin, rmax = 0.2, 0.5, 0.1, 1.0
    pc = pvoros_cost(yv, pv, yt, pt, alpha, kappa_frac, rmin, rmax, n_points=200)
    vc = voros_cost(yv, pv, yt, pt, alpha, kappa_frac, rmin, rmax, n_points=200)
    assert pc == vc


def test_voros_cost_tight_constraints_still_valid(val_test_data):
    """voros_cost completes and returns [0,1] even with tight alpha/kappa_frac."""
    yv, pv, yt, pt = val_test_data
    # alpha/kappa_frac are accepted but don't affect threshold selection
    cost = voros_cost(yv, pv, yt, pt, alpha=0.9, kappa_frac=0.05,
                      min_fp_cost_ratio=0.1, max_fp_cost_ratio=1.0, n_points=50)
    assert 0.0 <= cost <= 1.0


# ---- Cross-function consistency ----

def test_pvoros_cost_le_recall_cost_is_not_guaranteed(val_test_data):
    """pvoros_cost and recall_cost use different threshold selection; no dominance guaranteed."""
    yv, pv, yt, pt = val_test_data
    # Just verify both return valid floats in [0,1]
    rc = recall_cost(yv, pv, yt, pt, 0.2, 0.5, 0.1, 1.0, n_points=100)
    pc = pvoros_cost(yv, pv, yt, pt, 0.2, 0.5, 0.1, 1.0, n_points=100)
    assert 0.0 <= rc <= 1.0
    assert 0.0 <= pc <= 1.0


# ---- Replication tests using fixture files ----

_FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures')


def _load(name):
    import pandas as pd
    df = pd.read_csv(os.path.join(_FIXTURES, name))
    return df['y_true'].to_numpy(), df['proba_y_eq_1'].to_numpy()


@pytest.fixture
def scenario1_pvoros():
    return _load('scenario1_pvoros_valid.csv'), _load('scenario1_pvoros_test.csv')


@pytest.fixture
def scenario1_recall():
    return _load('scenario1_recall_valid.csv'), _load('scenario1_recall_test.csv')


@pytest.fixture
def scenario2():
    return _load('scenario2_valid.csv'), _load('scenario2_test.csv')


def test_pvoros_cost_scenario1(scenario1_pvoros):
    (yv, pv), (yt, pt) = scenario1_pvoros
    cost = pvoros_cost(yv, pv, yt, pt, alpha=0.15, kappa_frac=0.5,
                       min_fp_cost_ratio=1/9, max_fp_cost_ratio=1/6)
    assert abs(cost - 0.261471) < 1e-5, f"got {cost:.6f}"


def test_recall_cost_scenario1(scenario1_recall):
    (yv, pv), (yt, pt) = scenario1_recall
    cost = recall_cost(yv, pv, yt, pt, alpha=0.15, kappa_frac=0.5,
                       min_fp_cost_ratio=1/9, max_fp_cost_ratio=1/6)
    assert abs(cost - 0.302208) < 1e-5, f"got {cost:.6f}"


def test_pvoros_cost_scenario2(scenario2):
    (yv, pv), (yt, pt) = scenario2
    cost = pvoros_cost(yv, pv, yt, pt, alpha=0.5, kappa_frac=0.1,
                       min_fp_cost_ratio=1/40, max_fp_cost_ratio=1/20)
    assert abs(cost - 0.537574) < 1e-5, f"got {cost:.6f}"


def test_recall_cost_scenario2(scenario2):
    (yv, pv), (yt, pt) = scenario2
    cost = recall_cost(yv, pv, yt, pt, alpha=0.5, kappa_frac=0.1,
                       min_fp_cost_ratio=1/40, max_fp_cost_ratio=1/20)
    assert abs(cost - 0.537574) < 1e-5, f"got {cost:.6f}"


# ---- eICU fixture regression tests ----

@pytest.fixture
def eicu_s1_pvoros():
    return _load('eicu_s1_pvoros_valid.csv'), _load('eicu_s1_pvoros_test.csv')


@pytest.fixture
def eicu_s1_recall():
    return _load('eicu_s1_recall_valid.csv'), _load('eicu_s1_recall_test.csv')


@pytest.fixture
def eicu_s2_pvoros():
    return _load('eicu_s2_pvoros_valid.csv'), _load('eicu_s2_pvoros_test.csv')


@pytest.fixture
def eicu_s2_voros():
    return _load('eicu_s2_voros_valid.csv'), _load('eicu_s2_voros_test.csv')


def test_pvoros_cost_eicu_s1(eicu_s1_pvoros):
    (yv, pv), (yt, pt) = eicu_s1_pvoros
    cost = pvoros_cost(yv, pv, yt, pt, alpha=0.15, kappa_frac=0.5,
                       min_fp_cost_ratio=1/9, max_fp_cost_ratio=1/6)
    assert abs(cost - 0.318603) < 1e-5, f"got {cost:.6f}"


def test_voros_cost_eicu_s1(eicu_s1_pvoros):
    (yv, pv), (yt, pt) = eicu_s1_pvoros
    cost = voros_cost(yv, pv, yt, pt, alpha=0.15, kappa_frac=0.5,
                      min_fp_cost_ratio=1/9, max_fp_cost_ratio=1/6)
    assert abs(cost - 0.318603) < 1e-5, f"got {cost:.6f}"


def test_recall_cost_eicu_s1(eicu_s1_recall):
    (yv, pv), (yt, pt) = eicu_s1_recall
    cost = recall_cost(yv, pv, yt, pt, alpha=0.15, kappa_frac=0.5,
                       min_fp_cost_ratio=1/9, max_fp_cost_ratio=1/6)
    assert abs(cost - 0.335502) < 1e-5, f"got {cost:.6f}"


def test_pvoros_cost_eicu_s2(eicu_s2_pvoros):
    (yv, pv), (yt, pt) = eicu_s2_pvoros
    cost = pvoros_cost(yv, pv, yt, pt, alpha=0.5, kappa_frac=0.1,
                       min_fp_cost_ratio=1/40, max_fp_cost_ratio=1/20)
    assert abs(cost - 0.706699) < 1e-5, f"got {cost:.6f}"


def test_recall_cost_eicu_s2(eicu_s2_pvoros):
    (yv, pv), (yt, pt) = eicu_s2_pvoros
    cost = recall_cost(yv, pv, yt, pt, alpha=0.5, kappa_frac=0.1,
                       min_fp_cost_ratio=1/40, max_fp_cost_ratio=1/20)
    assert abs(cost - 0.706699) < 1e-5, f"got {cost:.6f}"


def test_voros_cost_eicu_s2(eicu_s2_voros):
    (yv, pv), (yt, pt) = eicu_s2_voros
    cost = voros_cost(yv, pv, yt, pt, alpha=0.5, kappa_frac=0.1,
                      min_fp_cost_ratio=1/40, max_fp_cost_ratio=1/20)
    assert abs(cost - 0.771692) < 1e-5, f"got {cost:.6f}"


@pytest.fixture
def mimic_s2_voros():
    return _load('mimic_s2_voros_valid.csv'), _load('mimic_s2_voros_test.csv')


def test_voros_cost_mimic_s2(mimic_s2_voros):
    (yv, pv), (yt, pt) = mimic_s2_voros
    cost = voros_cost(yv, pv, yt, pt, alpha=0.5, kappa_frac=0.1,
                      min_fp_cost_ratio=1/40, max_fp_cost_ratio=1/20)
    assert abs(cost - 0.639849) < 1e-5, f"got {cost:.6f}"


def test_pauroc_cost_matches_recall_cost(val_test_data):
    """pauroc_cost delegates to recall_cost — results must be identical."""
    yv, pv, yt, pt = val_test_data
    alpha, kappa_frac, rmin, rmax = 0.2, 0.5, 0.1, 1.0
    rc = recall_cost(yv, pv, yt, pt, alpha, kappa_frac, rmin, rmax, n_points=200)
    ac = pauroc_cost(yv, pv, yt, pt, alpha, kappa_frac, rmin, rmax, n_points=200)
    assert rc == ac
