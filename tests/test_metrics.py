"""Tests for pvoros.metrics: voros_score, pvoros_score, make_pvoros_scorer.

Uses synthetic data for correctness checks. Where real prediction CSVs are
available under workflows/, additional replication tests are run; they are
skipped automatically if the files are absent.
"""

import os
import numpy as np
import pytest

from pvoros import voros_score, pvoros_score, make_pvoros_scorer
from pvoros import _geometry


# ---- Synthetic data fixture ----

@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(0)
    n = 500
    y_true = rng.integers(0, 2, n)
    # moderately discriminative scores
    y_pred = rng.beta(2, 5, n)
    y_pred[y_true == 1] += rng.uniform(0, 0.3, int(y_true.sum()))
    y_pred = np.clip(y_pred, 0, 1)
    return y_true, y_pred


# ---- voros_score ----

def test_voros_score_range(synthetic_data):
    y_true, y_pred = synthetic_data
    score = voros_score(y_true, y_pred, min_fp_cost_ratio=0.1, max_fp_cost_ratio=1.0)
    assert 0.0 <= score <= 1.0


def test_voros_score_perfect_classifier():
    """Perfect classifier should achieve score near 1."""
    n = 200
    y_true = np.array([0] * 100 + [1] * 100)
    # Perfect separation: positives all score 1.0, negatives all score 0.0
    y_pred = np.array([0.0] * 100 + [1.0] * 100)
    score = voros_score(y_true, y_pred, min_fp_cost_ratio=0.1, max_fp_cost_ratio=1.0)
    assert score > 0.9


def test_voros_score_random_classifier(synthetic_data):
    """Random classifier should score lower than perfect."""
    y_true, _ = synthetic_data
    rng = np.random.default_rng(99)
    y_pred_random = rng.uniform(0, 1, len(y_true))
    perfect_pred = np.where(y_true == 1, 0.9, 0.1).astype(float)
    score_random = voros_score(y_true, y_pred_random, 0.1, 1.0, n_points=200)
    score_perfect = voros_score(y_true, perfect_pred, 0.1, 1.0, n_points=200)
    assert score_perfect > score_random


def test_voros_score_uses_full_roc(synthetic_data):
    """voros_score uses alpha=1e-8, kappa=P+N (effectively unconstrained)."""
    y_true, y_pred = synthetic_data
    P = int(np.sum(y_true == 1))
    N = int(np.sum(y_true == 0))
    from sklearn.metrics import roc_curve
    fprs, tprs, _ = roc_curve(y_true, y_pred)
    # Direct call with same parameters
    expected = _geometry.voros(fprs, tprs, float(P + N), 1e-8, P, N, 0.1, 1.0, n_points=500)
    got = voros_score(y_true, y_pred, 0.1, 1.0, n_points=500)
    assert abs(got - expected) < 1e-12


# ---- pvoros_score ----

def test_pvoros_score_range(synthetic_data):
    y_true, y_pred = synthetic_data
    score = pvoros_score(y_true, y_pred, alpha=0.3, kappa_frac=0.5,
                         min_fp_cost_ratio=0.1, max_fp_cost_ratio=1.0)
    assert 0.0 <= score <= 1.0


def test_pvoros_score_less_than_or_equal_voros(synthetic_data):
    """Partial VOROS with constraints should be <= full VOROS."""
    y_true, y_pred = synthetic_data
    full = voros_score(y_true, y_pred, 0.1, 1.0, n_points=200)
    partial = pvoros_score(y_true, y_pred, alpha=0.3, kappa_frac=0.5,
                           min_fp_cost_ratio=0.1, max_fp_cost_ratio=1.0, n_points=200)
    # Tighter constraints → score <= full (with small numerical tolerance)
    assert partial <= full + 1e-9


def test_pvoros_score_tight_constraints_reduce_score(synthetic_data):
    """Stricter alpha/kappa_frac should generally reduce the score."""
    y_true, y_pred = synthetic_data
    loose = pvoros_score(y_true, y_pred, alpha=0.1, kappa_frac=0.9,
                         min_fp_cost_ratio=0.1, max_fp_cost_ratio=1.0, n_points=200)
    tight = pvoros_score(y_true, y_pred, alpha=0.7, kappa_frac=0.1,
                         min_fp_cost_ratio=0.1, max_fp_cost_ratio=1.0, n_points=200)
    assert loose >= tight - 1e-9  # loose constraints give >= score


def test_pvoros_score_matches_manual(synthetic_data):
    """pvoros_score matches a manual step-by-step computation."""
    y_true, y_pred = synthetic_data
    alpha = 0.3
    kappa_frac = 0.5
    rmin, rmax = 0.1, 1.0
    n_points = 300

    from sklearn.metrics import roc_curve
    P = int(np.sum(y_true == 1))
    N = int(np.sum(y_true == 0))
    n = len(y_true)
    kappa = kappa_frac * n

    fprs, tprs, thrs = roc_curve(y_true, y_pred)
    fprs = fprs.astype(float)
    tprs = tprs.astype(float)
    thrs = thrs.astype(float)

    _, acc_fprs, acc_tprs, _, _ = _geometry._kept_on_valid(fprs, tprs, thrs, alpha, kappa, N, P)
    expected = _geometry.voros(acc_fprs, acc_tprs, kappa, alpha, P, N, rmin, rmax, n_points=n_points)

    got = pvoros_score(y_true, y_pred, alpha, kappa_frac, rmin, rmax, n_points=n_points)
    assert abs(got - expected) < 1e-12


# ---- make_pvoros_scorer ----

def test_make_pvoros_scorer_returns_callable(synthetic_data):
    scorer = make_pvoros_scorer(alpha=0.3, kappa_frac=0.5,
                                min_fp_cost_ratio=0.1, max_fp_cost_ratio=1.0)
    assert callable(scorer)


def test_make_pvoros_scorer_matches_pvoros_score(synthetic_data):
    y_true, y_pred = synthetic_data
    alpha, kappa_frac, rmin, rmax = 0.3, 0.5, 0.1, 1.0
    scorer = make_pvoros_scorer(alpha, kappa_frac, rmin, rmax, n_points=200)
    score_from_scorer = scorer(y_true, y_pred)
    score_direct = pvoros_score(y_true, y_pred, alpha, kappa_frac, rmin, rmax, n_points=200)
    assert abs(score_from_scorer - score_direct) < 1e-12


def test_make_pvoros_scorer_baked_params(synthetic_data):
    """Two scorers with different params produce different scores."""
    y_true, y_pred = synthetic_data
    scorer_loose = make_pvoros_scorer(0.1, 0.9, 0.1, 1.0, n_points=200)
    scorer_tight = make_pvoros_scorer(0.7, 0.1, 0.1, 1.0, n_points=200)
    assert scorer_loose(y_true, y_pred) != scorer_tight(y_true, y_pred)


# ---- Replication tests using fixture files ----

_FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures')


def _load_fixture(name):
    import pandas as pd
    df = pd.read_csv(os.path.join(_FIXTURES, name))
    return df['y_true'].to_numpy(), df['proba_y_eq_1'].to_numpy()


def test_pvoros_score_scenario1_on_fixture():
    """pvoros_score is in [0,1] and <= voros_score on real data."""
    y_true, y_pred = _load_fixture('scenario1_pvoros_valid.csv')
    partial = pvoros_score(y_true, y_pred, alpha=0.15, kappa_frac=0.5,
                           min_fp_cost_ratio=1/9, max_fp_cost_ratio=1/6)
    full = voros_score(y_true, y_pred, min_fp_cost_ratio=1/9, max_fp_cost_ratio=1/6)
    assert 0.0 <= partial <= 1.0
    assert 0.0 <= full <= 1.0
    assert full >= partial - 1e-9


def test_pvoros_score_scenario2_on_fixture():
    """pvoros_score is in [0,1] on scenario 2 fixture."""
    y_true, y_pred = _load_fixture('scenario2_valid.csv')
    score = pvoros_score(y_true, y_pred, alpha=0.5, kappa_frac=0.1,
                         min_fp_cost_ratio=1/40, max_fp_cost_ratio=1/20)
    assert 0.0 <= score <= 1.0


def test_make_pvoros_scorer_on_fixture():
    """make_pvoros_scorer result matches pvoros_score on real data."""
    y_true, y_pred = _load_fixture('scenario1_pvoros_valid.csv')
    scorer = make_pvoros_scorer(alpha=0.15, kappa_frac=0.5,
                                min_fp_cost_ratio=1/9, max_fp_cost_ratio=1/6)
    assert abs(scorer(y_true, y_pred) -
               pvoros_score(y_true, y_pred, 0.15, 0.5, 1/9, 1/6)) < 1e-12
