"""Public scoring functions for VOROS and partial VOROS.

These functions follow a scikit-learn-compatible interface:
    score = pvoros_score(y_true, y_pred, ...)
    scorer = make_pvoros_scorer(...)
    score = scorer(y_true, y_pred)
"""

import numpy as np
from sklearn.metrics import roc_curve

import _geometry


def voros_score(y_true, y_pred, min_fp_cost_ratio, max_fp_cost_ratio, n_points=1000):
    """Full VOROS score (no precision/capacity constraints).

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary labels.
    y_pred : array-like of shape (n,)
        Predicted probabilities.
    min_fp_cost_ratio : float
        Minimum C0/C1 (false-positive to false-negative cost ratio).
    max_fp_cost_ratio : float
        Maximum C0/C1 cost ratio.
    n_points : int
        Number of cost ratio grid points for integration.

    Returns
    -------
    float : VOROS score in [0, 1]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    P = int(np.sum(y_true == 1))
    N = int(np.sum(y_true == 0))
    fprs, tprs, _ = roc_curve(y_true, y_pred)
    fprs = fprs.astype(float)
    tprs = tprs.astype(float)
    # Full VOROS: no effective constraints (alpha=1e-8 ≈ 0, kappa=P+N covers all points)
    return float(_geometry.voros(
        fprs, tprs, float(P + N), 1e-8, P, N,
        min_fp_cost_ratio, max_fp_cost_ratio, n_points=n_points,
    ))


def pvoros_score(y_true, y_pred, alpha, kappa_frac, min_fp_cost_ratio, max_fp_cost_ratio,
                 n_points=1000):
    """Partial VOROS score with precision and capacity constraints.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary labels.
    y_pred : array-like of shape (n,)
        Predicted probabilities.
    alpha : float in (0, 1)
        Minimum precision (PPV) constraint.
    kappa_frac : float in (0, 1]
        Maximum predicted positive fraction (capacity = kappa_frac * len(y_true)).
    min_fp_cost_ratio : float
        Minimum C0/C1 cost ratio.
    max_fp_cost_ratio : float
        Maximum C0/C1 cost ratio.
    n_points : int
        Number of cost ratio grid points for integration.

    Returns
    -------
    float : pVOROS score in [0, 1]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    P = int(np.sum(y_true == 1))
    N = int(np.sum(y_true == 0))
    n = len(y_true)
    kappa = kappa_frac * float(n)

    fprs, tprs, thrs = roc_curve(y_true, y_pred)
    fprs = fprs.astype(float)
    tprs = tprs.astype(float)
    thrs = thrs.astype(float)

    # Filter to feasible ROC points before integrating (matches step7 behavior)
    _, acc_fprs, acc_tprs, _, _ = _geometry._kept_on_valid(fprs, tprs, thrs, alpha, kappa, N, P)

    return float(_geometry.voros(
        acc_fprs, acc_tprs, kappa, alpha, P, N,
        min_fp_cost_ratio, max_fp_cost_ratio, n_points=n_points,
    ))


def make_pvoros_scorer(alpha, kappa_frac, min_fp_cost_ratio, max_fp_cost_ratio, n_points=1000):
    """Return a pVOROS scorer with constraints baked in.

    The returned function has signature ``(y_true, y_pred) -> float`` and is
    compatible with ``sklearn.model_selection.cross_val_score`` and
    ``sklearn.metrics.make_scorer``.

    Parameters
    ----------
    alpha : float in (0, 1)
        Minimum precision constraint.
    kappa_frac : float in (0, 1]
        Maximum predicted positive fraction.
    min_fp_cost_ratio : float
        Minimum C0/C1 cost ratio.
    max_fp_cost_ratio : float
        Maximum C0/C1 cost ratio.
    n_points : int
        Number of integration grid points.

    Returns
    -------
    callable : scorer(y_true, y_pred) -> float

    Example
    -------
    >>> scorer = make_pvoros_scorer(alpha=0.15, kappa_frac=0.5,
    ...                              min_fp_cost_ratio=1/6, max_fp_cost_ratio=1/9)
    >>> score = scorer(y_true, y_pred)
    """
    def scorer(y_true, y_pred):
        return pvoros_score(
            y_true, y_pred, alpha, kappa_frac,
            min_fp_cost_ratio, max_fp_cost_ratio, n_points=n_points,
        )
    scorer.__name__ = (
        f"pvoros_a{alpha}_k{kappa_frac}_r{min_fp_cost_ratio:.4g}-{max_fp_cost_ratio:.4g}"
    )
    return scorer
