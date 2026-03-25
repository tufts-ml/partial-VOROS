"""Public cost functions for pVOROS-based threshold selection.

All four functions share the same pattern:
  - threshold(s) are selected on the validation set
  - cost is evaluated on the test set
  - expected cost is averaged over the cost ratio range

All return scalar float (lower is better), or (float, fprs_t, tprs_t) when
return_test_operating_points=True is passed.
"""

import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix

from . import _geometry


# ---- Internal helpers ----

def _fpr_tpr_at_thresholds(y_true, proba, thresholds):
    """Evaluate FPR/TPR on (y_true, proba) at each fixed threshold.

    Ported from src/step7_lucky_number.py. Uses max(1, ...) guards to
    avoid divide-by-zero on degenerate splits.
    """
    fprs = []
    tprs = []
    for thr in thresholds:
        yhat = (proba >= float(thr)).astype(int)
        cm = confusion_matrix(y_true, yhat, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        P = max(1, int(tp + fn))
        N = max(1, int(tn + fp))
        fprs.append(float(fp) / float(N))
        tprs.append(float(tp) / float(P))
    return np.asarray(fprs, dtype=float), np.asarray(tprs, dtype=float)


def _partial_auc(fprs, tprs):
    """Partial AUROC under the given (fpr, tpr) curve (trapezoid rule)."""
    if len(fprs) < 2:
        return 0.0
    df = __import__('pandas').DataFrame({'fpr': fprs, 'tpr': tprs})
    agg = df.groupby('fpr', as_index=False)['tpr'].max().sort_values('fpr')
    return float(np.trapz(agg['tpr'].to_numpy(), agg['fpr'].to_numpy()))


def _roc(y_true, proba):
    fprs, tprs, thrs = roc_curve(y_true, proba)
    return fprs.astype(float), tprs.astype(float), thrs.astype(float)


# ---- Public cost functions ----

def recall_cost(y_true_val, y_pred_val, y_true_test, y_pred_test,
                alpha, kappa_frac, min_fp_cost_ratio, max_fp_cost_ratio, n_points=1000,
                return_test_operating_points=False):
    """Expected test cost using the best-recall threshold selected on validation.

    From the constraint-feasible validation thresholds, selects the one that
    maximises TPR on **test**, then averages cost over the cost ratio range
    [min_fp_cost_ratio, max_fp_cost_ratio].

    Note: threshold selection uses test TPR (matching step7_lucky_number.py
    ``expected_cost_by_recall_test``). This is an oracle/upper-bound — a
    deployment system would instead select on validation TPR.

    Parameters
    ----------
    y_true_val, y_pred_val : array-like
        Validation labels and predicted probabilities.
    y_true_test, y_pred_test : array-like
        Test labels and predicted probabilities.
    alpha : float in (0, 1)
        Minimum precision (PPV) constraint.
    kappa_frac : float in (0, 1]
        Capacity as fraction of validation set size.
    min_fp_cost_ratio, max_fp_cost_ratio : float
        C0/C1 cost ratio range.
    n_points : int
        Number of integration grid points.
    return_test_operating_points : bool
        If True, return (cost, fprs_t, tprs_t) instead of just cost.
        fprs_t and tprs_t are length-1 arrays for the single selected threshold.

    Returns
    -------
    float or (float, ndarray, ndarray)
    """
    yv = np.asarray(y_true_val)
    pv = np.asarray(y_pred_val)
    yt = np.asarray(y_true_test)
    pt = np.asarray(y_pred_test)

    P_v = int(np.sum(yv == 1))
    N_v = int(np.sum(yv == 0))
    P_t = int(np.sum(yt == 1))
    N_t = int(np.sum(yt == 0))
    n_v = len(yv)

    kappa = kappa_frac * float(n_v)

    fprs_v, tprs_v, thrs_v = _roc(yv, pv)
    _, acc_fprs_v, acc_tprs_v, acc_thrs_v, _ = _geometry._kept_on_valid(
        fprs_v, tprs_v, thrs_v, alpha, kappa, N_v, P_v
    )

    # Apply kept val thresholds to test; pick threshold maximising test TPR
    # (oracle selection — matches step7 expected_cost_by_recall_test)
    acc_fprs_t, acc_tprs_t = _fpr_tpr_at_thresholds(yt, pt, acc_thrs_v)
    if len(acc_tprs_t) > 0:
        i_best_t = int(np.argmax(acc_tprs_t))
        fpr_best_t = acc_fprs_t[i_best_t]
        tpr_best_t = acc_tprs_t[i_best_t]
    else:
        fpr_best_t = 0.0
        tpr_best_t = 0.0

    # Average cost over r-grid on test
    rs_cost = np.linspace(min_fp_cost_ratio, max_fp_cost_ratio, num=max(2, n_points))
    t_t_cost = np.array([_geometry.ratio_to_t(r, P_t, N_t) for r in rs_cost], dtype=float)
    costs_t = t_t_cost * fpr_best_t + (1.0 - t_t_cost) * (1.0 - tpr_best_t)
    cost = float(np.mean(costs_t))

    if return_test_operating_points:
        return cost, np.array([fpr_best_t]), np.array([tpr_best_t])
    return cost


def pauroc_cost(y_true_val, y_pred_val, y_true_test, y_pred_test,
                alpha, kappa_frac, min_fp_cost_ratio, max_fp_cost_ratio, n_points=1000,
                return_test_operating_points=False):
    """Expected test cost for a model selected by partial AUROC.

    Computes cost the same way as recall_cost: selects the best-recall
    constraint-feasible val threshold and averages cost over the grid.

    The distinction from recall_cost is in MODEL SELECTION (caller provides
    predictions from the pAUROC-best model). The threshold selection and cost
    formula are identical to recall_cost.

    Parameters
    ----------
    y_true_val, y_pred_val : array-like
        Validation labels and predicted probabilities (from pAUROC-best model).
    y_true_test, y_pred_test : array-like
        Test labels and predicted probabilities.
    alpha : float in (0, 1)
        Minimum precision constraint.
    kappa_frac : float in (0, 1]
        Capacity as fraction of validation set size.
    min_fp_cost_ratio, max_fp_cost_ratio : float
        C0/C1 cost ratio range.
    n_points : int
        Number of integration grid points.
    return_test_operating_points : bool
        If True, return (cost, fprs_t, tprs_t).

    Returns
    -------
    float or (float, ndarray, ndarray)
    """
    return recall_cost(
        y_true_val, y_pred_val, y_true_test, y_pred_test,
        alpha, kappa_frac, min_fp_cost_ratio, max_fp_cost_ratio,
        n_points=n_points, return_test_operating_points=return_test_operating_points,
    )


def pvoros_cost(y_true_val, y_pred_val, y_true_test, y_pred_test,
                alpha, kappa_frac, min_fp_cost_ratio, max_fp_cost_ratio, n_points=1000,
                return_test_operating_points=False):
    """Expected test cost using pVOROS-optimal thresholds.

    At each cost ratio r in the grid, selects the threshold on validation
    (among constraint-feasible thresholds) that maximises reduced area.
    Applies the corresponding threshold to test and averages cost(r) over
    the grid.

    Parameters
    ----------
    y_true_val, y_pred_val : array-like
        Validation labels and predicted probabilities.
    y_true_test, y_pred_test : array-like
        Test labels and predicted probabilities.
    alpha : float in (0, 1)
        Minimum precision constraint.
    kappa_frac : float in (0, 1]
        Capacity as fraction of validation set size.
    min_fp_cost_ratio, max_fp_cost_ratio : float
        C0/C1 cost ratio range.
    n_points : int
        Number of integration grid points.
    return_test_operating_points : bool
        If True, return (cost, fprs_t, tprs_t) where fprs_t/tprs_t are the
        per-r test operating points (length n_points).

    Returns
    -------
    float or (float, ndarray, ndarray)

    Notes
    -----
    kappa is derived from the validation set size (not test set size), matching
    the step7_lucky_number.py implementation.
    """
    yv = np.asarray(y_true_val)
    pv = np.asarray(y_pred_val)
    yt = np.asarray(y_true_test)
    pt = np.asarray(y_pred_test)

    P_v = int(np.sum(yv == 1))
    N_v = int(np.sum(yv == 0))
    P_t = int(np.sum(yt == 1))
    N_t = int(np.sum(yt == 0))
    n_v = len(yv)

    # NOTE: kappa uses validation set size, not test set size
    kappa = kappa_frac * float(n_v)

    fprs_v, tprs_v, thrs_v = _roc(yv, pv)
    _, acc_fprs_v, acc_tprs_v, acc_thrs_v, _ = _geometry._kept_on_valid(
        fprs_v, tprs_v, thrs_v, alpha, kappa, N_v, P_v
    )

    if len(acc_fprs_v) == 0:
        empty = np.zeros(n_points)
        if return_test_operating_points:
            return 0.0, empty, empty
        return 0.0

    # pVOROS on val: get best threshold per cost ratio point
    _, ts_arr, best_thr_arr = _geometry.voros(
        acc_fprs_v, acc_tprs_v, kappa, alpha, P_v, N_v,
        min_fp_cost_ratio, max_fp_cost_ratio, n_points=n_points,
        return_best_thresholds=True, thresholds=acc_thrs_v,
    )
    rs_list = np.linspace(min_fp_cost_ratio, max_fp_cost_ratio, num=len(ts_arr))
    best_thr_np = np.asarray(best_thr_arr, dtype=float)

    # Evaluate val-selected thresholds on test
    fprs_t_sel, tprs_t_sel = _fpr_tpr_at_thresholds(yt, pt, best_thr_np)

    # Average cost on test over r-grid
    t_t_grid = np.array([_geometry.ratio_to_t(r, P_t, N_t) for r in rs_list], dtype=float)
    costs_t = t_t_grid * fprs_t_sel + (1.0 - t_t_grid) * (1.0 - tprs_t_sel)
    cost = float(np.mean(costs_t))

    if return_test_operating_points:
        return cost, fprs_t_sel, tprs_t_sel
    return cost


def voros_cost(y_true_val, y_pred_val, y_true_test, y_pred_test,
               alpha, kappa_frac, min_fp_cost_ratio, max_fp_cost_ratio, n_points=1000,
               return_test_operating_points=False):
    """Expected test cost using pVOROS-optimal thresholds (VOROS model selection).

    Identical to pvoros_cost in every way. The distinction is in MODEL SELECTION:
    the caller provides predictions from the VOROS-best model (selected by unconstrained
    voros_score), while pvoros_cost receives predictions from the pVOROS-best model.

    Parameters
    ----------
    y_true_val, y_pred_val : array-like
        Validation labels and predicted probabilities (from VOROS-best model).
    y_true_test, y_pred_test : array-like
        Test labels and predicted probabilities.
    alpha : float in (0, 1)
        Minimum precision constraint.
    kappa_frac : float in (0, 1]
        Capacity as fraction of validation set size.
    min_fp_cost_ratio, max_fp_cost_ratio : float
        C0/C1 cost ratio range.
    n_points : int
        Number of integration grid points.
    return_test_operating_points : bool
        If True, return (cost, fprs_t, tprs_t).

    Returns
    -------
    float or (float, ndarray, ndarray)
    """
    return pvoros_cost(
        y_true_val, y_pred_val, y_true_test, y_pred_test,
        alpha, kappa_frac, min_fp_cost_ratio, max_fp_cost_ratio,
        n_points=n_points, return_test_operating_points=return_test_operating_points,
    )
