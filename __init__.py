"""pvoros — partial VOROS scoring library.

Public API:

Scoring functions (scikit-learn compatible):
    voros_score(y_true, y_pred, min_fp_cost_ratio, max_fp_cost_ratio)
    pvoros_score(y_true, y_pred, alpha, kappa_frac, min_fp_cost_ratio, max_fp_cost_ratio)
    make_pvoros_scorer(alpha, kappa_frac, min_fp_cost_ratio, max_fp_cost_ratio)

Cost functions (threshold selection on val, cost evaluated on test):
    recall_cost(y_true_val, y_pred_val, y_true_test, y_pred_test, alpha, kappa_frac, ...)
    pauroc_cost(y_true_val, y_pred_val, y_true_test, y_pred_test, alpha, kappa_frac, ...)
    pvoros_cost(y_true_val, y_pred_val, y_true_test, y_pred_test, alpha, kappa_frac, ...)
    voros_cost(y_true_val, y_pred_val, y_true_test, y_pred_test, alpha, kappa_frac, ...)
"""

from .metrics import voros_score, pvoros_score, make_pvoros_scorer
from .cost import recall_cost, pauroc_cost, pvoros_cost, voros_cost

__all__ = [
    "voros_score",
    "pvoros_score",
    "make_pvoros_scorer",
    "recall_cost",
    "pauroc_cost",
    "pvoros_cost",
    "voros_cost",
]
