# pvoros

Python library implementing **VOROS** (Volume Over ROC Surface) and **partial VOROS** (pVOROS) metrics for evaluating binary classifiers under precision and capacity constraints.

## Background

VOROS summarizes classifier performance across a range of cost ratios by computing the volume under the ROC surface. Partial VOROS extends this with two constraints before computing the volume:

- **Precision constraint** (α): minimum PPV required at a threshold
- **Capacity constraint** (κ): maximum fraction of the population that can be flagged positive

These metrics are designed for clinical decision support settings where deployable thresholds must satisfy real-world operating constraints.

## Structure

```
pvoros/
├── metrics.py          # VOROS/pVOROS scoring functions (scikit-learn compatible)
├── cost.py             # Cost functions: threshold selected on val, cost evaluated on test
├── print_cost_table.py # Reproduce Table X from the paper
├── _geometry.py        # Internal geometry helpers (polygon clipping, reduced area, ROC filtering)
└── tests/
    ├── fixtures/       # Real MIMIC-IV and eICU prediction CSVs used in regression tests
    ├── test_geometry.py
    ├── test_metrics.py
    └── test_cost.py
```

### `metrics.py` — VOROS scoring

scikit-learn-compatible scoring functions. Use these to compare models during training/validation.

```python
from pvoros import voros_score, pvoros_score, make_pvoros_scorer

# Full VOROS (no constraints)
score = voros_score(y_true, y_pred, min_fp_cost_ratio=1/9, max_fp_cost_ratio=1/6)

# Partial VOROS with precision and capacity constraints
score = pvoros_score(y_true, y_pred,
                     alpha=0.15,            # minimum precision
                     kappa_frac=0.5,        # max flagged fraction
                     min_fp_cost_ratio=1/9, # Min/max ratios of cost of false positive 
                     max_fp_cost_ratio=1/6) #     to cost of false negative

# Bake constraints into a reusable scorer
scorer = make_pvoros_scorer(alpha=0.15, kappa_frac=0.5,
                             min_fp_cost_ratio=1/9, max_fp_cost_ratio=1/6)
score = scorer(y_true, y_pred)
```

### `cost.py` — deployment cost functions

For comparing model-selection strategies. Each function selects a threshold on a held-out validation set and evaluates expected cost on a test set, averaged over the cost-ratio range.

```python
from pvoros import recall_cost, pauroc_cost, pvoros_cost, voros_cost

# Shared signature:
cost = pvoros_cost(y_true_val, y_pred_val,
                   y_true_test, y_pred_test,
                   alpha=0.15, kappa_frac=0.5,
                   min_fp_cost_ratio=1/9, max_fp_cost_ratio=1/6)

# Return the test operating point(s) alongside cost
cost, fprs_t, tprs_t = pvoros_cost(..., return_test_operating_points=True)
```

| Function | Threshold selection | Intended use |
|---|---|---|
| `recall_cost` | Best-recall constraint-feasible val threshold (oracle) | Recall-maximizing strategy |
| `pauroc_cost` | Same as `recall_cost` | For pAUROC-selected models |
| `pvoros_cost` | Per-cost-ratio pVOROS-optimal val threshold | pVOROS strategy |
| `voros_cost`  | Same as `pvoros_cost` | For VOROS-selected models |

### `_geometry.py` — internal helpers

Polygon clipping (Sutherland-Hodgman), reduced area computation over the feasible ROC region, and `_kept_on_valid` which filters ROC curve points to the constraint-feasible set. Not part of the public API.

### `print_cost_table.py` — Reproduce Tables 1 and 2 from saved predictions

Runs all four cost calculation strategies across MIMIC-IV and eICU scenarios using saved labels and predictions and prints a formatted comparison table.

```bash
python pvoros/print_cost_table.py
```

Expected output:

```
----------------------------------------------------------------------------------------
Strategy               MIMIC S1           MIMIC S2            eICU S1            eICU S2
----------------------------------------------------------------------------------------
Full VOROS           0.261 (OK)         0.640 (OK)         0.319 (OK)         0.772 (OK)
Recall               0.302 (OK)         0.538 (OK)        0.336 (cap)       0.707 (prec)
pAUROC               0.303 (OK)         0.538 (OK)        0.336 (cap)       0.707 (prec)
pVOROS               0.261 (OK)         0.538 (OK)         0.319 (OK)       0.707 (prec)
----------------------------------------------------------------------------------------
```

Violation codes: `OK` = no constratints violated, `prec` = precision constraint violated on test, `cap` = capacity constraint violated on test.

## Installation

All dependencies are listed in `requirements.txt` (`numpy`, `scikit-learn`, `pandas`, `pytest`).

**pip + venv (standard)**
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**uv (fast drop-in for pip)**
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

**conda / micromamba**
```bash
conda create -n pvoros python=3.11
conda activate pvoros
pip install -r requirements.txt
```

## Running tests

```bash
pytest pvoros/tests/
```
