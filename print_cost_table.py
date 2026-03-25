"""Print a table of expected test costs by model-selection strategy and scenario.

Usage:
    python pvoros/print_cost_table.py

Rows   : model-selection strategy (full VOROS, recall, pAUROC, pVOROS)
Columns: dataset x scenario (MIMIC S1, MIMIC S2, eICU S1, eICU S2)

Each cell shows "cost(violation)" where violation is OK, prec, cap, or prec+cap.

Fixture files must exist under pvoros/tests/fixtures/.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pvoros import pvoros_cost, recall_cost, voros_cost, pauroc_cost

# ---- Config ----

FIXTURES = os.path.join(os.path.dirname(__file__), 'tests', 'fixtures')

# Scenario parameters
S1 = dict(alpha=0.15, kappa_frac=0.5,  min_fp_cost_ratio=1/9,  max_fp_cost_ratio=1/6)
S2 = dict(alpha=0.5,  kappa_frac=0.1,  min_fp_cost_ratio=1/40, max_fp_cost_ratio=1/20)

# Fixture file sets: (valid, test) pairs per strategy per scenario
#   Keys: (dataset, scenario, strategy)
#   Values: (valid_file, test_file) or None if unknown
FIXTURES_MAP = {
    # MIMIC
    ('mimic', 1, 'pvoros'): ('scenario1_pvoros_valid.csv',  'scenario1_pvoros_test.csv'),
    ('mimic', 1, 'recall'): ('scenario1_recall_valid.csv',  'scenario1_recall_test.csv'),
    ('mimic', 1, 'pauroc'): ('scenario1_pvoros_valid.csv',  'scenario1_pvoros_test.csv'),  # same model as pvoros-best
    ('mimic', 1, 'voros'):  ('scenario1_pvoros_valid.csv',  'scenario1_pvoros_test.csv'),  # same model as pvoros-best
    ('mimic', 2, 'pvoros'): ('scenario2_valid.csv',         'scenario2_test.csv'),
    ('mimic', 2, 'recall'): ('scenario2_valid.csv',         'scenario2_test.csv'),
    ('mimic', 2, 'pauroc'): ('scenario2_valid.csv',         'scenario2_test.csv'),          # same model as pvoros-best
    ('mimic', 2, 'voros'):  ('mimic_s2_voros_valid.csv',     'mimic_s2_voros_test.csv'),
    # eICU
    ('eicu',  1, 'pvoros'): ('eicu_s1_pvoros_valid.csv',    'eicu_s1_pvoros_test.csv'),
    ('eicu',  1, 'recall'): ('eicu_s1_recall_valid.csv',    'eicu_s1_recall_test.csv'),
    ('eicu',  1, 'pauroc'): ('eicu_s1_recall_valid.csv',    'eicu_s1_recall_test.csv'),  # same model as recall-best
    ('eicu',  1, 'voros'):  ('eicu_s1_pvoros_valid.csv',    'eicu_s1_pvoros_test.csv'),  # same model as pvoros-best
    ('eicu',  2, 'pvoros'): ('eicu_s2_pvoros_valid.csv',    'eicu_s2_pvoros_test.csv'),
    ('eicu',  2, 'recall'): ('eicu_s2_pvoros_valid.csv',    'eicu_s2_pvoros_test.csv'),  # same model as pvoros-best
    ('eicu',  2, 'pauroc'): ('eicu_s2_pvoros_valid.csv',    'eicu_s2_pvoros_test.csv'),  # same model as pvoros-best
    ('eicu',  2, 'voros'):  ('eicu_s2_voros_valid.csv',     'eicu_s2_voros_test.csv'),
}

SCENARIO_PARAMS = {1: S1, 2: S2}
STRATEGIES = ['voros', 'recall', 'pauroc', 'pvoros']
STRATEGY_LABELS = {'voros': 'Full VOROS', 'recall': 'Recall',
                   'pauroc': 'pAUROC', 'pvoros': 'pVOROS'}
COLUMNS = [('mimic', 1), ('mimic', 2), ('eicu', 1), ('eicu', 2)]
COLUMN_LABELS = {('mimic', 1): 'MIMIC S1', ('mimic', 2): 'MIMIC S2',
                 ('eicu',  1): 'eICU S1',  ('eicu',  2): 'eICU S2'}

COST_FN = {'voros': voros_cost, 'recall': recall_cost,
           'pauroc': pauroc_cost, 'pvoros': pvoros_cost}


# ---- Helpers ----

def load(filename):
    df = pd.read_csv(os.path.join(FIXTURES, filename))
    return df['y_true'].to_numpy(), df['proba_y_eq_1'].to_numpy()


def _constraint_violation_str(fprs_t, tprs_t, alpha, kappa_frac, P_t, N_t):
    """Return 'OK', 'prec', 'cap', or 'prec+cap' based on test operating points."""
    n_t = P_t + N_t
    kappa = kappa_frac * n_t
    prec_viol = False
    cap_viol = False
    for fpr, tpr in zip(fprs_t, tprs_t):
        predicted_pos = tpr * P_t + fpr * N_t
        if predicted_pos > 0 and tpr * P_t < alpha * predicted_pos:
            prec_viol = True
        if predicted_pos > kappa:
            cap_viol = True
    if prec_viol and cap_viol:
        return "prec+cap"
    elif prec_viol:
        return "prec"
    elif cap_viol:
        return "cap"
    return "OK"


def compute_cell(dataset, scenario, strategy):
    """Return (cost_str, violation_str), or ('N/A', '') if fixture is missing."""
    key = (dataset, scenario, strategy)
    files = FIXTURES_MAP.get(key)
    if files is None:
        return 'N/A', ''

    vf, tf = files
    yv, pv = load(vf)
    yt, pt = load(tf)
    params = SCENARIO_PARAMS[scenario]
    fn = COST_FN[strategy]

    cost, fprs_t, tprs_t = fn(
        yv, pv, yt, pt, **params,
        return_test_operating_points=True,
    )

    P_t = int(np.sum(yt == 1))
    N_t = int(np.sum(yt == 0))
    viol = _constraint_violation_str(fprs_t, tprs_t,
                                     params['alpha'], params['kappa_frac'],
                                     P_t, N_t)
    return f"{cost:.3f}", viol


# ---- Main ----

def main():
    cell_w = 19  # wide enough for "0.261 (prec+cap)" + padding
    row_label_w = 12

    header = f"{'Strategy':<{row_label_w}}" + "".join(
        f"{COLUMN_LABELS[c]:>{cell_w}}" for c in COLUMNS
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for strategy in STRATEGIES:
        row = f"{STRATEGY_LABELS[strategy]:<{row_label_w}}"
        for col in COLUMNS:
            dataset, scenario = col
            cost_str, viol = compute_cell(dataset, scenario, strategy)
            if cost_str == 'N/A':
                cell = 'N/A'
            else:
                cell = f"{cost_str} ({viol})"
            row += f"{cell:>{cell_w}}"
        print(row)

    print(sep)


if __name__ == '__main__':
    main()
