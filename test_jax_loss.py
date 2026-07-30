"""
Sanity-checks that jax_voros_loss agrees with pvoros_loss
across multiple seed datasets and randomized parameter pairings.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["JAX_PLATFORMS"] = "cpu"
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from metrics_jax import pv_loss, pvoros_loss_kept_on_valid



# Generate 10 reproducible random (theta, c) pairs
rng = np.random.default_rng(seed=42)

KAPPA_FRAC = 0.3
ALPHA = 0.6
MIN_FP_COST_RATIO = 1 / 9
MAX_FP_COST_RATIO = 1 / 6
N_POINTS = 1000

THETA_C_PAIRS = [
    (float(t), float(c))
    for t, c in zip(rng.uniform(0, 2 * np.pi, 1), rng.uniform(-1.0, 1.0, 1))
]

SEED_FILENAMES = [
    "seed_101_201.npy",
    "seed_301_101.npy",
    "seed_501_801.npy",
    "seed_601_201.npy",
    "seed_701_501.npy",
]


def load_seed_data(seed_filename):
    """Load data from a seed file."""
    data_dict = np.load(seed_filename, allow_pickle=True).item()
    x = data_dict["data"]["x"]
    y = data_dict["data"]["y"]
    return x, y


def get_thresholds(y_pred):
    """Compute thresholds for ROC curve based on unique predicted scores."""
    eps = 1e-5
    thresholds = np.unique(y_pred)[::-1]
    thresholds = np.clip(thresholds, eps, 1.0 - eps)
    return thresholds

def _theta_c_to_wb_and_thresholds(w_vec, b_val, x_val):
    logits = jnp.dot(x_val, w_vec) + b_val
    y_pred = jax.nn.sigmoid(logits)
    thresholds = get_thresholds(np.asarray(y_pred))
    return thresholds


def _theta_c_to_wb(theta, c, M=1.0):
    """Convert angular (theta, c) parametrization to (w, b) expected by pvoros_loss."""
    w1 = M * jnp.sin(theta)
    w2 = -M * jnp.cos(theta)
    w_vec = jnp.array([w1, w2], dtype=jnp.float64)
    b_val = jnp.array(M * c * jnp.cos(theta), dtype=jnp.float64)
    return w_vec, b_val


class TestJaxLossVsNonJaxVoros(unittest.TestCase):

    def setUp(self):
        # Cache loaded seed data across subTests within a single test method
        # so we don't reload the same .npy file 10x per seed.
        self._data_cache = {}

    def _get_data(self, seed_filename):
        if seed_filename not in self._data_cache:
            x_val, y_val = load_seed_data(seed_filename)
            x_val = jnp.asarray(x_val, dtype=jnp.float64)
            y_val = jnp.asarray(y_val, dtype=jnp.float64)
            self._data_cache[seed_filename] = (x_val, y_val)
        return self._data_cache[seed_filename]

    def test_jax_loss_close_to_nonjax_voros(self):
        for seed_filename in SEED_FILENAMES:
            x_val, y_val = self._get_data(seed_filename)

            for theta, c in THETA_C_PAIRS:
                with self.subTest(seed_filename=seed_filename, theta=theta, c=c):
                    w_vec, b_val = _theta_c_to_wb(theta, c)
                    params_wb = {'w': w_vec, 'b': b_val}

                    P = float(jnp.sum(y_val == 1.0))
                    N = float(jnp.sum(y_val == 0.0))
                    KAPPA = KAPPA_FRAC * (P + N)

                    thresholds = _theta_c_to_wb_and_thresholds(w_vec, b_val, x_val)

                    old_loss_val = float(pvoros_loss_kept_on_valid(
                        params=params_wb,
                        X=x_val,
                        y_true=y_val,
                        kappa=KAPPA,
                        alpha=ALPHA,
                        thresholds=thresholds,
                        min_fp_cost_ratio=MIN_FP_COST_RATIO,
                        max_fp_cost_ratio=MAX_FP_COST_RATIO
                    ))

                    new_loss_val = float(
                        pv_loss(
                            params_wb, x_val, y_val, P, N, KAPPA, ALPHA, thresholds,
                            MIN_FP_COST_RATIO, MAX_FP_COST_RATIO, N_POINTS
                        )
                    )

                    new_loss_val = float(new_loss_val)

                    print(
                        f"seed={seed_filename}, theta={theta:.4f}, c={c:.4f}: "
                        f"old_loss_val={old_loss_val:.7f}, new_loss_val={new_loss_val:.7f}, satisfy={satisfy}"
                    )

                    diff = abs(new_loss_val - old_loss_val)
                    if satisfy:
                        self.assertLessEqual(
                            diff,
                            1e-7,
                            msg=(
                                f"seed={seed_filename}, theta={theta:.4f}, c={c:.4f}: "
                                f"jax_loss={new_loss_val:.7f} vs pvoros_loss={old_loss_val:.7f} "
                                f"(diff={diff:.7f})"
                            ),
                        )
                    else:
                        self.assertEqual(new_loss_val, 0.0)

    def test_jax_loss_is_zero_when_no_point_satisfies_constraints(self):
        """If constraints are impossible to satisfy, both paths should treat
        the loss/VOROS as 0 (jax via the `satisfy` gate, non-jax via an
        all-empty/degenerate max_points curve)."""
        for seed_filename in SEED_FILENAMES:
            x_val, y_val = self._get_data(seed_filename)

            for theta, c in THETA_C_PAIRS:
                with self.subTest(seed_filename=seed_filename, theta=theta, c=c):
                    w_vec, b_val = _theta_c_to_wb(theta, c)
                    params_wb = {'w': w_vec, 'b': b_val}

                    P = float(jnp.sum(y_val == 1))
                    N = float(jnp.sum(y_val == 0))

                    thresholds = _theta_c_to_wb_and_thresholds(w_vec, b_val, x_val)

                    # Impossible alpha/kappa combo -> nothing should satisfy
                    impossible_alpha = 0.9999
                    impossible_kappa = -1.0

                    loss = float(
                        pv_loss(
                            params_wb, x_val, y_val, P, N, impossible_kappa, impossible_alpha,
                            thresholds, MIN_FP_COST_RATIO, MAX_FP_COST_RATIO, N_POINTS
                        )
                    )

                    loss = float(loss)
                    self.assertAlmostEqual(loss, 0.0, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)