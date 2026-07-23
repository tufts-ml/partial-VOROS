"""
Sanity-checks that jax_voros_loss agrees with pvoros_loss
across multiple seed datasets and randomized parameter pairings.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import train_and_voros  # imported to safely monkeypatch module constants
from train_and_voros import (  # noqa: adjust to real module
    jax_voros_loss,
    load_seed_data,
    ALPHA,
    KAPPA_FRAC,
    MIN_FP_COST_RATIO,
    MAX_FP_COST_RATIO,
    N_POINTS,
)

from metrics_jax import pvoros_loss  # noqa: adjust to real module


# Generate 10 reproducible random (theta, c) pairs
rng = np.random.default_rng(seed=42)
THETA_C_PAIRS = [
    (float(t), float(c)) 
    for t, c in zip(rng.uniform(0, 2 * np.pi, 10), rng.uniform(-1.0, 1.0, 10))
]

SEED_FILENAMES = [
    "seed_101_201.npy",
    "seed_301_101.npy",
    "seed_501_801.npy",
    "seed_601_201.npy",
    "seed_701_501.npy",
]


def _get_numeric_data(seed_filename):
    """Loads seed data and ensures feature/label arrays are float/int numeric types
    for JAX mathematical operations (prevents string abstract array errors)."""
    x_val, y_val = load_seed_data(seed_filename)
    x_val = jnp.asarray(x_val, dtype=jnp.float32)
    y_val = jnp.asarray(y_val, dtype=jnp.float32)
    return x_val, y_val


@pytest.mark.parametrize("seed_filename", SEED_FILENAMES)
@pytest.mark.parametrize("theta, c", THETA_C_PAIRS)
def test_jax_loss_close_to_nonjax_voros(seed_filename, theta, c):
    x_val, y_val = load_seed_data(seed_filename)
    x_val = jnp.asarray(x_val, dtype=jnp.float32)
    y_val = jnp.asarray(y_val, dtype=jnp.float32)

    # Convert angular (theta, c) parametrization to (w, b) expected by pvoros_loss
    M = 1.0
    w1 = M * jnp.sin(theta)
    w2 = -M * jnp.cos(theta)
    w_vec = jnp.array([w1, w2], dtype=jnp.float32)
    b_val = jnp.array(M * c * jnp.cos(theta), dtype=jnp.float32)

    # Tuple expected by pvoros_loss: (w, b)
    params_wb = (w_vec, b_val)

    # Dict expected by jax_voros_loss: {"theta": ..., "c": ...}
    params_dict = {
        "theta": jnp.array(theta, dtype=jnp.float32),
        "c": jnp.array(c, dtype=jnp.float32),
    }

    P = float(jnp.sum(y_val == 1.0))
    N = float(jnp.sum(y_val == 0.0))
    KAPPA = KAPPA_FRAC * (P + N)

    eps = 1e-5
    thresholds = jnp.linspace(eps, 1.0 - eps, 100)

    old_loss_val = float(
        pvoros_loss(
            params=params_wb,
            X=x_val,
            y_true=y_val,
            kappa=KAPPA,
            alpha=ALPHA,
            thresholds=thresholds,
            min_fp_cost_ratio=MIN_FP_COST_RATIO,
            max_fp_cost_ratio=MAX_FP_COST_RATIO,
            n_points=N_POINTS,
        )
    )

    new_loss_val = float(jax_voros_loss(params_dict, x_val, y_val, P, N))

    assert new_loss_val == pytest.approx(old_loss_val, abs=0.05), (
        f"seed={seed_filename}, theta={theta:.4f}, c={c:.4f}: "
        f"jax_loss={new_loss_val:.4f} vs pvoros_loss={old_loss_val:.4f} "
        f"(diff={abs(new_loss_val - old_loss_val):.4f})"
    )


@pytest.mark.parametrize("seed_filename", SEED_FILENAMES)
@pytest.mark.parametrize("theta, c", THETA_C_PAIRS)
def test_jax_loss_is_zero_when_no_point_satisfies_constraints(seed_filename, theta, c, monkeypatch):
    """If constraints are impossible to satisfy, both paths should treat
    the loss/VOROS as 0 (jax via the `satisfy` gate, non-jax via an
    all-empty/degenerate max_points curve)."""
    x_val, y_val = _get_numeric_data(seed_filename)
    params = {"theta": jnp.array(theta, dtype=jnp.float32), "c": jnp.array(c, dtype=jnp.float32)}
    P = float(jnp.sum(y_val == 1))
    N = float(jnp.sum(y_val == 0))

    # Safely override ALPHA and KAPPA_FRAC for this test execution only
    monkeypatch.setattr(train_and_voros, "ALPHA", 0.9999)
    monkeypatch.setattr(train_and_voros, "KAPPA_FRAC", -1.0)

    loss = jax_voros_loss(params, x_val, y_val, P, N)

    assert float(loss) == pytest.approx(0.0, abs=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])