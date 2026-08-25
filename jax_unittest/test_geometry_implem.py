import unittest
import numpy as np
import jax
import jax.numpy as jnp

# Ensure JAX float64 precision is enabled during testing
jax.config.update("jax_enable_x64", True)
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir) 
sys.path.append(parent_dir)
import _geometry
import _geometry_jax


class TestGeometryJaxParity(unittest.TestCase):
    """
    Rigorously test parity between _geometry (NumPy reference)
    and _geometry_jax (JAX vector/grad-friendly implementation).
    """

    def setUp(self):
        # Sample ROC curves covering canonical edge cases
        self.roc_curves = [
            # 1. Standard monotonic ROC
            (
                jnp.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
                jnp.array([0.0, 0.7, 0.8, 0.9, 0.9, 1.0]),
            ),
            # 2. Perfect classifier (step function)
            (
                jnp.array([0.0, 0.0, 1.0]),
                jnp.array([0.0, 1.0, 1.0]),
            ),
            # 3. Chance line / Diagonal
            (
                jnp.array([0.0, 0.25, 0.5, 0.75, 1.0]),
                jnp.array([0.0, 0.25, 0.5, 0.75, 1.0]),
            ),
            # 4. Dense synthetic curve with degenerate points
            (
                jnp.linspace(0.0, 1.0, 20),
                jnp.sqrt(jnp.linspace(0.0, 1.0, 20)),
            ),
        ]

        # Parameter combinations covering tight, relaxed, and binding constraints
        self.param_grid = [
            # (P, N, alpha, kappa)
            (10, 100, 0.2, 30),      # Standard binding capacity
            (100, 100, 0.5, 200),    # Non-binding capacity (kappa >= P+N)
            (500, 50, 0.01, 10),     # Severe class imbalance + tight capacity
            (1, 1000, 0.95, 100),    # High precision requirement
            (50, 50, 0.1, 5),        # Small capacity
        ]

        # Cost ratio ranges testing lower/upper boundary behavior
        self.r_ranges = [
            (0.0, 1.0),        # Standard including t=0
            (1.0, 10.0),       # Moderate ratio range
            (0.0, 1e6),        # Extreme upper cost ratio (t -> 1)
            (1e-6, 1e-3),      # Near-zero small cost ratios
            (0.5, 0.5),        # Zero-width range (min_r == max_r)
        ]

    # -------------------------------------------------------------------
    # 1. Tests for Total Region Geometry
    # -------------------------------------------------------------------
    def test_total_region_area_parity(self):
        """Verify total region area calculation across diverse P, N, alpha, kappa."""
        for P, N, alpha, kappa in self.param_grid:
            with self.subTest(P=P, N=N, alpha=alpha, kappa=kappa):
                ref_area, ref_poly = _geometry.total_region_area(P, N, alpha, kappa)
                jax_area, jax_poly = _geometry_jax.total_region_area(P, N, alpha, kappa)

                self.assertAlmostEqual(
                    float(jax_area), float(ref_area), places=6,
                    msg=f"Total area mismatch for P={P}, N={N}, α={alpha}, κ={kappa}"
                )

    # -------------------------------------------------------------------
    # 2. Tests for Isoperformance Line Coefficients
    # -------------------------------------------------------------------
    def test_iso_performance_line_sweep(self):
        """Test isoperformance line coefficients across extreme t values (0.0 to 0.999999)."""
        test_points = [(0.0, 0.0), (0.2, 0.8), (0.5, 0.5), (1.0, 1.0)]
        t_values = [0.0, 1e-12, 1e-5, 0.1, 0.5, 0.9, 1.0 - 1e-12]

        for h, k in test_points:
            for t in t_values:
                with self.subTest(h=h, k=k, t=t):
                    ref_a, ref_b, ref_c = _geometry._iso_performance_line(h, k, t)
                    jax_a, jax_b, jax_c = _geometry_jax._iso_performance_line(h, k, t)

                    self.assertAlmostEqual(float(jax_a), float(ref_a), places=6)
                    self.assertAlmostEqual(float(jax_b), float(ref_b), places=6)
                    self.assertAlmostEqual(float(jax_c), float(ref_c), places=6)

    # -------------------------------------------------------------------
    # 3. Tests for Reduced Area at Individual ROC Coordinates
    # -------------------------------------------------------------------
    def test_reduced_area_grid(self):
        """Evaluate reduced_area across combinations of (fpr, tpr) and cost ratios."""
        for P, N, alpha, kappa in self.param_grid[:3]:
            for r in [0.0, 0.1, 1.0, 100.0, 1e5]:
                for fpr, tpr in [(0.0, 0.0), (0.2, 0.7), (0.5, 0.5), (1.0, 1.0)]:
                    with self.subTest(P=P, N=N, alpha=alpha, kappa=kappa, r=r, fpr=fpr, tpr=tpr):
                        ref_val = _geometry.reduced_area(fpr, tpr, kappa, alpha, P, N, r)
                        jax_val = _geometry_jax.reduced_area(fpr, tpr, kappa, alpha, P, N, r)

                        self.assertAlmostEqual(
                            float(jax_val), float(ref_val), places=6,
                            msg=f"Reduced area mismatch at fpr={fpr}, tpr={tpr}, r={r}"
                        )

    # -------------------------------------------------------------------
    # 4. Tests for Max Area Per t Across Full Parameter Sweep
    # -------------------------------------------------------------------
    def test_max_area_per_t_sweeps(self):
        """Comprehensive sweep over curves, parameters, and ratio ranges for max_area_per_t."""
        n_points = 50

        for fprs, tprs in self.roc_curves:
            for P, N, alpha, kappa in self.param_grid:
                for min_r, max_r in self.r_ranges:
                    with self.subTest(min_r=min_r, max_r=max_r, P=P, N=N, α=alpha, κ=kappa):
                        # NumPy computation
                        ref_max, ref_ts = _geometry.max_area_per_t(
                            np.array(fprs), np.array(tprs), kappa, alpha, P, N,
                            min_r, max_r, n_points=n_points
                        )

                        # JAX computation
                        jax_max, jax_ts = _geometry_jax.max_area_per_t(
                            fprs, tprs, kappa, alpha, P, N,
                            min_r, max_r, n_points=n_points
                        )

                        # Validate output lengths
                        self.assertEqual(len(jax_max), len(ref_max))
                        self.assertEqual(len(jax_ts), len(ref_ts))

                        # Element-wise floating point assertions
                        np.testing.assert_allclose(
                            np.array(jax_ts, dtype=np.float64),
                            np.array(ref_ts, dtype=np.float64),
                            atol=1e-6, rtol=1e-5,
                            err_msg=f"t-vector mismatch for min_r={min_r}, max_r={max_r}"
                        )

                        np.testing.assert_allclose(
                            np.array(jax_max, dtype=np.float64),
                            np.array(ref_max, dtype=np.float64),
                            atol=1e-6, rtol=1e-5,
                            err_msg=f"Max area array mismatch for min_r={min_r}, max_r={max_r}"
                        )

    # -------------------------------------------------------------------
    # 5. Tests for VOROS Integration End-to-End
    # -------------------------------------------------------------------
    def test_voros_integrator_parity(self):
        """Verify integrated partial VOROS scalar output parity."""
        fprs, tprs = self.roc_curves[3]
        n_points = 100

        for P, N, alpha, kappa in self.param_grid:
            for min_r, max_r in self.r_ranges:
                with self.subTest(min_r=min_r, max_r=max_r, P=P, N=N):
                    ref_vor = _geometry.voros(
                        np.array(fprs), np.array(tprs), kappa, alpha, P, N,
                        min_r, max_r, n_points=n_points
                    )
                    jax_vor = _geometry_jax.voros_jax(
                        fprs, tprs, kappa, alpha, P, N,
                        min_r, max_r, n_points=n_points
                    )

                    self.assertAlmostEqual(
                        float(jax_vor), float(ref_vor), places=6,
                        msg=f"VOROS integral mismatch for min_r={min_r}, max_r={max_r}"
                    )


if __name__ == "__main__":
    unittest.main()