"""Unit tests comparing NumPy and JAX versions of _kept_on_valid."""

import unittest
import numpy as np
import jax.numpy as jnp
import _geometry
import _geometry_jax


class TestKeptOnValid(unittest.TestCase):
    """Tests for _kept_on_valid function comparison between NumPy and JAX versions."""
    
    def test_basic_filtering(self):
        """Test basic filtering with mixed valid/invalid points."""
        fprs = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        tprs = np.array([0.0, 0.3, 0.5, 0.7, 0.8, 1.0])
        thresholds = np.array([2.0, 1.8, 1.5, 1.2, 0.8, 0.0])
        
        alpha = 0.2
        kappa = 30
        P = 10
        N = 100
        
        # NumPy version
        mask_np, fprs_np, tprs_np, thrs_np, satisfy_np = _geometry._kept_on_valid(
            fprs, tprs, thresholds, alpha, kappa, N, P
        )
        
        # JAX version
        fprs_jax = jnp.array(fprs)
        tprs_jax = jnp.array(tprs)
        thresholds_jax = jnp.array(thresholds)
        
        mask_jax, fprs_jax_out, tprs_jax_out, thrs_jax, satisfy_jax = _geometry_jax._kept_on_valid(
            fprs_jax, tprs_jax, thresholds_jax, alpha, kappa, N, P
        )
        
        # Convert to numpy for comparison
        mask_jax = np.array(mask_jax)
        satisfy_jax = bool(satisfy_jax)
        
        # Test mask equivalence
        self.assertTrue(np.array_equal(mask_np, mask_jax), 
                       f"Masks differ: NumPy={mask_np}, JAX={mask_jax}")
        
        # Test satisfy equivalence
        self.assertEqual(satisfy_np, satisfy_jax,
                        f"Satisfy differs: NumPy={satisfy_np}, JAX={satisfy_jax}")
        
        if satisfy_np:
            # If points are filtered, check that filtered points match
            fprs_filtered_np = np.array(fprs_np)[~np.isnan(fprs_np)]
            fprs_filtered_jax = np.array(fprs_jax_out)[~np.isnan(fprs_jax_out)]
            
            np.testing.assert_array_almost_equal(fprs_filtered_np, fprs_filtered_jax,
                                                decimal=10)
            
            tprs_filtered_np = np.array(tprs_np)[~np.isnan(tprs_np)]
            tprs_filtered_jax = np.array(tprs_jax_out)[~np.isnan(tprs_jax_out)]
            
            np.testing.assert_array_almost_equal(tprs_filtered_np, tprs_filtered_jax,
                                                decimal=10)
    
    def test_all_valid(self):
        """Test when all points satisfy constraints."""
        fprs = np.array([0.0, 0.01, 0.02])
        tprs = np.array([0.0, 0.5, 0.8])
        thresholds = np.array([2.0, 1.5, 0.5])
        
        alpha = 0.1
        kappa = 100
        P = 10
        N = 100
        
        # NumPy version
        mask_np, _, _, _, satisfy_np = _geometry._kept_on_valid(
            fprs, tprs, thresholds, alpha, kappa, N, P
        )
        
        # JAX version
        mask_jax, _, _, _, satisfy_jax = _geometry_jax._kept_on_valid(
            jnp.array(fprs), jnp.array(tprs), jnp.array(thresholds), alpha, kappa, N, P
        )
        
        mask_jax = np.array(mask_jax)
        satisfy_jax = bool(satisfy_jax)
        
        # All points should be valid
        self.assertTrue(mask_np.all(), "NumPy: Not all points marked valid")
        self.assertTrue(mask_jax.all(), "JAX: Not all points marked valid")
        self.assertTrue(satisfy_np, "NumPy: satisfy should be True")
        self.assertTrue(satisfy_jax, "JAX: satisfy should be True")
    
    def test_no_valid(self):
        """Test when no points satisfy constraints."""
        fprs = np.array([0.9, 0.95, 1.0])
        tprs = np.array([0.0, 0.0, 0.0])
        thresholds = np.array([0.5, 0.3, 0.1])
        
        alpha = 0.9
        kappa = 1
        P = 10
        N = 100
        
        # NumPy version
        mask_np, fprs_np, tprs_np, thrs_np, satisfy_np = _geometry._kept_on_valid(
            fprs, tprs, thresholds, alpha, kappa, N, P
        )
        
        # JAX version
        mask_jax, fprs_jax_out, tprs_jax_out, thrs_jax, satisfy_jax = _geometry_jax._kept_on_valid(
            jnp.array(fprs), jnp.array(tprs), jnp.array(thresholds), alpha, kappa, N, P
        )
        
        satisfy_jax = bool(satisfy_jax)
        
        # All points should be invalid
        self.assertFalse(satisfy_np, "NumPy: satisfy should be False")
        self.assertFalse(satisfy_jax, "JAX: satisfy should be False")
        
        # When no points are valid, should fallback to all points
        np.testing.assert_array_equal(fprs_np, fprs)
        np.testing.assert_array_equal(tprs_np, tprs)
        np.testing.assert_array_equal(thrs_np, thresholds)
    
    def test_single_valid_point(self):
        """Test when only one point satisfies constraints."""
        fprs = np.array([0.0, 0.1, 0.5, 0.9])
        tprs = np.array([0.0, 0.5, 0.2, 0.0])
        thresholds = np.array([2.0, 1.5, 1.0, 0.5])
        
        alpha = 0.2
        kappa = 50
        P = 10
        N = 100
        
        # NumPy version
        mask_np, fprs_np, tprs_np, thrs_np, satisfy_np = _geometry._kept_on_valid(
            fprs, tprs, thresholds, alpha, kappa, N, P
        )
        
        # JAX version
        mask_jax, fprs_jax_out, tprs_jax_out, thrs_jax, satisfy_jax = _geometry_jax._kept_on_valid(
            jnp.array(fprs), jnp.array(tprs), jnp.array(thresholds), alpha, kappa, N, P
        )
        
        mask_jax = np.array(mask_jax)
        satisfy_jax = bool(satisfy_jax)
        
        # Check consistency
        self.assertTrue(np.array_equal(mask_np, mask_jax))
        self.assertEqual(satisfy_np, satisfy_jax)
        
        if satisfy_np:
            # Should have exactly one valid point
            n_valid_np = mask_np.sum()
            n_valid_jax = mask_jax.sum()
            self.assertEqual(n_valid_np, n_valid_jax)
            self.assertGreater(n_valid_np, 0)
    
    def test_from_seed_data(self):
        """Test with realistic parameters from seed data."""
        # Create realistic ROC curve
        n_points = 50
        fprs = np.linspace(0, 1, n_points)
        tprs = np.sqrt(fprs)  # Realistic curve
        thresholds = np.linspace(2.0, 0.0, n_points)
        
        # Parameters from seed_101_201.npy (P=490, N=3710 from 70/30 split of 5200 samples)
        alpha = 0.2
        kappa = 30
        P = 490
        N = 3710
        
        # NumPy version
        mask_np, fprs_np, tprs_np, thrs_np, satisfy_np = _geometry._kept_on_valid(
            fprs, tprs, thresholds, alpha, kappa, N, P
        )
        
        # JAX version
        mask_jax, fprs_jax_out, tprs_jax_out, thrs_jax, satisfy_jax = _geometry_jax._kept_on_valid(
            jnp.array(fprs), jnp.array(tprs), jnp.array(thresholds), alpha, kappa, N, P
        )
        
        mask_jax = np.array(mask_jax)
        satisfy_jax = bool(satisfy_jax)
        
        # Verify masks match
        self.assertTrue(np.array_equal(mask_np, mask_jax),
                       f"Masks differ at realistic scale")
        
        # Verify satisfy flags match
        self.assertEqual(satisfy_np, satisfy_jax)
        
        # If satisfied, verify filtered values match
        if satisfy_np:
            fprs_valid_np = fprs_np[mask_np]
            fprs_valid_jax = np.array(fprs_jax_out)[np.array(mask_jax)]
            
            np.testing.assert_array_almost_equal(fprs_valid_np, fprs_valid_jax,
                                                decimal=10)
            
            tprs_valid_np = tprs_np[mask_np]
            tprs_valid_jax = np.array(tprs_jax_out)[np.array(mask_jax)]
            
            np.testing.assert_array_almost_equal(tprs_valid_np, tprs_valid_jax,
                                                decimal=10)
    
    def test_edge_case_extreme_alpha(self):
        """Test with extreme alpha values."""
        fprs = np.array([0.0, 0.1, 0.2, 0.3])
        tprs = np.array([0.0, 0.1, 0.2, 0.3])
        thresholds = np.array([2.0, 1.5, 1.0, 0.5])
        P = 10
        N = 100
        
        for alpha in [0.001, 0.5, 0.99]:
            with self.subTest(alpha=alpha):
                kappa = 30
                
                mask_np, _, _, _, satisfy_np = _geometry._kept_on_valid(
                    fprs, tprs, thresholds, alpha, kappa, N, P
                )
                
                mask_jax, _, _, _, satisfy_jax = _geometry_jax._kept_on_valid(
                    jnp.array(fprs), jnp.array(tprs), jnp.array(thresholds), 
                    alpha, kappa, N, P
                )
                
                mask_jax = np.array(mask_jax)
                satisfy_jax = bool(satisfy_jax)
                
                self.assertTrue(np.array_equal(mask_np, mask_jax),
                               f"Masks differ for alpha={alpha}")
                self.assertEqual(satisfy_np, satisfy_jax,
                                f"Satisfy differs for alpha={alpha}")
    
    def test_edge_case_extreme_kappa(self):
        """Test with extreme kappa values."""
        fprs = np.array([0.0, 0.1, 0.2, 0.3])
        tprs = np.array([0.0, 0.1, 0.2, 0.3])
        thresholds = np.array([2.0, 1.5, 1.0, 0.5])
        P = 10
        N = 100
        alpha = 0.2
        
        for kappa in [0.5, 10, 100, 1000]:
            with self.subTest(kappa=kappa):
                mask_np, _, _, _, satisfy_np = _geometry._kept_on_valid(
                    fprs, tprs, thresholds, alpha, kappa, N, P
                )
                
                mask_jax, _, _, _, satisfy_jax = _geometry_jax._kept_on_valid(
                    jnp.array(fprs), jnp.array(tprs), jnp.array(thresholds), 
                    alpha, kappa, N, P
                )
                
                mask_jax = np.array(mask_jax)
                satisfy_jax = bool(satisfy_jax)
                
                self.assertTrue(np.array_equal(mask_np, mask_jax),
                               f"Masks differ for kappa={kappa}")
                self.assertEqual(satisfy_np, satisfy_jax,
                                f"Satisfy differs for kappa={kappa}")


if __name__ == '__main__':
    unittest.main()
