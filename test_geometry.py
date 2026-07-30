import unittest
import jax
import _geometry_jax
import _geometry
import jax.numpy as jnp
import numpy as np
import time
import grad
import train_and_voros
import metrics_jax

class TestGeometry(unittest.TestCase):
    def test_area_triangle(self):
        # Test case 1: Simple triangle
        pts = np.array([[0, 0], [1, 0], [0, 1]])
        # pts = [(0,0), (1,0), (0,1)]
        expected_area = 0.5
        test_area = _geometry_jax.area(pts)
        # print(f"Computed area: {test_area}, Expected area: {expected_area}")
        self.assertAlmostEqual(test_area, expected_area)
        self.assertAlmostEqual(test_area, _geometry.area(pts))

    def test_area_square(self):
        # Test case 2: Square
        pts = jnp.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        expected_area = 1.0
        test_area = _geometry_jax.area(pts)
        # print(f"Computed area: {test_area}, Expected area: {expected_area}")
        self.assertAlmostEqual(test_area, expected_area)
        self.assertAlmostEqual(test_area, _geometry.area(pts))

    def test_area_quad(self):
        # Test case 3: Irregular quadrilateral
        pts = jnp.array([[0, 0], [2, 0], [2, 1], [1, 2]])
        expected_area = 2.5
        test_area = _geometry_jax.area(pts)
        # print(f"Computed area: {test_area}, Expected area: {expected_area}")
        self.assertAlmostEqual(test_area, expected_area)
        self.assertAlmostEqual(test_area, _geometry.area(pts))
    
    def test_area_pentagon(self):
        # Test case 4: Pentagon
        pts = jnp.array([[0, 0], [2, 0], [3, 1], [1, 3], [0, 1]])
        expected_area = 5.5
        test_area = _geometry_jax.area(pts)
        # print(f"Computed area: {test_area}, Expected area: {expected_area}")
        self.assertAlmostEqual(test_area, expected_area)
        self.assertAlmostEqual(test_area, _geometry.area(pts))

    def test_area_single(self):
        # Test case 5: Single point (area should be 0)
        pts = jnp.array([[0, 0]])
        expected_area = 0.0
        test_area = _geometry_jax.area(pts)
        # print(f"Computed area: {test_area}, Expected area: {expected_area}")
        self.assertAlmostEqual(test_area, expected_area)
        self.assertAlmostEqual(test_area, _geometry.area(pts))

    def test_area_line(self):
        # Test case 6: Line (area should be 0)
        pts = jnp.array([[0, 0], [1, 1]])
        expected_area = 0.0
        test_area = _geometry_jax.area(pts)
        # print(f"Computed area: {test_area}, Expected area: {expected_area}")
        self.assertAlmostEqual(test_area, expected_area)
        self.assertAlmostEqual(test_area, _geometry.area(pts))
    
    def test_area_empty(self):
        # Test case 7: Empty array (area should be 0)
        pts = jnp.array([])
        expected_area = 0.0
        test_area = _geometry_jax.area(pts)
        # print(f"Computed area: {test_area}, Expected area: {expected_area}")
        self.assertAlmostEqual(test_area, expected_area)
        self.assertAlmostEqual(test_area, _geometry.area(pts))

    def clipping_equality(self, test_clipped_polygon, clipped_polygon):
        # Convert JAX output using Python's native round() to strip 32-bit artifacts
        jax_np = np.array(test_clipped_polygon)
        jax_set = {
            (round(float(p[0]), 4), round(float(p[1]), 4)) 
            for p in jax_np if not np.any(np.isnan(p))
        }

        expected_np = np.array(clipped_polygon)
        expected_set = {
            (round(float(p[0]), 4), round(float(p[1]), 4)) 
            for p in expected_np if not np.any(np.isnan(p))
        }

        # This will now match exactly and pass!
        self.assertEqual(jax_set, expected_set)

    def test_clipping_triangle(self):
        # Test case 8: Clipping a triangle with a line
        triangle = jnp.array([[0, 0], [2, 0], [1, 2]])
        # line: y = x-1 
        # x - y <= 1
        # a = 1, b = -1, c = 1
        test_clipped_polygon = _geometry_jax._clip_polygon_with_halfplane(triangle, 1, -1, 1)
        clipped_polygon = _geometry._clip_polygon_with_halfplane(triangle, 1, -1, 1)
        # print(f"Computed clipped polygon: {test_clipped_polygon}, Expected clipped polygon: {clipped_polygon}")
        self.clipping_equality(test_clipped_polygon, clipped_polygon)

    def test_clipping_square(self):
        # Test case 8: Clipping a square with a line
        square = jnp.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        # line: y = x-1 
        # x - y <= 1
        # a = 1, b = -1, c = 1
        test_clipped_polygon = _geometry_jax._clip_polygon_with_halfplane(square, -1, 1, 0)
        clipped_polygon = _geometry._clip_polygon_with_halfplane(square, -1, 1, 0)
        # print(f"Computed clipped polygon: {test_clipped_polygon}, Expected clipped polygon: {clipped_polygon}")
        self.clipping_equality(test_clipped_polygon, clipped_polygon)

    def test_clipping_square_outside(self):
        # Test case 8: Clipping a square with a line
        square = jnp.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        # line: y = x-1 
        # x - y <= 1
        # a = 1, b = -1, c = 1
        test_clipped_polygon = _geometry_jax._clip_polygon_with_halfplane(square, -1, 1, 1)
        clipped_polygon = _geometry._clip_polygon_with_halfplane(square, -1, 1, 1)
        # print(f"Computed clipped polygon: {test_clipped_polygon}, Expected clipped polygon: {clipped_polygon}")
        # self.assertAlmostEqual(test_clipped_polygon, clipped_polygon)
        self.clipping_equality(test_clipped_polygon, clipped_polygon)

    def test_clipping_triangle_triangle(self):
        # Test case 8: Clipping a triangle with a line
        triangle = jnp.array([[0, 0], [2, 0], [2, 1]])
        # line: y = x-1 
        # x - y <= 1
        # a = 1, b = -1, c = 1
        test_clipped_polygon = _geometry_jax._clip_polygon_with_halfplane(triangle, -1, 1, -1)
        clipped_polygon = _geometry._clip_polygon_with_halfplane(triangle, -1, 1, -1)
        # print(f"Computed clipped polygon: {test_clipped_polygon}, Expected clipped polygon: {clipped_polygon}")
        self.clipping_equality(test_clipped_polygon, clipped_polygon)

    def test_clipping_pentagon_triangle(self):
        # Test case 8: Clipping a pentagon into a triangle
        pentagon = jnp.array([[0, 0], [2,0], [2,1], [1,2], [0,1]])
        # line: y = x-1 
        # x - y <= 1
        # a = 1, b = -1, c = 1
        test_clipped_polygon = _geometry_jax._clip_polygon_with_halfplane(pentagon, -1, 1, -1)
        clipped_polygon = _geometry._clip_polygon_with_halfplane(pentagon, -1, 1, -1)

        self.clipping_equality(test_clipped_polygon, clipped_polygon)
    
    def test_clipping_pentagon_pentagon(self):
        # Test case 8: Clipping a pentagon into a triangle
        pentagon = jnp.array([[0, 0], [2,0], [2,1], [1,2], [0,1]])
        # line: y = x-1 
        # x - y <= 1
        # a = 1, b = -1, c = 1
        test_clipped_polygon = _geometry_jax._clip_polygon_with_halfplane(pentagon, 1, -1, 1)
        clipped_polygon = _geometry._clip_polygon_with_halfplane(pentagon, 1, -1, 1)
        # print(f"pentagonComputed clipped polygon: {test_clipped_polygon}, Expected clipped polygon: {clipped_polygon}")
        self.clipping_equality(test_clipped_polygon, clipped_polygon)

    def test_intersect_halfplanes(self):
        # Test case 9: Intersecting halfplanes to form a square
        halfplanes = jnp.array([[1, 0, 1], [-1, 0, 0], [0, 1, 1], [0, -1, 0]])
        test_polygon = _geometry_jax._intersect_halfplanes(halfplanes)
        polygon = _geometry._intersect_halfplanes(halfplanes)
        # print(f"Computed intersected polygon: {test_polygon}, Expected intersected polygon: {polygon}")
        self.clipping_equality(test_polygon, polygon)

    def test_intersect_halfplanes_triangle(self):
        # Test case 10: Intersecting halfplanes to form a triangle
        halfplanes = jnp.array([[1, 0, 1], [-1, 0, 0], [0, 1, 1]])
        test_polygon = _geometry_jax._intersect_halfplanes(halfplanes)
        polygon = _geometry._intersect_halfplanes(halfplanes)
        # print(f"Computed intersected polygon: {test_polygon}, Expected intersected polygon: {polygon}")
        self.clipping_equality(test_polygon, polygon)

    def test_intersect_halfplanes_pentagon(self):
        # Test case 11: Intersecting halfplanes to form a pentagon
        halfplanes = jnp.array([[1, 0, 1], [-1, 0, 0], [0, 1, 1], [0, -1, 0], [1, -1, 0]])
        test_polygon = _geometry_jax._intersect_halfplanes(halfplanes)
        polygon = _geometry._intersect_halfplanes(halfplanes)
        # print(f"Computed intersected polygon: {test_polygon}, Expected intersected polygon: {polygon}")
        self.clipping_equality(test_polygon, polygon)

    def test_intersect_halfplanes_empty(self):##
        # Test case 12: Intersecting halfplanes that do not intersect (empty polygon)
        halfplanes = jnp.array([[1, 0, 1], [-1, 0, -2], [0, 1, 1], [0, -1, -2]])
        test_polygon = _geometry_jax._intersect_halfplanes(halfplanes)
        polygon = _geometry._intersect_halfplanes(halfplanes)
        # print(f"Computed intersected polygon: {test_polygon}, Expected intersected polygon: {polygon}")
        self.clipping_equality(test_polygon, polygon)

    def test_feasible_polygon_precision(self):
        # Test case 13: Feasible polygon with given P, N, alpha, kappa
        P = 4
        N = 6
        alpha = 0.5
        kappa = 20
        test_polygon = _geometry_jax.compute_total_region_polygon(P, N, alpha, kappa)
        polygon = _geometry.compute_total_region_polygon(P, N, alpha, kappa)
        # print(f"Computed feasible polygon: {test_polygon}, Expected feasible polygon: {polygon}")
        self.clipping_equality(test_polygon, polygon)

    def test_feasible_polygon_capacity(self):
        # Test case 14: Feasible polygon, only capacity line
        P = 4
        N = 6
        alpha = 0.0001
        kappa = 5
        test_polygon = _geometry_jax.compute_total_region_polygon(P, N, alpha, kappa)
        polygon = _geometry.compute_total_region_polygon(P, N, alpha, kappa)
        # print(f"Computed feasible polygon: {test_polygon}, Expected feasible polygon: {polygon}")
        self.clipping_equality(test_polygon, polygon)

    def test_feasible_polygon(self):
        # Test case 14: Feasible polygon, only capacity line
        P = 4
        N = 6
        alpha = 0.5
        kappa = 5
        test_polygon = _geometry_jax.compute_total_region_polygon(P, N, alpha, kappa)
        polygon = _geometry.compute_total_region_polygon(P, N, alpha, kappa)
        # print(f"Computed feasible polygon: {test_polygon}, Expected feasible polygon: {polygon}")
        self.clipping_equality(test_polygon, polygon)

    def test_feasible_precision(self):
        # Test case 15:
        P = 4
        N = 6
        alpha = 0.5
        kappa = 20
        test_poly_area, test_polygon = _geometry_jax.total_region_area(P, N, alpha, kappa)
        poly_area, polygon = _geometry.total_region_area(P, N, alpha, kappa)
        self.clipping_equality(test_polygon, polygon)
        self.assertAlmostEqual(test_poly_area, poly_area)
    
    def test_feasible_area_capacity(self):
        # Test case 15:
        P = 4
        N = 6
        alpha = 0.1
        kappa = 5
        test_poly_area, test_polygon = _geometry_jax.total_region_area(P, N, alpha, kappa)
        poly_area, polygon = _geometry.total_region_area(P, N, alpha, kappa)
        self.clipping_equality(test_polygon, polygon)
        self.assertAlmostEqual(test_poly_area, poly_area)
    
    def test_feasible_area(self):
        # Test case 15:
        P = 4
        N = 6
        alpha = 0.5
        kappa = 5
        test_poly_area, test_polygon = _geometry_jax.total_region_area(P, N, alpha, kappa)
        poly_area, polygon = _geometry.total_region_area(P, N, alpha, kappa)
        self.clipping_equality(test_polygon, polygon)
        self.assertAlmostEqual(test_poly_area, poly_area)

    def test_feasible_area2(self):
        # Test case 15:
        P = 10
        N = 100
        alpha = 0.15
        kappa = 20
        test_poly_area, test_polygon = _geometry_jax.total_region_area(P, N, alpha, kappa)
        poly_area, polygon = _geometry.total_region_area(P, N, alpha, kappa)
        self.clipping_equality(test_polygon, polygon)
        self.assertAlmostEqual(test_poly_area, poly_area)

    def test_iso(self):
        h = 0.5
        k = 0.5
        t = 1/8

        a_test, b_test, c_test = _geometry_jax._iso_performance_line(h, k, t)
        a, b, c = _geometry._iso_performance_line(h, k, t)

        self.assertAlmostEqual(a_test, a)
        self.assertAlmostEqual(b_test, b)
        self.assertAlmostEqual(c_test, c)

    def test_iso_2(self):
        h = 0.5
        k = 0.5
        t = 0

        a_test, b_test, c_test = _geometry_jax._iso_performance_line(h, k, t)
        # a, b, c = _geometry._iso_performance_line(h, k, t)

        self.assertAlmostEqual(a_test, 0)
        self.assertAlmostEqual(b_test, -1)
        self.assertAlmostEqual(c_test, -k)
    
    def test_iso_3(self):
        h = 0.5
        k = 0.5
        t = 1

        a_test, b_test, c_test = _geometry_jax._iso_performance_line(h, k, t)
        a, b, c = _geometry._iso_performance_line(h, k, t)

        self.assertAlmostEqual(a_test, a)
        self.assertAlmostEqual(b_test, b)
        self.assertAlmostEqual(c_test, c)

    def test_reduced_outside(self):
        h = 0.5
        k = 0.5
        kappa = 30
        alpha = 0.2
        P = 10
        N = 100
        fp_cost_ratio = 1/6

        test_value, test_total_poly_area = _geometry_jax.reduced_area(h, k, kappa, alpha, P, N, fp_cost_ratio, True, False, True)
        value, total_poly_area = _geometry.reduced_area(h, k, kappa, alpha, P, N, fp_cost_ratio, True, False, True)

        self.assertAlmostEqual(test_value, value)
        self.assertAlmostEqual(test_total_poly_area, total_poly_area)
        print(f"reduced area: {test_value}, total area:{test_total_poly_area}")
        # self.assertEqual(test_details, details)
    
    def test_reduced(self):
        h = 0.1
        k = 0.5
        kappa = 30
        alpha = 0.2
        P = 10
        N = 100
        fp_cost_ratio = 1/6

        test_value, test_total_poly_area = _geometry_jax.reduced_area(h, k, kappa, alpha, P, N, fp_cost_ratio, True, False, True)
        value, total_poly_area = _geometry.reduced_area(h, k, kappa, alpha, P, N, fp_cost_ratio, True, False, True)

        self.assertAlmostEqual(test_value, value)
        self.assertAlmostEqual(test_total_poly_area, total_poly_area)
        print(f"reduced area: {test_value}, total area:{test_total_poly_area}")
        # self.assertEqual(test_details, details)

    def test_reduced_2(self):
        h = 0.1
        k = 0.5
        kappa = 30
        alpha = 0.2
        P = 10
        N = 100
        fp_cost_ratio = 1/8

        test_value, test_total_poly_area = _geometry_jax.reduced_area(h, k, kappa, alpha, P, N, fp_cost_ratio, True, False, True)
        value, total_poly_area = _geometry.reduced_area(h, k, kappa, alpha, P, N, fp_cost_ratio, True, False, True)
        print(f"reduced area: {test_value}, total area:{test_total_poly_area}")
        self.assertAlmostEqual(test_value, value)
        self.assertAlmostEqual(test_total_poly_area, total_poly_area)
        # self.assertEqual(test_details, details)

    def test_reduced_3(self):
        h = 0.1
        k = 0.8
        kappa = 30
        alpha = 0.2
        P = 10
        N = 100
        fp_cost_ratio = 1/60

        test_value, test_total_poly_area = _geometry_jax.reduced_area(h, k, kappa, alpha, P, N, fp_cost_ratio, True, False, True)
        value, total_poly_area = _geometry.reduced_area(h, k, kappa, alpha, P, N, fp_cost_ratio, True, False, True)
        print(f"reduced area: {test_value}, total area:{test_total_poly_area}")
        self.assertAlmostEqual(test_value, value)
        self.assertAlmostEqual(test_total_poly_area, total_poly_area)
    
    def test_keep_model(self):
        h = 0.1
        k = 0.5
        kappa = 30
        alpha = 0.2
        P = 10
        N = 100

        self.assertTrue(_geometry.keep_model(h, k, alpha, kappa, N, P))
        self.assertTrue(_geometry_jax.keep_model(h, k, alpha, kappa, N, P))

    def test_keep_model_false(self):
        h = 0.7
        k = 0.7
        kappa = 30
        alpha = 0.2
        P = 10
        N = 100

        self.assertFalse(_geometry.keep_model(h, k, alpha, kappa, N, P))
        self.assertFalse(_geometry_jax.keep_model(h, k, alpha, kappa, N, P))

    def test_ratio(self):
        r = 1/6
        P = 10
        N = 100

        r_to_t = _geometry.ratio_to_t(r, P, N)
        self.assertAlmostEqual(r, _geometry.t_to_ratio(r_to_t, P, N), places=12)

        test_r_to_t = _geometry_jax.ratio_to_t(r, P, N)
        self.assertAlmostEqual(r, _geometry_jax.t_to_ratio(test_r_to_t, P, N), places=12)

    def test_t(self):
        t = 0
        P = 10
        N = 100

        t_to_r = _geometry.t_to_ratio(t, P, N)
        self.assertAlmostEqual(t, _geometry.ratio_to_t(t, P, N), places=12)

        test_t_to_r = _geometry_jax.t_to_ratio(t, P, N)
        self.assertAlmostEqual(t, _geometry_jax.ratio_to_t(test_t_to_r, P, N), places=12)

    def test_t_2(self):
        t = 0.8
        P = 10
        N = 100

        t_to_r = _geometry.t_to_ratio(t, P, N)
        self.assertAlmostEqual(t, _geometry.ratio_to_t(t_to_r, P, N), places=12)

        test_t_to_r = _geometry_jax.t_to_ratio(t, P, N)
        self.assertAlmostEqual(t, _geometry_jax.ratio_to_t(test_t_to_r, P, N), places=12)

    def test_max_area(self):
        fprs = jnp.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        tprs = jnp.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        kappa = 30
        alpha = 0.2
        P = 10
        N = 100

        min_r = 1/9
        max_r = 1/6

        test_max_points, test_ts = _geometry_jax.max_area_per_t(fprs, tprs, kappa, alpha, P, N, min_r, max_r)
        max_points, ts = _geometry.max_area_per_t(fprs, tprs, kappa, alpha, P, N, min_r, max_r)

        self.assertEqual(len(test_max_points), len(max_points))
        self.assertEqual(len(test_ts), len(ts))

        for i in range(len(max_points)):
            self.assertAlmostEqual(float(test_max_points[i]), float(max_points[i]), places=6)
            self.assertAlmostEqual(float(test_ts[i]), float(ts[i]), places=6)

    def test_max_area_2(self):
        fprs = self.fprs
        tprs = self.tprs
        kappa = self.kappa
        alpha = self.alpha
        P = self.P
        N = self.N
        min_r = self.min_r
        max_r = self.max_r

        test_max_points, test_ts = _geometry_jax.max_area_per_t(fprs, tprs, kappa, alpha, P, N, min_r, max_r)
        max_points, ts = _geometry.max_area_per_t(fprs, tprs, kappa, alpha, P, N, min_r, max_r)

        self.assertEqual(len(test_max_points), len(max_points))
        self.assertEqual(len(test_ts), len(ts))

        for i in range(len(max_points)):
            self.assertAlmostEqual(float(test_max_points[i]), float(max_points[i]), places=6)
            self.assertAlmostEqual(float(test_ts[i]), float(ts[i]), places=6)

    def test_infeasible_area(self):
        fprs = jnp.array([0.9, 0.95, 1.0])
        tprs = jnp.array([0.0, 0.0, 0.0])
        
        # NumPy requires standard arrays
        fprs_np = np.array(fprs)
        tprs_np = np.array(tprs)

        # Strict constraints to force all points out of bounds
        alpha = 0.6
        kappa = 0.5
        P = 10
        N = 100

        min_r = 1/9
        max_r = 1/6

        # Execute JAX version
        test_max_points, test_ts = _geometry_jax.max_area_per_t(
            fprs, tprs, kappa, alpha, P, N, min_r, max_r
        )
        
        # Execute NumPy version
        max_points, ts = _geometry.max_area_per_t(
            fprs_np, tprs_np, kappa, alpha, P, N, min_r, max_r
        )

        # 1. Structural Validation
        self.assertEqual(len(test_max_points), len(max_points))
        self.assertEqual(len(test_ts), len(ts))

        # 2. Value Validation (Every single maximum reduced area must be exactly 0.0)
        for i in range(len(max_points)):
            # JAX version must evaluate to exactly 0.0 (no false -1.0 area leaks allowed)
            self.assertAlmostEqual(float(test_max_points[i]), 0.0, places=6,
                                   msg=f"JAX leaked area at cost ratio slice index {i}")
            
            # NumPy baseline check
            self.assertAlmostEqual(float(max_points[i]), 0.0, places=6)
            
            # Cost transformations must still match perfectly
            self.assertAlmostEqual(float(test_ts[i]), float(ts[i]), places=6)

    def setUp(self):
        self.fprs = jnp.array([0.0, 0.2, 0.3, 0.4, 0.8, 1.0])
        self.tprs = jnp.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        self.kappa = 20
        self.alpha = 0.3
        self.P = 10
        self.N = 100
        self.min_r = 1/9
        self.max_r = 1/6

    def test_voros(self):
        fprs = self.fprs
        tprs = self.tprs
        kappa = self.kappa
        alpha = self.alpha
        P = self.P
        N = self.N

        min_r = self.min_r
        max_r = self.max_r
        start = time.perf_counter()
        test_vor = _geometry_jax.voros_jax(fprs, tprs, kappa, alpha, P, N, min_r, max_r)
        end = time.perf_counter()
        print(f"jax computation time {end-start}")
        
        start = time.perf_counter()
        vor = _geometry.voros(fprs, tprs, kappa, alpha, P, N, min_r, max_r)
        end = time.perf_counter()
        print(f"ref computation time {end-start}")
        self.assertAlmostEqual(float(test_vor), float(vor), places=6)
        print(f"voros: {test_vor}")

    def test_voros_1(self):
        fprs = self.fprs
        tprs = self.tprs
        kappa = self.kappa
        alpha = self.alpha
        P = self.P
        N = self.N

        min_r = self.min_r
        max_r = self.max_r
        
        start_t = time.perf_counter()
        test_vor = _geometry_jax.voros_jax(fprs, tprs, kappa, alpha, P, N, min_r, max_r)
        end_t = time.perf_counter()
        print(f"jax computation time {end_t-start_t}")
        
        start = time.perf_counter()
        vor = _geometry.voros(fprs, tprs, kappa, alpha, P, N, min_r, max_r)
        end = time.perf_counter()
        print(f"ref computation time {end-start}")
        self.assertAlmostEqual(float(test_vor), float(vor), places=6)

        print(f"voros: {test_vor}")

    def test_grad_without_error(self):
        # just confirms no exceptions, no NaNs sneaking through the trace
        grad_fn = jax.grad(_geometry_jax.voros_jax, argnums=(0, 1))
        try:
            fprs = self.fprs
            tprs = self.tprs
            kappa = self.kappa
            alpha = self.alpha
            P = self.P
            N = self.N

            min_r = self.min_r
            max_r = self.max_r

            grad = grad_fn(
                fprs, tprs, kappa, alpha, P, N, min_r, max_r
            )
        except Exception as e:
            self.fail(f"jax.grad raised an exception: {e}")

    def test_grad_shape_and_dtype(self):
        grad_fn = jax.grad(_geometry_jax.voros_jax, argnums=(0, 1))
        fprs = self.fprs
        tprs = self.tprs

        kappa = self.kappa
        alpha = self.alpha
        P = self.P
        N = self.N

        min_r = self.min_r
        max_r = self.max_r
        g_fpr, g_tpr = grad_fn(
            fprs, tprs, kappa, alpha, P, N, min_r, max_r
        )
        self.assertEqual(g_fpr.shape, fprs.shape)
        self.assertEqual(g_tpr.shape, tprs.shape)
        self.assertEqual(g_fpr.dtype, fprs.dtype)

    def test_grad_no_nan_or_inf(self):
        grad_fn = jax.grad(_geometry_jax.voros_jax, argnums=(0, 1))
        fprs = self.fprs
        tprs = self.tprs

        kappa = self.kappa
        alpha = self.alpha
        P = self.P
        N = self.N

        min_r = self.min_r
        max_r = self.max_r
        g_fpr, g_tpr = grad_fn(
            fprs, tprs, kappa, alpha, P, N, min_r, max_r
        )
        for name, g in [("fpr", g_fpr), ("tpr", g_tpr)]:
            self.assertFalse(jnp.any(jnp.isnan(g)), f"NaN in grad w.r.t. {name}")
            self.assertFalse(jnp.any(jnp.isinf(g)), f"Inf in grad w.r.t. {name}")

    def compute_reference_pvoros_kept_on_valid(self, x_train, y_true, theta, c, M):
        """Computes reference pVOROS score directly using _kept_on_valid and voros_jax."""
        P = int(np.sum(y_true == 1))
        N = int(np.sum(y_true == 0))
        kappa = 0.5 * float(len(y_true))

        # Convert theta, c, M back to linear decision boundary weights
        w1 = M * np.sin(theta)
        w2 = -M * np.cos(theta)
        intercept = M * c * np.cos(theta)

        logits = x_train[:, 0] * w1 + x_train[:, 1] * w2 + intercept
        y_pred = 1.0 / (1.0 + np.exp(-logits))

        eps = 1e-5
        thresholds = np.linspace(eps, 1.0 - eps, 100)

        # Compute smoothed FPR and TPR arrays using grad helper
        fprs_smooth, tprs_smooth = grad.compute_smoothed_fprs_tprs_jax(y_true, y_pred, thresholds)

        # Reference filtering step via _kept_on_valid
        _, acc_fprs, acc_tprs, _, satisfy = _geometry_jax._kept_on_valid(
            fprs_smooth, tprs_smooth, thresholds, 0.6, kappa, N, P
        )

        if satisfy:
            voros_val = float(
                _geometry_jax.voros_jax(
                    acc_fprs,
                    acc_tprs,
                    kappa,
                    0.6,
                    P,
                    N,
                    1/9,
                    1/6,
                    n_points=1000,  # <-- MATCHES N_POINTS=50 in grad.py
                )
            )
            print("ref voros: ", voros_val)
            total_envelope_area, _ = _geometry_jax.total_region_area(P, N, 0.6, kappa)
            env_area_scalar = float(np.asarray(total_envelope_area).item())
            return min(voros_val, env_area_scalar)

        return 0.0

    def test_1_new_loss(self):
        """Verifies that grad.jax_voros_loss matches reference _kept_on_valid + voros_jax."""
        data_path = "heatmaps/sweep_meta_data.npy"

        data = np.load(data_path, allow_pickle=True).item()
        train_test = data["train_test"]

        # Deterministic seed for reproducible random parameter evaluations
        np.random.seed(42)

        print("\n" + "=" * 85)
        print("TESTING GRAD.JAX_VOROS_LOSS EQUIVALENCE AGAINST KEPT_ON_VALID REFERENCE")
        print("=" * 85)

        for seed_name, seed_data in train_test.items():
            X = np.asarray(seed_data[0])
            Y = np.asarray(seed_data[2])

            P = int(np.sum(Y == 1))
            N = int(np.sum(Y == 0))

            kappa = 0.5*(P+N)

            # Sample random evaluation parameters (theta, c, M)
            theta_val = np.random.uniform(0, np.pi/4)
            c_val = np.random.uniform(-3.0, 3.0)
            M_val = np.random.uniform(0.5, 3.0)
            thresholds = jnp.linspace(1e-5, 1.0-1e-5, 100)

            params = {
                "theta": jnp.array(theta_val, dtype=jnp.float64),
                "c": jnp.array(c_val, dtype=jnp.float64),
            }

            # 1. Compute reference score using C++/NumPy filtering logic (_kept_on_valid)
            ref_score = self.compute_reference_pvoros_kept_on_valid(X, Y, theta_val, c_val, M_val)

            # 2. Compute score using the JAX gradient loss function (-1 * loss)
            loss_val = float(metrics_jax.pv_loss_theta_c(params, X, Y, P, N, kappa, 0.6, thresholds, 1/9, 1/6, n_points=1000, temp = 0.02, M = M_val))
            jax_score = -loss_val

            print(
                f"Seed: {seed_name:<18} | θ: {np.degrees(theta_val):+6.1f}° | c: {c_val:+5.2f} | M: {M_val:.2f} "
                f"| Ref: {ref_score:.6f} | JAX: {jax_score:.6f}"
            )

            # Assert numerical equivalence up to 5 decimal places
            diff = abs(ref_score - jax_score)

            self.assertAlmostEqual(
                ref_score,
                jax_score,
                places=5,
                msg=(
                    f"\n[MISMATCH DETECTED on {seed_name}]"
                    f"\n  Reference Score (_kept_on_valid): {ref_score:.12f}"
                    f"\n  JAX Score (grad.jax_voros_loss): {jax_score:.12f}"
                    f"\n  Absolute Difference:            {diff:.12f}"
                    f"\n  Parameters: theta={np.degrees(theta_val):.2f}°, c={c_val:.4f}, M={M_val:.4f}"
                )
            )

        print("=" * 85)
        print("SUCCESS: All seeds matched the reference _kept_on_valid evaluation!")
        print("=" * 85)

    # def test_double_grad_without_error(self):
    #     # just confirms no exceptions, no NaNs sneaking through the trace
    #     double_grad_fn = jax.grad(jax.grad(_geometry_jax.voros_jax, argnums=(0, 1)), argnums=(0, 1))
    #     try:
    #         fprs = self.fprs
    #         tprs = self.tprs
    #         kappa = self.kappa
    #         alpha = self.alpha
    #         P = self.P
    #         N = self.N

    #         min_r = self.min_r
    #         max_r = self.max_r

    #         double_grad = double_grad_fn(
    #             fprs, tprs, kappa, alpha, P, N, min_r, max_r
    #         )
    #     except Exception as e:
    #         self.fail(f"jax.grad raised an exception: {e}")

    # def test_double_grad_shape_and_dtype(self):
    #     double_grad_fn = jax.grad(jax.grad(_geometry_jax.voros_jax, argnums=(0, 1)), argnums=(0, 1))
    #     fprs = self.fprs
    #     tprs = self.tprs

    #     kappa = self.kappa
    #     alpha = self.alpha
    #     P = self.P
    #     N = self.N

    #     min_r = self.min_r
    #     max_r = self.max_r
    #     g_fpr_fpr, g_fpr_tpr, g_tpr_fpr, g_tpr_tpr = double_grad_fn(
    #         fprs, tprs, kappa, alpha, P, N, min_r, max_r
    #     )
    #     self.assertEqual(g_fpr_fpr.shape, fprs.shape)
    #     self.assertEqual(g_fpr_tpr.shape, tprs.shape)
    #     self.assertEqual(g_tpr_fpr.shape, fprs.shape)
    #     self.assertEqual(g_tpr_tpr.shape, tprs.shape)
    #     self.assertEqual(g_fpr_fpr.dtype, fprs.dtype)
    #     self.assertEqual(g_fpr_tpr.dtype, tprs.dtype)
    #     self.assertEqual(g_tpr_fpr.dtype, fprs.dtype)
    #     self.assertEqual(g_tpr_tpr.dtype, tprs.dtype)
    
    # def test_double_grad_no_nan_or_inf(self):
    #     double_grad_fn = jax.grad(jax.grad(_geometry_jax.voros_jax, argnums=(0, 1)), argnums=(0, 1))
    #     fprs = self.fprs
    #     tprs = self.tprs

    #     kappa = self.kappa
    #     alpha = self.alpha
    #     P = self.P
    #     N = self.N

    #     min_r = self.min_r
    #     max_r = self.max_r
    #     g_fpr_fpr, g_fpr_tpr, g_tpr_fpr, g_tpr_tpr = double_grad_fn(
    #         fprs, tprs, kappa, alpha, P, N, min_r, max_r
    #     )
    #     for name, g in [("fpr_fpr", g_fpr_fpr), ("fpr_tpr", g_fpr_tpr), ("tpr_fpr", g_tpr_fpr), ("tpr_tpr", g_tpr_tpr)]:
    #         self.assertFalse(jnp.any(jnp.isnan(g)), f"NaN in double grad w.r.t. {name}")
    #         self.assertFalse(jnp.any(jnp.isinf(g)), f"Inf in double grad w.r.t. {name}")


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
            # NumPy returns filtered arrays (shape: n_valid)
            # JAX returns arrays with same shape as input, with repetition for non-matching
            # Extract the valid count from NumPy result
            n_valid = len(fprs_np)
            
            # Get the first n_valid elements from JAX output
            fprs_jax_valid = np.array(fprs_jax_out[:n_valid])
            tprs_jax_valid = np.array(tprs_jax_out[:n_valid])
            
            np.testing.assert_array_almost_equal(fprs_np, fprs_jax_valid,
                                                decimal=10)
            
            np.testing.assert_array_almost_equal(tprs_np, tprs_jax_valid,
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
        n_points = 1000
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
            # NumPy returns filtered arrays (shape: n_valid)
            # JAX returns arrays with same shape as input, with repetition for non-matching
            # Extract the valid count from NumPy result
            n_valid = len(fprs_np)
            
            # Get the first n_valid elements from JAX output
            fprs_valid_jax = np.array(fprs_jax_out[:n_valid])
            tprs_valid_jax = np.array(tprs_jax_out[:n_valid])
            
            np.testing.assert_array_almost_equal(fprs_np, fprs_valid_jax,
                                                decimal=10)
            np.testing.assert_array_almost_equal(tprs_np, tprs_valid_jax,
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
                
    def _reference_mask(self, fprs, tprs, alpha, kappa, N_v, P_v):
        """Ground truth: apply keep_model pointwise without vmap."""
        return jnp.array(
            [bool(_geometry_jax.keep_model(f, t, alpha, kappa, N_v, P_v)) for f, t in zip(fprs, tprs)]
        )
                
    def test_mask_matches_reference(self, alpha, kappa, N_v, P_v):
        fprs = jnp.linspace(0.0, 1.0, 11)
        tprs = jnp.linspace(0.0, 1.0, 11)
        thresholds = jnp.linspace(1.0, 0.0, 11)
    
        mask, _, _, _, satisfy = _geometry_jax._kept_on_valid(fprs, tprs, thresholds, alpha, kappa, N_v, P_v)
    
        expected_mask = self._reference_mask(fprs, tprs, alpha, kappa, N_v, P_v)
        expected_satisfy = jnp.any(expected_mask)
    
        assert jnp.array_equal(mask, expected_mask), (
            f"mask mismatch: got {mask}, expected {expected_mask}"
        )
        assert bool(satisfy) == bool(expected_satisfy), (
            f"satisfy mismatch: got {satisfy}, expected {expected_satisfy}"
        )
 
 
    def test_mask_matches_reference_nontrivial_roc_curve(self):
        # A more realistic, non-monotonic-looking ROC-ish curve
        fprs = jnp.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
        tprs = jnp.array([0.0, 0.3, 0.5, 0.6, 0.65, 0.8, 1.0])
        thresholds = jnp.linspace(1.0, 0.0, len(fprs))
        alpha, kappa, N_v, P_v = 0.4, 0.5, 100.0, 20.0
    
        mask, _, _, _, satisfy = _geometry_jax._kept_on_valid(fprs, tprs, thresholds, alpha, kappa, N_v, P_v)
    
        expected_mask = self._reference_mask(fprs, tprs, alpha, kappa, N_v, P_v)
        expected_satisfy = jnp.any(expected_mask)
    
        assert jnp.array_equal(mask, expected_mask)
        assert bool(satisfy) == bool(expected_satisfy)

if __name__ == '__main__':
    unittest.main()
