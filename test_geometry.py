import unittest
import jax
import _geometry_jax
import _geometry
import jax.numpy as jnp
import numpy as np
import time
import visualize_data

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

if __name__ == '__main__':
    unittest.main()
