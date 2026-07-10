import unittest
import jax
import _geometry_jax
import _geometry
import jax.numpy as jnp
import numpy as np

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
        jax_set = { (round(float(p[0]), 4), round(float(p[1]), 4)) for p in jax_np }

        # Convert expected output using the exact same logic
        expected_np = np.array(clipped_polygon)
        expected_set = { (round(float(p[0]), 4), round(float(p[1]), 4)) for p in expected_np }

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


if __name__ == '__main__':
    unittest.main()
