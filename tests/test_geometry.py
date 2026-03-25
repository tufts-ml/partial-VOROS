"""Tests for pvoros._geometry.

Covers the reduced_area return_details interface and constraint-checking helpers.
Note: tests that compared against the parent repo's src/utils_voros.py have been
removed since that file is not part of this repository.
"""

import pytest
import _geometry


def test_reduced_area_return_details():
    val, details = _geometry.reduced_area(0.1, 0.6, 400, 0.2, 100, 900, 1.0, return_details=True)
    assert 'total_polygon' in details
    assert 'iso_polygon' in details
    assert 'iso_line' in details
    assert 't' in details
    assert 0.0 <= val <= 1.0
    assert details['total_polygon']
    assert details['iso_polygon']
