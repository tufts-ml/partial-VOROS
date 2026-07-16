"""Internal geometry primitives for pVOROS computation.

Direct port of src/utils_voros.py (minus plot_reduced_area and
reduced_area_untested). Math and signatures are identical to the source.
Also includes _kept_on_valid, a private ROC-filtering helper ported from
src/step7_lucky_number.py for use by metrics.py and cost.py.
"""

import numpy as np
import scipy.integrate
from typing import Optional


# ---- Polygon area ----

def area(polygon_points):
    """
    Calculate the area of a polygon given its vertices using the trapezoid formula.

    Parameters:
    polygon_points (list of tuples): List of (x, y) coordinates of the polygon vertices

    Returns:
    float: Area of the polygon
    """
    n = len(polygon_points)
    area = 0
    for i in range(n):
        x, y = polygon_points[i]
        x_next, y_next = polygon_points[(i + 1) % n]
        area += x * y_next - y * x_next
    return area / 2


# ---- Sutherland-Hodgman clipping ----

def _clip_polygon_with_halfplane(poly, a, b, c):
    """Sutherland-Hodgman style clipping of convex polygon with half-plane a*x + b*y <= c.
    poly: list of (x,y) vertices in order (convex assumed)
    Returns new list of vertices (may be empty).
    """
    if not poly:
        return []

    def inside(pt):
        """Is point inside half-plane?"""
        x, y = pt
        return a * x + b * y <= c + 1e-12  # tolerance

    def intersection(p1, p2):
        """Compute intersection of segment p1-p2 with boundary a*x + b*y = c
        assumes one point inside, one outside."""
        x1, y1 = p1
        x2, y2 = p2
        v1 = a * x1 + b * y1 - c
        v2 = a * x2 + b * y2 - c
        # line param t in [0,1] where segment crosses boundary a*x+b*y=c
        denom = v1 - v2
        if abs(denom) < 1e-15:
            return p2  # nearly parallel; pick second
        t = v1 / denom
        xi = x1 + (x2 - x1) * t
        yi = y1 + (y2 - y1) * t
        return (xi, yi)

    out = []
    prev = poly[-1]
    prev_inside = inside(prev)
    for curr in poly:
        curr_inside = inside(curr)
        if prev_inside and curr_inside:
            # We stayed inside, keep current
            out.append(curr)
        elif prev_inside and not curr_inside:
            # We exited plane, add intersection only
            out.append(intersection(prev, curr))
        elif (not prev_inside) and curr_inside:
            # We entered plane, add intersection and current
            out.append(intersection(prev, curr))
            out.append(curr)
        prev, prev_inside = curr, curr_inside
    return out


def _intersect_halfplanes(halfplanes, bbox=((0, 0), (1, 1))):
    """Intersect half-planes (a,b,c) representing a*x + b*y <= c within initial bbox square.
    bbox: ((x0,y0),(x1,y1)) axis-aligned rectangle providing initial polygon.
    Returns list of vertices of resulting convex polygon in order.
    """
    (x0, y0), (x1, y1) = bbox
    poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    for (a, b, c) in halfplanes:
        poly = _clip_polygon_with_halfplane(poly, a, b, c)
        if not poly:
            break
    # sort vertices counter-clockwise for consistency
    if len(poly) > 2:
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        poly.sort(key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
    return poly


# ---- Feasible region polygon ----

def compute_total_region_polygon(P, N, α, κ):
    """Compute polygon of feasible region defined by:
    precision constraint: TP/(TP+FP) >= α -> y >= (α*N/( (1-α)*P)) x (derived earlier)
    written as y >= m_p x, where m_p = α*N/((1-α)*P)
    capacity constraint: predicted positives <= κ -> P*y + N*x <= κ -> N*x + P*y <= κ
    bounding box: 0 <= x <= 1, 0 <= y <= 1 (ROC space)
    Returns list of vertices.
    Handles all geometric cases automatically via half-plane intersection.
    """
    if not (0 < α < 1):
        raise ValueError("α must be in (0,1)")
    if not (0 < κ):
        raise ValueError("κ must be > 0")
    m_p = (α * N) / ((1 - α) * P)
    # Half-planes a*x + b*y <= c
    halfplanes = []
    # y >= m_p x  -> -m_p x + y >= 0 -> m_p x - y <= 0
    halfplanes.append((m_p, -1.0, 0.0))
    # capacity N*x + P*y <= κ (only if kappa < P+N; else it's non-binding inside ROC square)
    if κ < (P + N):
        halfplanes.append((N, P, κ))
    # x >= 0 -> -x <= 0
    halfplanes.append((-1.0, 0.0, 0.0))
    # y <= 1
    halfplanes.append((0.0, 1.0, 1.0))
    # x <= 1
    halfplanes.append((1.0, 0.0, 1.0))
    # y >= 0 -> -y <= 0
    halfplanes.append((0.0, -1.0, 0.0))
    poly = _intersect_halfplanes(halfplanes)
    # Deduplicate near-identical consecutive or global duplicates
    dedup = []
    for pt in poly:
        if not any(abs(pt[0] - q[0]) < 1e-12 and abs(pt[1] - q[1]) < 1e-12 for q in dedup):
            dedup.append(pt)
    return dedup


def total_region_area(P, N, α, κ):
    poly = compute_total_region_polygon(P, N, α, κ)
    if not poly:
        return 0.0, poly
    return abs(area(poly)), poly


# ---- Isoperformance line ----

def _iso_performance_line(h, k, t):
    """Return coefficients (a,b,c) for half-plane a*x + b*y <= c representing the lower-cost side
    of the isoperformance line passing through (h,k).
    Cost line derived from total expected cost = const. For now we use linearization:
    k - y = ((k-1 + (1-k)/t) - (h - x)) * ( (k - 1) / (h - (h+k-1+(1-k)/t)) )
    Simpler: we know another point where iso hits y=1: x_iso1 = h+k-1+(1-k)/t.
    Treat line through (h,k) and (x_iso1,1).
    Returns line in normalized form a*x + b*y = c with orientation such that region 'below' (toward origin) is kept.
    """
    x2 = h + k - 1 + (1 - k) / t
    y2 = 1.0
    x1, y1 = h, k
    # line through (x1,y1) & (x2,y2)
    if abs(x2 - x1) < 1e-12:
        # vertical line x = x1; keep right side (higher FPR): x >= x1 -> -x <= -x1
        return (-1.0, 0.0, -x1)
    m = (y2 - y1) / (x2 - x1)
    b0 = y1 - m * x1
    if m >= 0:
        # keep higher FPR side: x >= (y - b0)/m -> -m x + y <= b0
        a, b, c = -m, 1.0, b0
    else:
        # m < 0: x >= (y - b0)/m -> m x - y <= -b0
        a, b, c = m, -1.0, -b0
    return (a, b, c)


# ---- Reduced area ----

def reduced_area(h, k, κ, α, P, N, fp_cost_ratio, return_percent=True,
                 return_details=False, return_total_area=False):
    """Compute reduced area (fraction) using geometric clipping.
    Steps:
      1. Build total feasible region polygon via compute_total_region_polygon.
      2. Build half-plane for isoperformance line through (h,k) given cost ratio r = c0/c1.
         Parameter t = r*N/(r*N + P).
      3. Intersect total region with half-plane 'below' iso line (worse or equal cost) to get polygon.
      4. Area ratio = area(intersection)/area(total_region) if return_percent else raw area.
    Returns scalar, optionally (scalar, dict) with detailed geometry. If return_total_area is True returns
    (value, total_area) or (value, total_area, details) when return_details also True.
    """
    r = fp_cost_ratio
    t = r * N / (r * N + P)
    total_poly_area, total_poly = total_region_area(P, N, α, κ)
    if total_poly_area == 0:
        if return_total_area and return_details:
            return 0.0, 0.0, {"total_polygon": total_poly, "iso_polygon": [], "iso_line": None, "t": None}
        if return_total_area:
            return 0.0, 0.0
        return 0.0 if not return_details else (0.0, {"total_polygon": total_poly, "iso_polygon": [], "iso_line": None, "t": None})
    a, b, c = _iso_performance_line(h, k, t)
    # Intersect total polygon with iso half-plane
    iso_poly = _clip_polygon_with_halfplane(total_poly, a, b, c)
    raw_area = abs(area(iso_poly)) if iso_poly else 0.0
    value = raw_area / total_poly_area if return_percent else raw_area
    if return_total_area and return_details:
        return value, total_poly_area, {"total_polygon": total_poly, "iso_polygon": iso_poly, "iso_line": (a, b, c), "t": t}
    if return_total_area:
        return value, total_poly_area
    if return_details:
        return value, {"total_polygon": total_poly, "iso_polygon": iso_poly, "iso_line": (a, b, c), "t": t}
    return value


# ---- Threshold feasibility filter ----

def keep_model(fpr, tpr, target_prec, target_cap, count_N, count_P):
    """Return True if (fpr, tpr) satisfies capacity and precision constraints.

    Constraints in ROC space (x=fpr, y=tpr):
        - Capacity:     N*x + P*y <= kappa  -> y <= (kappa - N*x)/P
        - Precision:    TP/(TP+FP) >= alpha -> y >= (alpha*N*x)/((1-alpha)*P)

    Notes:
        - For numerical stability and to match downstream expectations, we round
            tpr and both computed bounds to 6 decimals before comparison.
        - Assumes 0 < target_prec < 1 and count_P > 0.
    """
    # Compute upper (capacity) and lower (precision) bounds for TPR at the given FPR
    upper_bound = (target_cap - count_N * fpr) / count_P
    lower_bound = (target_prec * count_N * fpr) / ((1 - target_prec) * count_P)

    # Round for stable comparisons
    tpr_r = round(float(tpr), 6)
    upper_r = round(float(upper_bound), 6)
    lower_r = round(float(lower_bound), 6)

    return (tpr_r <= upper_r) and (tpr_r >= lower_r)


# ---- Cost ratio / t conversions ----

def ratio_to_t(r: float, P: int, N: int) -> float:
    """Convert fp_cost_ratio r to t = r*N/(r*N + P).

    >>> round(ratio_to_t(0.0, 100, 100), 6)
    0.0
    >>> t = ratio_to_t(1.0, 100, 100); 0 < t < 1
    True
    """
    if r < 0:
        raise ValueError("ratio r must be >= 0")
    return (r * float(N)) / (r * float(N) + float(P))


def t_to_ratio(t: float, P: int, N: int) -> float:
    """Convert t to fp_cost_ratio r given P,N.

    Assumes 0 <= t < 1.

    >>> round(t_to_ratio(0.0, 100, 100), 6)
    0.0
    >>> r = t_to_ratio(0.5, 100, 100); r > 0
    True
    """
    if t < 0 or t >= 1:
        raise ValueError("t must satisfy 0 <= t < 1")
    if t == 0:
        return 0.0
    return (float(P) / float(N)) * (t / (1.0 - t))


def calc_cost(t_G, fpr_G, tpr_G):
    cost_G = t_G * fpr_G + (1.0 - t_G) * (1.0 - tpr_G)
    assert cost_G.max() <= 1.0
    assert cost_G.min() >= 0.0
    return cost_G

# ---- Max reduced area per cost ratio ----

def max_area_per_t(
    fprs,
    tprs,
    κ,
    α,
    P,
    N,
    min_fp_cost_ratio,
    max_fp_cost_ratio,
    n_points: int = 1000,
    return_best_thresholds: bool = False,
    thresholds: Optional[np.ndarray] = None,
    do_fast_threshold_sel_via_cost=False,
):
    """Calculate the maximum reduced area across ROC points for each cost ratio in a range.

    If return_best_thresholds=True, also returns the threshold (from the provided
    'thresholds' array) that achieved the max at each cost ratio. In that case,
    'thresholds' must be provided and aligned with fprs/tprs.
    Returns (max_points, ts) or (max_points, ts, best_thresholds).
    """
    fp_cost_ratios = np.linspace(min_fp_cost_ratio, max_fp_cost_ratio, n_points)
    ts = [ratio_to_t(fp_ratio, P, N) for fp_ratio in fp_cost_ratios]

    # for each fp_cost_ratio, calculate reduced area for all fpr,tpr pairs
    max_points = []
    best_thresh = [] if return_best_thresholds else None
    for fp_ratio, t in zip(fp_cost_ratios, ts):
        if do_fast_threshold_sel_via_cost:
            costs = calc_cost(t, fprs, tprs)
            if len(costs) > 0:
                imax = int(np.argmin(costs))
                bestarea = reduced_area(fprs[imax], tprs[imax], κ, α, P, N, fp_ratio)
                max_points.append(bestarea)
            else:
                imax = -1
                max_points.append(0.0) 
        else:
            vals = [reduced_area(fpr, tpr, κ, α, P, N, fp_ratio) for fpr, tpr in zip(fprs, tprs)]
            # find argmax
            imax = int(np.argmax(vals)) if len(vals) else -1
            max_points.append(vals[imax] if imax >= 0 else 0.0)
        if return_best_thresholds:
            if thresholds is None:
                raise ValueError("thresholds must be provided when return_best_thresholds=True")
            best_thresh.append(float(thresholds[imax]))
    if return_best_thresholds:
        return max_points, ts, np.array(best_thresh, dtype=float)
    return max_points, ts


# ---- VOROS integrator ----

def voros(
    fprs,
    tprs,
    κ,
    α,
    P,
    N,
    min_fp_cost_ratio,
    max_fp_cost_ratio,
    n_points: int = 1000,
    return_best_thresholds: bool = False,
    thresholds: Optional[np.ndarray] = None,
    do_fast_threshold_sel_via_cost=False,
):
    """Compute partial VOROS (average of max reduced area across t in range).

    If return_best_thresholds=True, returns (voros_value, ts, best_thresholds_per_t),
    where best_thresholds_per_t aligns with ts and contains the threshold (from
    provided 'thresholds') achieving the max area at each t.
    """
    if return_best_thresholds:
        max_points, ts, best_thresholds = max_area_per_t(
            fprs, tprs, κ, α, P, N, min_fp_cost_ratio, max_fp_cost_ratio,
            n_points=n_points, return_best_thresholds=True, thresholds=thresholds,
            do_fast_threshold_sel_via_cost=do_fast_threshold_sel_via_cost,
        )
    else:
        max_points, ts = max_area_per_t(
            fprs, tprs, κ, α, P, N, min_fp_cost_ratio, max_fp_cost_ratio,
            n_points=n_points, do_fast_threshold_sel_via_cost=do_fast_threshold_sel_via_cost,
        )

    # Integrate in r-space (cost-ratio space) where we have uniform sampling.
    # The expectation is E_{r ~ Uniform}[f(r)] = (1/(r_max - r_min)) * integral f(r) dr.
    fp_cost_ratios = np.linspace(min_fp_cost_ratio, max_fp_cost_ratio, n_points)
    r_range = max_fp_cost_ratio - min_fp_cost_ratio
    if len(fp_cost_ratios) > 1 and r_range > 0:
        integral_val = scipy.integrate.trapezoid(max_points, x=fp_cost_ratios)
        vor = float(integral_val) / r_range
    else:
        vor = float(max_points[0]) if max_points else 0.0
    if return_best_thresholds:
        return vor, np.array(ts, dtype=float), np.array(best_thresholds, dtype=float)
    return vor


# ---- Private ROC helpers (ported from step7_lucky_number.py) ----

def _kept_on_valid(fprs_v, tprs_v, thresholds_v, alpha, kappa, N_v, P_v):
    """Filter ROC points to those satisfying precision+capacity constraints on validation.

    Returns (mask, acc_fprs, acc_tprs, acc_thresholds, satisfy).
    If no points satisfy, falls back to all points with satisfy=False.
    """
    mask = np.array(
        [keep_model(fpr, tpr, alpha, kappa, N_v, P_v) for fpr, tpr in zip(fprs_v, tprs_v)],
        dtype=bool,
    )
    if mask.any():
        acc_fprs_v = fprs_v[mask]
        acc_tprs_v = tprs_v[mask]
        acc_thresholds_v = thresholds_v[mask]
        satisfy = True
    else:
        # fallback: no feasible point found; use full arrays
        acc_fprs_v = fprs_v
        acc_tprs_v = tprs_v
        acc_thresholds_v = thresholds_v
        satisfy = False
    return mask, acc_fprs_v, acc_tprs_v, acc_thresholds_v, satisfy
