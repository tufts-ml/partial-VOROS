import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
import scipy.integrate
from typing import Optional



def area(polygon_points):
    if polygon_points.shape[0] == 0:
        return 0.0
    
    # 1. Filter out empty polygon markers if the polygon completely disappeared
    is_nan = jnp.isnan(polygon_points[:, 0])
    clean_poly = jnp.where(is_nan[:, None], 0.0, polygon_points)
    
    # 2. Identify and keep only the truly UNIQUE consecutive vertices.
    # We compare each row with its preceding neighbor.
    diffs = jnp.abs(clean_poly - jnp.roll(clean_poly, 1, axis=0))
    is_unique_edge = jnp.any(diffs > 1e-5, axis=1)
    
    # 3. Pull out ONLY the valid unique coordinates
    # Non-unique duplicate rows collapse cleanly into (0.0, 0.0)
    valid_vertices = jnp.where(is_unique_edge[:, None], clean_poly, 0.0)
    
    # 4. Compute the geometric center (centroid) of the unique shape boundaries
    num_valid = jnp.sum(is_unique_edge)
    cx = jnp.sum(valid_vertices[:, 0]) / (num_valid + 1e-15)
    cy = jnp.sum(valid_vertices[:, 1]) / (num_valid + 1e-15)
    
    # 5. Sort indices counter-clockwise relative to the center.
    # This guarantees a perfect sequential perimeter loop for the Shoelace math.
    angles = jnp.arctan2(valid_vertices[:, 1] - cy, valid_vertices[:, 0] - cx)
    # Force unused padding points to a high dummy angle so they cluster at the end
    sorted_angles = jnp.where(is_unique_edge, angles, 5.0)
    
    sorted_idx = jnp.argsort(sorted_angles)
    ordered_poly = valid_vertices[sorted_idx]
    
    # 6. Apply Shoelace formula exclusively within the valid boundary segment.
    # Instead of rolling across the whole matrix (which includes the 0,0 padding tail),
    # we roll only across the active unique vertices.
    x = ordered_poly[:, 0]
    y = ordered_poly[:, 1]
    
    # Standard vector cross products
    term1 = x * jnp.roll(y, -1)
    term2 = y * jnp.roll(x, -1)
    
    # Zero-out terms associated with the padded indices at the end of the array
    # because they do not form structural parts of the polygon perimeter
    padded_mask = (jnp.arange(polygon_points.shape[0]) < num_valid)
    raw_area = 0.5 * jnp.abs(jnp.sum(jnp.where(padded_mask, term1 - term2, 0.0)))
    
    # Handle the boundary edge closure loop from the last unique vertex back to the first
    first_pt = ordered_poly[0]
    # Use dynamic gathering to extract the last unique vertex index safely
    last_pt = ordered_poly[jnp.clip(num_valid - 1, 0, polygon_points.shape[0] - 1)]
    closure_term = 0.5 * jnp.abs(last_pt[0] * first_pt[1] - last_pt[1] * first_pt[0])
    
    # If the polygon had less than 3 unique vertices, the real area is 0.0
    final_area = jnp.where(num_valid < 3, 0.0, raw_area + closure_term)
    
    return jnp.where(jnp.any(is_nan), 0.0, final_area)

# ---- Sutherland-Hodgman clipping ----

def _clip_polygon_with_halfplane(poly, a, b, c):
    """Sutherland-Hodgman style clipping of convex polygon with half-plane a*x + b*y <= c.
    poly: list of (x,y) vertices in order (convex assumed)
    Returns new list of vertices (may be empty).
    """

    N = poly.shape[0]
    
    curr_pts = poly
    next_pts = jnp.roll(poly, -1, axis=0)
    
    # Evaluate half-plane: value <= 1e-6 means INSIDE
    v_curr = a * curr_pts[:, 0] + b * curr_pts[:, 1] - c
    v_next = a * next_pts[:, 0] + b * next_pts[:, 1] - c
    
    curr_inside = v_curr <= 1e-6
    next_inside = v_next <= 1e-6
    
    # Calculate exact edge intersections
    denom = v_curr - v_next
    safe_denom = jnp.where(jnp.abs(denom) < 1e-15, 1e-15, denom)
    t = jnp.clip(v_curr / safe_denom, 0.0, 1.0)
    intersections = curr_pts + t[:, None] * (next_pts - curr_pts)
    
    # --- EDGE CONDITIONAL ASSIGNMENTS ---
    crossed = (curr_inside != next_inside)
    pt_A = jnp.where(crossed[:, None], intersections, 
                     jnp.where(next_inside[:, None], next_pts, intersections))
    
    use_next_B = (~curr_inside) & next_inside
    pt_B = jnp.where(use_next_B[:, None], next_pts, pt_A)
    
    # Interleave to build initial fixed array shape (2*N, 2)
    output = jnp.stack([pt_A, pt_B], axis=1).reshape(2 * N, 2)
    
    # --- OUTSIDE CLEANUP ---
    v_out = a * output[:, 0] + b * output[:, 1] - c
    # Relaxed to 1e-4 to prevent floating-point precision noise from killing valid boundary lines
    is_outside = v_out > 1e-4 
    
    # Find a locally valid anchor point from this specific clipping step
    first_inside_idx = jnp.argmax(curr_inside)
    any_survived = jnp.any(curr_inside)
    local_anchor = jnp.where(any_survived, poly[first_inside_idx], poly[0])
    
    # Force any remaining outside coordinate noise to snap directly to the local anchor
    clipped_poly = jnp.where(is_outside[:, None], local_anchor, output)
    
    # --- EMPTY OBLITERATION STEP ---
    has_nan = jnp.any(jnp.isnan(poly))
    is_still_valid = any_survived & (~has_nan)
    
    final_poly = jnp.where(is_still_valid, clipped_poly, jnp.nan)
    
    return final_poly


def _intersect_halfplanes(halfplanes, bbox=((0, 0), (1, 1))):
    """Intersect half-planes (a,b,c) representing a*x + b*y <= c within initial bbox square.
    bbox: ((x0,y0),(x1,y1)) axis-aligned rectangle providing initial polygon.
    Returns list of vertices of resulting convex polygon in order.
    """
    (x0, y0), (x1, y1) = bbox
    
    # 1. Initialize our starting box as a JAX array
    # We must explicitly pad/over-allocate the array size upfront.
    # Each half-plane clip can potentially double the number of slots needed.
    # For M half-planes, starting with 4 vertices, a safe allocation size is 4 * (2**M),
    # or you can set a safe maximum upper bound (e.g., 32 or 64 slots) depending on M.
    M = halfplanes.shape[0]
    max_vertices = 4 * (2 ** M)  
    
    # Start with a pristine CCW square, padded with duplicate coordinates
    poly = jnp.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=jnp.float32)
    # poly = jnp.tile(init_poly, (2 ** M, 1)) 
    
    # 2. Sequentially apply each halfplane step using jax.lax.fori_loop 
    # or a standard Python loop (Python loops are perfectly fine here because 
    # M, the number of half-planes, is a static structural property)
    for i in range(M):
        a = halfplanes[i, 0]
        b = halfplanes[i, 1]
        c = halfplanes[i, 2]
        poly = _clip_polygon_with_halfplane(poly, a, b, c)
        
    return poly


# ---- Feasible region polygon ----

def compute_total_region_polygon(P, N, α, κ):
    """
    Compute polygon of the feasible region in ROC space.
    Fully compatible with jax.grad.
    
    Parameters:
    P, N: Total positives and negatives (scalars/floats)
    α: Precision threshold in (0, 1)
    κ: Capacity threshold > 0
    
    Returns:
    jax.Array: Fixed-size padded polygon array
    """
    m_p = (α * N) / ((1.0 - α) * P)
    
    # 1. Define the 5 core boundaries that are always active
    # Format: [a, b, c] for ax + by <= c
    h0 = jnp.array([m_p, -1.0, 0.0])   # precision constraint: y >= m_p * x
    # h1 = jnp.array([-1.0, 0.0, 0.0])  # x >= 0
    # h2 = jnp.array([0.0, 1.0, 1.0])   # y <= 1
    # h3 = jnp.array([1.0, 0.0, 1.0])   # x <= 1
    # h4 = jnp.array([0.0, -1.0, 0.0])  # y >= 0
    
    # 2. Handle the conditional capacity constraint: N*x + P*y <= kappa
    # JAX must trace all paths, so we evaluate the constraint plane anyway.
    # If kappa >= P + N, it's non-binding, so we substitute a dummy plane 
    # that sits entirely outside the ROC square (e.g., 0*x + 0*y <= 1.0)
    capacity_plane = jnp.array([N, P, κ])
    dummy_plane = jnp.array([0.0, 0.0, 1.0])
    
    h_cap = jnp.where(κ < (P + N), capacity_plane, dummy_plane)
    
    # 3. Stack into a single static matrix (6 half-planes total)
    halfplanes = jnp.stack([h0, h_cap], axis=0)
    
    # 4. Intersect the halfplanes inside the standard ROC box [0,1]x[0,1]
    # This returns a static, padded array where duplicate points are clean
    # zero-area lines that don't disrupt downstream math.
    poly = _intersect_halfplanes(halfplanes, bbox=((0.0, 0.0), (1.0, 1.0)))
    
    return poly


def total_region_area(P, N, α, κ):
    # 1. Compute the padded polygon matrix
    poly = compute_total_region_polygon(P, N, α, κ)
    
    # 2. Check if the polygon is empty (contains NaNs)
    is_empty = jnp.any(jnp.isnan(poly))
    
    # 3. Compute the area using your JAX area function
    calculated_area = jnp.abs(area(poly))
    
    # 4. Use jnp.where to conditionally assign the final area value.
    # If it is empty, the area is 0.0, otherwise use the calculated area.
    final_area = jnp.where(is_empty, 0.0, calculated_area)
    
    return final_area, poly


# ---- Isoperformance line ----

def _iso_performance_line(h, k, t):
    """
    Computes coefficients (a, b, c) for the half-plane a*x + b*y <= c.
    Fully compatible with jax.grad and vectorized operations.
    """
    x1, y1 = h, k
    y2 = 1.0
    
    # 1. Protect against division by zero if t = 0
    is_t_zero = jnp.abs(t) < 1e-12
    safe_t = jnp.where(is_t_zero, 1e-12, t)
    
    # Calculate x2 using the protected t value
    x2 = h + k - 1.0 + (1.0 - k) / safe_t
    
    # 2. Safely check for vertical line slopes
    dx = x2 - x1
    is_vertical = jnp.abs(dx) < 1e-12
    safe_dx = jnp.where(is_vertical, 1e-12, dx)
    
    m = (y2 - y1) / safe_dx
    b0 = y1 - m * x1
    
    # 3. Handle slope orientations
    a_pos, b_pos, c_pos = -m, 1.0, b0
    a_neg, b_neg, c_neg = m, -1.0, -b0
    
    a_slope = jnp.where(m >= 0, a_pos, a_neg)
    b_slope = jnp.where(m >= 0, b_pos, b_neg)
    c_slope = jnp.where(m >= 0, c_pos, c_neg)
    
    # 4. Apply standard vertical line override flag if dx was near zero
    a_mid = jnp.where(is_vertical, -1.0, a_slope)
    b_mid = jnp.where(is_vertical, 0.0, b_slope)
    c_mid = jnp.where(is_vertical, -x1, c_slope)
    
    # 5. ULTIMATE OVERRIDE FOR t = 0
    # When t = 0, the line collapses geometrically to a pure horizontal boundary 
    # tracking the height of your current point k:
    # y >= k  ->  0*x - 1*y <= -k
    a_zero_t = 0.0
    b_zero_t = -1.0
    c_zero_t = -k
    
    a = jnp.where(is_t_zero, a_zero_t, a_mid)
    b = jnp.where(is_t_zero, b_zero_t, b_mid)
    c = jnp.where(is_t_zero, c_zero_t, c_mid)
    
    return a, b, c


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
    
    # 1. Compute total region polygon and area
    total_poly_area, total_poly = total_region_area(P, N, α, κ)
    
    # 2. Compute iso-performance line and clip
    a, b, c = _iso_performance_line(h, k, t)
    iso_poly = _clip_polygon_with_halfplane(total_poly, a, b, c)
    
    # 3. Calculate raw intersection area
    raw_area = area(iso_poly)
    
    # 4. Handle Area Scale Logic mathematically via jnp.where 
    # to protect against division-by-zero if total_poly_area is 0
    safe_total_area = jnp.where(total_poly_area == 0.0, 1.0, total_poly_area)
    percent_area = raw_area / safe_total_area
    
    # Select our baseline value target dynamically without branching
    selected_value = jnp.where(return_percent, percent_area, raw_area)
    
    # If the total area is strictly zero, override the final value to 0.0
    value = jnp.where(total_poly_area == 0.0, 0.0, selected_value)
    
    # 5. Handle Return Structures Statically
    # Since these flags are plain Python booleans passed at call-time, 
    # JAX unrolls these branches perfectly during tracing!
    if return_total_area and return_details:
        details = {"total_polygon": total_poly, "iso_polygon": iso_poly, "iso_line": (a, b, c), "t": t}
        return value, total_poly_area, details
        
    if return_total_area:
        return value, total_poly_area
        
    if return_details:
        details = {"total_polygon": total_poly, "iso_polygon": iso_poly, "iso_line": (a, b, c), "t": t}
        return value, details
        
    return value

    # if total_poly_area == 0:
    #     if return_total_area and return_details:
    #         return 0.0, 0.0, {"total_polygon": total_poly, "iso_polygon": [], "iso_line": None, "t": None}
    #     if return_total_area:
    #         return 0.0, 0.0
    #     return 0.0 if not return_details else (0.0, {"total_polygon": total_poly, "iso_polygon": [], "iso_line": None, "t": None})
    # a, b, c = _iso_performance_line(h, k, t)
    # # Intersect total polygon with iso half-plane
    # iso_poly = _clip_polygon_with_halfplane(total_poly, a, b, c)
    # raw_area = abs(area(jnp.array(iso_poly))) if iso_poly else 0.0
    # value = raw_area / total_poly_area if return_percent else raw_area
    # if return_total_area and return_details:
    #     return value, total_poly_area, {"total_polygon": total_poly, "iso_polygon": iso_poly, "iso_line": (a, b, c), "t": t}
    # if return_total_area:
    #     return value, total_poly_area
    # if return_details:
    #     return value, {"total_polygon": total_poly, "iso_polygon": iso_poly, "iso_line": (a, b, c), "t": t}
    # return value


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
    satisfies_capacity = (tpr - upper_bound) <= 1e-6
    satisfies_precision = (lower_bound - tpr) <= 1e-6

    return satisfies_precision and satisfies_capacity


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








# MAX_VERTS = 12  # provably sufficient upper bound on polygon vertex count

# # ---------------- geometry primitives (JAX, fixed-size, differentiable) ----------------

# def _area_jax(poly, n):
#     """Shoelace area for a polygon stored as (MAX_VERTS,2) with n valid leading vertices."""

#     # Computes area with all vertices in a vector at once (no for loop)
#     idx = jnp.arange(MAX_VERTS)
#     n_safe = jnp.maximum(n, 1)
#     next_idx = jnp.mod(idx + 1, n_safe)
#     x, y = poly[:, 0], poly[:, 1]
#     x_next, y_next = poly[next_idx, 0], poly[next_idx, 1]
#     terms = x * y_next - y * x_next
#     mask = idx < n
#     return 0.5 * jnp.sum(jnp.where(mask, terms, 0.0))


# def _clip_halfplane_jax(poly, n, a, b, c):
#     """Sutherland-Hodgman clip of a convex polygon (poly, n) by a*x+b*y<=c.
#     Returns (new_poly, new_n), both fixed shape.
#     """
#     def inside(pt):
#         x, y = pt
#         return a * x + b * y <= c + 1e-12

#     def intersect(p1, p2):
#         x1, y1 = p1
#         x2, y2 = p2
#         v1 = a * x1 + b * y1 - c
#         v2 = a * x2 + b * y2 - c
#         denom = v1 - v2
#         safe_denom = jnp.where(jnp.abs(denom) < 1e-15, 1.0, denom)
#         t = jnp.where(jnp.abs(denom) < 1e-15, 0.0, v1 / safe_denom)
#         return p1 + (p2 - p1) * t

#     def body(i, carry):
#         out_poly, out_count = carry
#         n_safe = jnp.maximum(n, 1)
#         active = i < n
#         idx_prev = jnp.mod(i - 1 + n_safe, n_safe)
#         curr = poly[i]
#         prev = poly[idx_prev]
#         p_in, c_in = inside(prev), inside(curr)
#         inter = intersect(prev, curr)

#         enter = (~p_in) & c_in
#         exit_ = p_in & (~c_in)
#         stay = p_in & c_in

#         n_add = jnp.where(active, jnp.where(enter, 2, jnp.where(exit_, 1, jnp.where(stay, 1, 0))), 0)
#         pt_first = jnp.where(stay, curr, inter)   # inter if enter/exit, curr if stay
#         pt_second = curr                          # only used if n_add == 2 (enter)

#         out_poly = jax.lax.cond(
#             n_add >= 1,
#             lambda op: jax.lax.dynamic_update_slice(op, pt_first[None, :], (out_count, 0)),
#             lambda op: op, out_poly)
#         out_poly = jax.lax.cond(
#             n_add == 2,
#             lambda op: jax.lax.dynamic_update_slice(op, pt_second[None, :], (out_count + 1, 0)),
#             lambda op: op, out_poly)
#         return (out_poly, out_count + n_add)

#     out_poly0 = jnp.zeros((MAX_VERTS, 2))
#     out_count0 = jnp.array(0, dtype=jnp.int32)
#     out_poly, out_count = jax.lax.fori_loop(0, MAX_VERTS, body, (out_poly0, out_count0))
#     return out_poly, out_count


# def _compute_total_region_jax(P, N, alpha, kappa):
#     m_p = (alpha * N) / ((1 - alpha) * P)
#     halfplanes = [
#         (m_p, -1.0, 0.0),     # precision
#         (N, P, kappa),        # capacity (always included; non-binding when kappa>=P+N)
#         (-1.0, 0.0, 0.0),     # x >= 0
#         (0.0, 1.0, 1.0),      # y <= 1
#         (1.0, 0.0, 1.0),      # x <= 1
#         (0.0, -1.0, 0.0),     # y >= 0
#     ]
#     poly = jnp.zeros((MAX_VERTS, 2)).at[:4].set(jnp.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]))
#     n = jnp.array(4, dtype=jnp.int32)
#     for (a, b, c) in halfplanes:
#         poly, n = _clip_halfplane_jax(poly, n, a, b, c)
#     return poly, n


# def _iso_line_jax(h, k, t):
#     x2 = h + k - 1 + (1 - k) / t
#     y2 = 1.0
#     x1, y1 = h, k
#     dx = x2 - x1
#     vertical = jnp.abs(dx) < 1e-12
#     m = jnp.where(vertical, 0.0, (y2 - y1) / jnp.where(vertical, 1.0, dx))
#     b0 = y1 - m * x1
#     a_v, b_v, c_v = -1.0, 0.0, -x1                       # vertical-line branch
#     a_pos, b_pos, c_pos = -m, 1.0, b0                     # m >= 0 branch
#     a_neg, b_neg, c_neg = m, -1.0, -b0                    # m < 0 branch
#     a = jnp.where(vertical, a_v, jnp.where(m >= 0, a_pos, a_neg))
#     b = jnp.where(vertical, b_v, jnp.where(m >= 0, b_pos, b_neg))
#     c = jnp.where(vertical, c_v, jnp.where(m >= 0, c_pos, c_neg))
#     return a, b, c


# def _reduced_area_jax(h, k, total_poly, total_n, total_area, t):
#     a, b, c = _iso_line_jax(h, k, t)
#     iso_poly, iso_n = _clip_halfplane_jax(total_poly, total_n, a, b, c)
#     raw = jnp.abs(_area_jax(iso_poly, iso_n))
#     return jnp.where(total_area > 1e-12, raw / total_area, 0.0)


# def _keep_model_jax(fpr, tpr, alpha, kappa, N, P):
#     upper = (kappa - N * fpr) / P
#     lower = (alpha * N * fpr) / ((1 - alpha) * P)
#     return (tpr <= upper) & (tpr >= lower)


# def _ratio_to_t_jax(r, P, N):
#     return (r * N) / (r * N + P)


# def _trapz_jax(y, x):
#     dx = x[1:] - x[:-1]
#     return jnp.sum((y[1:] + y[:-1]) * dx / 2.0)


# def voros_jax(fprs, tprs, feasible_mask, kappa, alpha, P, N,
#               min_fp_cost_ratio, max_fp_cost_ratio, n_points=200):
#     total_poly, total_n = _compute_total_region_jax(P, N, alpha, kappa)
#     total_area = jnp.abs(_area_jax(total_poly, total_n))

#     fp_cost_ratios = jnp.linspace(min_fp_cost_ratio, max_fp_cost_ratio, n_points)
#     ts = _ratio_to_t_jax(fp_cost_ratios, P, N)

#     def per_t(t):
#         vals = jax.vmap(lambda h, k: _reduced_area_jax(h, k, total_poly, total_n, total_area, t))(fprs, tprs)
#         masked = jnp.where(feasible_mask, vals, -jnp.inf)
#         m = jnp.max(masked)
#         return jnp.where(jnp.isfinite(m), m, 0.0)

#     max_points = jax.vmap(per_t)(ts)
#     r_range = max_fp_cost_ratio - min_fp_cost_ratio
#     return _trapz_jax(max_points, fp_cost_ratios) / r_range


