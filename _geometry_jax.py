import jax
import jax.numpy as jnp
from functools import partial

MAX_VERTS = 12  # provably sufficient upper bound on polygon vertex count

# ---------------- geometry primitives (JAX, fixed-size, differentiable) ----------------

def _area_jax(poly, n):
    """Shoelace area for a polygon stored as (MAX_VERTS,2) with n valid leading vertices."""
    idx = jnp.arange(MAX_VERTS)
    n_safe = jnp.maximum(n, 1)
    next_idx = jnp.mod(idx + 1, n_safe)
    x, y = poly[:, 0], poly[:, 1]
    x_next, y_next = poly[next_idx, 0], poly[next_idx, 1]
    terms = x * y_next - y * x_next
    mask = idx < n
    return 0.5 * jnp.sum(jnp.where(mask, terms, 0.0))


def _clip_halfplane_jax(poly, n, a, b, c):
    """Sutherland-Hodgman clip of a convex polygon (poly, n) by a*x+b*y<=c.
    Returns (new_poly, new_n), both fixed shape.
    """
    def inside(pt):
        return a * pt[0] + b * pt[1] <= c + 1e-9

    def intersect(p1, p2):
        v1 = a * p1[0] + b * p1[1] - c
        v2 = a * p2[0] + b * p2[1] - c
        denom = v1 - v2
        safe_denom = jnp.where(jnp.abs(denom) < 1e-15, 1.0, denom)
        t = jnp.where(jnp.abs(denom) < 1e-15, 0.0, v1 / safe_denom)
        return p1 + (p2 - p1) * t

    def body(i, carry):
        out_poly, out_count = carry
        n_safe = jnp.maximum(n, 1)
        active = i < n
        idx_prev = jnp.mod(i - 1 + n_safe, n_safe)
        curr = poly[i]
        prev = poly[idx_prev]
        p_in, c_in = inside(prev), inside(curr)
        inter = intersect(prev, curr)

        enter = (~p_in) & c_in
        exit_ = p_in & (~c_in)
        stay = p_in & c_in

        n_add = jnp.where(active, jnp.where(enter, 2, jnp.where(exit_, 1, jnp.where(stay, 1, 0))), 0)
        pt_first = jnp.where(stay, curr, inter)   # inter if enter/exit, curr if stay
        pt_second = curr                          # only used if n_add == 2 (enter)

        out_poly = jax.lax.cond(
            n_add >= 1,
            lambda op: jax.lax.dynamic_update_slice(op, pt_first[None, :], (out_count, 0)),
            lambda op: op, out_poly)
        out_poly = jax.lax.cond(
            n_add == 2,
            lambda op: jax.lax.dynamic_update_slice(op, pt_second[None, :], (out_count + 1, 0)),
            lambda op: op, out_poly)
        return (out_poly, out_count + n_add)

    out_poly0 = jnp.zeros((MAX_VERTS, 2))
    out_count0 = jnp.array(0, dtype=jnp.int32)
    out_poly, out_count = jax.lax.fori_loop(0, MAX_VERTS, body, (out_poly0, out_count0))
    return out_poly, out_count


def _compute_total_region_jax(P, N, alpha, kappa):
    m_p = (alpha * N) / ((1 - alpha) * P)
    halfplanes = [
        (m_p, -1.0, 0.0),     # precision
        (N, P, kappa),        # capacity (always included; non-binding when kappa>=P+N)
        (-1.0, 0.0, 0.0),     # x >= 0
        (0.0, 1.0, 1.0),      # y <= 1
        (1.0, 0.0, 1.0),      # x <= 1
        (0.0, -1.0, 0.0),     # y >= 0
    ]
    poly = jnp.zeros((MAX_VERTS, 2)).at[:4].set(jnp.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]))
    n = jnp.array(4, dtype=jnp.int32)
    for (a, b, c) in halfplanes:
        poly, n = _clip_halfplane_jax(poly, n, a, b, c)
    return poly, n


def _iso_line_jax(h, k, t):
    x2 = h + k - 1 + (1 - k) / t
    y2 = 1.0
    x1, y1 = h, k
    dx = x2 - x1
    vertical = jnp.abs(dx) < 1e-12
    m = jnp.where(vertical, 0.0, (y2 - y1) / jnp.where(vertical, 1.0, dx))
    b0 = y1 - m * x1
    a_v, b_v, c_v = -1.0, 0.0, -x1                       # vertical-line branch
    a_pos, b_pos, c_pos = -m, 1.0, b0                     # m >= 0 branch
    a_neg, b_neg, c_neg = m, -1.0, -b0                    # m < 0 branch
    a = jnp.where(vertical, a_v, jnp.where(m >= 0, a_pos, a_neg))
    b = jnp.where(vertical, b_v, jnp.where(m >= 0, b_pos, b_neg))
    c = jnp.where(vertical, c_v, jnp.where(m >= 0, c_pos, c_neg))
    return a, b, c


def _reduced_area_jax(h, k, total_poly, total_n, total_area, t):
    a, b, c = _iso_line_jax(h, k, t)
    iso_poly, iso_n = _clip_halfplane_jax(total_poly, total_n, a, b, c)
    raw = jnp.abs(_area_jax(iso_poly, iso_n))
    return jnp.where(total_area > 1e-12, raw / total_area, 0.0)


def _keep_model_jax(fpr, tpr, alpha, kappa, N, P):
    upper = (kappa - N * fpr) / P
    lower = (alpha * N * fpr) / ((1 - alpha) * P)
    return (tpr <= upper) & (tpr >= lower)


def _ratio_to_t_jax(r, P, N):
    return (r * N) / (r * N + P)


def _trapz_jax(y, x):
    dx = x[1:] - x[:-1]
    return jnp.sum((y[1:] + y[:-1]) * dx / 2.0)


def voros_jax(fprs, tprs, feasible_mask, kappa, alpha, P, N,
              min_fp_cost_ratio, max_fp_cost_ratio, n_points=200):
    total_poly, total_n = _compute_total_region_jax(P, N, alpha, kappa)
    total_area = jnp.abs(_area_jax(total_poly, total_n))

    fp_cost_ratios = jnp.linspace(min_fp_cost_ratio, max_fp_cost_ratio, n_points)
    ts = _ratio_to_t_jax(fp_cost_ratios, P, N)

    def per_t(t):
        vals = jax.vmap(lambda h, k: _reduced_area_jax(h, k, total_poly, total_n, total_area, t))(fprs, tprs)
        masked = jnp.where(feasible_mask, vals, -jnp.inf)
        m = jnp.max(masked)
        return jnp.where(jnp.isfinite(m), m, 0.0)

    max_points = jax.vmap(per_t)(ts)
    r_range = max_fp_cost_ratio - min_fp_cost_ratio
    return _trapz_jax(max_points, fp_cost_ratios) / r_range


