"""
hull
====

Concave-hull utilities used by the extract and process stages:

* :func:`alpha_hull` — replicates MATLAB's ``boundary(x, y, S)``
  with default shrink factor ``S = 0.5``.  Uses the same alpha-spectrum
  indexing logic as MATLAB's ``boundary.m``:

  1. Compute the Delaunay triangulation and per-triangle circumradii.
  2. Sort unique circumradii descending (= the *alpha spectrum*).
  3. Binary-search for the *critical alpha*: the smallest circumradius
     threshold that still yields a single connected boundary covering
     every triangulated point.
  4. Restrict the spectrum to values ≥ critical alpha.
  5. Index into this restricted spectrum with the shrink factor::

         idx = numA - max(ceil((1 - S) * numA), 1)
         alpha_use = Ahispec[idx]

  6. Keep all Delaunay triangles whose circumradius ≤ ``alpha_use``
     and extract the boundary edges.

* :func:`inpolygon` — point-in-polygon test with on-edge detection.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Tuple

import numpy as np
from scipy.spatial import ConvexHull, Delaunay


# ---------------------------------------------------------------------------
# Vectorized circumradius
# ---------------------------------------------------------------------------

def _circumradius_vec(pts: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    """Circumradii for all Delaunay triangles (vectorized)."""
    p1 = pts[simplices[:, 0]]
    p2 = pts[simplices[:, 1]]
    p3 = pts[simplices[:, 2]]
    a = np.linalg.norm(p2 - p3, axis=1)
    b = np.linalg.norm(p1 - p3, axis=1)
    c = np.linalg.norm(p1 - p2, axis=1)
    s = 0.5 * (a + b + c)
    area_sq = np.maximum(s * (s - a) * (s - b) * (s - c), 0.0)
    area = np.sqrt(area_sq)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = (a * b * c) / (4.0 * area)
    R[area < 1e-10] = np.inf
    return R


# ---------------------------------------------------------------------------
# Single-region check for a given alpha threshold
# ---------------------------------------------------------------------------

def _is_single_region(
    simplices: np.ndarray,
    radii: np.ndarray,
    alpha: float,
    n_triangulated: int,
) -> bool:
    """True if the alpha shape at the given circumradius threshold
    covers all triangulated points and forms a single connected loop.
    """
    keep = radii <= alpha
    covered = set()
    edge_count = {}
    for idx in np.where(keep)[0]:
        a, b, c = (
            int(simplices[idx, 0]),
            int(simplices[idx, 1]),
            int(simplices[idx, 2]),
        )
        covered.update((a, b, c))
        for u, v in ((min(a, b), max(a, b)),
                      (min(b, c), max(b, c)),
                      (min(a, c), max(a, c))):
            edge_count[(u, v)] = edge_count.get((u, v), 0) + 1
    if len(covered) < n_triangulated:
        return False
    boundary_edges = [e for e, cnt in edge_count.items() if cnt == 1]
    if not boundary_edges:
        return False
    adj = defaultdict(set)
    for u, v in boundary_edges:
        adj[u].add(v)
        adj[v].add(u)
    start = next(iter(adj))
    visited = {start}
    queue = [start]
    while queue:
        node = queue.pop()
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return len(visited) == len(adj)


# ---------------------------------------------------------------------------
# Boundary extraction (walk edges into an ordered ring)
# ---------------------------------------------------------------------------

def _extract_ring(
    simplices: np.ndarray,
    radii: np.ndarray,
    alpha: float,
) -> list[int]:
    """Extract boundary vertex ring at the given alpha threshold."""
    if alpha == float("inf"):
        keep = np.isfinite(radii)
    else:
        keep = radii <= alpha
    edge_count = {}
    for idx in np.where(keep)[0]:
        a, b, c = (
            int(simplices[idx, 0]),
            int(simplices[idx, 1]),
            int(simplices[idx, 2]),
        )
        for u, v in ((min(a, b), max(a, b)),
                      (min(b, c), max(b, c)),
                      (min(a, c), max(a, c))):
            edge_count[(u, v)] = edge_count.get((u, v), 0) + 1
    boundary_edges = [(u, v) for (u, v), cnt in edge_count.items() if cnt == 1]
    if not boundary_edges:
        return []
    adj = defaultdict(list)
    for u, v in boundary_edges:
        adj[u].append(v)
        adj[v].append(u)
    # Walk the longest loop (handles multi-component gracefully).
    def walk_from(start: int) -> list[int]:
        ring = [start]
        prev = -1
        current = start
        while True:
            nxt = None
            for c in adj.get(current, []):
                if c != prev:
                    nxt = c
                    break
            if nxt is None or nxt == start:
                break
            ring.append(nxt)
            prev, current = current, nxt
            if len(ring) > len(adj) + 1:
                break
        ring.append(start)
        return ring
    best = []
    seen = set()
    for s in adj:
        if s in seen:
            continue
        ring = walk_from(s)
        seen.update(ring)
        if len(ring) > len(best):
            best = ring
    return best


# ---------------------------------------------------------------------------
# Public API: alpha_hull  (replaces old implementation)
# ---------------------------------------------------------------------------

def alpha_hull(
    x: np.ndarray,
    y: np.ndarray,
    *,
    target_coverage: float = 1.0,
    shrink: float = 0.5,
) -> np.ndarray:
    """Compute the concave hull of a 2-D point cloud.

    Replicates MATLAB's ``boundary(x, y, shrink)`` using the exact
    same alpha-spectrum indexing logic.  The ``target_coverage``
    parameter is accepted for API compatibility but ignored; the
    ``shrink`` parameter controls boundary tightness (0 = convex hull,
    1 = tightest single-region boundary, default 0.5).

    Returns the 0-based indices of the points on the hull in traversal
    order, with the first index repeated at the end so the polygon is
    closed.  Indices refer to the *original* (pre-NaN-drop) input
    arrays.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    valid = ~(np.isnan(x) | np.isnan(y))
    pts = np.column_stack([x[valid], y[valid]])
    if len(pts) < 3:
        idx = np.where(valid)[0]
        if len(idx) == 0:
            return np.zeros(0, dtype=int)
        return np.append(idx, idx[0])

    # Remove duplicate points (MATLAB's boundary does this).
    _, uniq_map = np.unique(pts, axis=0, return_index=True)
    uniq_map = np.sort(uniq_map)
    pts_u = pts[uniq_map]
    if len(pts_u) < 3:
        idx = np.where(valid)[0][uniq_map]
        return np.append(idx, idx[0])

    try:
        tri = Delaunay(pts_u)
    except Exception:
        idx = np.where(valid)[0][uniq_map]
        return np.append(idx, idx[0])

    radii = _circumradius_vec(pts_u, tri.simplices)
    finite_r = radii[np.isfinite(radii)]
    if len(finite_r) == 0:
        idx = np.where(valid)[0][uniq_map]
        return np.append(idx, idx[0])

    # Alpha spectrum: unique finite circumradii, sorted descending.
    spectrum = np.sort(np.unique(finite_r))[::-1]

    # Number of points actually used in the triangulation (Qhull may
    # discard a few that are exactly coplanar with neighbours).
    n_triangulated = len(set(tri.simplices.ravel()))

    # Check that the convex-hull end of the spectrum works at all.
    if not _is_single_region(tri.simplices, radii, spectrum[0], n_triangulated):
        alpha_use = float("inf")
    else:
        # Binary-search for the critical alpha (smallest circumradius
        # threshold giving a single connected region).
        lo, hi = 0, len(spectrum) - 1
        crit_idx = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if _is_single_region(tri.simplices, radii, spectrum[mid], n_triangulated):
                crit_idx = mid
                lo = mid + 1
            else:
                hi = mid - 1

        Acritical = spectrum[crit_idx]
        Ahispec = spectrum[spectrum >= Acritical]
        numA = len(Ahispec)

        if numA <= 1 or (Ahispec[0] - Acritical) < 1e-3 * Ahispec[0]:
            alpha_use = float("inf")
        else:
            # MATLAB indexing:
            #   Ahispec(numA + 1 - max(ceil((1-S)*numA), 1))
            # Converted to 0-based:
            idx = numA - max(int(np.ceil((1 - shrink) * numA)), 1)
            alpha_use = Ahispec[idx]

    ring = _extract_ring(tri.simplices, radii, alpha_use)
    if not ring:
        # Fallback to convex hull.
        try:
            ch = ConvexHull(pts_u)
            ring = list(ch.vertices) + [int(ch.vertices[0])]
        except Exception:
            ring = list(range(len(pts_u))) + [0]

    # Map: ring (in pts_u space) → uniq_map → valid-index → original index.
    original_indices = np.where(valid)[0]
    return original_indices[uniq_map[ring]]


# ---------------------------------------------------------------------------
# inpolygon (unchanged from original)
# ---------------------------------------------------------------------------

def inpolygon(
    xq: np.ndarray,
    yq: np.ndarray,
    xv: np.ndarray,
    yv: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Point-in-polygon test with on-edge detection.

    Returns two boolean arrays of the same shape as ``xq``:

    * ``in``: True if the query point is strictly inside the polygon or on
      its edge.
    * ``on``: True if the query point is on the polygon edge (within a small
      numerical tolerance).

    Uses :class:`matplotlib.path.Path` for the point-in-polygon test, which
    handles concave polygons correctly.

    NaN query points return ``False`` in both arrays.
    """
    from matplotlib.path import Path

    xq = np.asarray(xq, dtype=float)
    yq = np.asarray(yq, dtype=float)
    xv = np.asarray(xv, dtype=float).ravel()
    yv = np.asarray(yv, dtype=float).ravel()

    out_shape = xq.shape
    xq_f = xq.ravel()
    yq_f = yq.ravel()

    valid = ~(np.isnan(xq_f) | np.isnan(yq_f))
    query = np.column_stack([xq_f, yq_f])
    query_valid = query[valid]

    poly = Path(np.column_stack([xv, yv]))
    inside = np.zeros(len(xq_f), dtype=bool)
    inside[valid] = poly.contains_points(query_valid, radius=1e-9)

    on = np.zeros(len(xq_f), dtype=bool)
    if valid.any():
        inside_plus = poly.contains_points(query_valid, radius=+1e-6)
        inside_minus = poly.contains_points(query_valid, radius=-1e-6)
        on_valid = inside_plus & ~inside_minus
        on[valid] = on_valid

    return inside.reshape(out_shape), on.reshape(out_shape)
