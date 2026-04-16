"""
hull
====

Concave-hull utilities used by the extract and process stages:

* :func:`alpha_hull` — an alpha-shape hull builder that interpolates
  between the convex hull (shrink factor 0) and the tightest
  enclosing shape (shrink factor 1).
* :func:`inpolygon` — point-in-polygon test with on-edge detection.

The alpha-shape threshold is chosen by calibration: pick the smallest
circumradius threshold that encloses at least ``target_coverage`` of
the input points. The implementation is deterministic and fully
reproducible given the same input points and ``target_coverage``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial import ConvexHull, Delaunay


def _triangle_circumradius(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Circumradius of the triangle with vertices ``p1``, ``p2``, ``p3``.

    Returns ``inf`` for degenerate (collinear) triangles, so they are always
    dropped by the alpha-shape filter.
    """
    a = float(np.linalg.norm(p2 - p3))
    b = float(np.linalg.norm(p1 - p3))
    c = float(np.linalg.norm(p1 - p2))
    s = 0.5 * (a + b + c)
    area_sq = s * (s - a) * (s - b) * (s - c)
    if area_sq <= 1e-20:
        return float("inf")
    area = float(np.sqrt(area_sq))
    return (a * b * c) / (4.0 * area)


def _alpha_shape_indices(
    pts: np.ndarray,
    *,
    target_coverage: float = 1.0,
) -> np.ndarray:
    """Return the indices of points on the boundary of an alpha shape.

    Proper alpha-shape construction: compute the Delaunay triangulation, find
    the smallest circumradius threshold ``alpha`` such that the set of
    triangles with circumradius ``≤ alpha`` covers at least
    ``target_coverage`` of the input points, then keep *all* triangles below
    that threshold (not just the ones we walked through looking for
    coverage). The boundary consists of edges that appear in exactly one
    surviving triangle.

    With ``target_coverage=1.0`` this is the classical "tightest alpha that
    still encloses every input point"; in the body-part cloud data from
    ``ground_normalized.mat`` the result effectively matches the convex hull
    (the clouds are compact and roughly convex per body part).

    Parameters
    ----------
    pts:
        ``(n, 2)`` array of 2-D points.
    target_coverage:
        Fraction of points that must be vertices of at least one surviving
        triangle. ``1.0`` by default.

    Returns
    -------
    np.ndarray
        Indices of the boundary vertices in traversal order, with the first
        index repeated at the end so the polygon is closed.
    """
    if len(pts) < 3:
        return np.arange(len(pts), dtype=int)

    # Protect against degenerate point clouds (e.g. all points at origin,
    # as happens for Body2 in ground_normalized.mat) which would otherwise
    # crash Qhull.
    try:
        tri = Delaunay(pts)
    except Exception:
        return np.arange(len(pts), dtype=int)
    simplices = tri.simplices  # (n_tri, 3) vertex indices

    # Compute circumradii once.
    radii = np.empty(len(simplices))
    for i, s in enumerate(simplices):
        radii[i] = _triangle_circumradius(pts[s[0]], pts[s[1]], pts[s[2]])

    # Find the alpha threshold: the smallest radius such that the surviving
    # triangles (all triangles with radius ≤ alpha) touch at least
    # `target_coverage` of the points. This differs from "add triangles in
    # radius order until coverage is reached" because we then include *all*
    # triangles below the discovered threshold, not just enough to reach it.
    order = np.argsort(radii)
    needed = int(np.ceil(target_coverage * len(pts)))
    covered = np.zeros(len(pts), dtype=bool)
    alpha_idx = len(order) - 1
    for rank, idx in enumerate(order):
        tri_verts = simplices[idx]
        covered[tri_verts] = True
        if int(covered.sum()) >= needed:
            alpha_idx = rank
            break
    alpha_threshold = radii[order[alpha_idx]]
    keep = radii <= alpha_threshold

    # Extract edges that belong to exactly one kept triangle.
    edges: dict[tuple[int, int], int] = {}
    for idx in np.where(keep)[0]:
        a, b, c = simplices[idx]
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            edges[key] = edges.get(key, 0) + 1
    boundary_edges = [k for k, count in edges.items() if count == 1]

    if not boundary_edges:
        # Fall back to the convex hull; this should only happen on degenerate
        # inputs (e.g. all points collinear).
        hull = ConvexHull(pts)
        ring = list(hull.vertices) + [int(hull.vertices[0])]
        return np.array(ring, dtype=int)

    # Walk the edges to build an ordered ring. The boundary of a simply
    # connected alpha shape is a single loop; if the shape turns out to be
    # multiply connected we walk the longest loop and drop the rest, which
    # matches the behaviour of returning a single outer contour when
    # called as ``k = boundary(x, y)``.
    adjacency: dict[int, list[int]] = {}
    for u, v in boundary_edges:
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)

    def walk_from(start: int) -> list[int]:
        ring = [start]
        prev = -1
        current = start
        while True:
            nxt = None
            for candidate in adjacency.get(current, []):
                if candidate != prev:
                    nxt = candidate
                    break
            if nxt is None or nxt == start:
                break
            ring.append(nxt)
            prev, current = current, nxt
            if len(ring) > len(pts) + 1:  # safety valve
                break
        ring.append(start)
        return ring

    starts = list(adjacency.keys())
    best_ring: list[int] = []
    visited_starts: set[int] = set()
    for s in starts:
        if s in visited_starts:
            continue
        ring = walk_from(s)
        visited_starts.update(ring)
        if len(ring) > len(best_ring):
            best_ring = ring
    return np.array(best_ring, dtype=int)


def alpha_hull(
    x: np.ndarray,
    y: np.ndarray,
    *,
    target_coverage: float = 1.0,
) -> np.ndarray:
    """Compute the concave hull of a 2-D point cloud.

    Returns the 0-based indices of the points on the hull, in
    traversal order, with the first index repeated at the end so the
    polygon is closed.

    NaN points are dropped before the computation; the returned
    indices refer to the *original* (pre-drop) array.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    mask = ~(np.isnan(x) | np.isnan(y))
    pts = np.column_stack([x[mask], y[mask]])
    if len(pts) == 0:
        return np.zeros(0, dtype=int)

    local_ring = _alpha_shape_indices(pts, target_coverage=target_coverage)

    # Map local (post-drop) indices back to the original array.
    original_indices = np.where(mask)[0]
    return original_indices[local_ring]


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

    # "On edge" — slightly dilated and slightly contracted. A point is "on"
    # the boundary if the inside test disagrees between dilated and contracted
    # versions (within the tolerance). This matches the ``on`` semantics
    # well enough for the body-part outlier test in ``extract_3``.
    on = np.zeros(len(xq_f), dtype=bool)
    if valid.any():
        inside_plus = poly.contains_points(query_valid, radius=+1e-6)
        inside_minus = poly.contains_points(query_valid, radius=-1e-6)
        on_valid = inside_plus & ~inside_minus
        on[valid] = on_valid

    return inside.reshape(out_shape), on.reshape(out_shape)
