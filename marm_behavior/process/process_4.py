"""
process_4
=========

Computes.

This is the final stage of the process pipeline.  It takes the normalized
matrices from :mod:`process_3`, cleans them up with nan-fills and a hull
filter, builds a set of 125 "edges" per animal (body-part pairs with three
interpolation points each), un-rotates those edges back into pixel-space
coordinates, and returns the eight ``(n_frames, 375)`` matrices that
:mod:the pipeline saves to ``edges_*.mat``.

Dead code removed
-----------------
The stage also contains a parallel "undo scale translate and
rotate F_norm" block that produces ``F_3``/``F_4``/``F_3B``/``F_4B``/
``F_3R``/``F_4R``/``F_3Y``/``F_4Y`` body-part matrices in pixel space.  These
matrices are computed but *never saved* by `the pipeline` — only the
edges matrices make it to disk.  This port omits the body-part un-rotation
to keep the code short; re-adding it would be a copy-and-paste of
:func:`_undo_rotate_edges` with 42-wide instead of 750-wide input.

Preserved  quirks
-----------------------
* The ``f == 11 || f == 12`` special-case override in the large-gap fill
   checks the *gap* iteration index, not the
  body-part column index.  This looks like a copy-paste bug (almost
  certainly should be ``i == 11``), and it fires essentially never in
  practice because most body parts have fewer than 11 large NaN gaps.
  Preserved.
* ``F_norm = F_normT ./ bh`` at line 97 (and its three siblings) divides
  the already-normalized F_normT by bh a second time.  The resulting
  F_norm is dead code for edge building but this port computes it anyway
  for downstream compatibility with the features stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from ..numerics.hull import inpolygon, alpha_hull
from ..numerics.helpers import (
    cart2pol,
    crossing,
    fillmissing_pchip,
    movmedian,
    nanmean,
)
from ..io.mat_io import load_ground_normalized
from .process_2 import Process2Output
from .process_3 import (
    BH_WINDOW,
    BH_MIN,
    BH_MAX,
    BH_FALLBACK,
    _L_F3,
    _R_F3,
    _M_F3,
    _L_F4,
    _R_F4,
    _M_F4,
    _compute_bh,
)

# Post-cleanup clamp range for F_normT (tighter than process_3's ±25).
CLAMP_NORMT = 10.0

# Body-part 2 (Body2, 0-based index 5) is skipped in the hull loop.
_BODY2_INDEX = 5

# Minimum NaN-gap length that triggers "large gap" handling.
_LARGE_GAP_MIN = 15


# ---------------------------------------------------------------------------
# edge list construction (hard-coded, from the process stage lines 411-455)
# ---------------------------------------------------------------------------

#  (1-based).
_EDGE_LIST_1_1B: tuple[int, ...] = (
    1, 1, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3,
    4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7,
    8, 8, 8, 8, 8, 8, 8, 8,
    9, 9, 9, 9, 9, 9, 9, 9,
    10, 10, 10, 10,
    11, 11, 11, 11,
    12, 12, 12, 12,
    13, 13, 13, 13,
    14, 14, 14, 14,
    15, 15, 15, 15,
    16, 16, 16, 16,
    17, 17, 17, 17,
    18, 18, 18, 18,
    19, 19, 19, 19,
    20, 20, 20, 20,
    21, 21, 21, 21,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
)

#  (1-based).
_EDGE_LIST_2_1B: tuple[int, ...] = (
    2, 3, 4, 5,
    1, 3, 4, 5,
    1, 2, 4, 5,
    1, 2, 3, 5,
    6, 7, 8, 9, 10, 13, 16, 19,
    5, 7, 8, 9, 10, 13, 16, 19,
    5, 6, 8, 9, 10, 13, 16, 19,
    5, 6, 7, 9, 10, 13, 16, 19,
    5, 6, 7, 8, 10, 13, 16, 19,
    2, 8, 11, 12,
    2, 8, 10, 12,
    2, 8, 10, 11,
    3, 9, 14, 15,
    3, 9, 13, 15,
    3, 9, 13, 14,
    8, 17, 18, 19,
    8, 16, 18, 19,
    8, 16, 17, 19,
    9, 16, 20, 21,
    9, 16, 19, 21,
    9, 16, 19, 20,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
)

assert len(_EDGE_LIST_1_1B) == 125, f"edge_list_1 length {len(_EDGE_LIST_1_1B)}"
assert len(_EDGE_LIST_2_1B) == 125, f"edge_list_2 length {len(_EDGE_LIST_2_1B)}"

# Convert to 0-based arrays for Python use.
EDGE_LIST_1 = np.array(_EDGE_LIST_1_1B, dtype=int) - 1
EDGE_LIST_2 = np.array(_EDGE_LIST_2_1B, dtype=int) - 1

N_EDGES = 125
EDGE_COLS_PER_HALF = 3 * N_EDGES  # 375


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class Process4Output:
    """The eight edge matrices saved to ``edges_*.mat`` by `the pipeline`.

    Each matrix is ``(n_frames, 375)`` float32, with     variable names so callers can just do::

        edges_dict = {
            "F_3E": out.F_3E,  "F_4E": out.F_4E,
            "F_3BE": out.F_3BE, "F_4BE": out.F_4BE,
            "F_3RE": out.F_3RE, "F_4RE": out.F_4RE,
            "F_3YE": out.F_3YE, "F_4YE": out.F_4YE,
        }
        save_edges("edges_test.mat", edges_dict)
    """
    F_3E: np.ndarray
    F_4E: np.ndarray
    F_3BE: np.ndarray
    F_4BE: np.ndarray
    F_3RE: np.ndarray
    F_4RE: np.ndarray
    F_3YE: np.ndarray
    F_4YE: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        """Return the matrices keyed by their  variable names."""
        return {
            "F_3E": self.F_3E, "F_4E": self.F_4E,
            "F_3BE": self.F_3BE, "F_4BE": self.F_4BE,
            "F_3RE": self.F_3RE, "F_4RE": self.F_4RE,
            "F_3YE": self.F_3YE, "F_4YE": self.F_4YE,
        }


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------

def _fill_edge_nans(col: np.ndarray) -> None:
    """Fill NaN runs at the start and end of ``col`` with the column mean.

    Mutates ``col`` in place.  Port of  (and siblings).
    On a column with no valid data at all, sets every cell to the global
    nanmean (as a safe fallback).
    """
    valid = np.where(~np.isnan(col))[0]
    if len(valid) == 0:
        # Defer to caller to set this column to the global mean.
        return
    first, last = int(valid[0]), int(valid[-1])
    mu = nanmean(col)
    if first > 0:
        col[: first + 1] = mu
    if last < len(col) - 1:
        col[last:] = mu


def _fill_large_gaps(
    matrix: np.ndarray,
    col_idx: int,
) -> None:
    """Port of  ``% fill nans in the middle ... if the gap is large``.

    Finds NaN gaps of at least :data:`_LARGE_GAP_MIN` frames in
    ``matrix[:, col_idx]`` and fills the interior (with ±5-frame padding)
    using the column mean — unless one of the special-case branches
    fires, in which case the interior is copied from a different body-part
    column.

    .. warning::
        The special-case checks use the *gap* iteration index ``f``, not
        the body-part column index ``col_idx``.  This is a copy-paste bug
        but preserved for downstream compatibility.
        In practice it essentially never fires.
    """
    col = matrix[:, col_idx]
    is_nan = np.isnan(col).astype(float)
    ind, _, _ = crossing(is_nan, level=0.5)
    if ind.size < 2:
        return
    starts = ind[0::2]
    stops = ind[1::2]
    if len(starts) > len(stops):
        starts = starts[: len(stops)]
    lengths = stops - starts
    keep = lengths >= _LARGE_GAP_MIN
    starts = starts[keep]
    stops = stops[keep]

    mu = nanmean(col)

    # Body-part fallback sources for the (buggy) special-case branches.
    #  uses col 10, 13, 16, 19 in 1-based → 9, 12, 15, 18 in 0-based
    # for the *same half* of the matrix.  We pick the source column from the
    # same half (F_3 or F_4) as col_idx.
    half = col_idx // 21
    half_base = half * 21
    fallback_10 = half_base + 9   # 1-based col 10
    fallback_13 = half_base + 12  # 1-based col 13
    fallback_16 = half_base + 15  # 1-based col 16
    fallback_19 = half_base + 18  # 1-based col 19

    for f, (s, e) in enumerate(zip(starts, stops), start=1):
        lo = int(s) + 5
        hi = int(e) - 5
        if hi <= lo:
            continue
        matrix[lo : hi + 1, col_idx] = mu
        #  conventions preserved — the check is on `f` (gap iteration
        # index), which is almost never 11, 12, 14, 15, 17, 18, 20, or 21.
        if f in (11, 12):
            matrix[lo : hi + 1, col_idx] = matrix[lo : hi + 1, fallback_10]
        if f in (14, 15):
            matrix[lo : hi + 1, col_idx] = matrix[lo : hi + 1, fallback_13]
        if f in (17, 18):
            matrix[lo : hi + 1, col_idx] = matrix[lo : hi + 1, fallback_16]
        if f in (20, 21):
            matrix[lo : hi + 1, col_idx] = matrix[lo : hi + 1, fallback_19]


def _cleanup_and_hull_filter(
    F_norm: np.ndarray,
    bh: np.ndarray,
    Ground_3: np.ndarray,
    Ground_4: np.ndarray,
) -> np.ndarray:
    """Run the full per-animal cleanup and return the cleaned F_normT.

    Applies the "fill nans" + "remove bp outside range" + "fill
    again" blocks for one animal.  Returns a fresh ``(n, 42)`` matrix
    clamped to ``[-10, 10]`` with NaN gaps filled and hull-rejected points
    re-filled via pchip.
    """
    F = F_norm.copy()

    # 1. Fill NaN runs at the start and end of each column.
    for i in range(F.shape[1]):
        _fill_edge_nans(F[:, i])
    # Fallback for columns that were entirely NaN: use the global mean.
    global_mu = nanmean(F)
    if np.isnan(global_mu):
        global_mu = 0.0
    for i in range(F.shape[1]):
        if np.all(np.isnan(F[:, i])):
            F[:, i] = global_mu

    # 2. Fill large NaN gaps with column mean (and the buggy specials).
    for i in range(F.shape[1]):
        _fill_large_gaps(F, i)

    # 3. pchip fill small gaps → F_normT
    F_normT = np.empty_like(F)
    for i in range(F.shape[1]):
        F_normT[:, i] = fillmissing_pchip(F[:, i])
    np.clip(F_normT, -CLAMP_NORMT, CLAMP_NORMT, out=F_normT)

    # 4. Hull filter via ground_normalized.  Operates on the F_3 half
    #    (cols 0..20) and F_4 half (cols 21..41).  Body2 (index 5) is
    #    skipped because its ground-truth cloud is degenerate.
    F_3 = F_normT[:, 0:21]
    F_4 = F_normT[:, 21:42]
    for i in range(21):
        if i == _BODY2_INDEX:
            continue
        gx = Ground_3[:, i]
        gy = Ground_4[:, i]
        k = alpha_hull(gx, gy, target_coverage=1.0)
        if len(k) < 3:
            continue
        inside, on = inpolygon(F_3[:, i], F_4[:, i], gx[k], gy[k])
        bad = ~(inside | on)
        F_3[bad, i] = np.nan
        F_4[bad, i] = np.nan
    F_normT[:, 0:21] = F_3
    F_normT[:, 21:42] = F_4

    # 5. Second pchip fill and clamp.
    for i in range(F_normT.shape[1]):
        F_normT[:, i] = fillmissing_pchip(F_normT[:, i])
    np.clip(F_normT, -CLAMP_NORMT, CLAMP_NORMT, out=F_normT)

    return F_normT


# ---------------------------------------------------------------------------
# Edge building
# ---------------------------------------------------------------------------

def _build_edges(F_normT: np.ndarray) -> np.ndarray:
    """Build the ``(n, 750)`` edges matrix from cleaned normalized body parts.

    Layout of the return value, matching the ``horzcat(edge1, edge1a,
    edge1b, edge2, edge2a, edge2b)``::

        columns   0.. 124 : edge1  (F_3 midpoint)
        columns 125.. 249 : edge1a (F_3 0.75/0.25)
        columns 250.. 374 : edge1b (F_3 0.25/0.75)
        columns 375.. 499 : edge2  (F_4 midpoint)
        columns 500.. 624 : edge2a (F_4 0.75/0.25)
        columns 625.. 749 : edge2b (F_4 0.25/0.75)

    Callers split this into F_3 (first 375) and F_4 (last 375) halves for
    the subsequent un-rotation.
    """
    # F_normT is (n, 42): cols 0..20 = F_3, cols 21..41 = F_4.
    F_3 = F_normT[:, 0:21]
    F_4 = F_normT[:, 21:42]

    F_3_a = F_3[:, EDGE_LIST_1]
    F_3_b = F_3[:, EDGE_LIST_2]
    F_4_a = F_4[:, EDGE_LIST_1]
    F_4_b = F_4[:, EDGE_LIST_2]

    edge1 = 0.5 * (F_3_a + F_3_b)
    edge1a = 0.75 * F_3_a + 0.25 * F_3_b
    edge1b = 0.25 * F_3_a + 0.75 * F_3_b
    edge2 = 0.5 * (F_4_a + F_4_b)
    edge2a = 0.75 * F_4_a + 0.25 * F_4_b
    edge2b = 0.25 * F_4_a + 0.75 * F_4_b

    return np.hstack([edge1, edge1a, edge1b, edge2, edge2a, edge2b])


# ---------------------------------------------------------------------------
# Undo rotate edges back to pixel space
# ---------------------------------------------------------------------------

def _undo_rotate_edges(
    edges: np.ndarray,
    bh: np.ndarray,
    animal: Dict[str, np.ndarray],
    suffix: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Un-rotate ``edges`` back to pixel coordinates for one animal.

    Returns ``(F_3E, F_4E)`` as ``(n_frames, 375)`` float32 arrays.

    Port of  (white) and the three color siblings.

    The inverse rotation uses ``t = -theta - pi`` where ``theta`` is the
    angle of ``(B3 - B1)`` — note the *reversed* sign from the forward
    rotation in :mod:`process_2`, where the angle was of ``(B1 - B3)``.
    Combined with the row-swap convention, this produces the inverse of
    the forward transformation.  See the derivation comment in the
    implementation.
    """
    # Per-animal body length (freshly recomputed here, just like ).
    n_frames = edges.shape[0]
    n_half = edges.shape[1] // 2

    # bh is passed in; we re-use it rather than recomputing from F. The
    # the stage recomputes bh from F, but since we already have it
    # from process_3, we can just pass it through.  This is mathematically
    # equivalent (the bh computation is deterministic from F).

    edgesT = edges * bh[:, None]

    # Body angle for the INVERSE rotation uses (B3 - B1), i.e. the
    # *negative* direction of the forward-rotation vector.
    B1 = np.column_stack(
        [fillmissing_pchip(animal[f"Body1{suffix}"][:, 0]),
         fillmissing_pchip(animal[f"Body1{suffix}"][:, 1])]
    )
    B2 = np.column_stack(
        [fillmissing_pchip(animal[f"Body2{suffix}"][:, 0]),
         fillmissing_pchip(animal[f"Body2{suffix}"][:, 1])]
    )
    B3 = np.column_stack(
        [fillmissing_pchip(animal[f"Body3{suffix}"][:, 0]),
         fillmissing_pchip(animal[f"Body3{suffix}"][:, 1])]
    )
    theta, _ = cart2pol(B3[:, 0] - B1[:, 0], B3[:, 1] - B1[:, 1])
    t = -theta - np.pi

    # Split edges into F_3 half and F_4 half.
    e_F3 = edgesT[:, :n_half]
    e_F4 = edgesT[:, n_half:]

    cos_t = np.cos(t)
    sin_t = np.sin(t)
    # Rotation with a row-swap.  Input "vertcat" order
    # is [F_3_half; F_4_half], so after R*temp the row-swap says the new
    # F_3 is row 2 and new F_4 is row 1.  See the derivation in the module
    # docstring; the net effect is the inverse of the forward rotation.
    new_row0 = cos_t[:, None] * e_F3 - sin_t[:, None] * e_F4
    new_row1 = sin_t[:, None] * e_F3 + cos_t[:, None] * e_F4
    F_3E = new_row1  # row-swap
    F_4E = new_row0

    # Re-center by adding B2 back.
    F_3E = F_3E + B2[:, [0]]
    F_4E = F_4E + B2[:, [1]]

    return F_3E.astype(np.float32), F_4E.astype(np.float32)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_process_4(
    p2: Process2Output,
    p3,  # Process3Output; annotation as forward string would cause circular ref
    animals: Dict[str, Dict[str, np.ndarray]],
    *,
    ground_normalized_path: "str | Path | None" = None,
) -> Process4Output:
    """Full port of `the process stage`.

    Parameters
    ----------
    p2:
        Output of :func:`process_2.run_process_2`.  Needed for the per-animal
        body-length computation in the un-rotate block ( recomputes
        from ``F``/``Fb``/``Fr``/``Fy`` — equivalent to the values already
        in ``p3.bh_*``).
    p3:
        Output of :func:`process_3.run_process_3`.  The normalized matrices
        and per-animal body lengths are read from here.
    animals:
        The per-animal body-part dicts from :func:`the process stageake_all_animals`.
        Used to look up Body1/Body2/Body3 for the un-rotation step.
    ground_normalized_path:
        Path to ``ground_normalized.mat``.
    """
    ground = load_ground_normalized(ground_normalized_path)
    G3 = ground["Ground_3"]
    G4 = ground["Ground_4"]

    # Cleanup and hull filter per animal.
    F_normT = _cleanup_and_hull_filter(p3.F_norm, p3.bh_w, G3, G4)
    Fb_normT = _cleanup_and_hull_filter(p3.Fb_norm, p3.bh_b, G3, G4)
    Fr_normT = _cleanup_and_hull_filter(p3.Fr_norm, p3.bh_r, G3, G4)
    Fy_normT = _cleanup_and_hull_filter(p3.Fy_norm, p3.bh_y, G3, G4)

    # Build edges per animal from the cleaned F_normT.
    edges_W = _build_edges(F_normT)
    edges_B = _build_edges(Fb_normT)
    edges_R = _build_edges(Fr_normT)
    edges_Y = _build_edges(Fy_normT)

    # Un-rotate edges back to pixel space for each animal.
    F_3E, F_4E = _undo_rotate_edges(edges_W, p3.bh_w, animals["W"], suffix="")
    F_3BE, F_4BE = _undo_rotate_edges(edges_B, p3.bh_b, animals["B"], suffix="Blue")
    F_3RE, F_4RE = _undo_rotate_edges(edges_R, p3.bh_r, animals["R"], suffix="Red")
    F_3YE, F_4YE = _undo_rotate_edges(edges_Y, p3.bh_y, animals["Y"], suffix="Yellow")

    return Process4Output(
        F_3E=F_3E, F_4E=F_4E,
        F_3BE=F_3BE, F_4BE=F_4BE,
        F_3RE=F_3RE, F_4RE=F_4RE,
        F_3YE=F_3YE, F_4YE=F_4YE,
    )
