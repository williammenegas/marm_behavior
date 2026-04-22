"""
extract_3
=========

Computes.

Stage-1 body-part cleanup.  For each animal (white, blue, red, yellow):

1. Compute the body angle from Body1→Body3.
2. Center every body part on Body2.
3. Rotate the centered cloud into the animal's own frame.
4. Normalize to a per-frame body length estimate (``bh``) obtained by
   combining three body-segment distances, smoothed with ``movmedian`` and
   clamped to ``[4, 40]``.
5. Load the per-body-part training hulls from ``ground_normalized.mat``
   and NaN out any body part whose normalized position falls outside the
   hull.  Body2 (index 5 in 0-based) is skipped because the ground data is
   degenerate at the origin.
6. Fill small gaps using the same ``crossing`` + ``movmedian`` + ``pchip``
   pattern used by :func:`extract_1._fill_small_gaps`, but with the larger
   thresholds from `the extract stage` (``time < 60``, ``dist < 120``).
7. Clamp all coordinates to ``[0, 1280]``.

The stage wraps each of the four per-animal blocks in a ``try/catch``
that silently swallows failures. This port uses ``try/except Exception`` in
the same places for parity; failures are logged to stderr rather than hidden.

Compatibility note
------------------
the ``boundary()`` is replaced by
:func:`marm_behavior.numerics.hull.alpha_hull`, which is a
Delaunay-based alpha shape calibrated to enclose ``target_coverage`` of the
training points. With ``target_coverage=0.999`` this produces a hull that
encloses 99.6-100% of ground_normalized points across all 21 body parts
(verified). It uses the alpha-hull builder.
``boundary(..., 0.5)``; see ``STATUS.md`` for the full caveat.
"""

from __future__ import annotations

import sys
from pathlib import Path

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


# Index of Body1/Body2/Body3 in the 0-based body-part ordering.
# These are used for the ROTATION ANGLE and the centering anchor.
BODY1_INDEX = 4
BODY2_INDEX = 5  # degenerate (all zeros) in ground_normalized — skipped
BODY3_INDEX = 6

# Indices of the head triangle Left/Right/Middle used by the extract_3
# for the body-length estimate.  Critical: the extract stage uses these
# head-triangle distances, not the Body1/Body2/Body3 distances, because the
# head triangle is more stable than the body axis.
LEFT_INDEX = 1
RIGHT_INDEX = 2
MIDDLE_INDEX = 3

#  ``extract_3`` thresholds for the small-gap fill and body-length
# normalization.
BODY_LEN_WIN = 10
BODY_LEN_MIN = 4.0
BODY_LEN_MAX = 40.0
GAP_TIME_LIMIT = 60
GAP_DIST_LIMIT = 120.0
COORD_MIN = 0.0
COORD_MAX = 1280.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _body_length(F_3: np.ndarray, F_4: np.ndarray, *, window: int = BODY_LEN_WIN) -> np.ndarray:
    """Per-frame body length in the rotated frame, smoothed and forward-filled.

    Computes ``bh = (bl_1 + bl_2 + bl_3)/3`` from
    `the extract stage` lines 55-58.  **Important:**  uses the
    Left/Right/Middle head-triangle distances here, NOT the Body1/Body2/Body3
    body-axis distances.  The head triangle is more stable than the body
    axis, so it produces a more reliable per-frame length estimate.
    """
    # Distances among the head triangle vertices, in the rotated frame.
    bl_1 = (
        np.sqrt(
            (F_3[:, LEFT_INDEX] - F_3[:, RIGHT_INDEX]) ** 2
            + (F_4[:, LEFT_INDEX] - F_4[:, RIGHT_INDEX]) ** 2
        )
        / 2.0
    )
    bl_2 = np.sqrt(
        (F_3[:, LEFT_INDEX] - F_3[:, MIDDLE_INDEX]) ** 2
        + (F_4[:, LEFT_INDEX] - F_4[:, MIDDLE_INDEX]) ** 2
    )
    bl_3 = np.sqrt(
        (F_3[:, RIGHT_INDEX] - F_3[:, MIDDLE_INDEX]) ** 2
        + (F_4[:, RIGHT_INDEX] - F_4[:, MIDDLE_INDEX]) ** 2
    )
    bh = (bl_1 + bl_2 + bl_3) / 3.0

    bh = movmedian(bh, window)
    bh[bh < BODY_LEN_MIN] = np.nan
    bh[bh > BODY_LEN_MAX] = np.nan

    if np.isnan(bh[0]):
        mean_val = nanmean(bh)
        bh[0] = 20.0 if np.isnan(mean_val) else float(mean_val)
    for ii in range(1, len(bh)):
        if np.isnan(bh[ii]):
            bh[ii] = bh[ii - 1]
    return bh


def _rotate_body_parts(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the normalized egocentric (F_3_norm, F_4_norm) for an animal.

    Replicates the "Find body angles ... centered ... Rotate ... normalize
    to head" block of `the extract stage` for one animal.
    """
    x = matrix[:, 0::2]  # (n, 21)
    y = matrix[:, 1::2]  # (n, 21)

    b1_x = fillmissing_pchip(x[:, BODY1_INDEX])
    b1_y = fillmissing_pchip(y[:, BODY1_INDEX])
    b2_x = fillmissing_pchip(x[:, BODY2_INDEX])
    b2_y = fillmissing_pchip(y[:, BODY2_INDEX])
    b3_x = fillmissing_pchip(x[:, BODY3_INDEX])
    b3_y = fillmissing_pchip(y[:, BODY3_INDEX])

    theta, _ = cart2pol(b1_x - b3_x, b1_y - b3_y)

    F_3 = x - b2_x[:, None]
    F_4 = y - b2_y[:, None]

    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    new_row0 = cos_t[:, None] * F_3 - sin_t[:, None] * F_4
    new_row1 = sin_t[:, None] * F_3 + cos_t[:, None] * F_4
    F_3_rot = new_row1  # row-swap convention; see postures.compute_egocentric
    F_4_rot = new_row0

    bh = _body_length(F_3_rot, F_4_rot)
    return F_3_rot / bh[:, None], F_4_rot / bh[:, None]


def _apply_hull_mask(
    matrix: np.ndarray,
    F_3_norm: np.ndarray,
    F_4_norm: np.ndarray,
    *,
    Ground_3: np.ndarray,
    Ground_4: np.ndarray,
    target_coverage: float,
) -> None:
    """NaN out body parts whose normalized position is outside the ground hull.

    Mutates ``matrix`` in place.  Body2 is skipped because the ground truth
    for that body part is degenerate at the origin.
    """
    for i in range(21):
        if i == BODY2_INDEX:
            continue
        gx = Ground_3[:, i]
        gy = Ground_4[:, i]
        k = alpha_hull(gx, gy, target_coverage=target_coverage)
        if len(k) < 3:
            continue
        inside, on = inpolygon(F_3_norm[:, i], F_4_norm[:, i], gx[k], gy[k])
        bad = ~(inside | on)
        matrix[bad, 2 * i] = np.nan
        matrix[bad, 2 * i + 1] = np.nan


def _fill_small_gaps_42(matrix: np.ndarray) -> None:
    """Small-gap fill on a 42-column body-part matrix.

    Generalisation of :func:`extract_1._fill_small_gaps` — same pattern,
    larger thresholds (``time < 60``, ``dist < 120``).
    """
    n = matrix.shape[0]
    for i in range(0, 42, 2):
        temp_r_1 = matrix[:, i : i + 2].copy()
        temp_fill = np.column_stack(
            [fillmissing_pchip(matrix[:, i]), fillmissing_pchip(matrix[:, i + 1])]
        )
        valid = (~np.isnan(matrix[:, i])).astype(float)
        valid[:2] = 1.0
        valid[-2:] = 1.0
        ind, _, _ = crossing(valid, level=0.5)
        if ind.size < 2:
            continue
        starts = ind[0::2]
        ends = ind[1::2]
        if len(starts) > len(ends):
            starts = starts[: len(ends)]
        times = ends - starts
        safe_starts = np.clip(starts - 1, 0, n - 1)
        safe_ends = np.clip(ends + 1, 0, n - 1)
        dists = np.sqrt(
            (temp_r_1[safe_ends, 0] - temp_r_1[safe_starts, 0]) ** 2
            + (temp_r_1[safe_ends, 1] - temp_r_1[safe_starts, 1]) ** 2
        )
        for s, e, t, d in zip(starts, ends, times, dists):
            if t < GAP_TIME_LIMIT and d < GAP_DIST_LIMIT:
                # MATLAB only fills column i (x), not i+1 (y), because
                # ``temp_fill(start:end)`` uses linear indexing on an
                # n×2 matrix and only returns column-1 values.
                matrix[s : e + 1, i] = temp_fill[s : e + 1, 0]


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_extract_3(
    *,
    Red: np.ndarray,
    White: np.ndarray,
    Blue: np.ndarray,
    Yellow: np.ndarray,
    ground_normalized_path: "str | Path | None" = None,
    target_coverage: float = 0.999,
) -> dict[str, np.ndarray]:
    """Full port of `the extract stage`.

    Parameters
    ----------
    Red, White, Blue, Yellow:
        42-column body-part matrices from ``extract_2``. Mutated in place
        and returned in the result dict.
    ground_normalized_path:
        Optional path to ``ground_normalized.mat``. If ``None`` (the
        default), the bundled copy shipped in
        ``marm_behavior/data/ground_normalized.npz`` is used.
    target_coverage:
        Fraction of the training points that the computed hull must enclose.
        Default 0.999.

    Returns
    -------
    dict[str, np.ndarray]
        ``{"Red": ..., "White": ..., "Blue": ..., "Yellow": ...}``
    """
    ground = load_ground_normalized(ground_normalized_path)
    G3 = ground["Ground_3"]
    G4 = ground["Ground_4"]

    for name, matrix in (("White", White), ("Blue", Blue), ("Red", Red), ("Yellow", Yellow)):
        try:
            F_3_norm, F_4_norm = _rotate_body_parts(matrix)
            _apply_hull_mask(
                matrix, F_3_norm, F_4_norm,
                Ground_3=G3, Ground_4=G4, target_coverage=target_coverage,
            )
        except Exception as err:
            print(
                f"[extract_3] {name}: hull filter failed ({err}); leaving as-is",
                file=sys.stderr,
            )

    for matrix in (White, Blue, Red, Yellow):
        for i in range(42):
            matrix[:, i] = movmedian(matrix[:, i], BODY_LEN_WIN)
        matrix[matrix == 0] = np.nan

    for matrix in (White, Blue, Red, Yellow):
        _fill_small_gaps_42(matrix)

    for matrix in (White, Blue, Red, Yellow):
        matrix[matrix < COORD_MIN] = COORD_MIN
        matrix[matrix > COORD_MAX] = COORD_MAX

    return {"Red": Red, "White": White, "Blue": Blue, "Yellow": Yellow}
