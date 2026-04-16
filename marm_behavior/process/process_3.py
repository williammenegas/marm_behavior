"""
process_3
=========

Computes.

Takes the egocentric / relative / movement matrices from :mod:`process_2`
and:

1. Computes a per-animal body-length estimate ``bh`` from the head triangle
   (Left/Right/Middle distances), smoothed with a 200-frame ``movmedian``
   and clamped to ``[6, 60]`` pixels.
2. Divides each egocentric matrix by its animal's ``bh`` to get
   scale-invariant ``F_norm``, ``Fb_norm``, ``Fr_norm``, ``Fy_norm``.
3. Does the same for the relative-to-white matrices ``FbR``/``FrR``/``FyR``
   (using the *other* animal's ``bh`` in each case — a  convention preserved
   for parity).
4. Divides the movement and relative-motion scalars by ``bh``.
5. Clamps each normalized matrix to its own range (the ranges differ
   between variants — see constants at the top of this module).
6. "Flip if center on left" — for each frame where the other animal's Body2
   is at negative ``F_3`` in white's frame (i.e., on white's left), negates
   the F_3 half of the row and swaps the left/right-named body parts so the
   rendering comes out consistent.

The heavy lifting is reused across four colors by the :func:`_compute_bh`
helper; the eight normalization blocks collapse into
a dozen lines of clip-and-divide code here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..numerics.helpers import fillmissing_pchip, movmedian, nanmean
from .process_2 import Process2Output


#  body-length parameters for process_3 (differ from extract_3!).
BH_WINDOW = 200
BH_MIN = 6.0
BH_MAX = 60.0
BH_FALLBACK = 20.0

#: Constant body-length used for animals that are *not* the focal
#: animal in one-animal mode. Matches the one-animal logic where
#: absent animals get ``bh = zeros(size(bh)) + 30`` instead of the
#: variable per-frame clamp + forward-fill applied to the present
#: animal. This is intentionally different from :data:`BH_FALLBACK`
#: (which is the all-NaN fallback for the four-animal pipeline).
BH_ONE_ANIMAL_CONSTANT = 30.0

# Clip ranges per matrix type.
CLIP_F_NORM = 25.0            # F_norm / Fb_norm / Fr_norm / Fy_norm
CLIP_FR_NORM = 75.0           # FbR_norm / FrR_norm / FyR_norm
CLIP_F1_WHITE = 500.0         # F1_norm ( note: 500, not 25)
CLIP_F1_OTHER = 25.0          # F1b_norm / F1r_norm / F1y_norm
CLIP_REL = 20.0               # Fb_rel / Fr_rel / Fy_rel

# Head-triangle column indices (0-based) within the 42-wide F matrices.
# The F matrices are laid out as ``[F_3 (21 cols) | F_4 (21 cols)]`` and the
# head triangle body parts are Left=1, Right=2, Middle=3 in BODY_PART_ORDER.
_L_F3 = 1
_R_F3 = 2
_M_F3 = 3
_L_F4 = 1 + 21
_R_F4 = 2 + 21
_M_F4 = 3 + 21


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class Process3Output:
    """Normalized matrices produced by :func:`run_process_3`.

    Each matrix keeps the same shape as its :class:`Process2Output`
    counterpart but is now dimensionless (divided by per-frame body length)
    and clipped to the appropriate range.  ``bh_*`` arrays are kept around
    because :mod:`process_4` needs them for the final un-normalization that
    reconstructs the pixel-space edges matrices.
    """

    # Egocentric, normalized
    F_norm: np.ndarray
    Fb_norm: np.ndarray
    Fr_norm: np.ndarray
    Fy_norm: np.ndarray
    # Relative-to-white, normalized
    FbR_norm: np.ndarray
    FrR_norm: np.ndarray
    FyR_norm: np.ndarray
    # Movement, normalized
    F1_norm: np.ndarray
    F1b_norm: np.ndarray
    F1r_norm: np.ndarray
    F1y_norm: np.ndarray
    # Relative motion scalars, normalized
    Fb_rel: np.ndarray
    Fr_rel: np.ndarray
    Fy_rel: np.ndarray
    # Body length per animal (for process_4)
    bh_w: np.ndarray
    bh_b: np.ndarray
    bh_r: np.ndarray
    bh_y: np.ndarray


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_bh_constant(F: np.ndarray) -> np.ndarray:
    """Return a constant body-length vector for an animal in one-animal
    mode where this animal is *not* the focal one.

    Replicates the ``bh = zeros(size(bh)) + 30`` pattern the
    one-animal version uses for non-focal animals. Skipping the
    movmedian + clamp + forward-fill is both faster and (more
    importantly) avoids producing meaningless body-length estimates
    from F matrices that are all NaN because the colour isn't
    actually in the video.
    """
    n = F.shape[0]
    return np.full(n, BH_ONE_ANIMAL_CONSTANT)


def _compute_bh(F: np.ndarray) -> np.ndarray:
    """Per-frame body length ``bh`` from the head triangle of an F matrix.

    Replicates `the process stage` lines 6-16 (and their five copies for
    blue/red/yellow/*R variants).  Uses a 200-frame ``movmedian`` window with
    ``[6, 60]`` clamping and a forward-fill of any residual NaNs.  If the
    signal is entirely NaN (e.g. this animal never appears in the video),
    returns a constant ``BH_FALLBACK`` vector.
    """
    n = F.shape[0]
    bl_1 = np.sqrt(
        (F[:, _L_F3] - F[:, _R_F3]) ** 2 + (F[:, _L_F4] - F[:, _R_F4]) ** 2
    ) / 2.0
    bl_2 = np.sqrt(
        (F[:, _L_F3] - F[:, _M_F3]) ** 2 + (F[:, _L_F4] - F[:, _M_F4]) ** 2
    )
    bl_3 = np.sqrt(
        (F[:, _R_F3] - F[:, _M_F3]) ** 2 + (F[:, _R_F4] - F[:, _M_F4]) ** 2
    )
    bh = (bl_1 + bl_2 + bl_3) / 3.0

    bh = movmedian(bh, BH_WINDOW)
    bh[bh < BH_MIN] = np.nan
    bh[bh > BH_MAX] = np.nan

    if np.isnan(bh[0]):
        mean_val = nanmean(bh)
        bh[0] = BH_FALLBACK if np.isnan(mean_val) else float(mean_val)
    if np.all(np.isnan(bh)):
        bh = np.full(n, BH_FALLBACK)
    for ii in range(1, n):
        if np.isnan(bh[ii]):
            bh[ii] = bh[ii - 1]
    return bh


def _normalize_clip(F: np.ndarray, bh: np.ndarray, clip: float) -> np.ndarray:
    """Return ``F / bh`` clamped to ``[-clip, clip]``."""
    out = F / bh[:, None]
    np.clip(out, -clip, clip, out=out)
    return out


def _normalize_clip_fill(F: np.ndarray, bh: np.ndarray, clip: float) -> np.ndarray:
    """Normalize, clip, pchip-fill NaNs per column, clip again.

    Used for ``FbR_norm`` / ``FrR_norm`` / ``FyR_norm`` where the input
    original applies the clip-fill-clip pattern to fill NaN gaps introduced
    by the larger (±75) clip range.
    """
    out = _normalize_clip(F, bh, clip)
    for i in range(out.shape[1]):
        out[:, i] = fillmissing_pchip(out[:, i])
    np.clip(out, -clip, clip, out=out)
    return out


# ---------------------------------------------------------------------------
# "Flip if center on left" — port of the process stage lines 337-458
# ---------------------------------------------------------------------------

# Body-part index pairs (0-based) to swap after negating the F_3 half.
# These are the mirror pairs across the body axis.  In 1-based  the
# pairs are {2,3}, {8,9}, {10,13}, {11,14}, {12,15}, {16,19}, {17,20},
# {18,21}; subtracting one gives these.
_FLIP_SWAP_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 2),    # Left <-> Right
    (7, 8),    # LS <-> RS
    (9, 12),   # FL1 <-> FR1
    (10, 13),  # FL2 <-> FR2
    (11, 14),  # FL3 <-> FR3
    (15, 18),  # BL1 <-> BR1
    (16, 19),  # BL2 <-> BR2
    (17, 20),  # BL3 <-> BR3
)


def _flip_if_left(M: np.ndarray) -> None:
    """Mirror the F_3 half of rows where Body2's F_3 is negative.

    Mutates ``M`` in place.  ``M`` is a ``(n, 42)`` relative-to-white matrix
    like ``FbR_norm``.  For each row where ``M[i, 5]`` (Body2's F_3
    component) is negative, negates columns 0..20 (the F_3 half) and swaps
    the left/right body-part pairs within that half.

    The F_4 half (cols 21..41, the body-axis direction) is left unchanged:
    Left and Right body parts have the *same* F_4 value by symmetry, so
    swapping their labels is a no-op on that half.
    """
    mask = M[:, 5] < 0
    if not mask.any():
        return
    rows = np.where(mask)[0]
    # Negate the F_3 half.
    M[rows, 0:21] = -M[rows, 0:21]
    # Swap the mirror pairs within the F_3 half.
    for a, b in _FLIP_SWAP_PAIRS:
        tmp = M[rows, a].copy()
        M[rows, a] = M[rows, b]
        M[rows, b] = tmp


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_process_3(
    p2: Process2Output,
    *,
    focal_animal: "str | None" = None,
) -> Process3Output:
    """Run the body-length normalization stage.

    Parameters
    ----------
    p2:
        Output of :func:`process_2.run_process_2`.
    focal_animal:
        Optional one-animal-mode switch. When set to one of
        ``'w'`` / ``'b'`` / ``'r'`` / ``'y'``, only that animal gets
        the full per-frame body-length computation (movmedian + clamp
        + forward-fill). The other three animals get a constant
        ``bh = 30`` vector, matching the one-animal pipeline. When
        ``None`` (the default), all four animals get the full
        treatment, matching the four-animal pipeline.

    Returns
    -------
    Process3Output
    """
    # Per-animal body-length: use the full clamp/fallback path for any
    # animal that's either the focal one or in four-animal mode; for
    # the non-focal animals in one-animal mode, substitute the
    # constant.
    if focal_animal is None:
        bh_w = _compute_bh(p2.F)
        bh_b = _compute_bh(p2.Fb)
        bh_r = _compute_bh(p2.Fr)
        bh_y = _compute_bh(p2.Fy)
    else:
        if focal_animal not in ("w", "b", "r", "y"):
            raise ValueError(
                f"focal_animal must be one of 'w', 'b', 'r', 'y' "
                f"or None; got {focal_animal!r}"
            )
        bh_w = _compute_bh(p2.F) if focal_animal == "w" else _compute_bh_constant(p2.F)
        bh_b = _compute_bh(p2.Fb) if focal_animal == "b" else _compute_bh_constant(p2.Fb)
        bh_r = _compute_bh(p2.Fr) if focal_animal == "r" else _compute_bh_constant(p2.Fr)
        bh_y = _compute_bh(p2.Fy) if focal_animal == "y" else _compute_bh_constant(p2.Fy)

    # Egocentric normalization (±25).
    F_norm = _normalize_clip(p2.F, bh_w, CLIP_F_NORM)
    Fb_norm = _normalize_clip(p2.Fb, bh_b, CLIP_F_NORM)
    Fr_norm = _normalize_clip(p2.Fr, bh_r, CLIP_F_NORM)
    Fy_norm = _normalize_clip(p2.Fy, bh_y, CLIP_F_NORM)

    # Relative-to-white, normalized by the *other* animal's bh ( uses
    # bhBlueR / bhRedR / bhYellowR which it recomputes from FbR/FrR/FyR; the
    # effect is essentially the same as using the other animal's bh because
    # rotation and translation preserve distances).  We recompute from the
    # FbR/FrR/FyR matrices directly to match  .
    bh_bR = _compute_bh(p2.FbR)
    bh_rR = _compute_bh(p2.FrR)
    bh_yR = _compute_bh(p2.FyR)
    FbR_norm = _normalize_clip_fill(p2.FbR, bh_bR, CLIP_FR_NORM)
    FrR_norm = _normalize_clip_fill(p2.FrR, bh_rR, CLIP_FR_NORM)
    FyR_norm = _normalize_clip_fill(p2.FyR, bh_yR, CLIP_FR_NORM)

    # Movement normalization.  White's F1_norm uses the convention of
    # clipping to ±500 rather than ±25.
    F1_norm = _normalize_clip(p2.F1, bh_w, CLIP_F1_WHITE)
    F1b_norm = _normalize_clip(p2.F1b, bh_b, CLIP_F1_OTHER)
    F1r_norm = _normalize_clip(p2.F1r, bh_r, CLIP_F1_OTHER)
    F1y_norm = _normalize_clip(p2.F1y, bh_y, CLIP_F1_OTHER)

    # Relative-motion scalars, normalized to the *other* animal's bh.
    Fb_rel = np.clip(p2.Fb_rel / bh_b, -CLIP_REL, CLIP_REL)
    Fr_rel = np.clip(p2.Fr_rel / bh_r, -CLIP_REL, CLIP_REL)
    Fy_rel = np.clip(p2.Fy_rel / bh_y, -CLIP_REL, CLIP_REL)

    # Flip other animals that sit on white's left.
    _flip_if_left(FbR_norm)
    _flip_if_left(FrR_norm)
    _flip_if_left(FyR_norm)

    return Process3Output(
        F_norm=F_norm, Fb_norm=Fb_norm, Fr_norm=Fr_norm, Fy_norm=Fy_norm,
        FbR_norm=FbR_norm, FrR_norm=FrR_norm, FyR_norm=FyR_norm,
        F1_norm=F1_norm, F1b_norm=F1b_norm, F1r_norm=F1r_norm, F1y_norm=F1y_norm,
        Fb_rel=Fb_rel, Fr_rel=Fr_rel, Fy_rel=Fy_rel,
        bh_w=bh_w, bh_b=bh_b, bh_r=bh_r, bh_y=bh_y,
    )
