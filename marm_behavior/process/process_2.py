"""
process_2
=========

Computes.

Takes the four per-animal body-part dicts produced by :mod:`process_1` and
computes the egocentric / relative / movement matrices used downstream by
:mod:`process_3` and :mod:`process_4`:

* **F / Fb / Fr / Fy** — each animal's 21 body parts centered on its own Body2
  anchor and rotated into its own egocentric frame, concatenated as
  ``[F_3 | F_4]`` into an ``(n, 42)`` matrix.
* **FbR / FrR / FyR** — the "other" animals' body parts centered on the
  *white* animal's Body2 and rotated into white's frame.  These are the
  "how does blue/red/yellow look from white's point of view" matrices.
* **F1 / F1b / F1r / F1y** — frame-to-frame movement vectors for each
  animal's body parts, rotated into that animal's own frame.
* **Fb_rel / Fr_rel / Fy_rel** — per-frame scalar "relative-to-white
  motion" traces.  Note: the original formula   `the process stage` uses the x component of B2 twice instead of x and
  y — this is preserved here (see ``_rel_motion`` for details).

The row-swap convention from ``compute_egocentric`` in
:mod:`process.postures` — that ``F_3`` in the output holds the rotated y
coordinate and ``F_4`` holds the rotated x — is reproduced in every block
for bit-identical parity with .
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from ..numerics.helpers import cart2pol, fillmissing_pchip
from .process_1 import BODY_PART_ORDER


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class Process2Output:
    """All the intermediate matrices produced by :func:`run_process_2`.

    The attribute names mirror conventional variable names so that a reader
    flipping between `the process stage` and this port can map between
    them without translation.

    Egocentric (``F``, ``Fb``, ``Fr``, ``Fy``) and relative-to-white
    (``FbR``, ``FrR``, ``FyR``) matrices are ``(n_frames, 42)`` with x
    coordinates in columns 0..20 and y coordinates in columns 21..41.  (Note:
    "x" here is the rotated-y and "y" is the rotated-x, per the row-swap
    convention.)

    ``theta_*`` and ``B2_*`` are stored because :mod:`process_4` needs them
    to undo the rotation and translation when reconstructing pixel-space
    coordinates for the final ``edges_*.mat`` file.
    """

    # Egocentric
    F: np.ndarray
    Fb: np.ndarray
    Fr: np.ndarray
    Fy: np.ndarray
    # Relative to white
    FbR: np.ndarray
    FrR: np.ndarray
    FyR: np.ndarray
    # Movement (frame-to-frame diff, own frame)
    F1: np.ndarray
    F1b: np.ndarray
    F1r: np.ndarray
    F1y: np.ndarray
    # Relative-to-white movement scalars
    Fb_rel: np.ndarray
    Fr_rel: np.ndarray
    Fy_rel: np.ndarray
    # Per-animal body angle and Body2 anchor (for process_4 un-rotation)
    theta_w: np.ndarray
    theta_b: np.ndarray
    theta_r: np.ndarray
    theta_y: np.ndarray
    B2_w: np.ndarray
    B2_b: np.ndarray
    B2_r: np.ndarray
    B2_y: np.ndarray


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _rotate_pair(
    xs: np.ndarray,
    ys: np.ndarray,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate ``(xs, ys)`` by ``-theta`` with a row-swap.

    Input shape: ``xs`` and ``ys`` are each ``(n_frames, n_cols)``.  Output
    ``(new_F3, new_F4)``: ``new_F3`` is the rotated y coordinate and
    ``new_F4`` is the rotated x coordinate, matching::

        F_3(i,:) = temp_new(2, :);  % from  parfor rotation
        F_4(i,:) = temp_new(1, :);
    """
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    new_row0 = cos_t[:, None] * xs - sin_t[:, None] * ys  # rotated x
    new_row1 = sin_t[:, None] * xs + cos_t[:, None] * ys  # rotated y
    return new_row1, new_row0  # (F_3, F_4)


def _body_parts_array(animal: Dict[str, np.ndarray], suffix: str) -> tuple[np.ndarray, np.ndarray]:
    """Return two ``(n, 21)`` matrices of x and y body-part coordinates.

    Columns are in :data:`BODY_PART_ORDER`.  Missing body parts (not
    present in the dict) raise ``KeyError``.
    """
    n_parts = len(BODY_PART_ORDER)
    first = animal[f"{BODY_PART_ORDER[0]}{suffix}"]
    n_frames = first.shape[0]
    xs = np.empty((n_frames, n_parts))
    ys = np.empty((n_frames, n_parts))
    for j, part in enumerate(BODY_PART_ORDER):
        arr = animal[f"{part}{suffix}"]
        xs[:, j] = arr[:, 0]
        ys[:, j] = arr[:, 1]
    return xs, ys


def _pchip_body_anchor(animal: Dict[str, np.ndarray], part: str, suffix: str) -> np.ndarray:
    """Return the (n, 2) pchip-filled anchor for ``part``.

    Matches the pattern::

        B1a = fillmissing(W.Body1(:,1),'pchip');
        B1b = fillmissing(W.Body1(:,2),'pchip');
        B1 = horzcat(B1a, B1b);
    """
    arr = animal[f"{part}{suffix}"]
    return np.column_stack(
        [fillmissing_pchip(arr[:, 0]), fillmissing_pchip(arr[:, 1])]
    )


# ---------------------------------------------------------------------------
# Per-animal egocentric + movement computations
# ---------------------------------------------------------------------------

def _egocentric_in_own_frame(
    animal: Dict[str, np.ndarray],
    suffix: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the (n, 42) egocentric matrix F for one animal in its own frame.

    Returns ``(F, theta, B2)``.  ``F`` is ``np.hstack([F_3, F_4])``.
    """
    B1 = _pchip_body_anchor(animal, "Body1", suffix)
    B2 = _pchip_body_anchor(animal, "Body2", suffix)
    B3 = _pchip_body_anchor(animal, "Body3", suffix)
    theta, _ = cart2pol(B1[:, 0] - B3[:, 0], B1[:, 1] - B3[:, 1])

    xs, ys = _body_parts_array(animal, suffix)
    xs_c = xs - B2[:, [0]]
    ys_c = ys - B2[:, [1]]
    F_3, F_4 = _rotate_pair(xs_c, ys_c, theta)
    F = np.hstack([F_3, F_4])
    return F, theta, B2


def _egocentric_in_frame_of(
    animal: Dict[str, np.ndarray],
    suffix: str,
    anchor_B2: np.ndarray,
    anchor_theta: np.ndarray,
) -> np.ndarray:
    """Compute the ``(n, 42)`` "other animal in white's frame" matrix.

    ``anchor_B2`` and ``anchor_theta`` are the *white* animal's Body2 and
    body angle.  The other animal's body parts are centered on white's B2
    and rotated by ``-white_theta``.
    """
    xs, ys = _body_parts_array(animal, suffix)
    xs_c = xs - anchor_B2[:, [0]]
    ys_c = ys - anchor_B2[:, [1]]
    F_3, F_4 = _rotate_pair(xs_c, ys_c, anchor_theta)
    return np.hstack([F_3, F_4])


def _frame_diff(animal: Dict[str, np.ndarray], suffix: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(F1_3, F1_4)`` where ``F1_3[i] = x[i+1] - x[i]``.

    The pipeline uses the raw (un-pchip-filled) body parts for this
    diff; the last row is padded with zeros.
    """
    xs, ys = _body_parts_array(animal, suffix)
    n = xs.shape[0]
    F1_3 = np.zeros_like(xs)
    F1_4 = np.zeros_like(ys)
    F1_3[:-1] = xs[1:] - xs[:-1]
    F1_4[:-1] = ys[1:] - ys[:-1]
    return F1_3, F1_4


def _rel_motion(B2_other: np.ndarray, B2_white: np.ndarray) -> np.ndarray:
    """``Fb_rel``/``Fr_rel``/``Fy_rel`` block .

    **Warning —  convention preserved.**  The formula is::

        Fb_rel(i) = sqrt((B2Blue(i,1)-B2(i-1,1))^2 + (B2Blue(i,1)-B2(i-1,1))^2)
                  - sqrt((B2Blue(i-1,1)-B2(i-1,1))^2 + (B2Blue(i-1,1)-B2(i-1,1))^2);

    Both terms in each ``sqrt`` use ``(..., 1)`` — the x coordinate — twice.
    This looks like a copy-paste bug (the second term should almost certainly
    be ``(..., 2)`` for the y coordinate), but the downstream deep-network
    features were trained with this value so we preserve it to keep
    Python output bit-compatible with .
    """
    n = B2_other.shape[0]
    out = np.zeros((n,))
    for i in range(1, n):
        temp1 = np.sqrt(
            (B2_other[i - 1, 0] - B2_white[i - 1, 0]) ** 2
            + (B2_other[i - 1, 0] - B2_white[i - 1, 0]) ** 2
        )
        temp2 = np.sqrt(
            (B2_other[i, 0] - B2_white[i - 1, 0]) ** 2
            + (B2_other[i, 0] - B2_white[i - 1, 0]) ** 2
        )
        out[i] = temp2 - temp1
    return out


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_process_2(
    animals: Dict[str, Dict[str, np.ndarray]],
) -> Process2Output:
    """Full port of `the process stage`.

    Parameters
    ----------
    animals:
        Output of :func:`the process stageake_all_animals` — a dict
        ``{'W': ..., 'B': ..., 'R': ..., 'Y': ...}`` where each value is a
        dict mapping body-part names (with the appropriate suffix) to
        ``(n, 2)`` x/y arrays.

    Returns
    -------
    Process2Output
    """
    # Own-frame egocentric for each animal.
    F, theta_w, B2_w = _egocentric_in_own_frame(animals["W"], "")
    Fb, theta_b, B2_b = _egocentric_in_own_frame(animals["B"], "Blue")
    Fr, theta_r, B2_r = _egocentric_in_own_frame(animals["R"], "Red")
    Fy, theta_y, B2_y = _egocentric_in_own_frame(animals["Y"], "Yellow")

    # Relative-to-white egocentric for the three non-white animals.
    FbR = _egocentric_in_frame_of(animals["B"], "Blue", B2_w, theta_w)
    FrR = _egocentric_in_frame_of(animals["R"], "Red", B2_w, theta_w)
    FyR = _egocentric_in_frame_of(animals["Y"], "Yellow", B2_w, theta_w)

    # Frame-to-frame movement vectors, each rotated into its own animal frame.
    F1_3, F1_4 = _frame_diff(animals["W"], "")
    F1b_3, F1b_4 = _frame_diff(animals["B"], "Blue")
    F1r_3, F1r_4 = _frame_diff(animals["R"], "Red")
    F1y_3, F1y_4 = _frame_diff(animals["Y"], "Yellow")
    F1 = np.hstack(_rotate_pair(F1_3, F1_4, theta_w))
    F1b = np.hstack(_rotate_pair(F1b_3, F1b_4, theta_b))
    F1r = np.hstack(_rotate_pair(F1r_3, F1r_4, theta_r))
    F1y = np.hstack(_rotate_pair(F1y_3, F1y_4, theta_y))

    # Relative-to-white motion scalars (preserving the historical x-twice convention).
    Fb_rel = _rel_motion(B2_b, B2_w)
    Fr_rel = _rel_motion(B2_r, B2_w)
    Fy_rel = _rel_motion(B2_y, B2_w)

    return Process2Output(
        F=F, Fb=Fb, Fr=Fr, Fy=Fy,
        FbR=FbR, FrR=FrR, FyR=FyR,
        F1=F1, F1b=F1b, F1r=F1r, F1y=F1y,
        Fb_rel=Fb_rel, Fr_rel=Fr_rel, Fy_rel=Fy_rel,
        theta_w=theta_w, theta_b=theta_b, theta_r=theta_r, theta_y=theta_y,
        B2_w=B2_w, B2_b=B2_b, B2_r=B2_r, B2_y=B2_y,
    )
