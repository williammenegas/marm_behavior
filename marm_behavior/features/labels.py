"""
features.labels
===============

Computes (and the thin Blue /
Red / Yellow wrapper variants, which just rename inputs before calling the
same logic).

Input
-----
Per animal the label script takes:

* ``White``, ``Blue``, ``Red``, ``Yellow`` — each ``(n_frames, 42)``, the
  ``horzcat(F_3X(:,105:125), F_4X(:,105:125))`` slice of the edge matrices
  produced by `the process stage`. Columns 1-21 are the x coordinates of
  the 21 body parts, columns 22-42 are the y coordinates.
* ``depths_White``, ``depths_Blue``, ``depths_Red``, ``depths_Yellow`` — each
  ``(n_frames, 375)``, the full depth array produced by
  `the depths stage`.
* ``F_3X``, ``F_4X`` for each animal — each ``(n_frames, 375)``, the two
  halves of the edge matrices. These are assembled into ``edges`` internally.

Output
------
``mid_layer`` — a binary ``(n_frames, 30)`` matrix. The column layout
(matching ``horzcat`` at the labels stage:2205-2218) is::

    0  : white_moving
    1  : white_moving_selfA
    2  : white_moving_toA
    3  : white_moving_stayA
    4  : white_moving_fromA
    5  : white_head_moving
    6  : white_look_selfA
    7  : white_look_toA
    8  : white_look_stayA
    9  : white_look_fromA
    10 : white_limb_moving
    11 : white_limb_mu
    12 : white_limb_FL
    13 : white_limb_FR
    14 : white_limb_BL
    15 : white_limb_BR
    16 : white_upright
    17 : white_upright_t1
    18 : white_upright_t2
    19 : white_upright_t3
    20 : other_nearA
    21 : other_nearA_t1_side
    22 : other_nearA_t2_att_att
    23 : other_nearA_t3_att_non
    24 : other_nearA_t4_non_att
    25 : other_nearA_t5_non_non
    26 : white_lookA
    27 : white_lookA_type1
    28 : white_lookA_type2
    29 : white_lookA_type3

This matrix gets written to ``w_description_<n>.csv`` by
`the pipeline`.

Stage 4 reuses ``fillmissing('pchip')``, ``movmedian``, ``movmean``,
``cart2pol`` and ``crossing`` from :mod:`compat.helpers`. It does not
use Wavelet Toolbox or any other exotic -only primitive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..numerics.helpers import (
    cart2pol,
    crossing,
    fillmissing_pchip,
    movmean,
    movmedian,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class AnimalInputs:
    """Per-animal inputs to the label script.

    Attributes
    ----------
    bp : ``(n_frames, 42)`` body-part matrix.  Cols ``[x1..x21, y1..y21]``.
    F_3 : ``(n_frames, 375)`` x-coordinate edge matrix (what  calls
        ``F_3E`` / ``F_3BE`` etc.).
    F_4 : ``(n_frames, 375)`` y-coordinate edge matrix.
    depths : ``(n_frames, 375)`` depth array from stage 3.
    """

    bp: np.ndarray
    F_3: np.ndarray
    F_4: np.ndarray
    depths: np.ndarray


# ---------------------------------------------------------------------------
# Section 1: Body position and angle (the labels stage:9-115)
# ---------------------------------------------------------------------------


def _body_pos_and_angle(animal: AnimalInputs) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """One of the per-animal blocks in the "Get body position and angle"
    .

    For each frame:

    1. Extract Body1, Body2, Body3 ``(n, 2)`` coordinates from columns
       ``5/6/7`` (+21 for the y side) with pchip fill.
    2. Compute the body angle ``theta = cart2pol(B3-B1)``.
    3. ``t = -theta - pi`` is the rotation that  applies.
    4. Translate the full ``edges = [F_3, F_4]`` matrix by ``-B2`` and rotate.
    5.  writes back ``horzcat(new_y, new_x)`` — rotated y row first,
       rotated x row second.  We keep that convention so the caller's column
       indexing matches the input indexing.

    Returns
    -------
    B1, B2, B3 : ``(n, 2)``
        pchip-filled body-part coordinates.
    theta : ``(n,)``
        Body angle from cart2pol.
    edges_rot : ``(n, 750)``
        Translated + rotated edges.  First 375 cols = new_y, last 375 =
        new_x.
    """
    bp = animal.bp
    n = bp.shape[0]

    # B1, B2, B3 in columns 5, 6, 7 (1-based) → 4, 5, 6 (0-based);
    # y-sides are at +21.
    B1 = np.column_stack([
        fillmissing_pchip(bp[:, 4]),
        fillmissing_pchip(bp[:, 4 + 21]),
    ])
    B2 = np.column_stack([
        fillmissing_pchip(bp[:, 5]),
        fillmissing_pchip(bp[:, 5 + 21]),
    ])
    B3 = np.column_stack([
        fillmissing_pchip(bp[:, 6]),
        fillmissing_pchip(bp[:, 6 + 21]),
    ])

    Ax = B3[:, 0] - B1[:, 0]
    Ay = B3[:, 1] - B1[:, 1]
    theta, _rho = cart2pol(Ax, Ay)
    t = -1.0 * theta - np.pi

    # Build the full (n, 750) edge buffer, translate by B2.
    ex = animal.F_3 - B2[:, 0:1]
    ey = animal.F_4 - B2[:, 1:2]

    ct = np.cos(t)
    st = np.sin(t)
    new_x = ct[:, None] * ex - st[:, None] * ey
    new_y = st[:, None] * ex + ct[:, None] * ey

    edges_rot = np.empty((n, 750), dtype=float)
    edges_rot[:, :375] = new_y   #  row 2
    edges_rot[:, 375:] = new_x   #  row 1
    return B1, B2, B3, theta, edges_rot


# ---------------------------------------------------------------------------
# Section 2: Head / body length (the labels stage:122-180)
# ---------------------------------------------------------------------------


def _head_radius(bp: np.ndarray) -> np.ndarray:
    """Port of the ``find head radius`` blocks.

    Combines three head-body segment lengths, smooths with
    ``movmedian(200)``, clamps to ``[6, 60]``, handles the edge cases and
    forward-fills remaining NaNs.  Returns a length-``n_frames`` vector
    ``bh``.
    """
    bl_1 = np.sqrt(
        (bp[:, 1] - bp[:, 2]) ** 2
        + (bp[:, 1 + 21] - bp[:, 2 + 21]) ** 2
    ) / 2.0
    bl_2 = np.sqrt(
        (bp[:, 1] - bp[:, 3]) ** 2
        + (bp[:, 1 + 21] - bp[:, 3 + 21]) ** 2
    )
    bl_3 = np.sqrt(
        (bp[:, 2] - bp[:, 3]) ** 2
        + (bp[:, 2 + 21] - bp[:, 3 + 21]) ** 2
    )
    bh = (bl_1 + bl_2 + bl_3) / 3.0

    bh = movmedian(bh, 200)
    bh[bh < 6] = np.nan
    bh[bh > 60] = np.nan

    if np.isnan(bh[0]):
        mean_val = np.nanmean(bh)
        if np.isnan(mean_val):
            mean_val = 20.0
        bh[0] = mean_val

    if np.all(np.isnan(bh)):
        return np.full_like(bh, 20.0)

    for i in range(1, len(bh)):
        if np.isnan(bh[i]):
            bh[i] = bh[i - 1]
    return bh


# ---------------------------------------------------------------------------
# Section 3: Sort depth data (the labels stage:188-265)
# ---------------------------------------------------------------------------


def _sort_depths(edges_rot: np.ndarray, depths: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray
]:
    """Port of the ``sort depths`` blocks.

    For each frame:

    1. Pair each of the 375 rotated-x edge samples with its depth value.
    2. Sort by edge-x ascending ( ``sortrows`` puts NaN keys last).
    3. Take every 18th sample → 21 samples.
    4. Fill NaNs with the nearest valid neighbour.

    Same procedure with edge-y. The two (n, 21) arrays are then centered
    on column 11 (1-based = 10 0-based) and have NaNs replaced with 0.
    """
    depths = np.where(depths == 0.0, np.nan, depths).astype(float)

    n = edges_rot.shape[0]
    edges_x = edges_rot[:, :375]
    edges_y = edges_rot[:, 375:]

    d_X = np.zeros((n, 21), dtype=float)
    d_Y = np.zeros((n, 21), dtype=float)

    for g in range(n):
        ex = edges_x[g]
        ey = edges_y[g]
        dep = depths[g]

        # sort-by-x.  numpy argsort places NaN at the end.
        order_x = np.argsort(ex, kind="stable")
        sorted_dep_x = dep[order_x]
        sub_x = sorted_dep_x[::18][:21]
        d_X[g] = _fill_nearest(sub_x)

        order_y = np.argsort(ey, kind="stable")
        sorted_dep_y = dep[order_y]
        sub_y = sorted_dep_y[::18][:21]
        d_Y[g] = _fill_nearest(sub_y)

    d_X = d_X - d_X[:, 10:11]
    d_Y = d_Y - d_Y[:, 10:11]
    d_X[np.isnan(d_X)] = 0.0
    d_Y[np.isnan(d_Y)] = 0.0
    return d_X, d_Y


def _fill_nearest(x: np.ndarray) -> np.ndarray:
    """Port of  ``fillmissing(x, 'nearest')`` for 1-D arrays.

    Replaces each NaN with the value of the nearest non-NaN sample,
    breaking ties by preferring the earlier one.
    """
    out = x.astype(float, copy=True)
    n = out.size
    if n == 0:
        return out
    valid_idx = np.flatnonzero(~np.isnan(out))
    if valid_idx.size == 0:
        return out
    if valid_idx.size == n:
        return out
    nan_idx = np.flatnonzero(np.isnan(out))
    for i in nan_idx:
        diffs = np.abs(valid_idx - i)
        j = valid_idx[int(np.argmin(diffs))]
        out[i] = out[j]
    return out


# ---------------------------------------------------------------------------
# Reusable helpers used across feature sections
# ---------------------------------------------------------------------------


def _rotate_to_anchor(
    other_bp: np.ndarray,
    white_bp: np.ndarray,
    pivot_col: int,
    anchor1_col: int,
    anchor2_col: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Translate ``other_bp`` by a white pivot and rotate into white's frame.

    Helper for the repeated pattern in
    `the labels stage`::

        d1 = Other(:,1:21) - White(:,pivot);
        d2 = Other(:,22:42) - White(:,pivot+21);
        Angles_X = White(:,anchor1) - White(:,anchor2);
        Angles_Y = White(:,anchor1+21) - White(:,anchor2+21);
        [theta, rho] = cart2pol(Angles_X, Angles_Y);
        t = -1 * theta;
        parfor i ...
            R = [cos(t) -sin(t); sin(t) cos(t)];
            temp = vertcat(d1, d2);
            temp_new = R * temp;
            d3(i,:) = temp_new(2,:);
            d4(i,:) = temp_new(1,:);

    The return triple is ``(d3, d4, theta)``.  Note that  stores
    ``temp_new(2,:)`` first and ``temp_new(1,:)`` second — that swap is
    Column indices are 0-based 
    is passed as ``4``).
    """
    d1 = other_bp[:, :21] - white_bp[:, pivot_col : pivot_col + 1]
    d2 = other_bp[:, 21:] - white_bp[:, pivot_col + 21 : pivot_col + 22]
    ax = white_bp[:, anchor1_col] - white_bp[:, anchor2_col]
    ay = white_bp[:, anchor1_col + 21] - white_bp[:, anchor2_col + 21]
    theta, _rho = cart2pol(ax, ay)
    t = -1.0 * theta
    ct = np.cos(t)[:, None]
    st = np.sin(t)[:, None]

    # new_row1 = ct*d1 - st*d2 = new_x
    # new_row2 = st*d1 + ct*d2 = new_y
    new_x = ct * d1 - st * d2
    new_y = st * d1 + ct * d2
    #  stores d3 = new_row2 (y), d4 = new_row1 (x).
    return new_y, new_x, theta


def _events_from_signal(
    signal: np.ndarray,
    level: float,
    *,
    iti_merge: int = 0,
    min_length: int = 0,
    area_reject: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Event segmentation that matches the pattern used throughout
    `the labels stage`.

    1. Force ``signal[0] = signal[-1] = 0``.
    2. Perturb exact-threshold samples (``signal == level``) slightly so
       ``crossing`` doesn't mis-handle tie cases ( does this with an
       explicit ``signal(signal==0.05)=0.05000001`` etc).
    3. Call :func:`crossing` at the given level.
    4. ``inds_start = ind[0::2]``, ``inds_stop = ind[1::2] + 1``.
    5. If ``iti_merge > 0``: whenever ``inds_iti(f) = inds_start(f+1) -
       inds_stop(f) < iti_merge``, drop the gap by NaN-ing ``inds_stop(f)``
       and ``inds_start(f+1)`` (then removing NaN entries).
    6. If ``min_length > 0``: drop any event where ``stop - start <
       min_length``.
    7. If ``area_reject`` is not None: drop any event where
       ``sum(signal[start:stop+1]) < area_reject``.

    Returns arrays of int start and stop indices (0-based, half-open like
    the inclusive-stop → Python's ``[start : stop + 1]``).
    """
    sig = np.asarray(signal, dtype=float).ravel().copy()
    sig[0] = 0.0
    sig[-1] = 0.0
    # Jitter exact-threshold samples in the same direction  does.
    eps = max(abs(level) * 1e-6, 1e-12)
    sig[sig == level] = level + eps

    ind, _, _ = crossing(sig, level=level)
    if ind.size < 2:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
        )
    starts = ind[0::2].astype(float)
    ends = ind[1::2].astype(float)
    #  does `inds_stop = inds_stop + 1`
    ends = ends + 1
    if starts.size > ends.size:
        starts = starts[: ends.size]

    # Merge-close events.
    if iti_merge > 0 and starts.size > 1:
        starts_nan = starts.astype(float).copy()
        ends_nan = ends.astype(float).copy()
        for f in range(starts.size - 1):
            iti = starts[f + 1] - ends[f]
            if iti < iti_merge:
                ends_nan[f] = np.nan
                starts_nan[f + 1] = np.nan
        starts = starts_nan[~np.isnan(starts_nan)]
        ends = ends_nan[~np.isnan(ends_nan)]

    # Length reject.
    if min_length > 0 and starts.size > 0:
        length = ends - starts
        keep = length >= min_length
        starts = starts[keep]
        ends = ends[keep]

    # Area reject.
    if area_reject is not None and starts.size > 0:
        keep = np.ones(starts.size, dtype=bool)
        for f in range(starts.size):
            s = int(starts[f])
            e = int(ends[f])
            area = float(np.sum(sig[s : e + 1]))
            if area < area_reject:
                keep[f] = False
        starts = starts[keep]
        ends = ends[keep]

    return starts.astype(int), ends.astype(int)


def _make_binary(n: int, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Utility: zero-filled length-``n`` vector with ``[start, end]``
    inclusive intervals set to 1."""
    out = np.zeros(n, dtype=float)
    for s, e in zip(starts, ends):
        out[int(s) : int(e) + 1] = 1.0
    return out


# ---------------------------------------------------------------------------
# Section 4: Near-other detection (the labels stage:272-354 etc)
# ---------------------------------------------------------------------------


def _near_other_detect(
    white_bp: np.ndarray,
    other_bp: np.ndarray,
    bh: np.ndarray,
    depths_white: np.ndarray,
    depths_other: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of the "Near other (blue/red/yellow)" per-animal section.

    Returns
    -------
    other_near : ``(n,)``
        Binary flag, 1 for frames where white and other are "near".
    inds_start, inds_stop : ``(k,)``
        Event start/stop indices used for type classification later.
    """
    n = white_bp.shape[0]

    # Vectorised per-frame 21x21 pairwise distance.  Shapes:
    #   w_x: (n, 21)  o_x: (n, 21)
    # We want d[f, i, j] = dist(w_xy[f, i], o_xy[f, j]) / bh[f]
    # Broadcast: w_xy[:, :, None] - o_xy[:, None, :]
    w_x = white_bp[:, :21]
    w_y = white_bp[:, 21:]
    o_x = other_bp[:, :21]
    o_y = other_bp[:, 21:]
    dx = w_x[:, :, None] - o_x[:, None, :]  # (n, 21, 21)
    dy = w_y[:, :, None] - o_y[:, None, :]
    with np.errstate(invalid="ignore"):
        dist = np.sqrt(dx * dx + dy * dy) / bh[:, None, None]
        # Per-frame min and mean, NaN-safe.
        running_min = np.nanmin(dist.reshape(n, -1), axis=1)
        running_mean = np.nanmean(dist.reshape(n, -1), axis=1)

    Near_other = np.zeros(n)
    Near_other[(running_min < 1.5) & (running_mean < 9)] = 1
    Near_other = movmedian(Near_other, 15)

    # Depth difference channel
    tempDW = movmean(np.nanmean(depths_white, axis=1), 30)
    tempDB = movmean(np.nanmean(depths_other, axis=1), 30)
    depDiff = np.abs(tempDW - tempDB)
    depDiff = _fill_nearest(depDiff)

    starts, ends = _events_from_signal(
        Near_other, level=0.5, iti_merge=22, min_length=22
    )

    # Remove events where mean depth diff > 12 (animals on different planes)
    keep = np.ones(starts.size, dtype=bool)
    for f in range(starts.size):
        s = int(starts[f])
        e = int(ends[f])
        if np.mean(depDiff[s : e + 1]) > 12:
            keep[f] = False
    starts = starts[keep]
    ends = ends[keep]

    other_near = _make_binary(n, starts, ends)
    return other_near, starts, ends


# ---------------------------------------------------------------------------
# Section 5: Near-other TYPE classification (the labels stage:356-496)
# ---------------------------------------------------------------------------


def _near_other_types(
    white_bp: np.ndarray,
    other_bp: np.ndarray,
    bh: np.ndarray,
    other_near: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Port of the "Near other types (blue/red/yellow)" per-animal section.

    Returns 5 binary vectors ``(t1_side, t2_att_att, t3_att_non, t4_non_att,
    t5_non_non)``.
    """
    n = white_bp.shape[0]

    # ---- Other animal POSITION (pivot = Body2 = col 6, anchor1/2 = B1/B3)
    d3, d4, _theta = _rotate_to_anchor(
        other_bp, white_bp, pivot_col=5, anchor1_col=4, anchor2_col=6
    )
    d3_abs = np.abs(d3) / bh[:, None]
    d4_scaled = d4 / bh[:, None]

    theta_pos, _rho = cart2pol(d3_abs, d4_scaled)

    #   atan2(mean(sind(theta_deg(:,5:9)),2),
    #         mean(cosd(theta_deg(:,5:9)),2)));
    # theta_rad = deg2rad(theta_deg);  (UNUSED for position — only used in
    # the "looking at" section; kept here because  computes it but
    # doesn't use it.)
    _ = theta_pos  # unused downstream in this block

    theta_col6 = theta_pos[:, 5]  #  col 6 → 0-based col 5

    # other_frontB = theta > 1 → 0, theta > 0 → 1 (the boundary case is a bit
    # tortured: it sets anything < 1 to 0 first, then anything > 0 to 1.
    # The net effect is:  1 iff theta >= 1.)
    other_front = np.zeros(n)
    other_front[theta_col6 >= 1] = 1

    other_behind = np.zeros(n)
    other_behind[theta_col6 <= -1] = 1

    other_side = np.zeros(n)
    other_side[np.abs(theta_col6) < 1] = 1

    # ---- Other animal ORIENTATION
    d3_o, d4_o, _ = _rotate_to_anchor(
        other_bp, white_bp, pivot_col=5, anchor1_col=4, anchor2_col=6
    )
    #  flips the sign of d3 row when d3(f,6) < 0.
    d3_o = d3_o.copy()
    d4_o = d4_o.copy()
    col6 = d3_o[:, 5]
    flip = col6 < 0
    d3_o[flip] = -1.0 * d3_o[flip]
    #  divides by bh in both branches.
    d3_o = d3_o / bh[:, None]
    d4_o = d4_o / bh[:, None]


    angles_x = d3_o[:, 4] - d3_o[:, 6]
    angles_y = d4_o[:, 4] - d4_o[:, 6]
    theta_ori, _ = cart2pol(angles_x, angles_y)

    other_same = np.zeros(n)
    other_same[(theta_ori > np.pi / 4) & (theta_ori < 3 * np.pi / 4)] = 1
    other_opp = np.zeros(n)
    other_opp[(theta_ori < -np.pi / 4) & (theta_ori > -3 * np.pi / 4)] = 1
    other_right = np.zeros(n)
    other_right[np.abs(theta_ori) < np.pi / 4] = 1
    other_left = np.zeros(n)
    other_left[np.abs(theta_ori) > 3 * np.pi / 4] = 1

    # ---- Classify per-frame into the 5 types
    t1 = np.zeros(n)
    t2 = np.zeros(n)
    t3 = np.zeros(n)
    t4 = np.zeros(n)
    t5 = np.zeros(n)
    for f in range(n):
        if other_near[f] != 1:
            continue
        if other_side[f] == 1 and other_same[f] == 1:
            t1[f] = 1
        if other_front[f] == 1 and other_opp[f] == 1:
            t2[f] = 1
        if other_front[f] == 1 and other_opp[f] != 1:
            t3[f] = 1
        if other_side[f] == 1 and other_left[f] == 1:
            t4[f] = 1
        if other_behind[f] == 1 and other_same[f] == 1:
            t4[f] = 1
        if other_side[f] == 1 and other_right[f] == 1:
            t5[f] = 1
        if other_side[f] == 1 and other_opp[f] == 1:
            t5[f] = 1
        if other_behind[f] == 1 and other_same[f] != 1:
            t5[f] = 1
    return t1, t2, t3, t4, t5


# ---------------------------------------------------------------------------
# Section 6: Moving (the labels stage:1047-1254)
# ---------------------------------------------------------------------------


def _moving_detect(
    white_bp: np.ndarray, bh: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of the ``Moving`` detection block.

    Returns ``(white_moving, inds_start, inds_stop)`` where
    ``white_moving`` is the binary per-frame flag.
    """
    n = white_bp.shape[0]
    # Vectorised per-frame 21-col displacement across sorted entries.
    dx = np.diff(white_bp[:, :21], axis=0)
    dy = np.diff(white_bp[:, 21:], axis=0)
    d = np.sqrt(dx * dx + dy * dy) / bh[1:, None]
    d_sorted = np.sort(d, axis=1)
    White_moving = np.zeros((n, 21))
    White_moving[1:] = d_sorted


    row_mean = np.nanmean(White_moving, axis=1)
    wm = movmean(row_mean, 10)
    starts, ends = _events_from_signal(wm, level=0.05, iti_merge=22, min_length=22)
    moving = _make_binary(n, starts, ends)
    return moving, starts, ends


def _movement_types_vs(
    inds_start: np.ndarray,
    inds_stop: np.ndarray,
    other_near: np.ndarray,
    n: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Port of the ``Movement types (re: X)`` blocks. Same logic as the
    ``(re: all other)`` block — here ``other_near`` is precomputed by the
    caller (either a single color or the union).

    Returns ``(self, to, stay, from)`` binary vectors. Note the argument
    order matches the input indexing ``self/to/stay/from`` grouping used inside the
    mid_layer horzcat — callers must be careful.

    Note: the window-building loop runs
    ``for f = 2:end-1`` so the very first and very last events are left
    with the zero-initialised temp_near_start/stop windows. That makes the
    ``mean(...) < 0.5`` condition trivially true for both ends → the first
    and last events always land in the ``self`` bin, regardless of what
    ``other_near`` actually is around them. Reproducing this exactly is
    essential to matching the CSV output.
    """
    m_self = np.zeros(n)
    m_to = np.zeros(n)
    m_stay = np.zeros(n)
    m_from = np.zeros(n)
    n_events = inds_start.size
    for f in range(n_events):
        s = int(inds_start[f])
        e = int(inds_stop[f])
        # the `for f = 2:end-1` means the very first and very last
        # events retain the zero-initialised windows.  Reproduce that.
        first_or_last = (f == 0) or (f == n_events - 1)
        if first_or_last or s - 2 < 0 or e + 2 >= n:
            start_win = np.zeros(5)
            stop_win = np.zeros(5)
        else:
            start_win = other_near[s - 2 : s + 3]
            stop_win = other_near[e - 2 : e + 3]
        mu_s = float(np.mean(start_win))
        mu_e = float(np.mean(stop_win))
        if mu_s < 0.5 and mu_e < 0.5:
            m_self[s : e + 1] = 1
        if mu_s < 0.5 and mu_e > 0.5:
            m_to[s : e + 1] = 1
        if mu_s > 0.5 and mu_e < 0.5:
            m_from[s : e + 1] = 1
        if mu_s > 0.5 and mu_e > 0.5:
            m_stay[s : e + 1] = 1
    return m_self, m_to, m_stay, m_from


# ---------------------------------------------------------------------------
# Section 7: Looking at other (the labels stage:1258-1676)
# ---------------------------------------------------------------------------


def _looking_at_other_detect(
    white_bp: np.ndarray, other_bp: np.ndarray, bh: np.ndarray
) -> np.ndarray:
    """Port of the per-animal "looking at other (X)" block.

    Returns a binary ``(n,)`` ``white_look`` vector.  The logic centers on
    ``cart2pol(abs(d3)/bh, d4/bh)``, then checks (a) per-frame max theta
    across 21 body parts > 1.44 and (b) the bh-weighted mean angle
    (``theta_rad``) over cols 5:9 is non-negative.
    """
    n = white_bp.shape[0]

    # col 4 (Middle).  Note:  writes White(:,1)-White(:,4) for the
    # anchor vector — i.e. Front - Middle.  We represent that as
    # anchor1=Front (0), anchor2=Middle (3) in 0-based.
    d3, d4, _ = _rotate_to_anchor(
        other_bp, white_bp, pivot_col=3, anchor1_col=0, anchor2_col=3
    )
    d3 = np.abs(d3) / bh[:, None]
    d4 = d4 / bh[:, None]

    theta, _rho = cart2pol(d3, d4)
    #  collapses cols 5:9 (1-based) → 4:9 (0-based exclusive) via
    # atan2(mean(sind), mean(cosd)). Use radians directly.
    cols = theta[:, 4:9]
    mean_rad = np.arctan2(np.mean(np.sin(cols), axis=1), np.mean(np.cos(cols), axis=1))

    look = np.zeros(n)
    look[np.max(theta, axis=1) > 1.44] = 1
    look[mean_rad < 0] = 0
    return look


def _looking_at_other_types(
    white_bp: np.ndarray, bh: np.ndarray, white_look_flag: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of "looking at other types (blue/red/yellow/all)".

    This is the block that reuses the *self* (white) body frame rotation
    — ``d1 = White(:,1:21) - White(:,6); d2 = White(:,22:42) - White(:,27);``
    and the same anchor ``White(:,5) - White(:,7)`` — then derives a single
    ``theta_pos`` per frame from ``cart2pol(d3(:,1)-d3(:,4), d4(:,1)-d4(:,4))``.

    Returns three binary vectors ``(type1, type2, type3)``.
    """
    n = white_bp.shape[0]
    # Self-rotation: Other := White here.
    d3, d4, _ = _rotate_to_anchor(
        white_bp, white_bp, pivot_col=5, anchor1_col=4, anchor2_col=6
    )
    d3 = d3 / bh[:, None]
    d4 = d4 / bh[:, None]
    ax = d3[:, 0] - d3[:, 3]
    ay = d4[:, 0] - d4[:, 3]
    theta, _ = cart2pol(ax, ay)

    type1 = np.zeros(n)
    type2 = np.zeros(n)
    type3 = np.zeros(n)
    for f in range(n):
        if white_look_flag[f] != 1:
            continue
        tp = theta[f]
        if tp > np.pi / 4 and tp < 3 * np.pi / 4:
            type1[f] = 1
        if abs(tp) < np.pi / 4:
            type2[f] = 1
        if abs(tp) > 3 * np.pi / 4:
            type2[f] = 1
        if tp > -3 * np.pi / 4 and tp < -np.pi / 4:
            type3[f] = 1

    #  collision resolution: if sum > 1 then type1 wins.
    x = type1 + type2 + type3
    type1[x > 1] = 1
    type2[x > 1] = 0
    type3[x > 1] = 0
    return type1, type2, type3


# ---------------------------------------------------------------------------
# Section 8: Head movement (the labels stage:1678-1926)
# ---------------------------------------------------------------------------


def _head_moving_detect(
    white_bp: np.ndarray, bh: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of "Head movement" detection block.

    Computes the per-frame angle of the Front→Middle axis (after
    self-rotation into the body frame), then differences across frames
    handling the ``±2pi`` wrap.  Returns the binary head-moving vector
    plus event start/stop indices.
    """
    n = white_bp.shape[0]
    d3, d4, _ = _rotate_to_anchor(
        white_bp, white_bp, pivot_col=5, anchor1_col=4, anchor2_col=6
    )
    d3 = d3 / bh[:, None]
    d4 = d4 / bh[:, None]
    ax = d3[:, 0] - d3[:, 3]  # Front - Middle (col 1 and col 4 )
    ay = d4[:, 0] - d4[:, 3]
    theta, _ = cart2pol(ax, ay)

    # Per-frame circular difference, vectorised.
    head_moving = np.zeros(n)
    delta = theta[1:] - theta[:-1]
    t1 = np.abs(delta)
    t2 = np.abs(delta - 2 * np.pi)
    t3 = np.abs(delta + 2 * np.pi)
    head_moving[1:] = np.minimum(np.minimum(t1, t2), t3)

    starts, ends = _events_from_signal(
        head_moving, level=0.1, iti_merge=11
    )
    # Extra filter: remove events where sum < 0.5 (not moving enough)
    keep = np.ones(starts.size, dtype=bool)
    for f in range(starts.size):
        s = int(starts[f])
        e = int(ends[f])
        if float(np.sum(head_moving[s : e + 1])) < 0.5:
            keep[f] = False
    starts = starts[keep]
    ends = ends[keep]

    hm_binary = _make_binary(n, starts, ends)
    return hm_binary, starts, ends


def _head_moving_types_vs(
    inds_start: np.ndarray,
    inds_stop: np.ndarray,
    look_flag: np.ndarray,
    n: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Port of "Head movement types (re: X)" blocks.

    Unlike the movement-types block which looks at a 5-sample window around
    each edge, head-movement-types uses:
        start window = look_flag[inds_start-2 : inds_start]  (3 samples)
        stop window  = look_flag[inds_stop    : inds_stop+2] (3 samples)
    — the window positioning is different.

    Same "skip first and last event" behavior as
    :func:`_movement_types_vs`.
    """
    h_self = np.zeros(n)
    h_to = np.zeros(n)
    h_stay = np.zeros(n)
    h_from = np.zeros(n)
    n_events = inds_start.size
    for f in range(n_events):
        s = int(inds_start[f])
        e = int(inds_stop[f])
        first_or_last = (f == 0) or (f == n_events - 1)
        if first_or_last or s - 2 < 0 or e + 2 >= n:
            start_win = np.zeros(3)
            stop_win = np.zeros(3)
        else:
            start_win = look_flag[s - 2 : s + 1]
            stop_win = look_flag[e : e + 3]
        mu_s = float(np.mean(start_win))
        mu_e = float(np.mean(stop_win))
        if mu_s < 0.5 and mu_e < 0.5:
            h_self[s : e + 1] = 1
        if mu_s < 0.5 and mu_e > 0.5:
            h_to[s : e + 1] = 1
        if mu_s > 0.5 and mu_e < 0.5:
            h_from[s : e + 1] = 1
        if mu_s > 0.5 and mu_e > 0.5:
            h_stay[s : e + 1] = 1
    return h_self, h_to, h_stay, h_from


# ---------------------------------------------------------------------------
# Section 9: Limb movement (the labels stage:1928-2075)
# ---------------------------------------------------------------------------


def _limb_moving_detect_and_types(
    white_bp: np.ndarray, bh: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Port of "Limb movement" + "Limb movement types".

    Uses the self-rotation frame. Returns ``(mu, FL, FR, BL, BR)`` binary
    vectors each shape ``(n,)``.  ``mu`` is the "multiple-limbs / default"
    bin — any limb event that isn't exclusively one of the four quadrants.
    """
    n = white_bp.shape[0]
    d3, d4, _ = _rotate_to_anchor(
        white_bp, white_bp, pivot_col=5, anchor1_col=4, anchor2_col=6
    )
    d3 = d3 / bh[:, None]
    d4 = d4 / bh[:, None]

    # Vectorised per-frame mean limb displacement across cols 9..20.
    diff_d3 = np.diff(d3[:, 9:21], axis=0)
    diff_d4 = np.diff(d4[:, 9:21], axis=0)
    step = np.sqrt(diff_d3 * diff_d3 + diff_d4 * diff_d4)  # (n-1, 12)
    limb_moving = np.zeros(n)
    limb_moving[1:] = step.mean(axis=1)
    limb_moving = movmean(limb_moving, 10)

    starts, ends = _events_from_signal(
        limb_moving, level=0.05, area_reject=1.0
    )

    # Per-quadrant displacement series (vectorised).
    def _quadrant_series(col_lo: int, col_hi: int) -> np.ndarray:
        d3_q = np.diff(d3[:, col_lo:col_hi], axis=0)
        d4_q = np.diff(d4[:, col_lo:col_hi], axis=0)
        series = np.sqrt(d3_q * d3_q + d4_q * d4_q).mean(axis=1)
        out = np.zeros(n)
        out[1:] = series
        return out

    L_FL = _quadrant_series(9, 12)   # cols 10-12  = 9-11 0-based
    L_FR = _quadrant_series(12, 15)
    L_BL = _quadrant_series(15, 18)
    L_BR = _quadrant_series(18, 21)

    sum_FL = np.zeros(starts.size)
    sum_FR = np.zeros(starts.size)
    sum_BL = np.zeros(starts.size)
    sum_BR = np.zeros(starts.size)
    #  `for f = 2:end-1` → Python 1:-1
    for f in range(1, max(0, starts.size - 1)):
        s = int(starts[f])
        e = int(ends[f])
        sum_FL[f] = float(np.sum(L_FL[s : e + 1]))
        sum_FR[f] = float(np.sum(L_FR[s : e + 1]))
        sum_BL[f] = float(np.sum(L_BL[s : e + 1]))
        sum_BR[f] = float(np.sum(L_BR[s : e + 1]))

    mu = np.zeros(n)
    FL = np.zeros(n)
    FR = np.zeros(n)
    BL = np.zeros(n)
    BR = np.zeros(n)
    for f in range(starts.size):
        s = int(starts[f])
        e = int(ends[f])
        mu[s : e + 1] = 1  # default bin
        a, b, c, d = sum_FL[f], sum_FR[f], sum_BL[f], sum_BR[f]
        if a > 5 and b < 5 and c < 5 and d < 5:
            FL[s : e + 1] = 1
            mu[s : e + 1] = 0
        if a < 5 and b > 5 and c < 5 and d < 5:
            FR[s : e + 1] = 1
            mu[s : e + 1] = 0
        if a < 5 and b < 5 and c > 5 and d < 5:
            BL[s : e + 1] = 1
            mu[s : e + 1] = 0
        if a < 5 and b < 5 and c < 5 and d > 5:
            BR[s : e + 1] = 1
            mu[s : e + 1] = 0

    #  collision resolution: mu wins on ties.
    x = mu + FL + FR + BL + BR
    mu[x > 1] = 1
    FL[x > 1] = 0
    FR[x > 1] = 0
    BL[x > 1] = 0
    BR[x > 1] = 0
    return mu, FL, FR, BL, BR


# ---------------------------------------------------------------------------
# Section 10: Upright (the labels stage:2077-2185)
# ---------------------------------------------------------------------------


def _upright_detect_and_types(
    dW_X: np.ndarray, dW_Y: np.ndarray, n: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Port of the "Upright" and "Upright types" blocks.

    Uses the sorted-depth matrices from ``_sort_depths``.  Returns
    ``(upright, t1, t2, t3)`` binary vectors.
    """
    temp_back = np.nanmean(dW_Y[:, :10], axis=1)
    temp_front = np.nanmean(dW_Y[:, 11:21], axis=1)
    temp_diff = temp_front - temp_back
    temp_diff = movmean(temp_diff, 10)

    starts, ends = _events_from_signal(
        temp_diff, level=5.0, iti_merge=22, min_length=22
    )

    # "Not upright enough" reject (inds_mean<11 && inds_max<11).
    keep = np.ones(starts.size, dtype=bool)
    for f in range(starts.size):
        s = int(starts[f])
        e = int(ends[f])
        m = float(np.mean(temp_diff[s : e + 1]))
        mx = float(np.max(temp_diff[s : e + 1]))
        if m < 11 and mx < 11:
            keep[f] = False
    starts = starts[keep]
    ends = ends[keep]

    upright = _make_binary(n, starts, ends)

    # Types: use dW_X, compare left vs right
    temp_left = np.nanmean(dW_X[:, :10], axis=1)
    temp_right = np.nanmean(dW_X[:, 11:21], axis=1)
    td2 = temp_left - temp_right
    td2 = movmean(td2, 10)
    inds_xy = np.zeros(starts.size)
    for f in range(starts.size):
        s = int(starts[f])
        e = int(ends[f])
        inds_xy[f] = float(np.mean(td2[s : e + 1]))

    t1 = np.zeros(n)
    t2 = np.zeros(n)
    t3 = np.zeros(n)
    for f in range(starts.size):
        s = int(starts[f])
        e = int(ends[f])
        v = inds_xy[f]
        if abs(v) < 4:
            t1[s : e + 1] = 1
        if v < -4:
            t2[s : e + 1] = 1
        if v > 4:
            t3[s : e + 1] = 1
    return upright, t1, t2, t3


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def label_white(
    white: AnimalInputs,
    blue: AnimalInputs,
    red: AnimalInputs,
    yellow: AnimalInputs,
) -> np.ndarray:
    """Top-level port of `the labels stage`.

    Composes all the per-section helpers and returns the ``mid_layer``
    ``(n_frames, 30)`` binary matrix that `the pipeline` writes
    to ``w_description_<name>.csv``.
    """
    n = white.bp.shape[0]

    # --- Section 1: body pos / angle (per animal) ---
    _, _, _, _, edgesW_rot = _body_pos_and_angle(white)
    # edgesB_rot, edgesR_rot, edgesY_rot are computed by  but only
    # used in their own ``sort_depths`` blocks (not relevant for the white
    # mid_layer output — those blocks produce dB_X/dB_Y/dR_X/dR_Y/dY_X/dY_Y
    # which are never referenced later). Skipped for speed.

    # --- Section 2: bh (head radius) ---
    bh = _head_radius(white.bp)

    # --- Section 3: sort depths (white only) ---
    dW_X, dW_Y = _sort_depths(edgesW_rot, white.depths)

    # --- Section 4-5: near-other per animal + types ---
    near_B, _, _ = _near_other_detect(white.bp, blue.bp, bh, white.depths, blue.depths)
    near_R, _, _ = _near_other_detect(white.bp, red.bp, bh, white.depths, red.depths)
    near_Y, _, _ = _near_other_detect(white.bp, yellow.bp, bh, white.depths, yellow.depths)

    tB1, tB2, tB3, tB4, tB5 = _near_other_types(white.bp, blue.bp, bh, near_B)
    tR1, tR2, tR3, tR4, tR5 = _near_other_types(white.bp, red.bp, bh, near_R)
    tY1, tY2, tY3, tY4, tY5 = _near_other_types(white.bp, yellow.bp, bh, near_Y)

    # near_A is 1 where ANY other is near
    near_A = ((near_B + near_R + near_Y) > 0).astype(float)

    # Nearest-per-frame per-animal distance for type resolution.
    w_x = white.bp[:, :21]
    w_y = white.bp[:, 21:]

    def _run_min(other_bp: np.ndarray) -> np.ndarray:
        o_x = other_bp[:, :21]
        o_y = other_bp[:, 21:]
        dx = w_x[:, :, None] - o_x[:, None, :]
        dy = w_y[:, :, None] - o_y[:, None, :]
        with np.errstate(invalid="ignore"):
            d = np.sqrt(dx * dx + dy * dy) / bh[:, None, None]
            out = np.nanmin(d.reshape(n, -1), axis=1)
        # Frames with all-NaN distances → +inf ( `min` of all-NaN is NaN,
        # but we only consult ``out`` inside ``near_A==1`` guards where that
        # case is benign — we'd rather never pick it).
        out[np.isnan(out)] = np.inf
        return out

    running_B = _run_min(blue.bp)
    running_R = _run_min(red.bp)
    running_Y = _run_min(yellow.bp)

    tA1 = np.zeros(n)
    tA2 = np.zeros(n)
    tA3 = np.zeros(n)
    tA4 = np.zeros(n)
    tA5 = np.zeros(n)
    for f in range(n):
        if near_A[f] != 1:
            continue
        rb, rr, ry = running_B[f], running_R[f], running_Y[f]
        if rb < rr and rb < ry:
            tA1[f] = tB1[f]; tA2[f] = tB2[f]; tA3[f] = tB3[f]; tA4[f] = tB4[f]; tA5[f] = tB5[f]
        elif rr < rb and rr < ry:
            tA1[f] = tR1[f]; tA2[f] = tR2[f]; tA3[f] = tR3[f]; tA4[f] = tR4[f]; tA5[f] = tR5[f]
        elif ry < rb and ry < rr:
            tA1[f] = tY1[f]; tA2[f] = tY2[f]; tA3[f] = tY3[f]; tA4[f] = tY4[f]; tA5[f] = tY5[f]

    # --- Section 6: moving ---
    moving, mv_starts, mv_ends = _moving_detect(white.bp, bh)
    # Movement types "re: all other" uses the UNION of per-color near flags,
    # clamped to binary.
    near_union_raw = near_B + near_R + near_Y
    near_union = (near_union_raw > 0).astype(float)
    mv_self, mv_to, mv_stay, mv_from = _movement_types_vs(
        mv_starts, mv_ends, near_union, n
    )

    # --- Section 7: looking at other per animal + types (all) ---
    lookB = _looking_at_other_detect(white.bp, blue.bp, bh)
    lookR = _looking_at_other_detect(white.bp, red.bp, bh)
    lookY = _looking_at_other_detect(white.bp, yellow.bp, bh)
    lookA = ((lookB + lookR + lookY) > 0).astype(float)
    lookA_t1, lookA_t2, lookA_t3 = _looking_at_other_types(white.bp, bh, lookA)

    # --- Section 8: head movement + types (all) ---
    head_moving, hm_starts, hm_ends = _head_moving_detect(white.bp, bh)
    # "re: all" uses the union of lookB/R/Y.
    look_union = ((lookB + lookR + lookY) > 0).astype(float)
    hm_self, hm_to, hm_stay, hm_from = _head_moving_types_vs(
        hm_starts, hm_ends, look_union, n
    )

    # --- Section 9: limb movement + types ---
    limb_mu, limb_FL, limb_FR, limb_BL, limb_BR = _limb_moving_detect_and_types(
        white.bp, bh
    )

    # --- Section 10: upright + types ---
    upright, up_t1, up_t2, up_t3 = _upright_detect_and_types(dW_X, dW_Y, n)

    # --- Assemble mid_layer (30 cols) ---
    mid = np.column_stack([
        moving,               # 0
        mv_self,              # 1 white_moving_selfA
        mv_to,                # 2 white_moving_toA
        mv_stay,              # 3 white_moving_stayA
        mv_from,              # 4 white_moving_fromA
        head_moving,          # 5 white_head_moving
        hm_self,              # 6 white_look_selfA
        hm_to,                # 7 white_look_toA
        hm_stay,              # 8 white_look_stayA
        hm_from,              # 9 white_look_fromA
        (limb_mu + limb_FL + limb_FR + limb_BL + limb_BR > 0).astype(float),  # 10 white_limb_moving
        limb_mu,              # 11
        limb_FL,              # 12
        limb_FR,              # 13
        limb_BL,              # 14
        limb_BR,              # 15
        upright,              # 16
        up_t1,                # 17
        up_t2,                # 18
        up_t3,                # 19
        near_A,               # 20
        tA1,                  # 21
        tA2,                  # 22
        tA3,                  # 23
        tA4,                  # 24
        tA5,                  # 25
        lookA,                # 26
        lookA_t1,             # 27
        lookA_t2,             # 28
        lookA_t3,             # 29
    ])
    return mid.astype(float)


# ---------------------------------------------------------------------------
# Thin wrappers: blue / red / yellow label files just swap White with the
# target color, then call the same logic.  The  sources
# (``) are 
# `the labels stage` apart from the ``deal`` swaps at the top.
# ---------------------------------------------------------------------------


def label_blue(
    white: AnimalInputs,
    blue: AnimalInputs,
    red: AnimalInputs,
    yellow: AnimalInputs,
) -> np.ndarray:
    """Port of `the labels stage`.  Equivalent to calling
    :func:`label_white` with ``White`` and ``Blue`` swapped."""
    return label_white(blue, white, red, yellow)


def label_red(
    white: AnimalInputs,
    blue: AnimalInputs,
    red: AnimalInputs,
    yellow: AnimalInputs,
) -> np.ndarray:
    """Port of `the labels stage`.  Equivalent to calling
    :func:`label_white` with ``White`` and ``Red`` swapped."""
    return label_white(red, blue, white, yellow)


def label_yellow(
    white: AnimalInputs,
    blue: AnimalInputs,
    red: AnimalInputs,
    yellow: AnimalInputs,
) -> np.ndarray:
    """Port of `the labels stage`.  Equivalent to calling
    :func:`label_white` with ``White`` and ``Yellow`` swapped."""
    return label_white(yellow, blue, red, white)


def loop_4_get_features(
    edges: "dict[str, np.ndarray]",
    depths: "dict[str, np.ndarray]",
    *,
    red_present: bool = True,
    white_present: bool = True,
    blue_present: bool = True,
    yellow_present: bool = True,
) -> "dict[str, np.ndarray]":
    """Port of `the pipeline`.

    Parameters
    ----------
    edges : dict
        Output of :func:`marm_behavior.io.mat_io.load_edges` — contains
        the 8 ``F_3E/F_4E/F_3BE/F_4BE/F_3RE/F_4RE/F_3YE/F_4YE`` matrices.
    depths : dict
        Output of :func:`marm_behavior.io.mat_io.load_depths` — contains
        ``depthWhite/depthBlue/depthRed/depthYellow``.
    *_present : bool
        One-flag-per-color gate, mirroring ``animals_present.txt``.  If an
        animal is not present, its mid_layer isn't computed and the returned
        dict omits that entry.

    Returns
    -------
    dict
        ``{'w': (n, 30), 'b': (n, 30), 'r': (n, 30), 'y': (n, 30)}`` (only
        keys for present animals).  Each value is the binary ``mid_layer``
        that `the pipeline` writes to
        ``{w,b,r,y}_description_<n>.csv``.
    """
    white = AnimalInputs(
        bp=np.column_stack([
            edges["F_3E"][:, 104:125].astype(float),
            edges["F_4E"][:, 104:125].astype(float),
        ]),
        F_3=edges["F_3E"].astype(float),
        F_4=edges["F_4E"].astype(float),
        depths=depths["depthWhite"].astype(float),
    )
    blue = AnimalInputs(
        bp=np.column_stack([
            edges["F_3BE"][:, 104:125].astype(float),
            edges["F_4BE"][:, 104:125].astype(float),
        ]),
        F_3=edges["F_3BE"].astype(float),
        F_4=edges["F_4BE"].astype(float),
        depths=depths["depthBlue"].astype(float),
    )
    red = AnimalInputs(
        bp=np.column_stack([
            edges["F_3RE"][:, 104:125].astype(float),
            edges["F_4RE"][:, 104:125].astype(float),
        ]),
        F_3=edges["F_3RE"].astype(float),
        F_4=edges["F_4RE"].astype(float),
        depths=depths["depthRed"].astype(float),
    )
    yellow = AnimalInputs(
        bp=np.column_stack([
            edges["F_3YE"][:, 104:125].astype(float),
            edges["F_4YE"][:, 104:125].astype(float),
        ]),
        F_3=edges["F_3YE"].astype(float),
        F_4=edges["F_4YE"].astype(float),
        depths=depths["depthYellow"].astype(float),
    )

    out: "dict[str, np.ndarray]" = {}
    if white_present:
        out["w"] = label_white(white, blue, red, yellow)
    if blue_present:
        out["b"] = label_blue(white, blue, red, yellow)
    if red_present:
        out["r"] = label_red(white, blue, red, yellow)
    if yellow_present:
        out["y"] = label_yellow(white, blue, red, yellow)
    return out
