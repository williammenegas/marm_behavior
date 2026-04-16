"""
extract_2
=========

Partial .

Status (Stage 0)
----------------
* :func:`assemble_frames` — **fully ported.**  Produces the ``(nframes, 42*5)``
  ``test_all`` matrix from the multi-animal CSV.
* :func:`midpoint_assignment` — **fully ported.**  This is the 24-permutation
  block that picks the best assignment of the four head markers to the four
  detected animal middles.  It is pure arithmetic, deterministic.
* :func:`assign_by_track` — **stub** with the original  pasted into the
  docstring for review.  Risky because of NaN-tie handling and the way tracks
  are written into ``White1``/``Blue1``/etc. with order-dependent updates.
* :func:`reconcile_assignments` — **stub.**  This is the "find points where
  Red and Red1 differ, choose the better fit" block.  Mechanical but long.
* :func:`remove_pairwise_overlaps` — **stub.**  Six near-identical blocks that
  remove overlapping points between pairs of colored animals.

Bringing the stubs to full implementations is Stage 1.  See ``STATUS.md``
for the full porting plan.
"""

from __future__ import annotations

from itertools import permutations
from typing import Tuple

import numpy as np

from ..numerics.helpers import nansum


# The full 24-permutation outcome matrix used in
# `the extract stage` lines 277-302.  Generated programmatically here
# (rather than hand-copied) so it cannot drift  if anyone
# regenerates it.  The order matches the ``perms([1 2 3 4])`` reversed
# into lexicographic order, which is also the order hard-coded .
OUTCOME_MAT = np.array(list(permutations([1, 2, 3, 4])), dtype=int)
assert OUTCOME_MAT.shape == (24, 4)


def assemble_frames(
    test_assem: np.ndarray,
    test_frames: np.ndarray,
    test_tracks: np.ndarray,
    n_frames: int,
    *,
    max_per_frame: int = 5,
) -> np.ndarray:
    """Reproduce the ``test_all_1 ... test_all_5`` block of extract_2.

    For each frame, picks up to the first ``max_per_frame`` detections (in the
    order they appear in ``test_assem``), reshapes each to a length-42 row,
    and concatenates them horizontally.

    Returns an ``(n_frames, 42 * max_per_frame)`` matrix, padded with zeros
    for frames that have fewer than ``max_per_frame`` detections.
    """
    n_cols = test_assem.shape[1]
    if n_cols != 42:
        raise ValueError(
            f"assemble_frames expects 42 body-part columns, got {n_cols}"
        )
    out = np.zeros((n_frames, 42 * max_per_frame))
    for n in range(1, n_frames + 1):  # 1-based to mirror 
        mask = test_frames == n
        rows = test_assem[mask]
        for k in range(min(max_per_frame, rows.shape[0])):
            out[n - 1, 42 * k : 42 * (k + 1)] = rows[k].reshape(-1)
    return out


def midpoint_assignment(
    test_all: np.ndarray,
    redHead: np.ndarray,
    whiteHead: np.ndarray,
    blueHead: np.ndarray,
    yellowHead: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the 24-permutation midpoint assignment problem.

    For each frame, computes the (R, W, B, Y) → (slot1..slot4) assignment that
    minimises the total distance between each color's head marker and the
    midpoint of the assigned slot's head body parts.  Returns the four
    ``(n_frames, 42)`` matrices ``Red``, ``White``, ``Blue``, ``Yellow`` after
    the assignment.

    This is a **fully ported** block; see `the extract stage` lines
    167-326.  It is pure floating-point arithmetic with no -specific
    edge cases, so the output is deterministic given valid input.
    order of operations in the inner ``nansum``.
    """
    import warnings

    n_frames = test_all.shape[0]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")

        # Compute the four candidate "middles" for each frame, using the head
        # body parts (columns 3:2:7 → x, 2:2:7 → y in the 1-based indexing,
        # which in Python is columns [2,4,6] for x and [1,3,5] for y).
        middles = np.full((n_frames, 12), np.nan)
        for slot in range(4):
            base = slot * 42
            x_cols = [base + 2, base + 4, base + 6]
            y_cols = [base + 1, base + 3, base + 5]
            with np.errstate(invalid="ignore"):
                middles[:, slot * 3] = np.nanmean(test_all[:, x_cols], axis=1)
                middles[:, slot * 3 + 1] = np.nanmean(test_all[:, y_cols], axis=1)
        # : Middle(Middle==0)=nan.  Empty slots produce 0 rather
        # than NaN from ``nanmean`` of all zeros.
        middles[middles == 0] = np.nan
        # The original also tries an "all body parts" fallback; both are
        # combined via ``Middle(isnan(Middle)) = Middle1(isnan(Middle));`` That
        # fallback is included here so the assignment cost matches.
        middles_alt = np.full((n_frames, 12), np.nan)
        for slot in range(4):
            base = slot * 42
            x_cols = list(range(base, base + 42, 2))
            y_cols = list(range(base + 1, base + 42, 2))
            with np.errstate(invalid="ignore"):
                middles_alt[:, slot * 3] = np.nanmean(test_all[:, x_cols], axis=1)
                middles_alt[:, slot * 3 + 1] = np.nanmean(test_all[:, y_cols], axis=1)
        # : Middle1(Middle1==0)=nan.
        middles_alt[middles_alt == 0] = np.nan
        nan_mask = np.isnan(middles)
        middles[nan_mask] = middles_alt[nan_mask]
        # : Middle(Middle==0)=nan (second pass after merge).
        middles[middles == 0] = np.nan

        # Distance from each color head to each of the four candidate middles.
        def dist_to_slot(head: np.ndarray, slot: int) -> np.ndarray:
            mx = middles[:, slot * 3]
            my = middles[:, slot * 3 + 1]
            return np.sqrt((head[:, 0] - mx) ** 2 + (head[:, 1] - my) ** 2)

        R = np.column_stack([dist_to_slot(redHead, s) for s in range(4)])
        W = np.column_stack([dist_to_slot(whiteHead, s) for s in range(4)])
        B = np.column_stack([dist_to_slot(blueHead, s) for s in range(4)])
        Y = np.column_stack([dist_to_slot(yellowHead, s) for s in range(4)])

        # For each frame, score each of the 24 (R,W,B,Y) → (s1,s2,s3,s4) choices.
        # Cost is the nansum of the four picked distances; we pick the argmin.
        sort_log = np.zeros(n_frames, dtype=int)
        for i in range(n_frames):
            costs = np.empty(24)
            for j, perm in enumerate(OUTCOME_MAT):
                r_id, w_id, b_id, y_id = perm - 1
                costs[j] = nansum(
                    [R[i, r_id], W[i, w_id], B[i, b_id], Y[i, y_id]]
                )
            sort_log[i] = int(np.nanargmin(costs))

        # Materialise Red/White/Blue/Yellow by slicing test_all with the chosen
        # permutation per frame.
        Red = np.zeros((n_frames, 42))
        White = np.zeros((n_frames, 42))
        Blue = np.zeros((n_frames, 42))
        Yellow = np.zeros((n_frames, 42))
        for i in range(n_frames):
            perm = OUTCOME_MAT[sort_log[i]] - 1
            Red[i] = test_all[i, perm[0] * 42 : (perm[0] + 1) * 42]
            White[i] = test_all[i, perm[1] * 42 : (perm[1] + 1) * 42]
            Blue[i] = test_all[i, perm[2] * 42 : (perm[2] + 1) * 42]
            Yellow[i] = test_all[i, perm[3] * 42 : (perm[3] + 1) * 42]

        for arr in (Red, White, Blue, Yellow):
            arr[arr == 0] = np.nan

    return Red, White, Blue, Yellow


# ---------------------------------------------------------------------------
# assign_by_track
# ---------------------------------------------------------------------------

def assign_by_track(
    test_assem: np.ndarray,
    test_tracks: np.ndarray,
    test_frames: np.ndarray,
    redHead: np.ndarray,
    whiteHead: np.ndarray,
    blueHead: np.ndarray,
    yellowHead: np.ndarray,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Implementation of
    `the extract stage` lines 331-447.

    Every operation is kept as close as possible to the the original, even
    at the cost of Python idiom, so that divergences can be audited against
    the loop directly.
    """
    import warnings as _warnings


    White1 = np.zeros((n_frames, 42))
    Blue1 = np.zeros((n_frames, 42))
    Red1 = np.zeros((n_frames, 42))
    Yellow1 = np.zeros((n_frames, 42))

    if test_assem.size == 0 or test_tracks.size == 0:
        ntracking = np.zeros(0, dtype=int)
        return (Red1, White1, Blue1, Yellow1,
                test_assem.copy(), test_tracks.copy(), test_frames.copy(),
                ntracking)


    n_tracks = int(test_tracks.max())
    ntracking = np.zeros(n_tracks, dtype=int)

    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", message="Mean of empty slice")
        _warnings.filterwarnings("ignore", message="All-NaN slice encountered")


        for n in range(1, n_tracks + 1):
            # current_track = test_assem(test_tracks==n,:);
            # current_frame = test_frames(test_tracks==n);
            mask = test_tracks == n
            current_track = test_assem[mask]
            current_frame_1 = test_frames[mask]  # 
            current_frame = current_frame_1 - 1  # 0-based for indexing

            # current_x = nanmean(current_track(:,3:2:7)');
            # current_y = nanmean(current_track(:,2:2:7)');
            #  1-based: 3:2:7 = [3,5,7] = 0-based [2,4,6]
            #                 2:2:7 = [2,4,6] = 0-based [1,3,5]
            # nanmean along dim 1 of transposed = mean across cols of original.
            with np.errstate(invalid="ignore"):
                current_x = np.nanmean(current_track[:, [2, 4, 6]], axis=1)
                current_y = np.nanmean(current_track[:, [1, 3, 5]], axis=1)

            # if size(current_frame,1)>2 && sum(~isnan(current_x))>0
            if not (current_frame.shape[0] > 2
                    and int(np.sum(~np.isnan(current_x))) > 0):
                continue

            # ntracking(n) = 1;
            ntracking[n - 1] = 1

            # d_current_white = sqrt((current_x'-whiteHead(current_frame,1)).^2
            #                      + (current_y'-whiteHead(current_frame,2)).^2);
            d_current_white = np.sqrt(
                (current_x - whiteHead[current_frame, 0]) ** 2
                + (current_y - whiteHead[current_frame, 1]) ** 2
            )
            d_current_blue = np.sqrt(
                (current_x - blueHead[current_frame, 0]) ** 2
                + (current_y - blueHead[current_frame, 1]) ** 2
            )
            d_current_red = np.sqrt(
                (current_x - redHead[current_frame, 0]) ** 2
                + (current_y - redHead[current_frame, 1]) ** 2
            )
            d_current_yellow = np.sqrt(
                (current_x - yellowHead[current_frame, 0]) ** 2
                + (current_y - yellowHead[current_frame, 1]) ** 2
            )

            # d_current_white = nanmedian(d_current_white);  (scalar)
            # If all-NaN,  nanmedian returns NaN.  Preserve that.
            with np.errstate(invalid="ignore"):
                d_current_white = float(np.nanmedian(d_current_white))
                d_current_blue = float(np.nanmedian(d_current_blue))
                d_current_red = float(np.nanmedian(d_current_red))
                d_current_yellow = float(np.nanmedian(d_current_yellow))

            # "mostly populated" guard — reference:             #   if sum(sum(White1(current_frame,:)>0)) > sum(sum(White1(current_frame,:)==0))
            #       d_current_white = 3000;
            #   end
            # Note: compares total populated cells vs total zero cells across
            # the sub-block White1(current_frame, :), which is a 42-wide slice
            # across the frames this track occupies.
            block_w = White1[current_frame, :]
            if int((block_w > 0).sum()) > int((block_w == 0).sum()):
                d_current_white = 3000.0
            block_b = Blue1[current_frame, :]
            if int((block_b > 0).sum()) > int((block_b == 0).sum()):
                d_current_blue = 3000.0
            block_r = Red1[current_frame, :]
            if int((block_r > 0).sum()) > int((block_r == 0).sum()):
                d_current_red = 3000.0
            block_y = Yellow1[current_frame, :]
            if int((block_y > 0).sum()) > int((block_y == 0).sum()):
                d_current_yellow = 3000.0

            # all_dists = [d_current_white, d_current_blue, d_current_red, d_current_yellow];
            # min(all_dists) in  ignores NaN by default.
            # NaN == anything is false, so a NaN distance is never a "min".
            all_dists = np.array([d_current_white, d_current_blue,
                                  d_current_red, d_current_yellow])
            if np.all(np.isnan(all_dists)):
                continue
            min_dist = float(np.nanmin(all_dists))

            def _write_temp_row(buf: np.ndarray) -> None:
                """Exact port of the inner  ``temp1`` write loop.

                reference:                     for d = 1:length(current_frame)
                        temp1 = buf(current_frame(d),:);
                        temp1(temp1>0)=nan;
                        temp1 = temp1+current_track(d,:);
                        temp1(isnan(temp1))=0;
                        buf(current_frame(d),:) = buf(current_frame(d),:)+temp1;
                    end
                """
                for d_idx in range(current_frame.shape[0]):
                    row = int(current_frame[d_idx])
                    temp1 = buf[row, :].copy()
                    temp1[temp1 > 0] = np.nan
                    temp1 = temp1 + current_track[d_idx, :]
                    temp1[np.isnan(temp1)] = 0.0
                    buf[row, :] = buf[row, :] + temp1

            #  uses `d_current_X == min(all_dists)`.  NaN == x is false.
            if d_current_white == min_dist:
                _write_temp_row(White1)
            if d_current_blue == min_dist:
                _write_temp_row(Blue1)
            if d_current_red == min_dist:
                _write_temp_row(Red1)
            if d_current_yellow == min_dist:
                _write_temp_row(Yellow1)

    #  cleanup:
    #   for f = 1:ntracks
    #       if ntracking(f)==1
    #           test_assem(test_tracks==f,:)=[];
    #           test_frames(test_tracks==f)=[];
    #           test_tracks(test_tracks==f)=[];
    #       end
    #   end
    keep = np.ones(len(test_tracks), dtype=bool)
    for f in range(1, n_tracks + 1):
        if ntracking[f - 1] == 1:
            keep &= test_tracks != f
    test_assem_out = test_assem[keep]
    test_tracks_out = test_tracks[keep]
    test_frames_out = test_frames[keep]


    Red1[Red1 == 0] = np.nan
    White1[White1 == 0] = np.nan
    Blue1[Blue1 == 0] = np.nan
    Yellow1[Yellow1 == 0] = np.nan

    return (Red1, White1, Blue1, Yellow1,
            test_assem_out, test_tracks_out, test_frames_out, ntracking)


# ---------------------------------------------------------------------------
# reconcile_assignments
# ---------------------------------------------------------------------------

def _reconcile_one_color(
    X: np.ndarray,
    X1: np.ndarray,
    headX: np.ndarray,
) -> np.ndarray:
    """Reconcile two candidate matrices for a single color.

    Reproduces one of the four  "find points where X and X1 differ,
    choose better fit" blocks.  Returns ``X2``, a copy of ``X1`` with
    disagreement runs replaced by whichever candidate has the lower mean
    step size (for runs < 120 frames) or lower mean distance to the head
    marker (for longer runs).
    """
    import warnings

    from ..numerics.helpers import crossing as _crossing

    n_frames, n_cols = X.shape
    X2 = X1.copy()
    nan_mask = np.isnan(X2)
    X2[nan_mask] = X[nan_mask]

    dif = np.abs(X - X1)
    dif[np.isnan(dif)] = 0.0
    asym = np.isnan(X) ^ np.isnan(X1)
    dif[asym] = 100.0
    dif[0, :] = 0.0
    dif[-1, :] = 0.0

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        for f in range(0, n_cols, 2):
            signal = dif[:, f] + dif[:, f + 1]
            ind, _, _ = _crossing(signal, level=10.0)
            if ind.size < 2:
                continue
            starts = ind[0::2]
            ends = ind[1::2] + 1
            if len(starts) > len(ends):
                starts = starts[: len(ends)]

            for g in range(1, len(starts) - 1):
                s = int(starts[g])
                e = int(ends[g])
                lo = max(0, s - 1)
                hi = min(n_frames, e + 2)
                if hi - lo < 2:
                    continue
                temp_x = X[lo:hi, f]
                temp_y = X[lo:hi, f + 1]
                temp_x1 = X1[lo:hi, f]
                temp_y1 = X1[lo:hi, f + 1]
                if e - s < 120:
                    with np.errstate(invalid="ignore"):
                        step = float(np.nanmean(
                            np.sqrt(np.diff(temp_x) ** 2 + np.diff(temp_y) ** 2)
                        ))
                        step1 = float(np.nanmean(
                            np.sqrt(np.diff(temp_x1) ** 2 + np.diff(temp_y1) ** 2)
                        ))
                else:
                    head_x = headX[lo:hi, 0]
                    head_y = headX[lo:hi, 1]
                    with np.errstate(invalid="ignore"):
                        step = float(np.nanmean(np.sqrt((temp_x - head_x) ** 2 + (temp_y - head_y) ** 2)))
                        step1 = float(np.nanmean(np.sqrt((temp_x1 - head_x) ** 2 + (temp_y1 - head_y) ** 2)))

                if not (np.isnan(step) or np.isnan(step1)) and step < step1:
                    X2[s : e + 1, f] = X[s : e + 1, f]
                    X2[s : e + 1, f + 1] = X[s : e + 1, f + 1]
    return X2


def reconcile_assignments(
    *,
    Red: np.ndarray, White: np.ndarray, Blue: np.ndarray, Yellow: np.ndarray,
    Red1: np.ndarray, White1: np.ndarray, Blue1: np.ndarray, Yellow1: np.ndarray,
    redHead: np.ndarray, whiteHead: np.ndarray, blueHead: np.ndarray, yellowHead: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run the four-color reconcile pass and return the reconciled matrices."""
    return {
        "White": _reconcile_one_color(White, White1, whiteHead),
        "Blue": _reconcile_one_color(Blue, Blue1, blueHead),
        "Red": _reconcile_one_color(Red, Red1, redHead),
        "Yellow": _reconcile_one_color(Yellow, Yellow1, yellowHead),
    }


# ---------------------------------------------------------------------------
# remove_pairwise_overlaps
# ---------------------------------------------------------------------------

def _remove_pair_overlap(A: np.ndarray, B: np.ndarray) -> None:
    """NaN out body parts that coincide (< 2 px) between A and B, in place.

    For each column pair (body part), finds runs where A and B sit within 2
    px of each other, then NaN's whichever animal has the larger mean
    distance to its own pchip smoother in the surrounding window.
    """
    import warnings

    from ..numerics.helpers import crossing as _crossing
    from ..numerics.helpers import fillmissing_pchip as _fpchip

    n_frames, n_cols = A.shape
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        for f in range(0, n_cols, 2):
            AX = _fpchip(A[:, f])
            AY = _fpchip(A[:, f + 1])
            BX = _fpchip(B[:, f])
            BY = _fpchip(B[:, f + 1])
            f_dists = np.sqrt((AX - BX) ** 2 + (AY - BY) ** 2)
            f_tracking = (f_dists < 2).astype(float)
            f_tracking[0] = 0
            f_tracking[-1] = 0

            AXF = A[:, f].copy()
            AYF = A[:, f + 1].copy()
            AXF[f_tracking > 0] = np.nan
            AYF[f_tracking > 0] = np.nan
            AXF = _fpchip(AXF)
            AYF = _fpchip(AYF)
            BXF = B[:, f].copy()
            BYF = B[:, f + 1].copy()
            BXF[f_tracking > 0] = np.nan
            BYF[f_tracking > 0] = np.nan
            BXF = _fpchip(BXF)
            BYF = _fpchip(BYF)

            ind, _, _ = _crossing(f_tracking, level=0.5)
            if ind.size < 2:
                continue
            starts = ind[0::2]
            ends = ind[1::2] + 1
            if len(starts) > len(ends):
                starts = starts[: len(ends)]

            for g in range(1, len(starts) - 1):
                s = int(starts[g])
                e = int(ends[g])
                lo = max(0, s - 1)
                hi = min(n_frames, e + 2)
                if hi - lo < 1:
                    continue
                tAx = A[lo:hi, f];   tAy = A[lo:hi, f + 1]
                tBx = B[lo:hi, f];   tBy = B[lo:hi, f + 1]
                tAxF = AXF[lo:hi];   tAyF = AYF[lo:hi]
                tBxF = BXF[lo:hi];   tByF = BYF[lo:hi]
                with np.errstate(invalid="ignore"):
                    a_dif = float(np.nanmean(np.sqrt((tAx - tAxF) ** 2 + (tAy - tAyF) ** 2)))
                    b_dif = float(np.nanmean(np.sqrt((tBx - tBxF) ** 2 + (tBy - tByF) ** 2)))
                if not (np.isnan(a_dif) or np.isnan(b_dif)):
                    if a_dif < b_dif:
                        B[s : e + 1, f] = np.nan
                        B[s : e + 1, f + 1] = np.nan
                    elif b_dif < a_dif:
                        A[s : e + 1, f] = np.nan
                        A[s : e + 1, f + 1] = np.nan


def remove_pairwise_overlaps(
    *,
    Red: np.ndarray, White: np.ndarray, Blue: np.ndarray, Yellow: np.ndarray,
) -> None:
    """Run all six pair-wise overlap removals in place.

    Mirrors the six  blocks W↔B, W↔R, W↔Y, B↔Y, B↔R, Y↔R in that order.
    """
    _remove_pair_overlap(White, Blue)
    _remove_pair_overlap(White, Red)
    _remove_pair_overlap(White, Yellow)
    _remove_pair_overlap(Blue, Yellow)
    _remove_pair_overlap(Blue, Red)
    _remove_pair_overlap(Yellow, Red)


# ---------------------------------------------------------------------------
# Final small-gap fill (the extract stage lines 1071-1220)
# ---------------------------------------------------------------------------

def fill_small_gaps_all(
    *,
    Red: np.ndarray, White: np.ndarray, Blue: np.ndarray, Yellow: np.ndarray,
) -> None:
    """Port of the small-gap fill block at the end of extract_2.

    5-frame ``movmedian`` followed by the same ``crossing``+``pchip`` small-gap
    fill used in ``extract_1`` / ``extract_3``, with thresholds ``time < 60``
    and ``dist < 120``.  Mutates all four matrices in place.
    """
    from .extract_3 import _fill_small_gaps_42
    from ..numerics.helpers import movmedian as _movmedian

    for matrix in (Red, White, Blue, Yellow):
        for i in range(42):
            matrix[:, i] = _movmedian(matrix[:, i], 5)
        _fill_small_gaps_42(matrix)
        matrix[matrix < 0] = 0
        matrix[matrix > 1280] = 1280


# ---------------------------------------------------------------------------
# One-animal mode: collapse all tracklets to a single focal animal
# ---------------------------------------------------------------------------

def assign_tracklets_one_animal(
    test_assem: np.ndarray,
    test_tracks: np.ndarray,
    test_frames: np.ndarray,
    n_frames: int,
) -> np.ndarray:
    """Build a single focal-animal body matrix from the multi-CSV tracklets.

    In one-animal mode all multi-CSV tracklets must belong to the focal
    animal regardless of which colour DLC's head classifier predicted —
    the head classifier was trained on four-animal data and will
    routinely mislabel a single animal across colours. The
    proximity-based ``assign_by_track`` is therefore the wrong tool
    here: it would split the focal animal's tracklets across the four
    colour slots based on whichever colour-misclassified head
    happened to be closest in each frame.

    This function instead:

    1. For each row of ``test_assem`` (one tracklet at one frame),
       counts how many of its 21 body parts survived the confidence
       threshold (i.e., are non-NaN). This count is a per-row quality
       proxy — high-confidence detections survive the
       :data:`c_thresh` filter and contribute non-NaN entries.
    2. For each frame index 1..``n_frames``, looks at every tracklet
       row that landed in that frame and picks the one with the most
       non-NaN body parts. Ties (e.g., two equally-good tracklets in
       the same frame) are broken by preferring the lower-numbered
       track id, which tends to favour longer-running tracks because
       DLC numbers tracks in birth order.
    3. Writes the chosen tracklet's 42-column row into the output
       matrix at that frame index.

    Frames where no tracklet exists at all (no detection by DLC) get
    a row of NaN, which the downstream stages handle the same way
    they handle missing detections in the four-animal pipeline.

    Parameters
    ----------
    test_assem:
        ``(n_rows, 42)`` matrix of confidence-thresholded tracklet
        body-part coordinates, one row per (tracklet, frame). NaN
        entries indicate body parts dropped by the confidence filter.
    test_tracks:
        ``(n_rows,)`` integer array of 1-based track ids for each row
        of ``test_assem``.
    test_frames:
        ``(n_rows,)`` integer array of 1-based frame indices for each
        row of ``test_assem``.
    n_frames:
        Total number of frames in the video. The returned matrix has
        exactly this many rows.

    Returns
    -------
    Focal:
        ``(n_frames, 42)`` matrix. Frame ``i`` contains the body
        parts of the highest-quality tracklet at frame ``i+1`` (the
        ``+1`` is because ``test_frames`` is 1-based), or all NaN
        if no tracklet existed at that frame.
    """
    Focal = np.full((n_frames, 42), np.nan)

    if test_assem.size == 0 or test_frames.size == 0:
        return Focal

    # Per-row quality proxy: count of non-NaN body parts (out of 42 slots).
    valid_count = np.sum(~np.isnan(test_assem), axis=1)

    # Group rows by frame using a sort + segment scan. This is O(n log n)
    # in the number of multi-CSV rows, which is much cheaper than the
    # O(n_frames × max_per_frame) Python loop used by the four-animal
    # ``assign_by_track``.
    order = np.argsort(test_frames, kind="stable")
    sorted_frames = test_frames[order]
    sorted_valid = valid_count[order]
    sorted_tracks = test_tracks[order]

    # Segment boundaries: each unique frame value defines one segment.
    unique_frames, segment_starts = np.unique(sorted_frames, return_index=True)
    segment_ends = np.append(segment_starts[1:], len(sorted_frames))

    for frame_1based, lo, hi in zip(unique_frames, segment_starts, segment_ends):
        # Skip frames that fell outside the head-length budget.
        if frame_1based < 1 or frame_1based > n_frames:
            continue
        seg_valid = sorted_valid[lo:hi]
        seg_tracks = sorted_tracks[lo:hi]
        # Pick the row with the highest non-NaN count; tie-break on the
        # smallest track id (favouring longer-running early tracks).
        best_in_segment = int(
            np.lexsort((seg_tracks, -seg_valid))[0]
        )
        original_row_index = order[lo + best_in_segment]
        Focal[frame_1based - 1] = test_assem[original_row_index]

    return Focal
