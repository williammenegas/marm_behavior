"""
extract_1
=========

Computes.

Stage-1 head-tracking cleanup.  Takes the per-frame single-animal CSV produced
by DeepLabCut, and returns four ``(n_frames, 2)`` head-position arrays
(``redHead``, ``whiteHead``, ``blueHead``, ``yellowHead``) after:

1. confidence thresholding,
2. cropping invalid (y < 1) rows,
3. resolving overlap between colors (priority: red > blue > yellow > white,
   yellow > red, yellow > blue, blue > red),
4. iterative outlier removal against a moving-median smoother,
5. small-gap fill (time < 10, displacement < 50) via PCHIP,
6. tracklet joining (drops short isolated runs),
7. final ``movmedian`` → ``fillmissing(pchip)`` → clamp ``[0, 1280]`` → ``movmean``.

The four-way color symmetry in the original  is collapsed here into
helper functions that operate on a single (n,2) array; each helper is called
once per color.  This shrinks the file from 537 lines of  to roughly
160 lines of Python while preserving the operation order.

Input column convention (matches the original )
-----------------------------------------------------
The ``*single.csv`` body-part file has columns::

    [x_red, y_red, c_red, x_white, y_white, c_white, x_blue, y_blue, c_blue,
     x_yellow, y_yellow, c_yellow, x_extra, y_extra, c_extra]

(Five (x,y,c) triples; the fifth is ignored downstream.)  The original
adds 100 to every x to align coordinates with the corresponding video frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from ..numerics.helpers import (
    crossing,
    fillmissing_pchip,
    movmean,
    movmedian,
)


C_THRESH_DEFAULT = 0.1
X_OFFSET = 100.0
COORD_MIN = 0.0
COORD_MAX = 1280.0
OUTLIER_DIST = 50.0
OUTLIER_PASSES = 10
GAP_TIME_LIMIT = 10
GAP_DIST_LIMIT = 50.0
TRACKLET_MIN_SIZE = 20
TRACKLET_GAP_LIMIT = 50.0
SMOOTH_WINDOW = 10


# ---------------------------------------------------------------------------
# CSV → per-color heads
# ---------------------------------------------------------------------------

def parse_single_csv_matrix(
    fullpickle: np.ndarray,
    *,
    c_thresh: float = C_THRESH_DEFAULT,
) -> Dict[str, np.ndarray]:
    """Reproduce lines 1-58 of `the extract stage`.

    Takes the raw matrix from ``readmatrix(filename)`` and returns a dict with
    keys ``red``, ``white``, ``blue``, ``yellow`` mapping to ``(n, 2)`` head
    arrays after confidence thresholding and the +100 x-offset.
    """
    test_full = fullpickle.astype(float, copy=True)
    test_full[test_full == 0] = np.nan

    #  iterates i = 1:3:15 over the first five (x,y,c) triples, dropping
    # rows where y < 1.
    n_cols = test_full.shape[1]
    for i in range(0, min(15, n_cols), 3):
        y = test_full[:, i + 1]
        bad = y < 1
        test_full[bad, i] = np.nan
        test_full[bad, i + 1] = np.nan
        test_full[bad, i + 2] = np.nan

    # Confidence threshold over the same five triples.
    for i in range(0, min(15, n_cols), 3):
        c = test_full[:, i + 2]
        bad = c < c_thresh
        test_full[bad, i] = np.nan
        test_full[bad, i + 1] = np.nan
        test_full[bad, i + 2] = np.nan

    # Drop the confidence columns and keep only the first 4 body parts (x,y).

    keep_cols = [c for c in range(test_full.shape[1]) if (c % 3) != 2]
    test_full = test_full[:, keep_cols]
    test_full = test_full[:, :10]
    test_full[:, 0:10:2] += X_OFFSET  # add 100 to every x

    return {
        "red": test_full[:, 0:2].copy(),
        "white": test_full[:, 2:4].copy(),
        "blue": test_full[:, 4:6].copy(),
        "yellow": test_full[:, 6:8].copy(),
    }


# ---------------------------------------------------------------------------
# Overlap resolution (color A "loses" to color B if they sit within 15 px)
# ---------------------------------------------------------------------------

def _resolve_overlap(loser: np.ndarray, winner: np.ndarray, *, max_dist: float = 15.0) -> None:
    """NaN out points in ``loser`` that are within ``max_dist`` of ``winner``.

    Mutates ``loser`` in place to mirror the 

        overlap_dist = sqrt( (fillmissing(winner_x,'pchip')-loser_x).^2 + ... );
        x = overlap_dist;
        x(x>15)=0; x(isnan(x))=0; x(1)=0; x(end)=0; x(x>0)=1;
        loser_x(x==1) = nan; loser_y(x==1) = nan;
    """
    wx = fillmissing_pchip(winner[:, 0])
    wy = fillmissing_pchip(winner[:, 1])
    d = np.sqrt((wx - loser[:, 0]) ** 2 + (wy - loser[:, 1]) ** 2)
    flag = (d > 0) & (d <= max_dist)
    flag[0] = False
    flag[-1] = False
    loser[flag, 0] = np.nan
    loser[flag, 1] = np.nan


# ---------------------------------------------------------------------------
# Outlier removal: 10 passes of "drop points >50 from movmedian smoother"
# ---------------------------------------------------------------------------

def _outlier_removal_pass(head: np.ndarray) -> None:
    smooth_x = fillmissing_pchip(movmedian(head[:, 0], SMOOTH_WINDOW))
    smooth_y = fillmissing_pchip(movmedian(head[:, 1], SMOOTH_WINDOW))
    d = np.sqrt((head[:, 0] - smooth_x) ** 2 + (head[:, 1] - smooth_y) ** 2)
    bad = d > OUTLIER_DIST
    head[bad, 0] = np.nan
    head[bad, 1] = np.nan


# ---------------------------------------------------------------------------
# Small-gap fill
# ---------------------------------------------------------------------------

def _fill_small_gaps(head: np.ndarray) -> None:
    """Executes::

        temp = head(:,1:2);
        temp_fill(:,1) = fillmissing(movmedian(head(:,1),10,'omitnan'),'pchip');
        temp_fill(:,2) = fillmissing(movmedian(head(:,2),10,'omitnan'),'pchip');
        temp(temp>0)=1; temp(1:2,:)=1; temp(end-1:end,:)=1; temp(isnan(temp))=0;
        temp_c = crossing(temp(:,1), 1:size(temp,1),0.5);
        ... fill if gap is short in time AND space ...
    """
    n = head.shape[0]
    fill_x = fillmissing_pchip(movmedian(head[:, 0], SMOOTH_WINDOW))
    fill_y = fillmissing_pchip(movmedian(head[:, 1], SMOOTH_WINDOW))

    # Build the 0/1 validity signal exactly , including
    # the "force first/last two samples to 1" sentinels.
    valid = (~np.isnan(head[:, 0])).astype(float)
    valid[:2] = 1.0
    valid[-2:] = 1.0

    ind, _, _ = crossing(valid, level=0.5)
    if ind.size < 2:
        return
    starts = ind[0::2]
    ends = ind[1::2]
    if len(starts) > len(ends):
        starts = starts[: len(ends)]

    times = ends - starts
    #  looks one sample before and after each gap to compute the
    # geometric distance across the gap.
    safe_starts = np.clip(starts - 1, 0, n - 1)
    safe_ends = np.clip(ends + 1, 0, n - 1)
    dists = np.sqrt(
        (head[safe_ends, 0] - head[safe_starts, 0]) ** 2
        + (head[safe_ends, 1] - head[safe_starts, 1]) ** 2
    )

    for s, e, t, d in zip(starts, ends, times, dists):
        if t < GAP_TIME_LIMIT and d < GAP_DIST_LIMIT:
            head[s : e + 1, 0] = fill_x[s : e + 1]
            head[s : e + 1, 1] = fill_y[s : e + 1]


# ---------------------------------------------------------------------------
# Tracklet joining: remove very short isolated runs of valid samples
# ---------------------------------------------------------------------------

def _remove_short_tracklets(head: np.ndarray) -> None:
    """Equivalent of the four  ``%%% tracklet joining ...`` blocks.

    Identifies maximal runs of non-NaN samples in ``head[:,0]``, then NaN-outs
    any run shorter than ``TRACKLET_MIN_SIZE`` whose distance to the previous
    OR next run exceeds ``TRACKLET_GAP_LIMIT``.

    MATLAB pre-computes all distances and sizes before the removal loop,
    so removing tracklet k does not affect the distance check for tracklet
    k+1.  We replicate that here.
    """
    n = head.shape[0]
    valid = ~np.isnan(head[:, 0])
    if not valid.any():
        return

    # Build start/end indices of each contiguous run.
    starts = []
    ends = []
    in_run = False
    for i in range(n):
        if valid[i] and not in_run:
            starts.append(i)
            in_run = True
        if in_run and (i == n - 1 or not valid[i + 1]):
            ends.append(i)
            in_run = False

    # Match the MATLAB loop bounds: ``for i = 3:tracksW-2`` where
    # ``tracksW = num_tracklets + 1``, i.e. 1-indexed tracklets 3 to
    # num_tracklets-1 = 0-indexed tracklets 2 to num_tracklets-2.
    n_runs = len(starts)
    if n_runs < 4:
        return

    # Pre-compute sizes and distances (MATLAB does this in a separate loop
    # before the removal loop, so NaN-ing out tracklet k doesn't affect
    # the distance calculation for tracklet k+1).
    sizes = [ends[k] - starts[k] for k in range(n_runs)]
    d_prev = [0.0] * n_runs
    d_next = [0.0] * n_runs
    for k in range(2, n_runs - 1):
        d_prev[k] = np.hypot(
            head[starts[k], 0] - head[ends[k - 1], 0],
            head[starts[k], 1] - head[ends[k - 1], 1],
        )
        d_next[k] = np.hypot(
            head[ends[k], 0] - head[starts[k + 1], 0],
            head[ends[k], 1] - head[starts[k + 1], 1],
        )

    for k in range(2, n_runs - 1):
        if sizes[k] >= TRACKLET_MIN_SIZE:
            continue
        if d_prev[k] > TRACKLET_GAP_LIMIT or d_next[k] > TRACKLET_GAP_LIMIT:
            head[starts[k] : ends[k] + 1, :] = np.nan


# ---------------------------------------------------------------------------
# Final smooth+fill+clamp+smooth pipeline
# ---------------------------------------------------------------------------

def _finalize(head: np.ndarray) -> None:
    """Final block of `the extract stage` (lines 482-530)."""
    head[:, 0] = movmedian(head[:, 0], SMOOTH_WINDOW)
    head[:, 1] = movmedian(head[:, 1], SMOOTH_WINDOW)
    head[:, 0] = fillmissing_pchip(head[:, 0])
    head[:, 1] = fillmissing_pchip(head[:, 1])
    head[head < COORD_MIN] = COORD_MIN
    head[head > COORD_MAX] = COORD_MAX
    head[:, 0] = movmean(head[:, 0], SMOOTH_WINDOW)
    head[:, 1] = movmean(head[:, 1], SMOOTH_WINDOW)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

@dataclass
class HeadTracks:
    """Output of :func:`run_extract_1`."""

    redHead: np.ndarray
    whiteHead: np.ndarray
    blueHead: np.ndarray
    yellowHead: np.ndarray


def run_extract_1(
    fullpickle: np.ndarray,
    *,
    c_thresh: float = C_THRESH_DEFAULT,
) -> HeadTracks:
    """Run the full `the extract stage` pipeline on a single video's CSV.

    Parameters
    ----------
    fullpickle:
        The raw matrix returned by :func:`...io.csv_io.read_single_csv`.
    c_thresh:
        Confidence threshold below which body-part detections are discarded.

    Returns
    -------
    HeadTracks
        Four ``(n_frames, 2)`` arrays, one per color.
    """
    heads = parse_single_csv_matrix(fullpickle, c_thresh=c_thresh)
    red = heads["red"]
    white = heads["white"]
    blue = heads["blue"]
    yellow = heads["yellow"]

    # Overlap resolution.  Resolution order: white loses first,
    # then yellow, then blue (each against earlier-priority colors).
    _resolve_overlap(white, red)
    _resolve_overlap(white, blue)
    _resolve_overlap(white, yellow)
    _resolve_overlap(yellow, red)
    _resolve_overlap(yellow, blue)
    _resolve_overlap(blue, red)

    # Outlier removal: 10 passes per color.
    for _ in range(OUTLIER_PASSES):
        _outlier_removal_pass(red)
        _outlier_removal_pass(white)
        _outlier_removal_pass(blue)
        _outlier_removal_pass(yellow)

    # Small-gap fill, applied to all four colors.
    for head in (red, white, blue, yellow):
        _fill_small_gaps(head)

    # Tracklet joining.
    for head in (white, red, blue, yellow):
        _remove_short_tracklets(head)

    # Final smooth + fill + clamp + smooth.
    for head in (white, red, blue, yellow):
        _finalize(head)

    return HeadTracks(redHead=red, whiteHead=white, blueHead=blue, yellowHead=yellow)
