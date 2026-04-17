"""
depths_1
========

Per-frame depth lookup for the edge coordinates of each animal.

The video is assumed to be a stereo 2560×720 pair: the left 1280
columns are RGB, and the right 1280 columns are a depth map. For
every frame, this stage reads the pixel value at
``(F_4, F_3 + 1280)`` for every ``(F_3, F_4)`` edge coordinate of
each animal and writes the result to the corresponding row of
``depth{Red,White,Blue,Yellow}``.

Two-part flow
-------------
Coordinate preparation (rounding, clamping, all-NaN column fallback)
lives in :func:`prepare_edges_for_depth_lookup` and depends only on
the edges matrices.

The per-frame pixel lookup lives in :func:`run_depths_1`, which reads
the video via :mod:`cv2` if available and falls back to
:mod:`imageio.v3` otherwise. Both backends return frames in
``(H, W, 3)`` layout, which are averaged along axis 2 before indexing.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterator

import numpy as np


# Pixel coordinate bounds (1-indexed to match the rounding/clamp
# convention; run_depths_1 subtracts 1 before indexing).
F3_MIN = 1
F3_MAX = 1280
F4_MIN = 1
F4_MAX = 720
DEPTH_X_OFFSET = 1280  # shift into the right half of the stereo pair

# Number of edge columns per animal.
N_EDGES_PER_ANIMAL = 375


# ---------------------------------------------------------------------------
# Coordinate preparation
# ---------------------------------------------------------------------------

def _round_clamp(
    F3: np.ndarray, F4: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Round to int and clamp to the valid pixel range.

    Returns new arrays (does not mutate inputs).
    """
    F3_out = np.round(F3).astype(np.int64)
    F4_out = np.round(F4).astype(np.int64)
    np.clip(F3_out, F3_MIN, F3_MAX, out=F3_out)
    np.clip(F4_out, F4_MIN, F4_MAX, out=F4_out)
    return F3_out, F4_out


def _replace_nan_columns(
    F3: np.ndarray, F4: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For any column that is entirely NaN in either half, replace
    both halves of that column with the per-frame mean of that half.

    This salvages columns where every frame is missing the edge by
    falling back to the average location across columns.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered"
        )
        F3_nan_cols = np.isnan(F3).all(axis=0)
        F4_nan_cols = np.isnan(F4).all(axis=0)
        bad_cols = F3_nan_cols | F4_nan_cols
        if not bad_cols.any():
            return F3, F4
        row_mean_F3 = np.round(np.nanmean(F3, axis=1)).astype(F3.dtype)
        row_mean_F4 = np.round(np.nanmean(F4, axis=1)).astype(F4.dtype)
        F3 = F3.copy()
        F4 = F4.copy()
        F3[:, bad_cols] = row_mean_F3[:, None]
        F4[:, bad_cols] = row_mean_F4[:, None]
    return F3, F4


def prepare_edges_for_depth_lookup(
    edges: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Round, clamp, and NaN-column-fallback the eight edge matrices.

    Takes a dict of the form ``{'F_3E': ..., 'F_4E': ..., ...,
    'F_4YE': ...}`` and returns a new dict with the same keys
    containing integer arrays suitable for direct pixel indexing.

    The output arrays use 1-based coordinates
    (``F_3 ∈ [1, 1280]``, ``F_4 ∈ [1, 720]``);
    :func:`run_depths_1` subtracts 1 before using them as Python
    array indices.
    """
    pairs = (
        ("F_3E", "F_4E"),
        ("F_3BE", "F_4BE"),
        ("F_3RE", "F_4RE"),
        ("F_3YE", "F_4YE"),
    )
    out: dict[str, np.ndarray] = {}
    for f3_key, f4_key in pairs:
        F3_clean, F4_clean = _replace_nan_columns(edges[f3_key], edges[f4_key])
        F3_int, F4_int = _round_clamp(F3_clean, F4_clean)
        out[f3_key] = F3_int
        out[f4_key] = F4_int
    return out


# ---------------------------------------------------------------------------
# Frame source
# ---------------------------------------------------------------------------

def _iter_video_frames_grayscale(
    video_path: str | Path,
    *,
    column_slice: "slice | None" = None,
) -> Iterator[np.ndarray]:
    """Yield grayscale ``(H, W)`` float frames from the video.

    Tries OpenCV first, then imageio.  Raises ``ImportError`` if neither is
    available.  Frames are returned as float arrays matching 's
    ``nanmean(readFrame(v), 3)`` — the mean across the three color channels,
    which is order-invariant so the BGR-vs-RGB distinction doesn't matter.

    Parameters
    ----------
    column_slice:
        Optional ``slice`` applied along axis 1 of each raw BGR frame
        *before* the float64 cast and channel mean. When used, only the
        selected columns are promoted to float64 and reduced, which
        halves the per-frame work if the caller only needs (say) the
        depth half of a stereo pair. The pre-slice is a numpy view on
        the uint8 frame, so it adds no copy of its own; the subsequent
        ``.astype(np.float64)`` materialises contiguous memory. Because
        the pixel values mean-reduced are exactly the same ones that
        would have been reduced then sliced, the yielded float values
        are bit-identical to the un-sliced version's ``[:, column_slice]``.
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        cv2 = None

    def _grayscale(bgr: np.ndarray) -> np.ndarray:
        if column_slice is not None:
            bgr = bgr[:, column_slice, :]
        return bgr.astype(np.float64).mean(axis=2)

    if cv2 is not None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open {video_path}")
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                yield _grayscale(bgr)
        finally:
            cap.release()
        return

    try:
        import imageio.v3 as iio  # type: ignore
    except ImportError as err:
        raise ImportError(
            "depths_1 requires cv2 or imageio for video reading"
        ) from err

    for frame in iio.imiter(str(video_path)):
        yield _grayscale(frame)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_depths_1(
    edges: dict[str, np.ndarray],
    frame_source,
    *,
    depth_x_offset: int = DEPTH_X_OFFSET,
    max_frames: int | None = None,
    red_present: bool = True,
    white_present: bool = True,
    blue_present: bool = True,
    yellow_present: bool = True,
) -> dict[str, np.ndarray]:
    """Look up the depth-map pixel value at every edge coordinate for
    each animal, frame by frame.

    Parameters
    ----------
    edges:
        Output of the process stage — a dict with the eight
        ``F_3*`` / ``F_4*`` edge matrices in float32 pixel coordinates.
    frame_source:
        Either a path to an AVI (will be opened via
        :func:`_iter_video_frames_grayscale`), or an iterable yielding
        grayscale ``(H, W)`` float frames. The iterable form is useful
        for testing with a synthetic frame stream.
    depth_x_offset:
        Column offset applied to every ``F_3*`` coordinate before
        indexing (the right half of the stereo pair holds the depth
        map). Defaults to 1280.
    max_frames:
        Optional cap on the number of frames processed. Defaults to
        ``edges['F_3E'].shape[0]``.
    red_present, white_present, blue_present, yellow_present:
        Per-animal gating flags. When an animal is marked not present,
        its depth lookup loop is skipped entirely and the corresponding
        ``depth*`` matrix is returned as a zero array of the correct
        shape. Skipping absent animals is the main optimization the
        single-animal mode relies on — the per-frame indexing work
        scales linearly with the number of present animals, and the
        depths stage is by far the slowest of the six.

    Returns
    -------
    dict[str, np.ndarray]
        ``{'depthRed': ..., 'depthWhite': ..., 'depthBlue': ...,
        'depthYellow': ...}`` with each entry an
        ``(n_frames, 375)`` float32 array. Absent animals' arrays are
        all zeros (still allocated so downstream code that expects all
        four keys keeps working).
    """
    prepared = prepare_edges_for_depth_lookup(edges)
    F_3E = prepared["F_3E"]
    F_4E = prepared["F_4E"]
    F_3BE = prepared["F_3BE"]
    F_4BE = prepared["F_4BE"]
    F_3RE = prepared["F_3RE"]
    F_4RE = prepared["F_4RE"]
    F_3YE = prepared["F_3YE"]
    F_4YE = prepared["F_4YE"]

    n_frames = F_3E.shape[0]
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)

    depthRed = np.zeros((n_frames, N_EDGES_PER_ANIMAL), dtype=np.float32)
    depthWhite = np.zeros((n_frames, N_EDGES_PER_ANIMAL), dtype=np.float32)
    depthBlue = np.zeros((n_frames, N_EDGES_PER_ANIMAL), dtype=np.float32)
    depthYellow = np.zeros((n_frames, N_EDGES_PER_ANIMAL), dtype=np.float32)

    # Fast exit: if no animals are present, we still need to walk
    # the frame source to keep the iterator consumed cleanly, but
    # there's nothing to look up.
    any_present = (
        red_present or white_present or blue_present or yellow_present
    )

    if isinstance(frame_source, (str, Path)):
        # Ask the iterator to crop each frame to the depth half
        # BEFORE the float64 promotion + channel mean. The per-frame
        # grayscale conversion is the hot spot of this stage (~75 ms
        # on the full 2560-wide frame vs ~20 ms on the 1280-wide
        # right half on a typical workstation, because the conversion
        # is memory-bandwidth-bound on the float64-promoted buffer).
        # With the crop baked into the iterator we skip promoting the
        # RGB half entirely. Values at the depth-half pixels are
        # bit-identical to the un-cropped version's because pre-slice
        # and post-slice operate on exactly the same three uint8
        # channel values at each (row, col) position.
        frame_iter = _iter_video_frames_grayscale(
            frame_source, column_slice=slice(depth_x_offset, None),
        )
    else:
        # Test / iterable path: assume full-width frames so the
        # downstream indexing stays uniform with the file path.
        # View-slice (no copy) to get the same (H, width - offset)
        # geometry the file path produces above.
        frame_iter = (
            f[:, depth_x_offset:] for f in iter(frame_source)
        )

    # Optional tqdm progress bar — cheap ImportError fallback keeps
    # tqdm a soft dep. This stage is typically the slowest of the
    # six (one video-decode + one grayscale-mean per frame), so a
    # bar is the main UX win from this change.
    try:
        from tqdm.auto import tqdm
        progress = tqdm(
            total=n_frames,
            desc="  depths",
            unit="frame",
            leave=True,
            mininterval=0.5,
        )
    except ImportError:
        progress = None

    try:
        for f, frame in enumerate(frame_iter):
            if f >= n_frames:
                break
            if not any_present:
                if progress is not None:
                    progress.update(1)
                continue

            # Per-animal lookup (skipped entirely for absent animals).
            # Each `try` catches out-of-bounds rows/cols and leaves that
            # frame as zeros for the affected animal.
            # Column indices are the original F_3 - 1 with no extra
            # depth_x_offset, because the iterator already dropped the
            # RGB half and `frame` starts at the depth-map column.
            if red_present:
                row_R = F_4RE[f] - 1
                col_R = F_3RE[f] - 1
                try:
                    depthRed[f] = frame[row_R, col_R]
                except IndexError:
                    pass
            if white_present:
                row_W = F_4E[f] - 1
                col_W = F_3E[f] - 1
                try:
                    depthWhite[f] = frame[row_W, col_W]
                except IndexError:
                    pass
            if blue_present:
                row_B = F_4BE[f] - 1
                col_B = F_3BE[f] - 1
                try:
                    depthBlue[f] = frame[row_B, col_B]
                except IndexError:
                    pass
            if yellow_present:
                row_Y = F_4YE[f] - 1
                col_Y = F_3YE[f] - 1
                try:
                    depthYellow[f] = frame[row_Y, col_Y]
                except IndexError:
                    pass

            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    return {
        "depthRed": depthRed,
        "depthWhite": depthWhite,
        "depthBlue": depthBlue,
        "depthYellow": depthYellow,
    }
