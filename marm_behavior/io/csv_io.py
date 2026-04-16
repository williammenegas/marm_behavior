"""
csv_io
======

Read the two pose-estimation CSV flavours consumed by the extract
stage:

* ``*single.csv`` — single-animal body-part predictions, one row per
  frame, whitespace-delimited, with ``(x, y, confidence, _)``
  quadruples for each body part.
* ``*multi.csv`` — multi-animal tracklet predictions with a leading
  ``[track_id, frame_idx]`` pair followed by ``(x, y, confidence, _)``
  quadruples for each body part, one row per (track, frame).

Both files are produced by :mod:`marm_behavior.el_to_csv` from the
DeepLabCut tracklet pickle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_single_csv(path: "str | Path") -> np.ndarray:
    """Read a ``*single.csv`` body-part file as a float matrix.

    Uses ``np.genfromtxt`` with whitespace delimiting. Empty cells
    become NaN.
    """
    return np.genfromtxt(str(path), delimiter=None, dtype=float)


def read_multi_csv(path: "str | Path") -> np.ndarray:
    """Read a ``*multi.csv`` tracklet file as a float matrix.

    Columns are ``[track_id, frame_idx, x1, y1, c1, _, x2, y2, c2, _, ...]``
    with the trailing underscore column dropped by the extract stage.
    Uses the same whitespace delimiter convention as
    :func:`read_single_csv`.
    """
    return np.genfromtxt(str(path), delimiter=None, dtype=float)
