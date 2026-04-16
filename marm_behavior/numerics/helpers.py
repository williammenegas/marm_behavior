"""
helpers
=======

Numerical primitives used throughout the pipeline: rolling-window
reductions, level-crossing detection, piecewise-cubic NaN fill, and
small nan-aware wrappers around NumPy aggregations.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# level crossings
# ---------------------------------------------------------------------------

def crossing(
    S: np.ndarray,
    t: Optional[np.ndarray] = None,
    level: float = 0.0,
    method: str = "linear",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the indices at which ``S`` crosses ``level``.

    Parameters
    ----------
    S:
        1-D signal.
    t:
        Optional abscissa; defaults to 1..len(S).
    level:
        Crossing level (default 0).
    method:
        ``"linear"`` for linearly-interpolated crossing times, or any
        other value for raw sample indices.

    Returns
    -------
    (ind, t0, s0):
        ``ind`` is a 0-based array of sample indices where a crossing
        occurred. ``t0`` is the corresponding interpolated crossing
        times (same length as ``ind``); ``s0`` is the signal residual
        at each crossing (always 0 for ``method="linear"``).
    """
    S = np.asarray(S, dtype=float).ravel()
    if t is None:
        t = np.arange(1, len(S) + 1, dtype=float)
    else:
        t = np.asarray(t, dtype=float).ravel()
        if len(t) != len(S):
            raise ValueError("t and S must be of identical length")

    S = S - level
    ind0 = np.where(S == 0)[0]
    S1 = S[:-1] * S[1:]
    ind1 = np.where(S1 < 0)[0]
    ind = np.sort(np.concatenate([ind0, ind1])).astype(int)

    t0 = t[ind].astype(float).copy()
    s0 = S[ind].astype(float).copy()

    if method == "linear":
        for ii in range(len(t0)):
            here = S[ind[ii]]
            if abs(here) > np.spacing(here):
                num = t[ind[ii] + 1] - t[ind[ii]]
                den = S[ind[ii] + 1] - S[ind[ii]]
                delta = num / den
                t0[ii] = t0[ii] - here * delta
                s0[ii] = 0.0

    return ind, t0, s0


# ---------------------------------------------------------------------------
# rolling-window reductions
# ---------------------------------------------------------------------------

def _centered_window_bounds(i: int, n: int, k: int) -> Tuple[int, int]:
    """Centered-window bounds for a length-``k`` window at position
    ``i`` in a length-``n`` array.

    For odd ``k``, the window is symmetric. For even ``k``, the window
    is biased backward by one element. Edges truncate.
    """
    half_back = k // 2
    half_fwd = k - half_back - 1
    return max(0, i - half_back), min(n, i + half_fwd + 1)


def movmedian(x: np.ndarray, k: int, omitnan: bool = True) -> np.ndarray:
    """Centered rolling median over a length-``k`` window.

    When ``omitnan=True`` and a window is entirely NaN, the result at
    that index is NaN. Vectorised via
    :func:`numpy.lib.stride_tricks.sliding_window_view`.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n == 0:
        return x.copy()

    half_back = k // 2
    half_fwd = k - half_back - 1

    if omitnan:
        pad = np.concatenate(
            [np.full(half_back, np.nan), x, np.full(half_fwd, np.nan)]
        )
        from numpy.lib.stride_tricks import sliding_window_view
        wins = sliding_window_view(pad, k)
        import warnings
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"All-NaN slice encountered",
                category=RuntimeWarning,
            )
            return np.nanmedian(wins, axis=1)

    out = np.empty(n)
    with np.errstate(invalid="ignore"):
        for i in range(n):
            lo, hi = _centered_window_bounds(i, n, k)
            out[i] = np.median(x[lo:hi])
    return out


def movmean(x: np.ndarray, k: int, omitnan: bool = True) -> np.ndarray:
    """Centered rolling mean over a length-``k`` window.

    Fully vectorised in O(n) via a cumulative-sum trick, independent
    of the window size. Handles NaN by tracking a per-window count of
    valid samples and dividing only by that count.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n == 0:
        return x.copy()

    half_back = k // 2
    half_fwd = k - half_back - 1

    i = np.arange(n)
    lo = np.maximum(0, i - half_back)
    hi = np.minimum(n, i + half_fwd + 1)

    if omitnan:
        mask = (~np.isnan(x)).astype(np.int64)
        x_zero = np.where(mask == 1, x, 0.0)
        csum = np.concatenate(([0.0], np.cumsum(x_zero)))
        cmask = np.concatenate(([0], np.cumsum(mask)))
        win_sum = csum[hi] - csum[lo]
        win_count = cmask[hi] - cmask[lo]
        with np.errstate(invalid="ignore"):
            out = np.where(
                win_count > 0,
                win_sum / np.maximum(win_count, 1),
                np.nan,
            )
    else:
        csum = np.concatenate(([0.0], np.cumsum(x)))
        win_sum = csum[hi] - csum[lo]
        out = win_sum / (hi - lo)
    return out


# ---------------------------------------------------------------------------
# NaN-aware interpolation
# ---------------------------------------------------------------------------

def fillmissing_pchip(x: np.ndarray) -> np.ndarray:
    """Fill NaN entries in ``x`` by piecewise cubic Hermite interpolation.

    Uses the Fritsch-Carlson monotonic formulation via
    :class:`scipy.interpolate.PchipInterpolator`, with extrapolation
    enabled so leading and trailing NaNs are filled too. If the entire
    input is NaN the output is unchanged; if exactly one sample is
    non-NaN the output is filled with that value.
    """
    from scipy.interpolate import PchipInterpolator

    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    valid = ~np.isnan(x)
    nvalid = int(valid.sum())
    if nvalid == 0:
        return x.copy()
    if nvalid == n:
        return x.copy()
    if nvalid == 1:
        return np.full(n, x[valid][0])

    idx = np.arange(n)
    interp = PchipInterpolator(idx[valid], x[valid], extrapolate=True)
    out = x.copy()
    out[~valid] = interp(idx[~valid])
    return out


def fillmissing_pchip_2d(x: np.ndarray) -> np.ndarray:
    """Apply :func:`fillmissing_pchip` independently to every column
    of a 2-D array.
    """
    x = np.asarray(x, dtype=float)
    out = x.copy()
    for j in range(x.shape[1]):
        out[:, j] = fillmissing_pchip(x[:, j])
    return out


# ---------------------------------------------------------------------------
# coordinate conversions
# ---------------------------------------------------------------------------

def cart2pol(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian ``(x, y)`` to polar ``(theta, rho)``.

    ``theta = atan2(y, x)`` in radians, ``rho = hypot(x, y)``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.arctan2(y, x), np.hypot(x, y)


def pol2cart(
    theta: np.ndarray, rho: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert polar ``(theta, rho)`` to Cartesian ``(x, y)``."""
    theta = np.asarray(theta, dtype=float)
    rho = np.asarray(rho, dtype=float)
    return rho * np.cos(theta), rho * np.sin(theta)


# ---------------------------------------------------------------------------
# NaN-aware aggregations
# ---------------------------------------------------------------------------

def nanmean(x, axis=None):
    """Wrapper around :func:`numpy.nanmean` that suppresses all-NaN
    warnings.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered"
        )
        with np.errstate(invalid="ignore"):
            return np.nanmean(x, axis=axis)


def nansum(x, axis=None):
    """Thin wrapper around :func:`numpy.nansum`."""
    return np.nansum(x, axis=axis)


def nanmin(x, axis=None):
    """Return ``(min_value, argmin_index)`` ignoring NaNs.

    For a 1-D input, returns ``(float, int)``. For a 2-D input with
    ``axis=None``, reduces along axis 0 and returns ``(array, array)``.
    """
    x = np.asarray(x, dtype=float)
    with np.errstate(invalid="ignore"):
        if x.ndim == 1:
            i = int(np.nanargmin(x))
            return float(x[i]), i
        if axis is None:
            axis = 0
        return np.nanmin(x, axis=axis), np.nanargmin(x, axis=axis)


def nanmedian(x, axis=None):
    """Thin wrapper around :func:`numpy.nanmedian`."""
    with np.errstate(invalid="ignore"):
        return np.nanmedian(x, axis=axis)
