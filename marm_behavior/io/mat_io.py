"""
mat_io
======

Read and write ``.mat`` files used throughout the pipeline.

Three file types are handled:

* **tracks** — per-animal body-part tracks produced by the extract
  stage. Contains the ``Red``, ``White``, ``Blue``, ``Yellow`` arrays.
* **edges** — per-animal edge matrices produced by the process stage.
  Contains the eight ``F_{3,4}{_,B,R,Y}E`` arrays.
* **depths** — per-animal pixel-depth lookups produced by the depths
  stage. Contains the four ``depth{Red,White,Blue,Yellow}`` arrays.

Files are written in v5/v7 format via :mod:`scipy.io` (zlib
compressed for ``edges`` and ``depths``, uncompressed for ``tracks``).
On read, the loader tries :mod:`scipy.io` first and falls back to
:mod:`h5py` for v7.3 (HDF5-based) files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# tracks
# ---------------------------------------------------------------------------

def load_tracks(path: "str | Path") -> Dict[str, np.ndarray]:
    """Load a ``tracks_*.mat`` file and return its named matrices.

    Returns a dict with whichever of ``Red``, ``White``, ``Blue``,
    ``Yellow`` are present. Missing variables are simply absent from
    the returned dict.
    """
    from scipy.io import loadmat

    raw = loadmat(str(path), squeeze_me=False, mat_dtype=True)
    return {
        name: np.asarray(raw[name], dtype=float)
        for name in ("Red", "White", "Blue", "Yellow")
        if name in raw
    }


def save_tracks(
    path: "str | Path", tracks: "Mapping[str, np.ndarray]"
) -> None:
    """Save a ``tracks_*.mat`` file.

    Only ``Red``, ``White``, ``Blue``, ``Yellow`` keys from ``tracks``
    are written; any other keys are silently skipped.
    """
    from scipy.io import savemat

    payload = {
        name: np.asarray(tracks[name], dtype=float)
        for name in ("Red", "White", "Blue", "Yellow")
        if name in tracks
    }
    savemat(str(path), payload, do_compression=False, oned_as="column")


# ---------------------------------------------------------------------------
# ground_normalized reference
# ---------------------------------------------------------------------------

def load_ground_normalized(
    path: "str | Path | None" = None,
) -> Dict[str, np.ndarray]:
    """Load ``Ground_3`` / ``Ground_4`` training-hull body-part clouds.

    Parameters
    ----------
    path:
        Optional path to a ``ground_normalized.mat`` file or a
        ``.npz`` file. If ``None`` (the default), loads the bundled
        copy shipped at ``marm_behavior/data/ground_normalized.npz``.

    Returns
    -------
    dict
        ``{'Ground_3': (N, 21) float, 'Ground_4': (N, 21) float}``.
    """
    if path is None:
        from .._data_files import ensure_data_file
        bundled = ensure_data_file("ground_normalized.npz")
        loaded = np.load(str(bundled))
        return {
            "Ground_3": np.asarray(loaded["Ground_3"], dtype=float),
            "Ground_4": np.asarray(loaded["Ground_4"], dtype=float),
        }

    path = Path(path)
    if path.suffix == ".npz":
        loaded = np.load(str(path))
        return {
            "Ground_3": np.asarray(loaded["Ground_3"], dtype=float),
            "Ground_4": np.asarray(loaded["Ground_4"], dtype=float),
        }

    from scipy.io import loadmat

    raw = loadmat(str(path), squeeze_me=False, mat_dtype=True)
    return {
        "Ground_3": np.asarray(raw["Ground_3"], dtype=float),
        "Ground_4": np.asarray(raw["Ground_4"], dtype=float),
    }


# ---------------------------------------------------------------------------
# edges / depths
# ---------------------------------------------------------------------------

#: The eight float32 matrices produced by the process stage and stored
#: in ``edges_*.mat``.
EDGE_VARS: Tuple[str, ...] = (
    "F_3E", "F_4E",
    "F_3BE", "F_4BE",
    "F_3RE", "F_4RE",
    "F_3YE", "F_4YE",
)

#: The four float32 matrices produced by the depths stage and stored
#: in ``depths_*.mat``.
DEPTH_VARS: Tuple[str, ...] = (
    "depthRed", "depthWhite", "depthBlue", "depthYellow",
)


def _load_mat_variables(
    path: "str | Path", var_names: "Iterable[str]"
) -> Dict[str, np.ndarray]:
    """Load a subset of variables from a ``.mat`` file.

    Tries :mod:`scipy.io` first (handles v5/v6/v7) and falls back to
    :mod:`h5py` for v7.3 (HDF5) files. Returns each variable as a
    float32 row-major ndarray.
    """
    var_list = list(var_names)

    try:
        from scipy.io import loadmat

        raw = loadmat(str(path), squeeze_me=False, mat_dtype=True)
        out: Dict[str, np.ndarray] = {}
        for name in var_list:
            if name in raw:
                out[name] = np.asarray(raw[name], dtype=np.float32)
        if out:
            return out
    except NotImplementedError:
        # scipy raises NotImplementedError for v7.3 files; fall through
        # to the HDF5 path below.
        pass
    except Exception:
        # Any other scipy failure: try h5py as a fallback.
        pass

    import h5py

    out = {}
    with h5py.File(str(path), "r") as f:
        for name in var_list:
            if name not in f:
                continue
            # v7.3 uses column-major on disk; transpose so downstream
            # Python code sees a row-major ``(n_rows, n_cols)`` shape.
            out[name] = np.asarray(f[name][...]).T.astype(
                np.float32, copy=False
            )
    return out


def load_edges(
    path: "str | Path",
    *,
    var_names: "Iterable[str]" = EDGE_VARS,
) -> Dict[str, np.ndarray]:
    """Load an ``edges_*.mat`` file and return the named matrices.

    Returns each array as float32 with shape ``(n_frames, n_edges)``.
    Handles both v7 (scipy-written) and v7.3 (HDF5) files.
    """
    return _load_mat_variables(path, var_names)


def save_edges(
    path: "str | Path", edges: "Mapping[str, np.ndarray]"
) -> None:
    """Write an ``edges_*.mat`` file.

    Uses v7 format with zlib compression. All arrays are cast to
    float32 before writing.
    """
    from scipy.io import savemat

    mat_dict: Dict[str, np.ndarray] = {}
    for name in EDGE_VARS:
        if name in edges:
            mat_dict[name] = np.asarray(edges[name], dtype=np.float32)
    savemat(
        str(path),
        mat_dict,
        format="5",
        do_compression=True,
        oned_as="column",
    )


def load_depths(
    path: "str | Path",
    *,
    var_names: "Iterable[str]" = DEPTH_VARS,
) -> Dict[str, np.ndarray]:
    """Load a ``depths_*.mat`` file into a ``{name: array}`` dict.

    Returns each array as float32 with shape ``(n_frames, 375)``.
    Handles both v7 (scipy-written) and v7.3 (HDF5) files.
    """
    return _load_mat_variables(path, var_names)


def save_depths(
    path: "str | Path", depths: "Mapping[str, np.ndarray]"
) -> None:
    """Write a ``depths_*.mat`` file.

    Uses v7 format with zlib compression. All arrays are cast to
    float32 before writing.
    """
    from scipy.io import savemat

    mat_dict: Dict[str, np.ndarray] = {}
    for name in DEPTH_VARS:
        if name in depths:
            mat_dict[name] = np.asarray(depths[name], dtype=np.float32)
    savemat(
        str(path),
        mat_dict,
        format="5",
        do_compression=True,
        oned_as="column",
    )
