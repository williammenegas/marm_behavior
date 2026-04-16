"""
marm_behavior.el_to_csv
=======================

Convert a DeepLabCut ``_el.pickle`` tracklet file into the two
whitespace-delimited CSV files that the extract stage reads:

* ``<video>DLC_..._el.picklesingle.csv`` — unique body parts
  (5 parts × 3 coords, right-padded with zeros to 20 columns)
* ``<video>DLC_..._el.picklemulti.csv`` — multi-animal tracklets, one
  row per (track_id, frame) with columns
  ``[track_id, frame_idx, x1, y1, c1, _, x2, y2, c2, _, ...]`` for the
  21 multi-animal body parts.

This step runs between ``deeplabcut.convert_detections2tracklets``
and the extract stage, as part of the ``dlc`` pipeline stage.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np


#: Number of unique animals represented in the ``single`` section of
#: the pickle (one "head" body part per animal).
NUM_ANIMALS = 5

#: Number of multi-animal body parts per tracklet.
NUM_JOINTS = 21


def el_pickle_to_csvs(
    el_pickle_path: "str | Path",
    *,
    output_dir: "str | Path | None" = None,
) -> "Tuple[Path, Path]":
    """Convert a DLC ``_el.pickle`` file into the two extract-stage CSVs.

    Parameters
    ----------
    el_pickle_path:
        Path to the ``<stem>DLC_..._el.pickle`` file produced by
        ``deeplabcut.convert_detections2tracklets``.
    output_dir:
        Where to write the two CSVs. Defaults to the pickle's parent
        folder.

    Returns
    -------
    (single_csv, multi_csv):
        Paths to the two CSV files this function wrote. Filenames are
        ``<el_pickle_stem>single.csv`` and ``<el_pickle_stem>multi.csv``
        — the ``.pickle`` in the middle of the name is literally part
        of the output filename, not a file extension.
    """
    el_pickle_path = Path(el_pickle_path).expanduser().resolve()
    if not el_pickle_path.exists():
        raise FileNotFoundError(f"_el.pickle not found: {el_pickle_path}")
    if output_dir is None:
        output_dir = el_pickle_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the pickle. It's a dict with keys:
    #   'single' → dict[frame_index → (5, 3) ndarray]
    #   'header' → pandas MultiIndex (we don't use it)
    #   <int>   → dict[frame_label → (21, 4) ndarray]  per track
    with open(el_pickle_path, "rb") as f:
        edata = pickle.load(f)

    e_single = edata.pop("single")
    edata.pop("header", None)

    # ------------------------------------------------------------------
    # Collect single-animal body parts: one row per frame, 20 cols.
    # ------------------------------------------------------------------
    nframes = list(e_single)[-1]
    # 20 cols = NUM_ANIMALS * 4 (x, y, confidence, _ per animal).
    collected_single = np.zeros((nframes, NUM_ANIMALS * 4))
    single_skipped = 0
    for n in range(nframes):
        try:
            flat = np.reshape(e_single[n], (1, -1))
            bound = flat.size
            collected_single[n, 0:bound] = flat
        except KeyError:
            single_skipped += 1

    # ------------------------------------------------------------------
    # Collect multi-animal tracklets: one row per (track_id, frame),
    # 2 + NUM_JOINTS*4 = 86 cols. Over-allocate a scratch buffer, then
    # trim to the number of rows actually written.
    # ------------------------------------------------------------------
    ntracks = max(
        (k for k in edata.keys() if isinstance(k, (int, np.integer))),
        default=-1,
    )
    collected_multi = np.zeros((9_000_000, NUM_JOINTS * 4 + 2))
    multi_skipped = 0
    multi_count = 0
    for n in range(ntracks + 1):
        try:
            track = edata[n]
            frame_keys = list(track.keys())
            for fk in frame_keys:
                collected_multi[multi_count, 0] = n
                # Frame labels look like 'frame00042'; chars [5:12]
                # are the frame number.
                current_frame_number = int(fk[5:12])
                collected_multi[multi_count, 1] = current_frame_number
                frame_data = np.reshape(track[fk], (1, -1))
                collected_multi[multi_count, 2:86] = frame_data
                multi_count += 1
        except KeyError:
            multi_skipped += 1
    collected_multi = collected_multi[0:multi_count, :]

    # ------------------------------------------------------------------
    # Write the two CSVs with np.savetxt defaults (%.18e scientific
    # notation, space delimiter).
    # ------------------------------------------------------------------
    stem = el_pickle_path.name
    single_csv = output_dir / f"{stem}single.csv"
    multi_csv = output_dir / f"{stem}multi.csv"
    np.savetxt(single_csv, collected_single)
    np.savetxt(multi_csv, collected_multi)
    return single_csv, multi_csv
