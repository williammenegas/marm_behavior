"""
orchestrators
=============

Batch orchestrators that walk a folder of videos and run each
`the pipeline` / `the pipeline` driver scripts, and of


Each function walks the current working directory's ``*.avi`` files (matching
a ``*.avi`` glob) and runs the corresponding stage on
each.  Stages that are not yet implemented in this Stage-0/1 cut raise
``NotImplementedError``; the orchestrators are still useful as the eventual
"plug in" point and as documentation of which loop calls which module.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..depths.depths_1 import run_depths_1
from ..extract.extract_1 import run_extract_1
from ..extract.extract_2 import (
    assemble_frames,
    assign_by_track,
    fill_small_gaps_all,
    midpoint_assignment,
    reconcile_assignments,
    remove_pairwise_overlaps,
)
from ..extract.extract_3 import run_extract_3
from ..io.csv_io import read_multi_csv, read_single_csv
from ..io.mat_io import load_tracks, load_edges, save_depths, save_edges, save_tracks
from ..process.process_1 import make_all_animals
from ..process.process_2 import run_process_2
from ..process.process_3 import run_process_3
from ..process.process_4 import run_process_4


def list_avi_basenames(folder: str | Path = ".") -> List[str]:
    """Return ``*.avi`` filenames in ``folder``, without the ``.avi`` suffix.

    Mirrors ``dir('*.avi')``  scripts.  Sorted for determinism.
    """
    return sorted(
        os.path.splitext(p)[0]
        for p in glob.glob(os.path.join(str(folder), "*.avi"))
    )


# ---------------------------------------------------------------------------
# Full extract pipeline: single CSV + multi CSV -> tracks dict
# ---------------------------------------------------------------------------

def run_extract_pipeline(
    *,
    single_csv_path: str | Path,
    multi_csv_path: str | Path,
    ground_normalized_path: "str | Path | None" = None,
    c_thresh: float = 0.1,
    target_coverage: float = 1.0,
    focal_animal: "str | None" = None,
) -> dict[str, np.ndarray]:
    """Run the full Stage-1 extract pipeline on one video's CSVs.

    Parameters
    ----------
    single_csv_path:
        Path to the DLC single-animal ``*single.csv`` body-part file.
    multi_csv_path:
        Path to the DLC multi-animal ``*multi.csv`` track file.
    ground_normalized_path:
        Optional path to the ``ground_normalized.mat`` reference used by
        :mod:`extract_3` for the hull filter.
    c_thresh:
        Confidence threshold (default 0.1).
    target_coverage:
        Alpha-shape coverage target for the hull filter.
    focal_animal:
        Optional one-animal-mode switch. When set to one of
        ``'r'`` / ``'w'`` / ``'b'`` / ``'y'``, the extract pipeline
        skips the proximity-based ``assign_by_track`` chain and
        instead collapses every multi-CSV tracklet onto the focal
        animal via :func:`assign_tracklets_one_animal`. The other
        three colours' output matrices are returned as all-NaN. This
        is the correct behaviour when only one animal is in the
        video — DLC's head colour classifier was trained on
        four-animal data and routinely mislabels a single animal
        across colours, so the proximity-based assignment would
        otherwise scatter the focal animal's tracklets across
        whichever colours got mislabeled.

    Returns
    -------
    dict[str, np.ndarray]
        ``{'Red': ..., 'White': ..., 'Blue': ..., 'Yellow': ...}``,
        ready to hand to :func:`marm_behavior.io.mat_io.save_tracks`.
    """
    # Stage 1: head extraction. We still call this in one-animal mode
    # to share the multi-CSV preprocessing path below, but the heads
    # themselves are unused on the one-animal branch.
    fullpickle = read_single_csv(single_csv_path)
    heads = run_extract_1(fullpickle, c_thresh=c_thresh)

    # Stage 2a: assemble multi-CSV into per-(track, frame) rows.
    multi_raw = read_multi_csv(multi_csv_path)
    test_tracks = (multi_raw[:, 0] + 1).astype(int)
    test_frames = (multi_raw[:, 1] + 1).astype(int)
    body = multi_raw[:, 2:]
    keep_cols = [c for c in range(body.shape[1]) if (c % 4) != 3]
    body = body[:, keep_cols]

    n_frames = heads.redHead.shape[0]
    within = test_frames <= n_frames
    body = body[within]
    test_tracks = test_tracks[within]
    test_frames = test_frames[within]

    # Confidence threshold: NaN out body parts below the threshold.
    for g in range(2, body.shape[1], 3):
        bad = body[:, g] < c_thresh
        body[bad, g - 2] = np.nan
        body[bad, g - 1] = np.nan
    keep2 = [c for c in range(body.shape[1]) if (c % 3) != 2]
    test_assem = body[:, keep2]
    test_assem[test_assem == 0] = np.nan
    test_assem[:, 0::2] += 100.0

    # ------------------------------------------------------------------
    # Stage 2 branch: four-animal vs one-animal
    # ------------------------------------------------------------------
    if focal_animal is not None:
        if focal_animal not in ("r", "w", "b", "y"):
            raise ValueError(
                f"focal_animal must be one of 'r', 'w', 'b', 'y' or "
                f"None; got {focal_animal!r}"
            )

        from ..extract.extract_2 import assign_tracklets_one_animal

        # Pick the highest-quality tracklet per frame and stack into a
        # single 42-column matrix for the focal animal. Skips
        # midpoint_assignment, assign_by_track, reconcile_assignments,
        # and remove_pairwise_overlaps entirely — all of which assume
        # four animals are present and use the colour-classified heads
        # from the single CSV as anchors.
        Focal = assign_tracklets_one_animal(
            test_assem, test_tracks, test_frames, n_frames=n_frames
        )

        empty = np.full((n_frames, 42), np.nan)
        slots: dict[str, np.ndarray] = {
            "r": empty.copy(),
            "w": empty.copy(),
            "b": empty.copy(),
            "y": empty.copy(),
        }
        slots[focal_animal] = Focal
        Red, White, Blue, Yellow = slots["r"], slots["w"], slots["b"], slots["y"]

        # Small-gap fill on all four matrices. The three absent ones
        # are all-NaN and the fill is a no-op for them; the focal one
        # gets the same treatment it would receive in four-animal
        # mode.
        fill_small_gaps_all(Red=Red, White=White, Blue=Blue, Yellow=Yellow)
    else:
        # Four-animal mode: original logic.
        test_all = assemble_frames(test_assem, test_frames, test_tracks, n_frames)
        Red, White, Blue, Yellow = midpoint_assignment(
            test_all, heads.redHead, heads.whiteHead, heads.blueHead, heads.yellowHead
        )

        (Red1, White1, Blue1, Yellow1,
         test_assem, test_tracks, test_frames, _ntracking) = assign_by_track(
            test_assem, test_tracks, test_frames,
            heads.redHead, heads.whiteHead, heads.blueHead, heads.yellowHead,
            n_frames=n_frames,
        )

        reconciled = reconcile_assignments(
            Red=Red, White=White, Blue=Blue, Yellow=Yellow,
            Red1=Red1, White1=White1, Blue1=Blue1, Yellow1=Yellow1,
            redHead=heads.redHead, whiteHead=heads.whiteHead,
            blueHead=heads.blueHead, yellowHead=heads.yellowHead,
        )
        Red = reconciled["Red"]
        White = reconciled["White"]
        Blue = reconciled["Blue"]
        Yellow = reconciled["Yellow"]

        remove_pairwise_overlaps(Red=Red, White=White, Blue=Blue, Yellow=Yellow)
        fill_small_gaps_all(Red=Red, White=White, Blue=Blue, Yellow=Yellow)

    # Stage 3: hull filter (extract_3). Wraps each animal in a
    # try/except so absent ones (all-NaN) are skipped cleanly.
    final = run_extract_3(
        Red=Red, White=White, Blue=Blue, Yellow=Yellow,
        ground_normalized_path=ground_normalized_path,
        target_coverage=target_coverage,
    )
    return final


# ---------------------------------------------------------------------------
# loop_1_extract: process every *.avi in a folder
# ---------------------------------------------------------------------------

def loop_1_extract(
    folder: str | Path = ".",
    *,
    ground_normalized_path: "str | Path | None" = None,
    c_thresh: float = 0.1,
    target_coverage: float = 1.0,
    dlc_csv_suffix_single: str = "single.csv",
    dlc_csv_suffix_multi: str = "multi.csv",
) -> List[str]:
    """Python equivalent of `the pipeline`.

    For every ``*.avi`` in ``folder``, locate the matching
    ``*single.csv`` / ``*multi.csv`` pair produced by DeepLabCut, run the
    full extract pipeline, and save the result as ``tracks_<basename>.mat``.

    Returns the list of basenames that were processed successfully.
    """
    folder = Path(folder)
    bases = list_avi_basenames(folder)
    processed: List[str] = []
    for base in bases:
        # list_avi_basenames returns full paths (for backwards compat);
        # we need just the filename stem for globbing inside ``folder``.
        stem = os.path.basename(base)
        # The DLC CSVs are named ``<base>DLC_...<suffix>``; find the closest
        # match on disk.
        single_candidates = sorted(folder.glob(f"{stem}*{dlc_csv_suffix_single}"))
        multi_candidates = sorted(folder.glob(f"{stem}*{dlc_csv_suffix_multi}"))
        if not single_candidates or not multi_candidates:
            print(f"[loop_1_extract] missing CSV pair for {stem}, skipping")
            continue
        try:
            tracks = run_extract_pipeline(
                single_csv_path=single_candidates[0],
                multi_csv_path=multi_candidates[0],
                ground_normalized_path=ground_normalized_path,
                c_thresh=c_thresh,
                target_coverage=target_coverage,
            )
        except Exception as err:
            print(f"[loop_1_extract] {stem}: pipeline failed — {err}")
            continue
        out_path = folder / f"tracks_{stem}.mat"
        save_tracks(out_path, tracks)
        processed.append(stem)
    return processed


# ---------------------------------------------------------------------------
# Full process pipeline: tracks dict -> edges dict
# ---------------------------------------------------------------------------

def run_process_pipeline(
    tracks: Dict[str, np.ndarray],
    *,
    ground_normalized_path: "str | Path | None" = None,
    focal_animal: "str | None" = None,
) -> Dict[str, np.ndarray]:
    """Run the Stage-2 process pipeline on a tracks dict.

    Parameters
    ----------
    tracks:
        Dict with keys ``'Red'``, ``'White'``, ``'Blue'``, ``'Yellow'`` —
        the 42-column body-part matrices.
    ground_normalized_path:
        Path to the ground-normalized reference data.
    focal_animal:
        Optional one-animal-mode switch passed through to
        :func:`marm_behavior.process.process_3.run_process_3`. When set
        to one of ``'w'`` / ``'b'`` / ``'r'`` / ``'y'``, only that
        animal gets the full per-frame body-length computation; the
        other three get a constant ``bh = 30``. ``None`` (the default)
        applies the full computation to all four animals.

    Returns
    -------
    dict[str, np.ndarray]
        ``{'F_3E': ..., 'F_4E': ..., ..., 'F_4YE': ...}`` ready to be
        saved to an edges ``.mat`` file.
    """
    animals = make_all_animals(**tracks)
    p2 = run_process_2(animals)
    p3 = run_process_3(p2, focal_animal=focal_animal)
    p4 = run_process_4(p2, p3, animals, ground_normalized_path=ground_normalized_path)
    return p4.as_dict()


# ---------------------------------------------------------------------------
# loop_2_process: process every tracks_*.mat in a folder
# ---------------------------------------------------------------------------

def loop_2_process(
    folder: str | Path = ".",
    *,
    ground_normalized_path: "str | Path | None" = None,
) -> List[str]:
    """Python equivalent of `the pipeline`.

    For every ``tracks_*.mat`` in ``folder``, run :func:`run_process_pipeline`
    and save the result as ``edges_<basename>.mat``.  Returns the list of
    basenames that were processed successfully.
    """
    folder = Path(folder)
    processed: List[str] = []
    for tracks_file in sorted(folder.glob("tracks_*.mat")):
        base = tracks_file.stem.removeprefix("tracks_")
        try:
            tracks = load_tracks(tracks_file)
            edges = run_process_pipeline(
                tracks, ground_normalized_path=ground_normalized_path
            )
        except Exception as err:
            print(f"[loop_2_process] {base}: pipeline failed — {err}")
            continue
        out_path = folder / f"edges_{base}.mat"
        save_edges(out_path, edges)
        processed.append(base)
    return processed


# ---------------------------------------------------------------------------
# loop_3_get_depths: compute pixel depths from edges + video
# ---------------------------------------------------------------------------

def loop_3_get_depths(
    folder: str | Path = ".",
) -> List[str]:
    """Python equivalent of `the pipeline`.

    For every ``*.avi`` in ``folder`` that has a matching ``edges_*.mat``
    but no ``depths_*.mat``, read the video, run
    :func:`marm_behavior.depths.depths_1.run_depths_1` on its edges, and
    save the result as ``depths_<basename>.mat``.

    Returns the list of basenames that were processed.  Skips videos whose
    AVI file is missing or whose edges file cannot be read.
    """
    folder = Path(folder)
    processed: List[str] = []
    for avi_path in sorted(folder.glob("*.avi")):
        base = avi_path.stem
        edges_path = folder / f"edges_{base}.mat"
        depths_path = folder / f"depths_{base}.mat"
        if not edges_path.exists():
            print(f"[loop_3_get_depths] {base}: no edges_*.mat, skipping")
            continue
        if depths_path.exists():
            print(f"[loop_3_get_depths] {base}: depths already exist, skipping")
            continue
        try:
            edges = load_edges(edges_path)
            depths = run_depths_1(edges, avi_path)
        except Exception as err:
            print(f"[loop_3_get_depths] {base}: failed — {err}")
            continue
        save_depths(depths_path, depths)
        processed.append(base)
    return processed


# ---------------------------------------------------------------------------
# loop_4_get_features: compute description CSVs from edges + depths
# ---------------------------------------------------------------------------

def loop_4_get_features(
    folder: str | Path = ".",
    *,
    red_present: bool = True,
    white_present: bool = True,
    blue_present: bool = True,
    yellow_present: bool = True,
) -> List[str]:
    """Python equivalent of `the pipeline`.

    For every ``*.avi`` in ``folder`` that has matching ``edges_*.mat``
    and ``depths_*.mat`` files, compute the four binary description
    matrices via :func:`marm_behavior.features.labels.loop_4_get_features`
    and save them as ``{w,b,r,y}_description_<basename>.csv`` using
    -compatible comma-separated integer formatting.

    The per-color presence flags mirror ``animals_present.txt`` from the
     driver: colors set to False are skipped and produce no CSV.

    Returns
    -------
    list
        The basenames that were successfully processed.
    """
    from ..features.labels import loop_4_get_features as _label_loop

    folder = Path(folder)
    processed: List[str] = []
    for avi_path in sorted(folder.glob("*.avi")):
        base = avi_path.stem
        edges_path = folder / f"edges_{base}.mat"
        depths_path = folder / f"depths_{base}.mat"
        if not edges_path.exists():
            print(f"[loop_4_get_features] {base}: no edges_*.mat, skipping")
            continue
        if not depths_path.exists():
            print(f"[loop_4_get_features] {base}: no depths_*.mat, skipping")
            continue
        try:
            edges = load_edges(edges_path)
            from ..io.mat_io import load_depths
            depths = load_depths(depths_path)
            out = _label_loop(
                edges,
                depths,
                red_present=red_present,
                white_present=white_present,
                blue_present=blue_present,
                yellow_present=yellow_present,
            )
        except Exception as err:
            print(f"[loop_4_get_features] {base}: failed — {err}")
            continue
        # Write out each present color as {w,b,r,y}_description_<base>.csv
        # matching the csvwrite output: integers, comma-separated,
        # no header.
        for key, prefix in [("w", "w"), ("b", "b"), ("r", "r"), ("y", "y")]:
            if key not in out:
                continue
            csv_path = folder / f"{prefix}_description_{base}.csv"
            # csvwrite writes the values as reals with up to 15 significant
            # digits, but since mid_layer is {0,1} we save ints for
            # compactness (the csvread will still read them as doubles).
            np.savetxt(csv_path, out[key].astype(int), delimiter=",", fmt="%d")
        processed.append(base)
    return processed


# ---------------------------------------------------------------------------
# extract_process_4_animal / extract_process_1_animal
# ---------------------------------------------------------------------------

def extract_process_4_animal(
    folder: str | Path = ".",
    *,
    ground_normalized_path: "str | Path | None" = None,
    red_present: bool = True,
    white_present: bool = True,
    blue_present: bool = True,
    yellow_present: bool = True,
) -> None:
    """Python replacement for `the batch script`.

    Runs the four loops in sequence:

    1. `the pipeline` — DLC CSVs → ``tracks_<base>.mat``
    2. `the pipeline` — tracks → ``edges_<base>.mat``
    3. `the pipeline` — edges + AVI → ``depths_<base>.mat``
    4. `the pipeline` — edges + depths → ``{w,b,r,y}_description_<base>.csv``

     (there to let 's
    parallel pool settle between stages) are simply omitted.
    """
    loop_1_extract(folder, ground_normalized_path=ground_normalized_path)
    loop_2_process(folder, ground_normalized_path=ground_normalized_path)
    loop_3_get_depths(folder)
    loop_4_get_features(
        folder,
        red_present=red_present,
        white_present=white_present,
        blue_present=blue_present,
        yellow_present=yellow_present,
    )


def extract_process_1_animal(
    folder: str | Path = ".",
    *,
    ground_normalized_path: "str | Path | None" = None,
    red_present: bool = True,
    white_present: bool = True,
    blue_present: bool = True,
    yellow_present: bool = True,
) -> None:
    """Batch orchestrator for single-animal videos.

    The  project ships ``_ONE_ANIMAL`` variants of every loop and
    ``behavior_extract_*`` script.  Diffing them against the four-animal
    versions shows that  is byte-identical
    to `the pipeline`,  differs
    only in tolerating missing animals, and the rest are identical. In the
    Python port we collapse those differences into the per-color
    ``*_present`` flags plumbed through ``extract_process_4_animal``.
    """
    extract_process_4_animal(
        folder,
        ground_normalized_path=ground_normalized_path,
        red_present=red_present,
        white_present=white_present,
        blue_present=blue_present,
        yellow_present=yellow_present,
    )
