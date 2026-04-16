"""
marm_behavior.run
=================

Single-function entry point for the full pipeline.

:func:`run` takes a video path (plus optional overrides) and runs all
six stages end-to-end: DLC pose inference → body-part extraction →
posture processing → depth lookup → per-frame feature labelling → NN
cluster projection. Returns a dict of produced artifact paths.

The only required argument is ``video_path``; everything else has a
sensible default:

* the DLC CSVs default to ``<basename>*single.csv`` / ``<basename>*multi.csv``
  next to the video (and are produced by the ``dlc`` stage if missing)
* the output directory defaults to the video's parent folder
* the ``ground_normalized`` reference defaults to the copy bundled
  inside the package
* the NN reference folder defaults to the bundled
  ``marm_behavior/data/nn_reference/`` (ships pre-populated)
* all four animals are assumed present by default

Example
-------

>>> from marm_behavior import run
>>> result = run("experiments/test_4.avi")
>>> result["descriptions"]
{'w': Path('experiments/w_description_test_4.csv'),
 'b': Path('experiments/b_description_test_4.csv'),
 'r': Path('experiments/r_description_test_4.csv'),
 'y': Path('experiments/y_description_test_4.csv')}

Running just one stage against existing intermediates:

>>> run("experiments/test_4.avi", stages=("labels",))

Skipping a missing animal:

>>> run("experiments/test_4.avi", yellow_present=False)
"""

from __future__ import annotations

# Silence TensorFlow's chatty C++ logger before any submodule
# import path can pull in tensorflow. Set ``MARM_BEHAVIOR_TF_VERBOSE=1``
# to skip the silencing for debugging.
from ._tf_quiet import silence_tensorflow_logging
silence_tensorflow_logging()

from pathlib import Path
from typing import Iterable, Sequence

__all__ = ["run"]


#: Canonical stage names, in the order they run.
_STAGE_ORDER = ("dlc", "extract", "process", "depths", "labels", "nn")


def _resolve_csv(
    explicit: "str | Path | None",
    video_stem: str,
    parent: Path,
    suffix: str,
) -> Path:
    """Return an absolute path to one of the DLC CSV inputs.

    If ``explicit`` is given, use it. Otherwise search ``parent`` for files
    matching ``<video_stem>*<suffix>.csv``. DeepLabCut writes CSVs like
    ``<video>DLC_resnet50_<model>_shuffle<n>_<iters>_el.pickle<suffix>.csv``
    so the glob has to accept anything between the video stem and the
    ``single.csv`` / ``multi.csv`` trailing token.

    Returns the lexically-first match, or — if none exist — the simple
    ``<video_stem><suffix>.csv`` path so the caller's error message points
    at a reasonable default location.
    """
    if explicit is not None:
        return Path(explicit).expanduser().resolve()

    matches = sorted(parent.glob(f"{video_stem}*{suffix}.csv"))
    if matches:
        return matches[0].resolve()

    # Fall back to the simple un-suffixed form for the error message.
    return (parent / f"{video_stem}{suffix}.csv").resolve()


def _stage_set(stages: "Iterable[str] | None") -> set[str]:
    if stages is None:
        return set(_STAGE_ORDER)
    bad = [s for s in stages if s not in _STAGE_ORDER]
    if bad:
        raise ValueError(
            f"unknown stage(s) {bad!r}; choose from {_STAGE_ORDER}"
        )
    return set(stages)


def run(
    video_path: "str | Path",
    *,
    single_csv_path: "str | Path | None" = None,
    multi_csv_path: "str | Path | None" = None,
    output_dir: "str | Path | None" = None,
    ground_normalized_path: "str | Path | None" = None,
    dlc_config: "str | Path | None" = None,
    nn_reference_dir: "str | Path | None" = None,
    nn_artifacts: "dict[str, object] | None" = None,
    nn_buddies: "dict[str, list[str]] | None" = None,
    red_present: bool = True,
    white_present: bool = True,
    blue_present: bool = True,
    yellow_present: bool = True,
    c_thresh: float = 0.1,
    target_coverage: float = 1.0,
    stages: "Sequence[str] | None" = None,
    verbose: bool = True,
) -> "dict[str, object]":
    """Run the full marm_behavior pipeline on one video.

    A single-function entry point that runs DLC inference, body-part
    extraction, posture processing, depth lookup, feature labelling,
    and NN cluster projection in sequence.

    Parameters
    ----------
    video_path:
        Path to the input ``.avi`` (stereo RGB-plus-depth). This is the
        only required argument.
    single_csv_path, multi_csv_path:
        Paths to the DLC ``<name>single.csv`` and ``<name>multi.csv``
        files. If either is omitted, the function looks for them next to
        the video with the usual naming convention (``<stem>single.csv``
        / ``<stem>multi.csv``, falling back to ``<stem>_single.csv`` /
        ``<stem>_multi.csv``).
    output_dir:
        Directory to write intermediate and final artifacts into. If
        omitted, uses the video's parent folder. Created if missing.
    ground_normalized_path:
        Path to the ``ground_normalized`` training-hull reference.
        Accepts either a ``.mat`` or ``.npz`` file. If omitted (the
        default), uses the copy bundled inside ``marm_behavior/data/``.
    dlc_config:
        Optional path to a user-supplied DLC ``config.yaml``. If given,
        the Stage 0 ``dlc`` step uses this project instead of the bundled
        model — useful for swapping in a retrained model without
        rebuilding the package.
    nn_reference_dir:
        Folder containing the reference files needed by the ``nn``
        stage. Defaults to the bundled ``data/nn_reference/`` inside
        the package, which ships pre-populated with the canonical
        reference set (4 small CSVs + native t-SNE cache). Override
        only if you have your own reference data.
    nn_artifacts:
        Optional pre-built nn-stage artifacts dict from
        :func:`marm_behavior.nn_postprocess.prepare_nn_artifacts`.
        When passed, the per-call setup work (load references, fit
        openTSNE, build KNN — about 5 minutes) is skipped, and the
        provided objects are used directly. Used by batch mode to
        amortize the openTSNE fit across many videos. Output is
        bit-identical either way.
    nn_buddies:
        Optional per-animal override for which *other* animal's
        description CSV gets concatenated with each self animal's in
        the NN stage. Format: ``{self_color: [buddy_color, ...]}`` where
        colors are short keys (``'r'``, ``'w'``, ``'b'``, ``'y'``).
        Missing keys keep their defaults (Red->Yellow/Blue,
        White->Blue/Red, Blue->White/Red, Yellow->Red/Blue). Example::

            nn_buddies={'r': ['b'], 'y': ['w', 'b']}

        says "always pair Red with Blue; pair Yellow with White first,
        falling back to Blue if White isn't available". Only matters
        if the ``nn`` stage runs.
    red_present, white_present, blue_present, yellow_present:
        Whether each colour is expected to be present in the video. A
        missing colour's downstream artifacts are written as all-zeros.
    c_thresh:
        DLC confidence threshold applied in the extract stage
        (default: 0.1).
    target_coverage:
        Alpha-shape hull coverage target in the extract stage
        (default: 1.0).
    stages:
        Which pipeline stages to run, as a subset of
        ``("dlc", "extract", "process", "depths", "labels", "nn")``.
        Defaults to all six. Useful for re-running a single stage
        against existing intermediates, or for skipping ``dlc`` when
        you already have the pose CSVs on disk.
    verbose:
        If True (the default), prints a one-line progress message before
        each stage. Set to False to run silently.

    Returns
    -------
    dict
        ``{``
        ``  "video": Path,``
        ``  "output_dir": Path,``
        ``  "stages_run": list[str],``
        ``  "single_csv": Path | None,     # if dlc ran or auto-found``
        ``  "multi_csv": Path | None,``
        ``  "tracks": Path | None,         # if extract ran``
        ``  "edges": Path | None,          # if process ran``
        ``  "depths": Path | None,         # if depths ran``
        ``  "descriptions": dict[str, Path],  # if labels ran``
        ``  "nn": dict[str, tuple],            # if nn ran``
        ``}``
    """
    video = Path(video_path).expanduser().resolve()
    stages_to_run = _stage_set(stages)
    video_needed = bool(stages_to_run & {"dlc", "depths"})
    if video_needed and not video.exists():
        raise FileNotFoundError(f"video not found: {video}")

    # Auto-download any missing bundled data files before doing
    # anything else. Idempotent and fast on the common path; on a
    # fresh install this triggers a one-time ~310 MB download from
    # the Hugging Face Hub. See marm_behavior._data_files for
    # override env vars (MARM_BEHAVIOR_DATA_DIR for shared installs,
    # MARM_BEHAVIOR_HF_REPO for a lab-internal mirror).
    from ._data_files import ensure_all
    ensure_all(verbose=verbose)

    # ------------------------------------------------------------------
    # One-animal mode auto-detection.
    #
    # When exactly one of the four ``*_present`` flags is True, switch
    # the process and depths stages into a focal-animal-only code path:
    #   * process_3 uses constant ``bh = 30`` for the three absent
    #     animals and the full clamp/fallback only for the focal one
    #   * depths_1's per-frame inner loop only runs for the focal
    #     animal (already gated by the existing ``*_present`` kwargs)
    #   * the nn stage is skipped automatically because there is no
    #     buddy animal to pair with
    #
    # When zero, two, three, or four animals are present, behavior is
    # unchanged from the four-animal pipeline.
    # ------------------------------------------------------------------
    _present_map = {
        "r": red_present,
        "w": white_present,
        "b": blue_present,
        "y": yellow_present,
    }
    _present_keys = [k for k, v in _present_map.items() if v]
    one_animal_mode = len(_present_keys) == 1
    focal_animal: "str | None" = _present_keys[0] if one_animal_mode else None
    _color_full_names = {
        "r": "Red", "w": "White", "b": "Blue", "y": "Yellow"
    }

    # Resolve output folder.
    if output_dir is None:
        out_dir = video.parent
    else:
        out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base = video.stem

    # Resolve the DLC CSVs (which may not exist yet if the dlc stage is
    # going to create them). If the user gave explicit paths, take them.
    # Otherwise we first look in out_dir (where the dlc stage writes)
    # before falling back to the video's parent folder.
    single_csv = _resolve_csv(single_csv_path, base, out_dir, "single")
    if not single_csv.exists():
        single_csv = _resolve_csv(single_csv_path, base, video.parent, "single")
    multi_csv = _resolve_csv(multi_csv_path, base, out_dir, "multi")
    if not multi_csv.exists():
        multi_csv = _resolve_csv(multi_csv_path, base, video.parent, "multi")

    # We deliberately import inside the function so the module itself is
    # importable with zero side effects (so tooling like ``inspect.signature``
    # and IDE linters don't trip on scipy / opencv being unavailable).
    from .extract.extract_1 import run_extract_1
    from .extract.extract_2 import (
        assemble_frames,
        assign_by_track,
        fill_small_gaps_all,
        midpoint_assignment,
        reconcile_assignments,
        remove_pairwise_overlaps,
    )
    from .extract.extract_3 import run_extract_3
    from .io.csv_io import read_multi_csv, read_single_csv
    from .io.mat_io import (
        load_depths,
        load_edges,
        load_tracks,
        save_depths,
        save_edges,
        save_tracks,
    )
    from .features.labels import loop_4_get_features as _label_loop
    from .pipeline.orchestrators import run_extract_pipeline, run_process_pipeline

    # Artifact file paths (computed up-front so subsequent stages can load them).
    tracks_path = out_dir / f"tracks_{base}.mat"
    edges_path = out_dir / f"edges_{base}.mat"
    depths_path = out_dir / f"depths_{base}.mat"

    def _log(msg: str) -> None:
        if verbose:
            print(f"[marm_behavior] {msg}")

    # Announce one-animal mode prominently.
    if one_animal_mode:
        _log(
            f"ONE-ANIMAL MODE: only {_color_full_names[focal_animal]} "
            f"present"
        )
        _log(
            "  extract: all multi-CSV tracklets are assigned to the "
            "focal animal (no proximity matching to colour-classified "
            "heads)"
        )
        _log(
            "  process: non-focal animals will use constant bh = 30"
        )
        _log(
            "  depths:  per-frame lookup runs only for the focal animal"
        )
        if "nn" in stages_to_run:
            _log(
                "  nn:      stage will be skipped (no buddy animal "
                "available in one-animal mode)"
            )

    stages_run: list[str] = []

    # ------------------------------------------------------------------
    # Stage 0: dlc (video → <base>DLC_..._el.pickle{single,multi}.csv)
    # ------------------------------------------------------------------
    if "dlc" in stages_to_run:
        _log(f"dlc:     {video.name} -> <stem>DLC_..._el.pickle{{single,multi}}.csv")
        from .dlc_inference import run_dlc_inference

        dlc_out = run_dlc_inference(
            video_path=video,
            output_dir=out_dir,
            dlc_config_override=dlc_config,
            verbose=verbose,
        )
        single_csv = dlc_out["single_csv"]
        multi_csv = dlc_out["multi_csv"]
        stages_run.append("dlc")

    # ------------------------------------------------------------------
    # Stage 1: extract (DLC CSVs → tracks_<base>.mat)
    # ------------------------------------------------------------------
    if "extract" in stages_to_run:
        _log(f"extract: {single_csv.name} + {multi_csv.name} -> {tracks_path.name}")
        if not single_csv.exists():
            raise FileNotFoundError(
                f"single CSV not found: {single_csv} "
                f"(searched next to {video.name})"
            )
        if not multi_csv.exists():
            raise FileNotFoundError(
                f"multi CSV not found: {multi_csv} "
                f"(searched next to {video.name})"
            )
        tracks = run_extract_pipeline(
            single_csv_path=single_csv,
            multi_csv_path=multi_csv,
            ground_normalized_path=ground_normalized_path,
            c_thresh=c_thresh,
            target_coverage=target_coverage,
            focal_animal=focal_animal,
        )
        save_tracks(str(tracks_path), tracks)
        stages_run.append("extract")

    # ------------------------------------------------------------------
    # Stage 2: process (tracks_<base>.mat → edges_<base>.mat)
    # ------------------------------------------------------------------
    if "process" in stages_to_run:
        _log(f"process: {tracks_path.name} -> {edges_path.name}")
        if not tracks_path.exists():
            raise FileNotFoundError(
                f"tracks file not found: {tracks_path}; "
                f"run stage 'extract' first or provide it manually"
            )
        tracks = load_tracks(str(tracks_path))
        edges = run_process_pipeline(
            tracks=tracks,
            ground_normalized_path=ground_normalized_path,
            focal_animal=focal_animal,
        )
        save_edges(str(edges_path), edges)
        stages_run.append("process")

    # ------------------------------------------------------------------
    # Stage 3: depths (edges + video → depths_<base>.mat)
    # ------------------------------------------------------------------
    if "depths" in stages_to_run:
        _log(f"depths:  {edges_path.name} + {video.name} -> {depths_path.name}")
        if not edges_path.exists():
            raise FileNotFoundError(
                f"edges file not found: {edges_path}; "
                f"run stage 'process' first or provide it manually"
            )
        if not video.exists():
            raise FileNotFoundError(
                f"video not found: {video}; cannot run 'depths' stage "
                f"without an input video"
            )
        from .depths.depths_1 import run_depths_1

        edges = load_edges(str(edges_path))
        depths = run_depths_1(
            edges,
            str(video),
            red_present=red_present,
            white_present=white_present,
            blue_present=blue_present,
            yellow_present=yellow_present,
        )
        save_depths(str(depths_path), depths)
        stages_run.append("depths")

    # ------------------------------------------------------------------
    # Stage 4: labels (edges + depths → {w,b,r,y}_description_<base>.csv)
    # ------------------------------------------------------------------
    description_paths: "dict[str, Path]" = {}
    if "labels" in stages_to_run:
        _log(
            f"labels:  {edges_path.name} + {depths_path.name} -> "
            f"{{w,b,r,y}}_description_{base}.csv"
        )
        if not edges_path.exists():
            raise FileNotFoundError(
                f"edges file not found: {edges_path}; "
                f"run stage 'process' first or provide it manually"
            )
        if not depths_path.exists():
            raise FileNotFoundError(
                f"depths file not found: {depths_path}; "
                f"run stage 'depths' first or provide it manually"
            )
        edges = load_edges(str(edges_path))
        depths = load_depths(str(depths_path))
        out = _label_loop(
            edges,
            depths,
            red_present=red_present,
            white_present=white_present,
            blue_present=blue_present,
            yellow_present=yellow_present,
        )
        import numpy as np

        for key in ("w", "b", "r", "y"):
            if key not in out:
                continue
            csv_path = out_dir / f"{key}_description_{base}.csv"
            np.savetxt(
                csv_path,
                out[key].astype(int),
                delimiter=",",
                fmt="%d",
            )
            description_paths[key] = csv_path
        stages_run.append("labels")

    # ------------------------------------------------------------------
    # Stage 5: nn (descriptions → hcoord_* + hlabel_* per animal)
    # ------------------------------------------------------------------
    nn_results: dict = {}
    if "nn" in stages_to_run and one_animal_mode:
        _log(
            "nn:      skipped (one-animal mode; no buddy animal "
            "available)"
        )
    elif "nn" in stages_to_run:
        _log(f"nn:      {base} -> hcoord_* + hlabel_* CSVs")
        from .nn_postprocess import run_nn_stage

        # Assemble the description-CSV map: either from the labels
        # stage we just ran, or (if the user re-ran only the nn stage)
        # from files already on disk.
        desc_for_nn: dict = dict(description_paths)
        for k in ("r", "w", "b", "y"):
            if k not in desc_for_nn:
                p = out_dir / f"{k}_description_{base}.csv"
                if p.exists():
                    desc_for_nn[k] = p

        nn_results = run_nn_stage(
            description_csvs=desc_for_nn,
            video_path=video,
            output_dir=out_dir,
            reference_dir=nn_reference_dir,
            artifacts=nn_artifacts,
            buddies=nn_buddies,
            red_present=red_present,
            white_present=white_present,
            blue_present=blue_present,
            yellow_present=yellow_present,
            verbose=verbose,
        )
        stages_run.append("nn")

    return {
        "video": video,
        "output_dir": out_dir,
        "stages_run": stages_run,
        "single_csv": single_csv if single_csv.exists() else None,
        "multi_csv": multi_csv if multi_csv.exists() else None,
        "tracks": tracks_path if "extract" in stages_run else (
            tracks_path if tracks_path.exists() else None
        ),
        "edges": edges_path if "process" in stages_run else (
            edges_path if edges_path.exists() else None
        ),
        "depths": depths_path if "depths" in stages_run else (
            depths_path if depths_path.exists() else None
        ),
        "descriptions": description_paths,
        "nn": nn_results,
    }
