"""
Command-line entry point for marm_behavior.

Invoke as::

    python -m marm_behavior                       # process all .avi in CWD
    python -m marm_behavior path/to/video.avi     # process one video
    python -m marm_behavior path/to/video_dir/    # process all .avi in dir

See ``python -m marm_behavior --help`` for all options.
"""

from __future__ import annotations

# Silence TensorFlow's chatty C++ logger BEFORE any import path
# that could pull in tensorflow (.run -> .nn_postprocess -> tf, or
# .run -> .dlc_inference -> tf via deeplabcut). Must run before
# `import tensorflow` happens anywhere in the process.
from ._tf_quiet import silence_tensorflow_logging
silence_tensorflow_logging()

import argparse
import sys
from pathlib import Path

from .run import run


#: Short colour keys and their full names, used for the per-animal
#: buddy-override CLI flags.
_COLOR_FLAGS = (
    ("red", "r"),
    ("white", "w"),
    ("blue", "b"),
    ("yellow", "y"),
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m marm_behavior",
        description=(
            "Run the marmoset behavioral analysis pipeline on one or "
            "more videos. With no positional argument, processes every "
            "``.avi`` in the current working directory. Outputs land "
            "next to each input video (or in --output-dir)."
        ),
    )
    p.add_argument(
        "video",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Path to an input ``.avi`` file, or a directory containing "
            "``.avi`` files. If omitted, uses the current working "
            "directory."
        ),
    )
    p.add_argument(
        "--single-csv",
        type=Path,
        default=None,
        help=(
            "Path to the DLC *single.csv file. Default: auto-discovered "
            "next to the video via glob '<video_stem>*single.csv'. "
            "Ignored in multi-video (batch) mode."
        ),
    )
    p.add_argument(
        "--multi-csv",
        type=Path,
        default=None,
        help=(
            "Path to the DLC *multi.csv file. Default: auto-discovered "
            "next to the video via glob '<video_stem>*multi.csv'. "
            "Ignored in multi-video (batch) mode."
        ),
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write artifacts into. Default: each video's parent folder.",
    )
    p.add_argument(
        "--ground-normalized",
        type=Path,
        default=None,
        help=(
            "Path to a ground_normalized.mat or .npz file. Default: the copy "
            "bundled inside the package — no external file needed."
        ),
    )
    p.add_argument(
        "--dlc-config",
        type=Path,
        default=None,
        help=(
            "Path to a user-supplied DLC config.yaml. Default: use the "
            "trained model bundled inside the package. Only matters if "
            "the 'dlc' stage runs."
        ),
    )
    p.add_argument(
        "--nn-reference-dir",
        type=Path,
        default=None,
        help=(
            "Override the folder containing the NN reference files "
            "(out_inner_mean1.csv, out_inner_std1.csv, "
            "tsne_temp1_1.csv, dbscan_temp1_1.csv, plus the native "
            "t-SNE cache files). Default: the canonical reference "
            "data bundled inside the package at "
            "marm_behavior/data/nn_reference/. You normally don't "
            "need this flag — only override if you have your own "
            "reference set (e.g. for a different cohort). Only "
            "matters if the 'nn' stage runs."
        ),
    )
    p.add_argument(
        "--prefetch-data",
        action="store_true",
        help=(
            "Download all bundled data (DLC model, NN encoder, NN "
            "reference set, ground_normalized) from the Hugging Face "
            "Hub and exit, without running any pipeline stages. "
            "Mostly redundant — every pipeline invocation already "
            "auto-downloads any missing files before starting work — "
            "but useful if you want to warm the cache up-front or "
            "verify the download succeeds without running on a real "
            "video. Files are cached under ~/.cache/huggingface/hub/ "
            "(override with MARM_BEHAVIOR_DATA_DIR=/path/for/shared/install)."
        ),
    )

    # Per-colour buddy overrides for the NN stage.
    for long_name, short_key in _COLOR_FLAGS:
        p.add_argument(
            f"--{long_name}-buddy",
            nargs="+",
            choices=("r", "w", "b", "y"),
            metavar="COLOR",
            default=None,
            help=(
                f"Override which other animal's description CSV is "
                f"paired with {long_name.capitalize()}'s during NN "
                f"encoding. Takes one or more short colour keys "
                f"(r/w/b/y) in preference order. Default chains: "
                f"Red→Yellow/Blue, White→Blue/Red, Blue→White/Red, "
                f"Yellow→Red/Blue."
            ),
        )

    # Per-colour "present" flags: default True, --no-<color> disables.
    for long_name, _ in _COLOR_FLAGS:
        p.add_argument(
            f"--no-{long_name}",
            dest=f"{long_name}_present",
            action="store_false",
            help=f"Mark {long_name} as not present in the video.",
        )
        p.set_defaults(**{f"{long_name}_present": True})

    p.add_argument(
        "--c-thresh",
        type=float,
        default=0.1,
        help="DLC confidence threshold for the extract stage. Default: 0.1.",
    )
    p.add_argument(
        "--target-coverage",
        type=float,
        default=1.0,
        help="Alpha-shape hull coverage target for the extract stage. Default: 1.0.",
    )
    p.add_argument(
        "--stages",
        nargs="+",
        choices=("dlc", "extract", "process", "depths", "labels", "nn"),
        default=None,
        metavar="STAGE",
        help=(
            "Which pipeline stages to run. Default: all six "
            "(dlc, extract, process, depths, labels, nn). Use e.g. "
            "'--stages labels' to re-run only the labels stage, or "
            "'--stages extract process depths labels' to skip both "
            "DLC inference and the NN postprocessing stage."
        ),
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress the per-stage progress messages.",
    )
    return p


def _discover_videos(target: "Path | None") -> "list[Path]":
    """Resolve the positional argument into a list of video files.

    * ``None`` → glob ``*.avi`` (and ``*.AVI``) in the current working
      directory.
    * a file path → ``[that file]``.
    * a directory path → glob ``*.avi`` / ``*.AVI`` inside it.

    Returns a sorted list. Raises :class:`FileNotFoundError` if no
    videos are found.
    """
    if target is None:
        search = Path.cwd()
    else:
        target = target.expanduser().resolve()
        if target.is_file():
            return [target]
        if not target.exists():
            raise FileNotFoundError(f"path not found: {target}")
        search = target

    matches = sorted(
        set(search.glob("*.avi")) | set(search.glob("*.AVI"))
    )
    if not matches:
        raise FileNotFoundError(
            f"no .avi files found in {search}. Pass an explicit path "
            f"or cd to a folder containing videos."
        )
    return matches


def _build_nn_buddies(args: argparse.Namespace) -> "dict[str, list[str]] | None":
    """Collect the four per-colour ``--<colour>-buddy`` flags into a
    dict suitable for :func:`marm_behavior.run.run`'s ``nn_buddies``
    argument.

    Returns ``None`` if no overrides were passed, so the downstream
    code path uses the built-in default chains.
    """
    overrides: dict[str, list[str]] = {}
    for long_name, short_key in _COLOR_FLAGS:
        flag_value = getattr(args, f"{long_name}_buddy")
        if flag_value:
            overrides[short_key] = list(flag_value)
    return overrides or None


def _print_video_summary(result: dict, quiet: bool) -> None:
    if quiet:
        return
    print()
    print(f"done. stages run: {', '.join(result['stages_run'])}")
    if result.get("tracks") is not None:
        print(f"  tracks:       {result['tracks']}")
    if result.get("edges") is not None:
        print(f"  edges:        {result['edges']}")
    if result.get("depths") is not None:
        print(f"  depths:       {result['depths']}")
    for key, path in result.get("descriptions", {}).items():
        print(f"  description {key}: {path}")
    for color, (hcoord, hlabel) in result.get("nn", {}).items():
        print(f"  nn {color}: {hcoord.name} + {hlabel.name}")


def main(argv: "list[str] | None" = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Auto-download any missing bundled data files before doing
    # anything else. On a fresh install this triggers a one-time
    # ~310 MB download from the Hugging Face Hub; on every
    # subsequent run this returns in milliseconds because every
    # file is already cached. We do this up front (instead of
    # inside each stage) so the user sees the "downloading..."
    # notice before any stage logs scroll past, and so a network
    # failure aborts cleanly instead of mid-pipeline.
    from ._data_files import ensure_all, _hf_repo_id

    if args.prefetch_data:
        # --prefetch-data is now mostly redundant since ensure_all
        # runs unconditionally, but we keep the flag as an explicit
        # "download and exit" mode for users who want to warm the
        # cache without running a pipeline.
        print(
            f"[marm_behavior] prefetching data files from "
            f"https://huggingface.co/datasets/{_hf_repo_id()} ..."
        )
        try:
            ensure_all(verbose=False)
        except Exception as err:
            print(
                f"error: prefetch failed: "
                f"{type(err).__name__}: {err}",
                file=sys.stderr,
            )
            return 2
        print("[marm_behavior] prefetch complete.")
        return 0

    try:
        ensure_all(verbose=not args.quiet)
    except Exception as err:
        print(
            f"error: failed to fetch bundled data: "
            f"{type(err).__name__}: {err}\n"
            f"Pipeline cannot run without the data files. Check your "
            f"network connection, or set MARM_BEHAVIOR_DATA_DIR to a "
            f"folder containing a manual copy.",
            file=sys.stderr,
        )
        return 2

    try:
        videos = _discover_videos(args.video)
    except FileNotFoundError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    nn_buddies = _build_nn_buddies(args)
    batch_mode = len(videos) > 1

    if batch_mode and not args.quiet:
        print(f"[marm_behavior] batch mode: {len(videos)} videos found")
        for v in videos:
            print(f"  - {v.name}")
        print()

    # In batch mode, --single-csv and --multi-csv don't make sense
    # because they'd point at a single video's files. Warn and ignore.
    if batch_mode and (args.single_csv or args.multi_csv):
        print(
            "warning: --single-csv / --multi-csv are ignored in batch "
            "mode; each video uses its own auto-discovered CSVs.",
            file=sys.stderr,
        )

    failures: list[tuple[Path, str]] = []
    successes: list[Path] = []

    # In batch mode, prebuild the nn-stage artifacts once so the ~5
    # min openTSNE fit doesn't get repeated per video. Skip the
    # prebuild if the user explicitly asked to omit the nn stage, or
    # if it's a single-video run (no batching benefit).
    nn_artifacts = None
    nn_will_run = (
        args.stages is None or "nn" in args.stages
    )
    if batch_mode and nn_will_run:
        try:
            from .nn_postprocess import prepare_nn_artifacts
            if not args.quiet:
                print(
                    "[marm_behavior] batch mode: building shared "
                    "nn artifacts (~5 min openTSNE fit, done once "
                    "for all videos)"
                )
            nn_artifacts = prepare_nn_artifacts(
                reference_dir=args.nn_reference_dir,
                verbose=not args.quiet,
            )
        except Exception as err:
            # If artifact prep fails we don't want to take the whole
            # batch down — fall back to per-video prep, which will
            # surface the same error per video and let the per-video
            # try/except below handle it.
            print(
                f"warning: failed to prebuild nn artifacts "
                f"({type(err).__name__}: {err}); falling back to "
                f"per-video setup",
                file=sys.stderr,
            )
            nn_artifacts = None

    for i, video in enumerate(videos, start=1):
        if batch_mode and not args.quiet:
            print(
                f"[marm_behavior] === [{i}/{len(videos)}] "
                f"{video.name} ==="
            )
        try:
            result = run(
                video_path=video,
                single_csv_path=args.single_csv if not batch_mode else None,
                multi_csv_path=args.multi_csv if not batch_mode else None,
                output_dir=args.output_dir,
                ground_normalized_path=args.ground_normalized,
                dlc_config=args.dlc_config,
                nn_reference_dir=args.nn_reference_dir,
                nn_artifacts=nn_artifacts,
                nn_buddies=nn_buddies,
                red_present=args.red_present,
                white_present=args.white_present,
                blue_present=args.blue_present,
                yellow_present=args.yellow_present,
                c_thresh=args.c_thresh,
                target_coverage=args.target_coverage,
                stages=args.stages,
                verbose=not args.quiet,
            )
        except FileNotFoundError as err:
            print(f"error: {video.name}: {err}", file=sys.stderr)
            failures.append((video, f"FileNotFoundError: {err}"))
            if not batch_mode:
                return 2
            continue
        except Exception as err:  # pragma: no cover
            print(
                f"error: {video.name}: {type(err).__name__}: {err}",
                file=sys.stderr,
            )
            failures.append((video, f"{type(err).__name__}: {err}"))
            if not batch_mode:
                return 1
            continue

        _print_video_summary(result, args.quiet)
        successes.append(video)

    if batch_mode and not args.quiet:
        print()
        print(
            f"[marm_behavior] batch done: "
            f"{len(successes)}/{len(videos)} succeeded"
        )
        if failures:
            print("failures:")
            for v, reason in failures:
                print(f"  - {v.name}: {reason}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
