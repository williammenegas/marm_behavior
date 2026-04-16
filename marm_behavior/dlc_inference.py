"""
marm_behavior.dlc_inference
===========================

Thin wrapper around the user-installed DeepLabCut runtime that runs
inference with a **bundled** trained model.

The model lives in ``marm_behavior/data/dlc_model/`` as a drop-in DLC
project tree::

    dlc_model/
    ├── config.yaml
    └── dlc-models/
        └── iteration-0/
            └── marm_4Sep5-trainset99shuffle1/
                ├── train/
                │   ├── pose_cfg.yaml
                │   ├── snapshot-15000.data-00000-of-00001
                │   ├── snapshot-15000.index
                │   └── snapshot-15000.meta
                └── test/
                    ├── inference_cfg.yaml
                    └── pose_cfg.yaml

DLC's ``config.yaml`` and ``pose_cfg.yaml`` contain an absolute
``project_path`` field that's baked in at project-creation time. DLC
uses it internally to resolve ``dlc-models/iteration-{N}/...``
relative paths, so leaving that field wrong would make
``analyze_videos`` fail to find the snapshot.

When the ``dlc`` stage runs, we copy the bundled model tree to a
temporary directory, rewrite ``project_path`` to point at that temp
directory, and hand the temp ``config.yaml`` to
``deeplabcut.analyze_videos``. The temp copy is torn down after the
stage finishes, so there are no persistent user-cache side effects
and the installed package never needs filesystem-write access.

The stage runs three DLC calls in sequence:

1. ``analyze_videos(..., auto_track=False, save_as_csv=False)`` —
   runs the ResNet backbone + part-affinity-field head on every
   frame, writes ``<stem>DLC_..._full.pickle``.
2. ``convert_detections2tracklets(..., track_method='ellipse')`` —
   groups per-frame detections into short tracklets, writes
   ``<stem>DLC_..._el.pickle``.
3. :func:`marm_behavior.el_to_csv.el_pickle_to_csvs` — reads the
   tracklet pickle and writes the two whitespace-delimited CSVs that
   the extract stage consumes:
   ``<stem>DLC_..._el.picklesingle.csv`` and
   ``<stem>DLC_..._el.picklemulti.csv``.
"""

from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


DEFAULT_SHUFFLE = 1
DEFAULT_TRAINING_SET_INDEX = 0
DEFAULT_TRACK_METHOD = "ellipse"


def _bundled_model_root() -> Path:
    """Return a :class:`Path` pointing at the DLC project root.

    The DLC project tree is hosted on the Hugging Face Hub at
    ``williammenegas/data`` (dataset repo) under ``dlc_model/`` and
    materialized into a consolidated local directory on first call.
    Subsequent calls return immediately.

    The returned path is read-only as far as callers are concerned,
    so callers that need to write (rewriting ``project_path`` etc.)
    must first copy the tree somewhere writable, as this function's
    existing consumers do.
    """
    from ._data_files import ensure_data_dir

    # Materialize the three subtrees of the DLC project. They all
    # share the same top-level directory (``data/dlc_model/``), so
    # the first call's return value is the project root we want;
    # the remaining two calls just fill in subdirectories within it.
    project_root = ensure_data_dir("dlc_model", ["config.yaml"])
    ensure_data_dir(
        "dlc_model/dlc-models/iteration-0/marm_4Sep5-trainset99shuffle1/train",
        [
            "pose_cfg.yaml",
            "snapshot-15000.index",
            "snapshot-15000.meta",
            "snapshot-15000.data-00000-of-00001",
        ],
    )
    ensure_data_dir(
        "dlc_model/dlc-models/iteration-0/marm_4Sep5-trainset99shuffle1/test",
        ["inference_cfg.yaml", "pose_cfg.yaml"],
    )
    return project_root


def _rewrite_project_path(yaml_path: Path, new_project_path: Path) -> None:
    """Rewrite the ``project_path:`` field in a DLC YAML config in place.

    This is a surgical text edit rather than a full YAML round-trip so we
    don't accidentally reformat the file — DLC parses the YAML strictly
    and occasionally emits warnings when list-style keys are unquoted.
    We only touch the single ``project_path`` line.

    If the file doesn't contain a ``project_path:`` line at all (as is the
    case for ``test/pose_cfg.yaml``, which only has training-time fields),
    this is a no-op — we leave the file untouched rather than appending a
    spurious field.
    """
    text = yaml_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    out_lines = []
    replaced = False
    # Use forward slashes in the rewritten path even on Windows so that
    # DLC's internal ``os.path.join`` doesn't end up with mixed separators.
    new_value = str(new_project_path).replace("\\", "/")
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("project_path:") and not replaced:
            indent = line[: len(line) - len(stripped)]
            newline = "\n" if line.endswith("\n") else ""
            out_lines.append(f"{indent}project_path: {new_value}{newline}")
            replaced = True
        else:
            out_lines.append(line)
    if replaced:
        yaml_path.write_text("".join(out_lines), encoding="utf-8")
    # else: file has no project_path field — leave it alone.


@contextmanager
def materialised_model(
    source_root: "Path | None" = None,
) -> Iterator[Path]:
    """Context manager: materialise the bundled DLC project into a temp
    directory with ``project_path`` rewritten, yield the path to the
    temp ``config.yaml``, and clean up on exit.

    Parameters
    ----------
    source_root:
        Optional override of the DLC project root to materialise. If
        ``None`` (the default), uses :func:`_bundled_model_root`.

    Yields
    ------
    Path
        Absolute path to the rewritten ``config.yaml`` in the temp dir,
        suitable for passing to ``deeplabcut.analyze_videos``.
    """
    src = source_root or _bundled_model_root()
    if not src.exists():
        raise FileNotFoundError(
            f"bundled DLC model tree not found at {src} — is the package "
            f"installed correctly?"
        )
    # Use a top-level temp dir rather than mkdtemp() + copy, because
    # DLC's analyze_videos creates sibling files inside the model tree
    # (cached h5 detection scores etc.) that need to be writable.
    tmp = Path(tempfile.mkdtemp(prefix="bfpy_dlc_"))
    try:
        dst = tmp / "project"
        shutil.copytree(src, dst)
        # Rewrite the two configs with the new project_path.
        _rewrite_project_path(dst / "config.yaml", dst)
        # pose_cfg.yaml in train/ also carries project_path.
        for sub in ("train", "test"):
            pc = (
                dst
                / "dlc-models"
                / "iteration-0"
                / "marm_4Sep5-trainset99shuffle1"
                / sub
                / "pose_cfg.yaml"
            )
            if pc.exists():
                _rewrite_project_path(pc, dst)
        yield dst / "config.yaml"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_dlc_inference(
    video_path: "str | Path",
    *,
    output_dir: "str | Path | None" = None,
    dlc_config_override: "str | Path | None" = None,
    shuffle: int = DEFAULT_SHUFFLE,
    training_set_index: int = DEFAULT_TRAINING_SET_INDEX,
    track_method: str = DEFAULT_TRACK_METHOD,
    verbose: bool = True,
) -> "dict[str, Path]":
    """Run DLC inference on one video, producing the CSVs Stage 1 reads.

    Parameters
    ----------
    video_path:
        Path to the input ``.avi``.
    output_dir:
        Directory to write DLC outputs into. If omitted, DLC writes them
        next to the video (DLC's default).
    dlc_config_override:
        Optional path to a user-supplied DLC ``config.yaml``. If given,
        the bundled model is bypassed and the user's project is used
        directly. Useful for swapping in a retrained model without
        rebuilding the package.
    shuffle, training_set_index, track_method:
        Passed through to DLC. Defaults match the bundled model's
        configuration (shuffle=1, trainset99, ellipse tracker).
    verbose:
        If True, log each DLC sub-step.

    Returns
    -------
    dict
        ``{'single_csv': Path, 'multi_csv': Path}`` pointing at the two
        CSV files Stage 1 consumes. Raises
        :class:`FileNotFoundError` if DLC runs cleanly but the expected
        outputs don't appear (which would indicate a DLC version mismatch
        — the CSV filename convention has changed across releases).
    """
    try:
        import deeplabcut as dlc
    except ImportError as err:
        raise RuntimeError(
            "The 'dlc' stage requires deeplabcut to be installed in the "
            "current environment. Install it with one of:\n"
            "    pip install 'deeplabcut[tf]'     # TensorFlow backend\n"
            "    pip install 'deeplabcut[pytorch]'  # PyTorch backend\n"
            "Or skip this stage by passing stages=('extract','process',"
            "'depths','labels') to run() / --stages on the CLI if you "
            "already have DLC CSVs."
        ) from err

    video_path = Path(video_path).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    if output_dir is not None:
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        if verbose:
            print(f"[marm_behavior.dlc] {msg}")

    # If the user pointed at their own config, use it directly — skip
    # the bundled-model path rewriting entirely.
    if dlc_config_override is not None:
        config_path = Path(dlc_config_override).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(
                f"--dlc-config points at a missing file: {config_path}"
            )
        _log(f"using user-supplied DLC config: {config_path}")
        return _invoke_dlc(
            dlc,
            config_path,
            video_path,
            output_dir,
            shuffle,
            training_set_index,
            track_method,
            _log,
        )

    # Bundled path: materialise a temp project with project_path rewritten.
    _log("materialising bundled DLC model to temp directory")
    with materialised_model() as config_path:
        _log(f"config: {config_path}")
        return _invoke_dlc(
            dlc,
            config_path,
            video_path,
            output_dir,
            shuffle,
            training_set_index,
            track_method,
            _log,
        )


def _invoke_dlc(
    dlc,
    config_path: Path,
    video_path: Path,
    output_dir: "Path | None",
    shuffle: int,
    training_set_index: int,
    track_method: str,
    log,
) -> "dict[str, Path]":
    """Internal helper: the DLC inference calls + output discovery.

    This matches the call signature used successfully against DLC 2.2.0.6
    on the original marm_behavior workflow::

        deeplabcut.analyze_videos(
            config_path, video_path,
            videotype='avi', use_shelve=True, auto_track=False,
        )
        deeplabcut.convert_detections2tracklets(
            config_path, video_path,
            videotype='avi', shuffle=1, trainingsetindex=0,
        )
        # Then stitch_tracklets to get the CSVs that Stage 1 reads.
        deeplabcut.stitch_tracklets(
            config_path, video_path,
            shuffle=1, trainingsetindex=0, track_method='ellipse',
        )

    Split out so both the bundled-model and user-config code paths can
    call it without duplication.
    """
    videos = [str(video_path)]

    # Stage 0.1: detector inference → ..._full.pickle
    log("analyze_videos: running ResNet detector on every frame")
    dlc.analyze_videos(
        str(config_path),
        videos,
        videotype="avi",
        shuffle=shuffle,
        trainingsetindex=training_set_index,
        use_shelve=True,
        auto_track=False,
        save_as_csv=False,
    )

    # Stage 0.2: tracklet formation → ..._el.pickle
    log("convert_detections2tracklets: grouping detections into tracklets")
    dlc.convert_detections2tracklets(
        str(config_path),
        videos,
        videotype="avi",
        shuffle=shuffle,
        trainingsetindex=training_set_index,
        track_method=track_method,
    )

    # Stage 0.3: convert the tracklet pickle into the two CSVs the
    # extract stage reads.
    log("el_to_csv: converting _el.pickle -> picklesingle.csv + picklemulti.csv")
    from .el_to_csv import el_pickle_to_csvs

    pickle_matches = sorted(video_path.parent.glob(f"{video_path.stem}DLC*el.pickle"))
    if not pickle_matches:
        # Also look in output_dir if it's different.
        if output_dir is not None:
            pickle_matches = sorted(Path(output_dir).glob(f"{video_path.stem}DLC*el.pickle"))
    if not pickle_matches:
        raise FileNotFoundError(
            f"convert_detections2tracklets ran but no _el.pickle was "
            f"written. Looked in {video_path.parent} and "
            f"{output_dir}. Check the DLC output manually."
        )
    single_csv, multi_csv = el_pickle_to_csvs(
        pickle_matches[0],
        output_dir=output_dir or video_path.parent,
    )
    return {
        "single_csv": single_csv,
        "multi_csv": multi_csv,
    }
