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

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


DEFAULT_SHUFFLE = 1
DEFAULT_TRAINING_SET_INDEX = 0
DEFAULT_TRACK_METHOD = "ellipse"


def _materialisation_root() -> Path:
    """Return the directory under which per-run materialised DLC
    projects should be created.

    We deliberately avoid ``tempfile.gettempdir()`` (i.e. ``%TEMP%``
    on Windows, ``/tmp`` on Linux) because Windows cleanup tools
    (Storage Sense, Disk Cleanup, CCleaner, enterprise endpoint
    agents, and users running "clear temp files" manually) routinely
    sweep that directory mid-run. A DLC inference pass on a long
    video can take hours, during which time the snapshot files under
    ``train/`` are opened exactly once (at the start, to load weights
    into TF) and then never touched again — so any cleanup tool that
    uses access-time heuristics will classify them as stale and
    delete them while the run is still in flight. The next DLC call
    in the same run (``convert_detections2tracklets``) then re-does
    ``os.listdir(modelfolder/train)``, finds nothing matching
    ``*.index``, and crashes with
    ``IndexError: index -1 is out of bounds for axis 0 with size 0``.

    Putting the materialisation under the user's local cache instead
    keeps it out of the default scope of every cleanup tool we know
    of, at the cost of ~130 MB of disk that's already consumed by
    the bundled model's own cache anyway (and which the existing
    ``finally: shutil.rmtree(...)`` on the context manager still
    reclaims on normal exit).

    Override with ``MARM_BEHAVIOR_DLC_TMPDIR`` if a specific location
    is needed (e.g. a large scratch volume, or a shared-cluster
    node-local path).
    """
    override = os.environ.get("MARM_BEHAVIOR_DLC_TMPDIR")
    if override:
        root = Path(override).expanduser().resolve()
    else:
        # Same fallback chain as _data_files._writable_data_root's
        # user-cache branch, so the DLC materialisation sits next to
        # the rest of marm_behavior's cache data.
        root = (
            Path(
                os.environ.get(
                    "XDG_CACHE_HOME",
                    str(Path.home() / ".cache"),
                )
            )
            / "marm_behavior"
            / "dlc_runs"
        )
    root.mkdir(parents=True, exist_ok=True)
    return root


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
    """Context manager: materialise the bundled DLC project into a
    per-run directory under the user's local cache with
    ``project_path`` rewritten, yield the path to the ``config.yaml``,
    and clean up on exit.

    The materialisation lives under :func:`_materialisation_root`
    (``~/.cache/marm_behavior/dlc_runs/`` by default), **not**
    ``%TEMP%``, so mid-run Windows temp cleanup can't delete the
    snapshot files out from under a long-running DLC pass. See
    :func:`_materialisation_root` for the full rationale.

    Parameters
    ----------
    source_root:
        Optional override of the DLC project root to materialise. If
        ``None`` (the default), uses :func:`_bundled_model_root`.

    Yields
    ------
    Path
        Absolute path to the rewritten ``config.yaml`` in the
        materialised dir, suitable for passing to
        ``deeplabcut.analyze_videos``.
    """
    src = source_root or _bundled_model_root()
    if not src.exists():
        raise FileNotFoundError(
            f"bundled DLC model tree not found at {src} — is the package "
            f"installed correctly?"
        )
    # Use a top-level per-run dir rather than mkdtemp() + copy, because
    # DLC's analyze_videos creates sibling files inside the model tree
    # (cached h5 detection scores etc.) that need to be writable.
    tmp = Path(
        tempfile.mkdtemp(prefix="bfpy_dlc_", dir=str(_materialisation_root()))
    )
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
    in_subprocess: bool = True,
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
    in_subprocess:
        If True (the default), run the DLC calls in a freshly-spawned
        subprocess so that all GPU memory DLC claims is returned to
        the driver when the subprocess exits. This is the only
        reliable way to free the memory on DLC 2.2.x: TF's BFC
        allocator keeps its pool mapped for the lifetime of the
        process even after ``clear_session()``, and DLC holds module-
        level references to its TF v1 session that ``clear_session()``
        cannot reach. Set to False only for debugging (e.g. to get a
        Python traceback in-process); expect nvidia-smi to show DLC's
        GPU footprint persisting until the Python process exits, and
        expect the nn stage's ``load_model`` to have less GPU memory
        available as a result.

    Returns
    -------
    dict
        ``{'single_csv': Path, 'multi_csv': Path}`` pointing at the two
        CSV files Stage 1 consumes. Raises
        :class:`FileNotFoundError` if DLC runs cleanly but the expected
        outputs don't appear (which would indicate a DLC version mismatch
        — the CSV filename convention has changed across releases).
    """
    # Lightweight importability check in the parent so callers get a
    # quick, readable error on a misconfigured env without paying the
    # ~5 s deeplabcut/tensorflow import cost up front (the subprocess
    # path does the real import in the child).
    import importlib.util
    if importlib.util.find_spec("deeplabcut") is None:
        raise RuntimeError(
            "The 'dlc' stage requires deeplabcut to be installed in the "
            "current environment. Install it with one of:\n"
            "    pip install 'deeplabcut[tf]'     # TensorFlow backend\n"
            "    pip install 'deeplabcut[pytorch]'  # PyTorch backend\n"
            "Or skip this stage by passing stages=('extract','process',"
            "'depths','labels') to run() / --stages on the CLI if you "
            "already have DLC CSVs."
        )

    video_path = Path(video_path).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    if output_dir is not None:
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        if verbose:
            print(f"[marm_behavior.dlc] {msg}")

    invoke = _invoke_dlc_in_subprocess if in_subprocess else _invoke_dlc_inprocess

    # If the user pointed at their own config, use it directly — skip
    # the bundled-model path rewriting entirely.
    if dlc_config_override is not None:
        config_path = Path(dlc_config_override).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(
                f"--dlc-config points at a missing file: {config_path}"
            )
        _log(f"using user-supplied DLC config: {config_path}")
        return invoke(
            config_path,
            video_path,
            output_dir,
            shuffle,
            training_set_index,
            track_method,
            verbose,
            _log,
        )

    # Bundled path: materialise a temp project with project_path
    # rewritten. The materialisation *must* live in the parent: the
    # temp dir has to outlive the subprocess (DLC writes sibling
    # files inside it during inference) and must be torn down
    # regardless of how the subprocess exits.
    _log("materialising bundled DLC model to temp directory")
    with materialised_model() as config_path:
        _log(f"config: {config_path}")
        return invoke(
            config_path,
            video_path,
            output_dir,
            shuffle,
            training_set_index,
            track_method,
            verbose,
            _log,
        )


def _invoke_dlc_inprocess(
    config_path: Path,
    video_path: Path,
    output_dir: "Path | None",
    shuffle: int,
    training_set_index: int,
    track_method: str,
    verbose: bool,
    log,
) -> "dict[str, Path]":
    """In-process variant of :func:`_invoke_dlc_in_subprocess`.

    Here only for debugging (``in_subprocess=False`` in
    :func:`run_dlc_inference`). On DLC 2.2.x the subprocess variant
    is the only reliable way to release GPU memory before later
    stages run, so this path is not recommended for production runs.
    """
    import deeplabcut as dlc
    return _invoke_dlc(
        dlc,
        config_path,
        video_path,
        output_dir,
        shuffle,
        training_set_index,
        track_method,
        log,
    )


def _dlc_subprocess_worker(
    config_path_str: str,
    video_path_str: str,
    output_dir_str: "str | None",
    shuffle: int,
    training_set_index: int,
    track_method: str,
    verbose: bool,
    send_end,
) -> None:
    """Runs in the spawned subprocess. Imports DLC, runs the three DLC
    calls via :func:`_invoke_dlc`, sends the resulting CSV paths (as
    strings) back through ``send_end``, and lets the interpreter exit
    — at which point the OS reclaims all GPU memory DLC allocated.

    Defined at module scope (not nested inside another function) so
    it's picklable across the ``spawn`` start-method boundary.

    Exception handling: any exception raised during DLC execution is
    captured along with its formatted traceback and sent back through
    the same pipe as a ``("err", repr, tb)`` tuple. The parent side
    re-raises it as a :class:`RuntimeError` with the child traceback
    embedded in the message, so the caller still sees where things
    went wrong inside DLC.
    """
    try:
        # Re-silence TF inside the child; the parent's silence doesn't
        # carry across a spawn since the child re-imports from scratch.
        try:
            from ._tf_quiet import silence_tensorflow_logging
            silence_tensorflow_logging()
        except Exception:
            pass

        try:
            import deeplabcut as dlc
        except ImportError as err:
            raise RuntimeError(
                f"deeplabcut import failed in subprocess: {err}"
            ) from err

        def _child_log(msg: str) -> None:
            if verbose:
                print(f"[marm_behavior.dlc] {msg}")

        result = _invoke_dlc(
            dlc,
            Path(config_path_str),
            Path(video_path_str),
            Path(output_dir_str) if output_dir_str is not None else None,
            shuffle,
            training_set_index,
            track_method,
            _child_log,
        )
        send_end.send(
            ("ok", str(result["single_csv"]), str(result["multi_csv"]))
        )
    except BaseException as err:  # noqa: BLE001 — want to ship any failure back
        import traceback
        try:
            send_end.send(("err", repr(err), traceback.format_exc()))
        except Exception:
            # Pipe already broken; best we can do is exit non-zero and
            # let the parent's EOFError branch handle it.
            pass
    finally:
        try:
            send_end.close()
        except Exception:
            pass


def _invoke_dlc_in_subprocess(
    config_path: Path,
    video_path: Path,
    output_dir: "Path | None",
    shuffle: int,
    training_set_index: int,
    track_method: str,
    verbose: bool,
    log,
) -> "dict[str, Path]":
    """Spawn a fresh Python subprocess to run the DLC calls.

    This is the only reliable way to release GPU memory held by DLC
    on DLC 2.2.x / TF 2.9. Even after ``tf.keras.backend.clear_session()``
    plus ``gc.collect()``, TF's BFC allocator keeps its GPU pool mapped
    to the process (by design — so it can reuse pages cheaply), and
    DLC keeps its own module-level references to a
    ``tf.compat.v1.Session`` that ``clear_session()`` can't reach. As
    a result nvidia-smi continues to show DLC's full footprint, and a
    later Keras model load (the nn stage's LSTM encoder) has to fit
    in whatever's left rather than in the full GPU. Running DLC in a
    child process cuts this Gordian knot: when the child exits, the
    OS unconditionally returns everything the child was holding back
    to the CUDA driver.

    The ``spawn`` start method is required (not ``fork``) because
    ``fork`` doesn't work reliably with an already-initialised CUDA
    context in the parent, and we can't guarantee the parent hasn't
    touched CUDA before this call (the nn stage or an interactive
    session might have). ``spawn`` starts a clean interpreter.

    Cost: ~5 s of subprocess startup per call (Python interpreter +
    TF + DLC import in the child). For DLC runs lasting tens of
    seconds to minutes per video, that overhead is immaterial.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    recv_end, send_end = ctx.Pipe(duplex=False)

    log("launching DLC in isolated subprocess; GPU memory will be released when it exits")
    proc = ctx.Process(
        target=_dlc_subprocess_worker,
        args=(
            str(config_path),
            str(video_path),
            str(output_dir) if output_dir is not None else None,
            shuffle,
            training_set_index,
            track_method,
            verbose,
            send_end,
        ),
        daemon=False,
    )
    proc.start()
    # Close the parent's copy of the write end so that recv() below
    # raises EOFError promptly if the child dies without sending,
    # rather than hanging forever waiting on a pipe that still has a
    # live writer (us).
    send_end.close()

    try:
        try:
            payload = recv_end.recv()
        except EOFError:
            proc.join()
            raise RuntimeError(
                f"DLC subprocess exited without returning a result "
                f"(exit code {proc.exitcode}). The child's stdout / "
                f"stderr was inherited from this process — any "
                f"traceback DLC printed should appear above."
            )
    finally:
        recv_end.close()

    proc.join()
    log(f"DLC subprocess exited (code {proc.exitcode}); GPU memory reclaimed")

    tag = payload[0]
    if tag == "err":
        _, child_repr, child_tb = payload
        raise RuntimeError(
            f"DLC subprocess failed: {child_repr}\n"
            f"--- child traceback ---\n{child_tb}"
        )
    if tag != "ok":
        raise RuntimeError(
            f"DLC subprocess returned an unexpected payload: {payload!r}"
        )
    _, single_csv_str, multi_csv_str = payload
    return {
        "single_csv": Path(single_csv_str),
        "multi_csv": Path(multi_csv_str),
    }


def _release_gpu_after_dlc(log) -> None:
    """Release TF (and torch) GPU memory held by the dlc stage.

    DLC runs its ResNet detector through TensorFlow, which adds nodes
    to the global default graph and reserves GPU memory via TF's BFC
    allocator. When ``analyze_videos`` / ``convert_detections2tracklets``
    return, DLC's Python handles drop, but the *global* TF state at
    module scope stays alive — so the GPU memory is not reusable by a
    later stage that instantiates a fresh Keras model (e.g. the nn
    stage's LSTM encoder). On modestly-sized GPUs that causes the nn
    stage's ``load_model`` to OOM or fall back to a slow path even
    though DLC is no longer doing any work.

    ``tf.keras.backend.clear_session()`` destroys the global Keras /
    TF-v1-compat default-graph state, which returns DLC's tensors to
    TF's allocator pool where a later ``load_model`` can actually
    reuse them. ``gc.collect()`` then sweeps any Python-side handles
    still pinning device tensors. Both calls are no-ops if TF isn't
    actually imported, so this is safe to call on the PyTorch backend
    too; the PyTorch half is a separate best-effort ``empty_cache``.

    Kept best-effort (swallowed ImportError) so a missing deep-learning
    dep never turns post-stage cleanup into a crash.
    """
    import gc
    try:
        from tensorflow.keras.backend import clear_session
        clear_session()
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()
    log("released GPU memory held by dlc stage")


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
    try:
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
    finally:
        # Runs whether DLC finished cleanly or raised -- releasing GPU
        # memory is just as important on the error path, where the
        # caller may retry or fall through to a later stage.
        _release_gpu_after_dlc(log)
