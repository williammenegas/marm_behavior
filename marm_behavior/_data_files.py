"""Lazy downloader for marm_behavior's bundled data files.

The package's runtime data — the trained DLC model (~128 MB), the
LSTM encoder (~2 MB), the canonical NN reference set (~190 MB), and
the ``ground_normalized.npz`` body-part cloud — is hosted on the
Hugging Face Hub at the dataset repo
`williammenegas/data <https://huggingface.co/datasets/williammenegas/data>`_
rather than shipped in the wheel. This keeps the installed package
small (a few hundred KB) while still giving users a one-step setup:
the first run silently downloads the files into the local cache and
every subsequent run reuses them.

This module exposes a single helper, :func:`ensure_data_file`, which
returns the on-disk ``Path`` to a requested data file. If the file
already exists in the bundled ``marm_behavior/data/`` directory
(useful for offline environments and lab-internal mirrors), that path
is returned directly with no network access. Otherwise the file is
downloaded from the HF Hub via :func:`huggingface_hub.hf_hub_download`,
which caches it under ``~/.cache/huggingface/hub/`` so subsequent
calls are instant.

Override the source repo with the ``MARM_BEHAVIOR_HF_REPO`` env var if
you mirror it elsewhere, and override the local cache root with
``MARM_BEHAVIOR_DATA_DIR`` if you want a self-contained install on a
shared filesystem.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

#: Default HF Hub dataset repo holding the data files. Override via
#: the ``MARM_BEHAVIOR_HF_REPO`` env var for lab-internal mirrors.
DEFAULT_HF_REPO = "williammenegas/data"

#: Repo type passed to ``huggingface_hub``. The data lives in a
#: dataset repo (not a model repo) because it's mostly reference
#: arrays + a fixed checkpoint, not a trainable model.
HF_REPO_TYPE = "dataset"


def _bundled_data_root() -> Path:
    """Return the in-package ``data/`` folder.

    Files in here take priority over Hub downloads, so air-gapped
    deployments can ship a pre-populated wheel and never call out to
    the network.
    """
    return Path(__file__).resolve().parent / "data"


def _override_data_dir() -> "Path | None":
    """Return ``MARM_BEHAVIOR_DATA_DIR`` as a Path if set, else None.

    When set, downloaded files are placed under this directory
    (mirroring the ``data/`` layout) instead of the HF cache. Useful
    for shared cluster installs where every user should hit the same
    on-disk copy.
    """
    val = os.environ.get("MARM_BEHAVIOR_DATA_DIR")
    if not val:
        return None
    return Path(val).expanduser().resolve()


def _hf_repo_id() -> str:
    """Return the configured HF repo id."""
    return os.environ.get("MARM_BEHAVIOR_HF_REPO", DEFAULT_HF_REPO)


def _writable_data_root() -> Path:
    """Return the directory where downloaded data files should be
    written. Falls back through:

    1. ``$MARM_BEHAVIOR_DATA_DIR`` if set — for shared cluster installs
    2. ``marm_behavior/data/`` inside the package — preferred default,
       keeps everything next to the code, no separate cache anywhere
    3. ``~/.cache/marm_behavior/`` if the package data dir isn't
       writable (e.g. system-wide install with read-only
       site-packages)

    The chosen directory is created if it doesn't exist.
    """
    override = _override_data_dir()
    if override is not None:
        override.mkdir(parents=True, exist_ok=True)
        return override

    bundled = _bundled_data_root()
    # Try to use the package data dir. Probe writability by attempting
    # to create the directory and touch a sentinel file. If the
    # package was installed with read-only site-packages, fall back
    # to the user cache.
    try:
        bundled.mkdir(parents=True, exist_ok=True)
        sentinel = bundled / ".write_test"
        sentinel.touch()
        sentinel.unlink()
        return bundled
    except (PermissionError, OSError):
        pass

    # Fallback: per-user cache dir.
    cache = (
        Path(
            os.environ.get(
                "XDG_CACHE_HOME",
                str(Path.home() / ".cache"),
            )
        )
        / "marm_behavior"
    )
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def ensure_data_file(relpath: str) -> Path:
    """Return a local Path to the requested data file.

    Lookup order:

    1. ``$MARM_BEHAVIOR_DATA_DIR/<relpath>`` if the env var is set
       and the file exists there
    2. ``marm_behavior/data/<relpath>`` (or the user cache fallback)
       if the file is already present
    3. Download from the HF Hub dataset repo *directly* into the
       writable data root — no separate HF cache, no cross-drive
       copying, no double-storage

    Parameters
    ----------
    relpath:
        Path relative to the data root, e.g.
        ``"nn_model/pred_model_marm_encoderD.h5"`` or
        ``"nn_reference/out_inner1.csv"``. Forward slashes on every
        platform.

    Returns
    -------
    Path
        Absolute path to a readable file.

    Raises
    ------
    RuntimeError
        If ``huggingface_hub`` isn't installed and the file isn't
        already on disk, with instructions on how to fix.
    """
    rel = relpath.lstrip("/").replace("\\", "/")

    # 1+2. Check every plausible local location before downloading.
    for root in _candidate_data_roots():
        candidate = root / rel
        if candidate.exists():
            return candidate

    # 3. Need to download. Pick the writable target root.
    target_root = _writable_data_root()
    target = target_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as err:
        raise RuntimeError(
            f"Data file '{rel}' is not on disk and huggingface_hub "
            f"is not available to fetch it. Either:\n"
            f"  1. pip install huggingface_hub  (recommended)\n"
            f"  2. Manually download {rel} from "
            f"https://huggingface.co/datasets/{_hf_repo_id()} into "
            f"{target_root}/\n"
            f"  3. Set MARM_BEHAVIOR_DATA_DIR=/path/to/local/data "
            f"and place the file at "
            f"$MARM_BEHAVIOR_DATA_DIR/{rel}"
        ) from err

    # Download directly into the target directory using local_dir.
    # This avoids HF's content-addressed cache layout entirely and
    # writes the file straight to its final location, so there's no
    # second copy and no risk of running out of space on the C: drive
    # (the default HF cache root) when the actual install is on H:.
    #
    # We retry on transient network errors. Multi-gigabyte downloads
    # over flaky connections regularly fail with truncation
    # ("Consistency check failed: file should be of size X but has
    # size Y"); HF's error message tells the user to retry with
    # force_download=True, which deletes the partial file and starts
    # over. We do that automatically so users don't have to babysit
    # a long initial download.
    last_err: "Exception | None" = None
    for attempt in range(1, _DOWNLOAD_MAX_ATTEMPTS + 1):
        force = attempt > 1  # force re-download after any failure
        try:
            _hf_download_one(
                rel=rel,
                target_root=target_root,
                hf_hub_download=hf_hub_download,
                force_download=force,
            )
            return target
        except Exception as err:  # noqa: BLE001 — log and retry
            last_err = err
            if attempt < _DOWNLOAD_MAX_ATTEMPTS:
                print(
                    f"[marm_behavior] download of {rel} failed "
                    f"(attempt {attempt}/{_DOWNLOAD_MAX_ATTEMPTS}): "
                    f"{type(err).__name__}: {err}. Retrying with "
                    f"force_download=True..."
                )
                # Brief backoff so a flaky link gets a moment to settle.
                import time
                time.sleep(2 * attempt)
            else:
                # Out of attempts; let the caller see the last error.
                raise RuntimeError(
                    f"download of {rel} failed after "
                    f"{_DOWNLOAD_MAX_ATTEMPTS} attempts. "
                    f"Last error: {type(err).__name__}: {err}\n"
                    f"You can resume by re-running the same command "
                    f"(already-downloaded files will be skipped), or "
                    f"manually download {rel} from "
                    f"https://huggingface.co/datasets/"
                    f"{_hf_repo_id()} into {target_root}/."
                ) from err
    # Unreachable, but appeases type checkers.
    raise RuntimeError(f"unreachable: {last_err}")


#: Maximum number of attempts per file before giving up.
_DOWNLOAD_MAX_ATTEMPTS = 4


def _hf_download_one(
    *,
    rel: str,
    target_root: Path,
    hf_hub_download,
    force_download: bool,
) -> None:
    """Single attempt at downloading one file via hf_hub_download.

    Splits this out from the retry loop in :func:`ensure_data_file`
    so the retry logic stays readable. Handles the
    ``local_dir_use_symlinks`` deprecation: required on older
    huggingface_hub (< 0.23) to actually skip the cache backing,
    deprecated-but-accepted on current versions, removed on some
    future version. We pass it inside a warnings.catch_warnings to
    suppress the deprecation noise on every download.
    """
    import warnings

    kwargs = dict(
        repo_id=_hf_repo_id(),
        filename=rel,
        repo_type=HF_REPO_TYPE,
        local_dir=str(target_root),
        force_download=force_download,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*local_dir_use_symlinks.*",
            category=UserWarning,
        )
        try:
            hf_hub_download(**kwargs, local_dir_use_symlinks=False)
        except TypeError:
            # Future version that removed the kwarg entirely.
            hf_hub_download(**kwargs)


def _candidate_data_roots() -> "list[Path]":
    """All directories where bundled data files may already exist,
    in priority order. Used by the lookup pass in
    :func:`ensure_data_file` and by :func:`_file_is_present`.
    """
    roots: "list[Path]" = []
    override = _override_data_dir()
    if override is not None:
        roots.append(override)
    roots.append(_bundled_data_root())
    user_cache = (
        Path(
            os.environ.get(
                "XDG_CACHE_HOME",
                str(Path.home() / ".cache"),
            )
        )
        / "marm_behavior"
    )
    if user_cache.exists():
        roots.append(user_cache)
    return roots


def ensure_data_dir(reldir: str, filenames: Iterable[str]) -> Path:
    """Ensure every named file under ``reldir`` is on disk and return
    a directory containing them all.

    Because :func:`ensure_data_file` now writes downloads directly to
    the writable data root (no separate HF cache, no
    content-addressed storage), every file requested for the same
    ``reldir`` lands in the same parent directory. So this function
    is effectively just a loop over :func:`ensure_data_file` plus a
    parent-directory return — no consolidation copy, no
    double-storage.

    Parameters
    ----------
    reldir:
        Subdirectory under the data root, e.g. ``"nn_reference"``.
    filenames:
        Files relative to ``reldir`` that must all be present.

    Returns
    -------
    Path
        Directory containing every file in ``filenames``.
    """
    rel = reldir.strip("/").replace("\\", "/")
    filenames = list(filenames)
    if not filenames:
        # Nothing to download; just return the writable root + reldir.
        target_dir = _writable_data_root() / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    # Ensure each file. The first call's return value tells us the
    # parent directory; subsequent calls will land in the same place
    # because ensure_data_file is deterministic about its target root.
    first_path = ensure_data_file(f"{rel}/{filenames[0]}")
    parent = first_path.parent
    for fn in filenames[1:]:
        ensure_data_file(f"{rel}/{fn}")
    return parent


#: Authoritative manifest of every data file marm_behavior needs at
#: runtime. The first item of each tuple is the directory the files
#: live under (relative to the data root); the second is the list of
#: filenames in that directory. Used by both :func:`ensure_all` (the
#: auto-download check that runs at the start of every CLI
#: invocation) and :func:`missing_files` (the dry-run lookup). When
#: you add a new bundled file, add it here so auto-download picks
#: it up.
DATA_MANIFEST: "list[tuple[str, list[str]]]" = [
    ("", ["ground_normalized.npz"]),
    ("nn_model", ["pred_model_marm_encoderD.h5"]),
    (
        "nn_reference",
        [
            "out_inner1.csv",
            "out_inner_mean1.csv",
            "out_inner_std1.csv",
            "tsne_temp1_1.csv",
            "dbscan_temp1_1.csv",
        ],
    ),
    ("dlc_model", ["config.yaml"]),
    (
        "dlc_model/dlc-models/iteration-0/marm_4Sep5-trainset99shuffle1/train",
        [
            "pose_cfg.yaml",
            "snapshot-15000.index",
            "snapshot-15000.meta",
            "snapshot-15000.data-00000-of-00001",
        ],
    ),
    (
        "dlc_model/dlc-models/iteration-0/marm_4Sep5-trainset99shuffle1/test",
        ["inference_cfg.yaml", "pose_cfg.yaml"],
    ),
]


def _file_is_present(relpath: str) -> bool:
    """Return True iff ``relpath`` is already on disk somewhere we
    can read it without hitting the network.

    Checks every directory in :func:`_candidate_data_roots` —
    matches the early-return behaviour of :func:`ensure_data_file`.
    """
    rel = relpath.lstrip("/").replace("\\", "/")
    for root in _candidate_data_roots():
        if (root / rel).exists():
            return True
    return False


def missing_files() -> "list[str]":
    """Return the list of manifest files not yet present locally.

    Does not check the HF cache (``~/.cache/huggingface/hub/``)
    directly; for cached files, the bundled / override checks are
    fast no-ops, and a real download attempt against an already-
    cached file is also a fast no-op (HF returns the cached path
    immediately). So this list is conservative — it may include
    files that are actually downloaded but live in the HF cache
    rather than in a directory we own. The returned list is mostly
    useful for deciding whether to print a "first-run download" log
    message vs staying silent.
    """
    rels = []
    for reldir, filenames in DATA_MANIFEST:
        for fn in filenames:
            rel = f"{reldir}/{fn}" if reldir else fn
            if not _file_is_present(rel):
                rels.append(rel)
    return rels


def ensure_all(verbose: bool = True) -> None:
    """Make sure every manifest file is downloaded, downloading any
    that aren't.

    Idempotent and cheap on the common path: if every file is
    already present locally, this returns in milliseconds without
    touching the network. On a fresh install it downloads everything
    in the manifest (~3 GB total) with HF's standard progress
    bars. Safe to call at the start of every CLI invocation.

    Parameters
    ----------
    verbose:
        If True (default), print a short notice before downloading
        so the user knows where the wait time and disk usage are
        going. Suppress with ``verbose=False`` for callers that
        don't want any extra output (e.g. test harnesses).
    """
    todo = missing_files()
    if not todo:
        return

    if verbose:
        # Avoid spamming the full file list — just say what's
        # happening and where the data comes from.
        target_root = _writable_data_root()
        print(
            f"[marm_behavior] first-run setup: downloading "
            f"{len(todo)} bundled data file(s) from "
            f"https://huggingface.co/datasets/{_hf_repo_id()} "
            f"(~3 GB total) into {target_root}. "
            f"This only happens once."
        )

    for reldir, filenames in DATA_MANIFEST:
        if reldir:
            ensure_data_dir(reldir, filenames)
        else:
            for fn in filenames:
                ensure_data_file(fn)


# Backwards-compatible alias for callers that imported the old name.
prefetch_all = ensure_all
