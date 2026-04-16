"""
marm_behavior.nn_postprocess
===============================

Stage 5: neural-network postprocessing. Literal Python port of the
original ``run_nn_1.py`` script.

For each present animal, this stage reads the 30-column
``<color>_description_<stem>.csv`` produced by Stage 4, concatenates it
with a "buddy" animal's description to form a 60-feature input, slides a
1200-frame window over it, runs each window through a bundled LSTM
encoder to produce a 256-dim latent vector, projects those latents into
2D via openTSNE, and assigns each window to one of N behavioral
clusters via a KNN classifier. The per-animal outputs are two CSVs::

    hcoord_<Color>_<stem>.csv   (n_windows, 2)   2D t-SNE coordinates
    hlabel_<Color>_<stem>.csv   (n_windows,)     integer cluster labels

Reference files (read from ``reference_dir``)::

    out_inner1.csv       (N, 256)   raw latent features from training data
    out_inner_mean1.csv  (256,)     mean for per-run normalization
    out_inner_std1.csv   (256,)     std  for per-run normalization
    tsne_temp1_1.csv     (N, 2)     reference 2D coordinates (for KNN)
    dbscan_temp1_1.csv   (N,)       reference cluster labels   (for KNN)

The default ``reference_dir`` resolves the four required files plus the
3 GB ``out_inner1.csv`` from the bundled HF Hub dataset (downloaded
lazily by :mod:`marm_behavior._data_files`). All five files are
required on every run — the openTSNE fit is repeated each time, exactly
as the original script does. Caching the fit to disk to skip the
~5-minute t-SNE step on subsequent runs is *not* implemented here on
purpose: it changes results by a few percent in some videos due to
optimizer-state and openTSNE-version sensitivity, and the only
acceptable behavior for this stage is bit-for-bit equivalence with
``run_nn_1.py``.

Dependencies
------------
Importing this module is cheap. The heavy dependencies — tensorflow
(for the encoder), openTSNE (for t-SNE), and scikit-learn (for KNN)
— are only imported inside :func:`run_nn_stage` and raise clear
``RuntimeError`` messages if unavailable.
"""

from __future__ import annotations

# Silence TensorFlow's chatty Grappler/oneDNN logging before any
# tensorflow import in this module. Set ``MARM_BEHAVIOR_TF_VERBOSE=1``
# to skip the silencing for debugging.
from ._tf_quiet import silence_tensorflow_logging
silence_tensorflow_logging()

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants (matching run_nn_1.py exactly)
# ---------------------------------------------------------------------------

#: Sliding-window length in frames (= 20 seconds at 60 fps).
SEQUENCE_SIZE = 1200

#: Prediction horizon (unused by this stage but used to derive
#: ``WINDOW_SIZE``, which is the small-chunk threshold below).
PREDICTION_SIZE = 300

#: Threshold for the "skip small chunks" check in the encoder loop.
#: An outer iteration whose effective ``current_size`` is at or below
#: this value is silently dropped — matching the original's
#: ``if current_size > window_size`` guard. With the default
#: SEQUENCE_SIZE+PREDICTION_SIZE this is 1500 frames.
WINDOW_SIZE = SEQUENCE_SIZE + PREDICTION_SIZE

#: Features per frame in the encoder input. 30 from self + 30 from buddy.
NUM_FEATURES = 60

#: Number of columns in each ``<color>_description_<stem>.csv``.
NUM_OUTPUT_FEATURES = 30

#: Encoder latent dimensionality.
N_LATENT = 256

#: Outer-loop chunk size for encoder batching.
MAX_TO_ENCODE = 20000

#: Inner-loop (GPU transfer) chunk size.
GPU_SIZE = 5000

#: Keras ``batch_size`` for ``encoder.predict`` calls.
BATCH_SIZE = 32

#: openTSNE perplexity, matching the original.
TSNE_PERPLEXITY = 250

#: Stride at which to subsample ``out_inner`` before fitting openTSNE.
#: The original uses ``out_inner[0:500000:5, :]`` — that exact slice
#: (capped at 500,000 rows) is reproduced in :func:`_fit_tsne`.
TSNE_SUBSAMPLE_STRIDE = 5

#: Cap on the number of rows fed to ``tsne.fit``. The original slices
#: ``out_inner[0:500000:5, :]``, which gives at most 100,000 rows when
#: ``out_inner`` has >=500,000 rows. We reproduce both halves of that:
#: cap at 500,000 rows then take every fifth.
TSNE_FIT_CAP_ROWS = 500_000

#: The five reference files the nn stage requires on every run.
REFERENCE_FILES: Tuple[str, ...] = (
    "out_inner1.csv",
    "out_inner_mean1.csv",
    "out_inner_std1.csv",
    "tsne_temp1_1.csv",
    "dbscan_temp1_1.csv",
)

#: Default buddy chain: which other animal's description CSV to
#: concatenate with each self animal's, in order of preference. The
#: first available buddy (by file existence) is used. Matches the
#: try/except fallbacks in the original ``run_nn_1.py``: Red->Yellow,
#: White->Blue, Blue->White, Yellow->Red, with the cross-color
#: fallback being the second entry in each tuple.
BUDDY_CHAIN: Dict[str, Tuple[str, ...]] = {
    "r": ("y", "b"),  # Red    -> Yellow / Blue
    "w": ("b", "r"),  # White  -> Blue   / Red
    "b": ("w", "r"),  # Blue   -> White  / Red
    "y": ("r", "b"),  # Yellow -> Red    / Blue
}

#: Short color key -> full title-case name used in output filenames.
COLOR_NAMES: Dict[str, str] = {
    "r": "Red",
    "w": "White",
    "b": "Blue",
    "y": "Yellow",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _default_reference_dir() -> Path:
    """Return the default NN reference directory.

    Returns the on-disk directory containing the canonical NN
    reference set. The files are hosted on the Hugging Face Hub at
    ``williammenegas/data`` (dataset repo) under ``nn_reference/``;
    the first call downloads them into the user's HF cache and
    materializes a consolidated mirror directory. Subsequent calls
    return immediately.

    Override the location entirely with ``--nn-reference-dir``
    (CLI) or ``reference_dir=...`` (Python API), or override the
    download root with the ``MARM_BEHAVIOR_DATA_DIR`` env var. See
    :mod:`marm_behavior._data_files` for the full lookup logic.
    """
    from ._data_files import ensure_data_dir
    return ensure_data_dir(
        "nn_reference",
        list(REFERENCE_FILES),
    )


def _bundled_encoder_path() -> Path:
    """Path to the LSTM encoder hosted on the HF Hub.

    Triggers a download on first call; cached locally afterward.
    """
    from ._data_files import ensure_data_file
    return ensure_data_file("nn_model/pred_model_marm_encoderD.h5")


def _load_reference_dir(reference_dir: Path) -> Dict[str, np.ndarray]:
    """Read all five reference CSV files from ``reference_dir`` into
    a dict keyed by filename. Uses :func:`numpy.genfromtxt` to match
    the original's loader exactly.
    """
    out: Dict[str, np.ndarray] = {}
    for name in REFERENCE_FILES:
        path = reference_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"required NN reference file not found: {path}\n"
                f"All five reference files must be present in "
                f"reference_dir: {list(REFERENCE_FILES)}"
            )
        out[name] = np.genfromtxt(str(path), delimiter=",")
    return out


# ---------------------------------------------------------------------------
# t-SNE fit (literal port of run_nn_1.py lines 58-74)
# ---------------------------------------------------------------------------

def _fit_tsne(out_inner_norm: np.ndarray, log) -> "object":
    """Fit openTSNE on a subsample of the normalized training latents,
    then pre-center the embedding by transforming the full set.

    This is a literal port of the original::

        tsne = TSNE(n_components=2, perplexity=250, ...)
        embedding_train = tsne.fit(out_inner[0:500000:5, :])
        embedding_temp = embedding_train.transform(out_inner)

    The third line's result is unused, but the call has a side
    effect: ``transform()`` centers the training embedding in place
    (``self -= (max + min) / 2``). Without that pre-centering call,
    the very first downstream ``embedding_train.transform(...)``
    triggers the centering instead, and any subsequent transforms
    see the same centered embedding. Mathematically the result is
    the same -- the centering shift depends only on the embedding's
    own max/min, not on the input -- but we reproduce the call here
    verbatim to eliminate it as a possible source of drift.

    Parameters
    ----------
    out_inner_norm:
        Normalized training latents -- i.e. ``(out_inner - mean) / std``.
        The fit subsamples ``out_inner_norm[0:500000:5, :]``.
    log:
        Logger callable.

    Returns
    -------
    The fitted ``TSNEEmbedding`` (already pre-centered).
    """
    from openTSNE import TSNE

    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        learning_rate="auto",
        early_exaggeration=2,
        early_exaggeration_iter=100,
        exaggeration=1,
        n_iter=1000,
        metric="cosine",
        n_jobs=-1,
        random_state=42,
        verbose=True,
    )

    # Reproduce out_inner[0:500000:5, :] exactly -- first cap rows at
    # TSNE_FIT_CAP_ROWS, then stride.
    capped = out_inner_norm[:TSNE_FIT_CAP_ROWS, :]
    fit_data = capped[::TSNE_SUBSAMPLE_STRIDE, :]
    log(
        f"fitting openTSNE on {fit_data.shape[0]} samples "
        f"(perplexity={TSNE_PERPLEXITY}; this takes several minutes)"
    )
    embedding_train = tsne.fit(fit_data)

    # Pre-center the embedding via the dummy full transform. Result
    # is intentionally discarded.
    log(
        f"pre-centering embedding by transforming full "
        f"out_inner ({out_inner_norm.shape[0]} points)"
    )
    _ = embedding_train.transform(out_inner_norm)

    return embedding_train


# ---------------------------------------------------------------------------
# Encoder loop (literal port of the per-animal processing block in
# run_nn_1.py, lines 108-133 / 163-188 / 218-243 / 273-298)
# ---------------------------------------------------------------------------

def _encode_windows(
    encoder_path: Path,
    x_in: np.ndarray,
    *,
    log,
    color_name: str = "",
) -> np.ndarray:
    """Run a ``(n_frames, 60)`` uint8 feature sequence through the
    LSTM encoder, returning a stacked ``(N, 256)`` latent matrix.

    Literal port of the original's outer/inner loop::

        for loop_encode in range(0, num_to_encode, max_to_encode):
            me1 = load_model(...)
            current_min = loop_encode
            current_max = loop_encode + max_to_encode
            if current_max > num_to_encode - sequence_size - 1:
                current_max = num_to_encode - sequence_size - 1
            current_size = current_max - current_min
            if current_size > window_size:
                x_input = np.zeros((max_to_encode, sequence_size, 60), uint8)
                x_count = 0
                for g in range(current_min, current_max):
                    x_input[x_count] = x_in[g:g + sequence_size]
                    x_count += 1
                x_input = x_input[:x_count]
                m = np.zeros((current_size, 256))
                for g in range(0, current_size, gpu_size):
                    temp = me1.predict(x_input[g:g+gpu_size], ...)
                    temp = temp[:, 0, :]      # FIRST timestep (intentional)
                    m[g:g + temp.shape[0]] = temp
                if loop_encode == 0:
                    m_running = m
                else:
                    m_running = vstack([m_running, m])
                del me1

    Quirks preserved verbatim:

    * **Outer iteration is over frames, not windows.** ``range(0,
      num_to_encode, max_to_encode)`` steps in frame units; the
      effective window count per iteration is then derived inside.
    * **Small trailing chunks are silently dropped.** If the capped
      ``current_size`` is at or below ``window_size`` (1500 frames),
      the iteration is skipped entirely -- no windows from it
      appear in the output.
    * **The encoder is reloaded every outer iteration.** The original
      used ``load_model`` inside the loop; this reset any
      Keras-internal state (e.g. stateful LSTM hidden states) at
      every iteration. We do the same. For typical-sized videos with
      <=20,000 frames this fires once and is essentially a no-op,
      but for longer videos it materially affects results if the
      encoder graph carries any persistent state.
    * **The first timestep is taken from the encoder output**
      (``temp[:, 0, :]``), not the last. This matches the original
      and is annotated as "important (0 or -1)" in the source.

    Parameters
    ----------
    encoder_path:
        Path to ``pred_model_marm_encoderD.h5``. Reloaded every outer
        iteration via ``keras.models.load_model``.
    x_in:
        Concatenated ``(n_frames, 60)`` uint8 self+buddy feature
        matrix.
    log:
        Logger callable.

    Returns
    -------
    np.ndarray
        Stacked latent matrix. The number of rows equals the sum of
        ``current_size`` across all non-skipped outer iterations.
    """
    from tensorflow.keras.models import load_model

    num_to_encode = int(x_in.shape[0])
    if num_to_encode < SEQUENCE_SIZE + 2:
        raise ValueError(
            f"not enough frames ({num_to_encode}) for sequence_size="
            f"{SEQUENCE_SIZE}; need at least {SEQUENCE_SIZE + 2}"
        )

    # Make sure x_in is contiguous uint8 for fast slicing.
    if x_in.dtype != np.uint8:
        x_in = x_in.astype(np.uint8)
    x_in = np.ascontiguousarray(x_in)

    # Pre-compute the exact total number of windows we'll process,
    # honouring the same skip-small-chunk guard the loop below uses.
    # This is the tqdm total — a lying total would either show the
    # bar finishing early (overcounted) or hanging at <100% forever
    # (undercounted), both of which are worse than no bar.
    total_windows = 0
    for _loop in range(0, num_to_encode, MAX_TO_ENCODE):
        _cmax = min(_loop + MAX_TO_ENCODE, num_to_encode - SEQUENCE_SIZE - 1)
        _csize = _cmax - _loop
        if _csize > WINDOW_SIZE:
            total_windows += _csize

    # tqdm is optional — no hard dependency, just a nicer bar when
    # available. Falls back to a no-op shim that still works.
    try:
        from tqdm.auto import tqdm
        bar_desc = (
            f"  encoding {color_name}" if color_name else "  encoding"
        )
        progress = tqdm(
            total=total_windows,
            desc=bar_desc,
            unit="win",
            unit_scale=True,
            leave=True,
            mininterval=0.5,
        )
    except ImportError:
        progress = None

    m_running: "Optional[np.ndarray]" = None

    try:
        for loop_encode in range(0, num_to_encode, MAX_TO_ENCODE):
            # Reload encoder every outer iteration (matches original).
            me1 = load_model(str(encoder_path), compile=False)

            current_min = loop_encode
            current_max = loop_encode + MAX_TO_ENCODE
            if current_max > num_to_encode - SEQUENCE_SIZE - 1:
                current_max = num_to_encode - SEQUENCE_SIZE - 1
            current_size = current_max - current_min

            # Skip small trailing chunks. This silently drops up to
            # WINDOW_SIZE-1 windows at the end of long videos. Matches
            # original ``if current_size > window_size`` guard.
            if current_size > WINDOW_SIZE:
                x_input = np.zeros(
                    (MAX_TO_ENCODE, SEQUENCE_SIZE, NUM_FEATURES),
                    dtype=np.uint8,
                )
                x_count = 0
                for g in range(current_min, current_max):
                    x_input[x_count, :, :] = x_in[g : g + SEQUENCE_SIZE, :]
                    x_count += 1
                x_input = x_input[0:x_count, :, :]

                m = np.zeros((current_size, N_LATENT))
                for g in range(0, current_size, GPU_SIZE):
                    sub = x_input[g : g + GPU_SIZE, :, :]
                    temp = me1.predict(sub, verbose=0, batch_size=BATCH_SIZE)
                    # First timestep -- see docstring quirk above.
                    if temp.ndim == 3:
                        temp = temp[:, 0, :]
                    m[g : g + sub.shape[0], :] = temp
                    if progress is not None:
                        progress.update(sub.shape[0])

                if loop_encode == 0:
                    m_running = m
                else:
                    assert m_running is not None
                    m_running = np.vstack((m_running, m))

            # Free the per-iteration model. The original does ``del me1``
            # at the same spot.
            del me1
    finally:
        if progress is not None:
            progress.close()

    if m_running is None:
        # Every outer iteration was skipped -- pathological short input.
        raise ValueError(
            f"no windows produced: num_to_encode={num_to_encode} "
            f"with max_to_encode={MAX_TO_ENCODE}, sequence_size="
            f"{SEQUENCE_SIZE} produces no chunk above the "
            f"window_size={WINDOW_SIZE} threshold"
        )
    return m_running


# ---------------------------------------------------------------------------
# Description-CSV resolution
# ---------------------------------------------------------------------------

def _resolve_description_csv(
    color_key: str,
    description_csvs: Mapping[str, Path],
    output_dir: Path,
    base: str,
) -> Optional[Path]:
    """Return the path to ``<color>_description_<base>.csv`` if it
    exists, looking first in the caller-supplied map and then in
    ``output_dir`` (where Stage 4 wrote the files).
    """
    p = description_csvs.get(color_key)
    if p is not None and Path(p).exists():
        return Path(p)
    fallback = output_dir / f"{color_key}_description_{base}.csv"
    if fallback.exists():
        return fallback
    return None


def _resolve_buddy_csv(
    color_key: str,
    description_csvs: Mapping[str, Path],
    output_dir: Path,
    base: str,
    buddy_chain: "Sequence[str]",
) -> "Tuple[Optional[Path], Optional[str]]":
    """Resolve the buddy description CSV for ``color_key`` by walking
    ``buddy_chain`` in order. Returns ``(path, color_key)`` for the
    first buddy whose description CSV is on disk, or ``(None, None)``.
    """
    for candidate in buddy_chain:
        if candidate == color_key:
            continue
        p = _resolve_description_csv(
            candidate, description_csvs, output_dir, base
        )
        if p is not None:
            return p, candidate
    return None, None


def _effective_buddy_chain(
    buddies: "Optional[Mapping[str, Sequence[str]]]",
) -> Dict[str, Tuple[str, ...]]:
    """Merge user buddy overrides with the built-in defaults."""
    merged: Dict[str, Tuple[str, ...]] = {
        k: tuple(v) for k, v in BUDDY_CHAIN.items()
    }
    if not buddies:
        return merged
    for self_key, override in buddies.items():
        if self_key not in BUDDY_CHAIN:
            raise ValueError(
                f"unknown self-color in buddy override: {self_key!r} "
                f"(expected one of {sorted(BUDDY_CHAIN)})"
            )
        if not override:
            continue
        chain: list[str] = []
        for c in override:
            c = str(c).lower()
            if c not in BUDDY_CHAIN:
                raise ValueError(
                    f"unknown buddy color in override for "
                    f"{self_key!r}: {c!r} (expected one of "
                    f"{sorted(BUDDY_CHAIN)})"
                )
            if c == self_key:
                continue
            if c not in chain:
                chain.append(c)
        if chain:
            merged[self_key] = tuple(chain)
    return merged


# ---------------------------------------------------------------------------
# Per-animal processing (literal port of the per-color blocks in
# run_nn_1.py)
# ---------------------------------------------------------------------------

def _process_animal(
    color_key: str,
    description_csvs: Mapping[str, Path],
    output_dir: Path,
    base: str,
    encoder_path: Path,
    embedding_train,
    knn,
    out_inner_mean: np.ndarray,
    out_inner_std: np.ndarray,
    buddy_chain: "Sequence[str]",
    log,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Run the encoder -> normalize -> t-SNE -> KNN pipeline for one
    animal, write ``hcoord_*`` and ``hlabel_*`` CSVs, and return their
    paths (or ``(None, None)`` if the animal was skipped).

    Literal port of one of the per-color blocks in ``run_nn_1.py``.
    Differences from the original:

    * The original wraps each color block in a bare ``try/except``
      that silently sets ``failed_<color>=1`` on any exception and
      continues. We let exceptions propagate to :func:`run_nn_stage`,
      which catches per-animal so one animal's failure doesn't kill
      the rest, and logs the traceback. Same semantics, better
      diagnostics.
    * The original's idempotence check (``hlabel_<Color>_<base>.csv``
      already exists -> skip) is reproduced here.
    * No bootstrap fallback, no zero-buddy fallback, no caching.
      Missing buddy CSV -> animal is skipped with a log message.
    """
    color_name = COLOR_NAMES[color_key]
    hcoord_path = output_dir / f"hcoord_{color_name}_{base}.csv"
    hlabel_path = output_dir / f"hlabel_{color_name}_{base}.csv"

    # Idempotence: matches the original ``complete_check`` test.
    if hlabel_path.exists():
        log(f"  {color_name}: hlabel already exists, skipping")
        return (
            hcoord_path if hcoord_path.exists() else None,
            hlabel_path,
        )

    self_csv = _resolve_description_csv(
        color_key, description_csvs, output_dir, base
    )
    if self_csv is None:
        log(f"  {color_name}: no self description CSV found, skipping")
        return None, None

    buddy_csv, buddy_key = _resolve_buddy_csv(
        color_key, description_csvs, output_dir, base, buddy_chain
    )
    if buddy_csv is None:
        log(
            f"  {color_name}: no buddy description CSV available "
            f"(tried {list(buddy_chain)}), skipping"
        )
        return None, None

    # Load both description CSVs, cast to uint8, and concatenate. Matches
    # the original's ``genfromtxt(...).astype(np.uint8) + np.hstack``.
    x_in = np.genfromtxt(str(self_csv), delimiter=",").astype(np.uint8)
    y_in = np.genfromtxt(str(buddy_csv), delimiter=",").astype(np.uint8)
    if x_in.shape != y_in.shape:
        raise ValueError(
            f"{color_name}: shape mismatch between self ({x_in.shape}) "
            f"and buddy ({y_in.shape}) description CSVs. Both should "
            f"be (n_frames, {NUM_OUTPUT_FEATURES})."
        )
    x_in = np.hstack((x_in, y_in))
    log(
        f"  {color_name}: self={self_csv.name} "
        f"buddy={buddy_csv.name} ({COLOR_NAMES[buddy_key]}) "
        f"-> features {x_in.shape}"
    )

    # Encode.
    log(f"  {color_name}: running encoder")
    m_running = _encode_windows(
        encoder_path, x_in, log=log, color_name=color_name,
    )

    # Normalize AFTER encoding (matches original lines 135-136).
    m_running = m_running - out_inner_mean
    m_running = m_running / out_inner_std

    # Project into the reference t-SNE space.
    log(f"  {color_name}: t-SNE transform ({m_running.shape[0]} points)")
    t_embedded = np.asarray(embedding_train.transform(m_running))

    # KNN classify.
    log(f"  {color_name}: KNN cluster assignment")
    t_dbscan = knn.predict(t_embedded)

    # Write outputs. ``np.savetxt`` defaults match the original
    # (no header, default float format).
    np.savetxt(str(hcoord_path), t_embedded, delimiter=",")
    np.savetxt(str(hlabel_path), t_dbscan, delimiter=",")
    log(
        f"  {color_name}: wrote {hcoord_path.name} + {hlabel_path.name}"
    )
    return hcoord_path, hlabel_path


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

def prepare_nn_artifacts(
    reference_dir: "str | Path | None" = None,
    *,
    verbose: bool = True,
) -> Dict[str, "object"]:
    """Build the per-batch reusable artifacts for the nn stage.

    The expensive part of the nn stage is the openTSNE fit on the
    ~100k-row training subsample (~5 minutes). That fit depends only
    on the reference files, not on any per-video input, so when
    processing many videos in a row we can do it once and reuse the
    fitted embedding across all of them.

    This helper does the work that's ``video``-independent — load
    references, fit openTSNE, build the KNN classifier — and returns
    the results in a dict suitable for passing as the ``artifacts``
    kwarg to :func:`run_nn_stage`. Calling
    ``run_nn_stage(..., artifacts=prepare_nn_artifacts())`` on the
    same inputs as ``run_nn_stage(...)`` produces bit-identical
    outputs; the only difference is that the per-call setup cost
    moves from inside the call to outside it.

    Parameters
    ----------
    reference_dir:
        Folder containing the five required reference files. If
        ``None``, uses the bundled set from the HF Hub (downloaded
        on first use; cached afterward).
    verbose:
        If True, log each step of the setup.

    Returns
    -------
    dict
        ``{
            "embedding_train": <openTSNE TSNEEmbedding>,
            "knn": <fitted KNeighborsClassifier>,
            "out_inner_mean": np.ndarray,
            "out_inner_std": np.ndarray,
            "reference_dir": Path,
        }``

        Pass the whole dict to :func:`run_nn_stage` via the
        ``artifacts`` kwarg.
    """
    try:
        from sklearn.neighbors import KNeighborsClassifier
    except ImportError as err:  # pragma: no cover
        raise RuntimeError(
            "The 'nn' stage requires scikit-learn. Install it with:\n"
            "    pip install 'scikit-learn>=1.0'"
        ) from err

    if reference_dir is None:
        reference_dir = _default_reference_dir()
    reference_dir = Path(reference_dir).expanduser().resolve()

    def _log(msg: str) -> None:
        if verbose:
            print(f"[marm_behavior.nn] {msg}")

    _log(f"loading reference files from {reference_dir}")
    ref = _load_reference_dir(reference_dir)
    out_inner_raw = ref["out_inner1.csv"]
    out_inner_mean = ref["out_inner_mean1.csv"]
    out_inner_std = ref["out_inner_std1.csv"]
    out_inner_tsne = ref["tsne_temp1_1.csv"]
    out_inner_labels = ref["dbscan_temp1_1.csv"]
    _log(
        f"reference shapes: out_inner={out_inner_raw.shape} "
        f"tsne={out_inner_tsne.shape} labels={out_inner_labels.shape}"
    )

    out_inner_norm = out_inner_raw - out_inner_mean
    out_inner_norm = out_inner_norm / out_inner_std

    embedding_train = _fit_tsne(out_inner_norm, _log)

    _log(
        f"fitting KNN (k=5) on {out_inner_tsne.shape[0]} reference points"
    )
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(out_inner_tsne, out_inner_labels)

    return {
        "embedding_train": embedding_train,
        "knn": knn,
        "out_inner_mean": out_inner_mean,
        "out_inner_std": out_inner_std,
        "reference_dir": reference_dir,
    }


def run_nn_stage(
    description_csvs: Mapping[str, "str | Path"],
    video_path: "str | Path",
    output_dir: "str | Path",
    *,
    reference_dir: "str | Path | None" = None,
    artifacts: "Optional[Dict[str, object]]" = None,
    buddies: "Optional[Mapping[str, Sequence[str]]]" = None,
    red_present: bool = True,
    white_present: bool = True,
    blue_present: bool = True,
    yellow_present: bool = True,
    verbose: bool = True,
) -> Dict[str, Tuple[Path, Path]]:
    """Run the NN postprocessing stage on a single video.

    Literal port of ``run_nn_1.py``. By default the openTSNE fit
    happens fresh on every call (no on-disk caching of the fitted
    embedding), so expect ~5 minutes of t-SNE work at the start of
    every invocation.

    For batch processing, call :func:`prepare_nn_artifacts` once
    upfront and pass the result via the ``artifacts`` kwarg —
    that's bit-identical to calling without it but skips the
    redundant ~5 min of t-SNE work per video.

    Parameters
    ----------
    description_csvs:
        ``{'r': path, 'w': path, 'b': path, 'y': path}`` -- paths to
        the per-animal description CSVs from Stage 4. Missing keys
        are looked up in ``output_dir`` as a fallback.
    video_path:
        Original video path. Only used to derive the ``<stem>`` for
        output filenames -- no video I/O happens in this stage.
    output_dir:
        Where to write ``hcoord_<Color>_<stem>.csv`` and
        ``hlabel_<Color>_<stem>.csv``. Created if missing.
    reference_dir:
        Folder containing the five required reference files. If
        ``None``, uses the bundled set from the HF Hub. Ignored if
        ``artifacts`` is provided (the artifacts dict carries its
        own resolved reference_dir).
    artifacts:
        Optional pre-built artifacts dict from
        :func:`prepare_nn_artifacts`. When supplied, skips the
        load-references / fit-tsne / build-knn setup and uses the
        provided objects directly. Use this when processing many
        videos in batch to amortize the ~5 min openTSNE fit cost.
    buddies:
        Optional per-animal override of the buddy chain.
    red_present, white_present, blue_present, yellow_present:
        Whether each animal is present in the video.
    verbose:
        If True, log each step.

    Returns
    -------
    dict
        ``{color_name: (hcoord_path, hlabel_path)}`` for each animal
        that was successfully processed.
    """
    # Heavy dep imports with clear errors.
    try:
        from tensorflow.keras.models import load_model  # noqa: F401
    except ImportError as err:  # pragma: no cover
        raise RuntimeError(
            "The 'nn' stage requires tensorflow. Install it with:\n"
            "    pip install 'tensorflow>=2.5'\n"
            "Or skip this stage via stages=('dlc','extract','process',"
            "'depths','labels') / '--stages ... (no nn)' on the CLI."
        ) from err
    try:
        from sklearn.neighbors import KNeighborsClassifier
    except ImportError as err:  # pragma: no cover
        raise RuntimeError(
            "The 'nn' stage requires scikit-learn. Install it with:\n"
            "    pip install 'scikit-learn>=1.0'"
        ) from err

    video_path = Path(video_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve reference_dir only when we'll actually need it (no
    # artifacts supplied). Resolving it eagerly via
    # _default_reference_dir() can trigger an unwanted HF download
    # in batch mode where the artifacts dict already has everything.
    if artifacts is None:
        if reference_dir is None:
            reference_dir = _default_reference_dir()
        reference_dir = Path(reference_dir).expanduser().resolve()

    base = video_path.stem
    present_flags = {
        "r": red_present,
        "w": white_present,
        "b": blue_present,
        "y": yellow_present,
    }
    buddy_chains = _effective_buddy_chain(buddies)

    def _log(msg: str) -> None:
        if verbose:
            print(f"[marm_behavior.nn] {msg}")

    # Log any non-default buddy overrides.
    for self_key in ("r", "w", "b", "y"):
        if buddy_chains[self_key] != BUDDY_CHAIN[self_key]:
            chain_names = [
                COLOR_NAMES[k] for k in buddy_chains[self_key]
            ]
            _log(
                f"buddy override: {COLOR_NAMES[self_key]} -> "
                f"{chain_names}"
            )

    # Resolve encoder path. The encoder file is reloaded inside
    # _encode_windows on every outer iteration (matching the
    # original), so we just need the path here.
    encoder_path = _bundled_encoder_path()
    if not encoder_path.exists():
        raise FileNotFoundError(
            f"bundled encoder not found at {encoder_path}"
        )
    _log(f"encoder: {encoder_path.name}")

    # Either use pre-built artifacts (batch mode) or build them inline
    # (single-video mode). Behaviour is identical either way; the
    # difference is just where the ~5 min openTSNE fit happens.
    if artifacts is not None:
        # The artifacts dict carries its own resolved reference_dir;
        # any caller-supplied reference_dir is silently ignored
        # because changing it without rebuilding the embedding would
        # produce inconsistent outputs.
        embedding_train = artifacts["embedding_train"]
        knn = artifacts["knn"]
        out_inner_mean = artifacts["out_inner_mean"]
        out_inner_std = artifacts["out_inner_std"]
        _log(
            "reusing pre-built nn artifacts (skipping the ~5 min "
            "openTSNE fit and KNN setup)"
        )
    else:
        # Build artifacts inline. Equivalent to calling
        # prepare_nn_artifacts() and using the result.
        _log(f"loading reference files from {reference_dir}")
        ref = _load_reference_dir(reference_dir)
        out_inner_raw = ref["out_inner1.csv"]
        out_inner_mean = ref["out_inner_mean1.csv"]
        out_inner_std = ref["out_inner_std1.csv"]
        out_inner_tsne = ref["tsne_temp1_1.csv"]
        out_inner_labels = ref["dbscan_temp1_1.csv"]
        _log(
            f"reference shapes: out_inner={out_inner_raw.shape} "
            f"tsne={out_inner_tsne.shape} labels={out_inner_labels.shape}"
        )

        # Normalize training latents (matches original lines 50-51).
        out_inner_norm = out_inner_raw - out_inner_mean
        out_inner_norm = out_inner_norm / out_inner_std

        # Fit openTSNE + pre-center via dummy transform.
        embedding_train = _fit_tsne(out_inner_norm, _log)

        # Fit KNN on the reference 2D coords + cluster labels.
        _log(
            f"fitting KNN (k=5) on {out_inner_tsne.shape[0]} "
            f"reference points"
        )
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(out_inner_tsne, out_inner_labels)

    # Process each present animal. The original processes them in the
    # order Red -> White -> Blue -> Yellow; we match that.
    results: Dict[str, Tuple[Path, Path]] = {}
    for color_key in ("r", "w", "b", "y"):
        if not present_flags[color_key]:
            _log(f"  {COLOR_NAMES[color_key]}: marked not-present, skipping")
            continue
        try:
            hcoord, hlabel = _process_animal(
                color_key,
                description_csvs,
                output_dir,
                base,
                encoder_path,
                embedding_train,
                knn,
                out_inner_mean,
                out_inner_std,
                buddy_chains[color_key],
                _log,
            )
            if hcoord is not None and hlabel is not None:
                results[COLOR_NAMES[color_key]] = (hcoord, hlabel)
        except Exception as err:
            # One animal's failure shouldn't kill the whole stage.
            _log(
                f"  {COLOR_NAMES[color_key]}: FAILED with "
                f"{type(err).__name__}: {err}"
            )
            if verbose:
                import traceback
                traceback.print_exc()

    return results
