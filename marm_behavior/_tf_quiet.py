"""Helper to silence the very chatty TensorFlow / Grappler / CUDA
loggers without dropping anything that would actually matter to the
user.

TensorFlow's C++ logger spams hundreds of "Error in PredictCost() for
the op: ..." warnings during ``predict()`` calls. These are Grappler
(the graph optimizer) saying it can't estimate the cost of an op
for its scheduling heuristics — they're not errors, they don't
affect the result, and there's no way to fix them at the user level.
We suppress them by setting the standard TF log-level env var
*before* tensorflow is imported.

The Python-level ``tf.get_logger()`` is also bumped up to ERROR after
import to catch a few additional warning categories that go through
the Python logger rather than the C++ one.

Call :func:`silence_tensorflow_logging` once at startup, before any
``import tensorflow`` happens. Idempotent — safe to call multiple
times. Honors a ``MARM_BEHAVIOR_TF_VERBOSE=1`` escape hatch for
users who want the noise back when debugging.
"""

from __future__ import annotations

import os


def silence_tensorflow_logging() -> None:
    """Silence TensorFlow's C++ logger and a few related sources.

    Sets ``TF_CPP_MIN_LOG_LEVEL=3`` (only fatal C++ messages get
    through) and ``TF_ENABLE_ONEDNN_OPTS=0`` (suppresses oneDNN's
    chatty optimization notices on x86). Both env vars must be set
    *before* tensorflow is first imported, otherwise they're
    silently ignored — that's why this lives in a tiny standalone
    module that can be imported from anywhere without dragging in
    other dependencies.

    If TF is already imported by the time we're called, we still
    bump the Python-level logger to ERROR. Some warnings get
    through that way even when the C++ var was missed.

    Set ``MARM_BEHAVIOR_TF_VERBOSE=1`` to skip the silencing
    entirely (useful for debugging encoder issues).
    """
    if os.environ.get("MARM_BEHAVIOR_TF_VERBOSE") == "1":
        return

    # C++ logger threshold:
    #   0 = all messages (default)
    #   1 = no INFO
    #   2 = no INFO or WARNING  <-- silences "Error in PredictCost"
    #   3 = no INFO/WARNING/ERROR (only FATAL)
    # Use 2 — that kills the Grappler spam without hiding genuine
    # tensorflow errors users would want to see.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    # oneDNN optimization notices.
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    # If TF is already imported, also bump the Python-side logger.
    # This catches a few warnings that route through Python rather
    # than C++. We do this best-effort — if TF isn't installed yet
    # or the import fails, just move on.
    import sys
    if "tensorflow" in sys.modules:
        try:
            import tensorflow as tf  # type: ignore
            tf.get_logger().setLevel("ERROR")
        except Exception:
            pass
