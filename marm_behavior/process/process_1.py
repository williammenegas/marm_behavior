"""
process_1
=========

Computes.

Adapts the input matrices ``Red`` / ``White`` / ``Blue`` / ``Yellow`` (each
``n_frames × 42``) into per-animal dictionaries with named body parts.  The
the process stage constructs four  structs (``W``, ``B``, ``R``, ``Y``)
whose fields differ between animals — for the white animal the fields are
``Front``, ``Left``, ``Right``, ..., while for blue/red/yellow the fields are
suffixed (``FrontBlue``, ``FrontRed``, ``FrontYellow``).  The Python port
keeps both the unsuffixed and suffixed field names so that subsequent
modules ported from `the process stage` and friends can use the names
they were written against.

This file is a pure data-shape transformation; there is no numerical logic.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


#: Order of the 21 body parts in the 42-column matrices.  The slice for
#: body-part ``i`` is ``columns[2*i : 2*i + 2]`` (x, y).
BODY_PART_ORDER = (
    "Front",
    "Left",
    "Right",
    "Middle",
    "Body1",
    "Body2",
    "Body3",
    "LS",
    "RS",
    "FL1",
    "FL2",
    "FL3",
    "FR1",
    "FR2",
    "FR3",
    "BL1",
    "BL2",
    "BL3",
    "BR1",
    "BR2",
    "BR3",
)


def split_body_parts(matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """Slice an ``n × 42`` matrix into a dict of ``(n, 2)`` body-part arrays.

    Keys are the 21 names in :data:`BODY_PART_ORDER`.
    """
    if matrix.shape[1] != 42:
        raise ValueError(f"expected 42 columns, got {matrix.shape[1]}")
    return {
        name: matrix[:, 2 * i : 2 * i + 2]
        for i, name in enumerate(BODY_PART_ORDER)
    }


def make_animal(
    matrix: np.ndarray,
    *,
    suffix: str = "",
) -> Dict[str, np.ndarray]:
    """Return a dict keyed by ``<BodyPart><suffix>`` (matches the input indexing structs).

    ``suffix=""`` produces the white-animal field names (``Front``, ``Left``,
    ...); ``suffix="Blue"`` etc. reproduce the suffixed names used for the
    other three animals in `the process stage`.
    """
    parts = split_body_parts(matrix)
    return {f"{name}{suffix}": arr for name, arr in parts.items()}


def make_all_animals(
    *,
    Red: np.ndarray,
    White: np.ndarray,
    Blue: np.ndarray,
    Yellow: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Convenience wrapper producing ``{'W': ..., 'B': ..., 'R': ..., 'Y': ...}``.

    Shape mirrors how `the process stage` ends up structured: four 
    structs containing the 21 named body parts each.
    """
    return {
        "W": make_animal(White, suffix=""),
        "B": make_animal(Blue, suffix="Blue"),
        "R": make_animal(Red, suffix="Red"),
        "Y": make_animal(Yellow, suffix="Yellow"),
    }
