"""
postures
========

Egocentric posture computation.  This module factors the four-times-repeated
rotation/centering math from ``step_2_process/the process stage`` into a
single function that runs on one animal at a time.

The same ~80-line block is unrolled four times, once
each for white / blue / red / yellow.  Everything below is the white block,
generalised: "given an animal's 21 named body parts, compute its body angle
from Body1→Body3, center every body part on Body2, rotate the centered cloud
into the animal's own frame, and return the rotated F_3/F_4 matrices plus the
body angle ``theta``."

The same function is also useful for `the process stage`'s "other animal in
white coordinates" block — there you simply pass the *other* animal's body
parts together with the *white* animal's centroid and angle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..numerics.helpers import cart2pol, fillmissing_pchip
from .process_1 import BODY_PART_ORDER


@dataclass
class Egocentric:
    """Result of :func:`compute_egocentric`.

    ``F_3`` and ``F_4`` are ``n × 21`` matrices of x and y coordinates in the
    animal's own frame, after centering on Body2 and rotating so the body
    axis points along the +x direction.  ``theta`` is the per-frame body angle
    (radians).  ``B2`` is the per-frame Body2 anchor used as the centering
    origin (handy when later code needs to undo the transform).
    """

    F_3: np.ndarray  # (n, 21) rotated x in egocentric frame
    F_4: np.ndarray  # (n, 21) rotated y in egocentric frame
    theta: np.ndarray  # (n,) body angle
    B2: np.ndarray  # (n, 2) Body2 anchor


def _body_anchor(animal: Dict[str, np.ndarray], part_name: str, suffix: str) -> np.ndarray:
    """Return the (n, 2) ``fillmissing(pchip)`` anchor for ``part_name``.

    Replicates the pattern::

        B1a = fillmissing(W.Body1(:,1),'pchip');
        B1b = fillmissing(W.Body1(:,2),'pchip');
        B1 = horzcat(B1a, B1b);
    """
    arr = animal[f"{part_name}{suffix}"]
    return np.column_stack(
        [fillmissing_pchip(arr[:, 0]), fillmissing_pchip(arr[:, 1])]
    )


def compute_egocentric(
    animal: Dict[str, np.ndarray],
    *,
    suffix: str = "",
    body_parts: tuple = BODY_PART_ORDER,
) -> Egocentric:
    """Compute the egocentric body-part frame for one animal.

    Parameters
    ----------
    animal:
        A dict like the one returned by
        :func:`marm_behavior.process.the process stageake_animal`.  Field names
        are expected to be ``"<part><suffix>"``.
    suffix:
        Empty string for white, ``"Blue"``/``"Red"``/``"Yellow"`` for the
        other animals.  Matches the convention in `the process stage`.
    body_parts:
        Override the body-part ordering if needed; defaults to
        :data:`BODY_PART_ORDER`.

    Returns
    -------
    Egocentric
    """
    B1 = _body_anchor(animal, "Body1", suffix)
    B2 = _body_anchor(animal, "Body2", suffix)
    B3 = _body_anchor(animal, "Body3", suffix)

    # Body angle from B1 -> B3.  reference:     #   Angles_X = B1(:,1)-B3(:,1);  Angles_Y = B1(:,2)-B3(:,2);
    #   [theta, rho] = cart2pol(Angles_X, Angles_Y);
    angles_x = B1[:, 0] - B3[:, 0]
    angles_y = B1[:, 1] - B3[:, 1]
    theta, _ = cart2pol(angles_x, angles_y)

    # Center every body part on Body2.
    n = B2.shape[0]
    n_parts = len(body_parts)
    F_3 = np.empty((n, n_parts))
    F_4 = np.empty((n, n_parts))
    for j, part in enumerate(body_parts):
        arr = animal[f"{part}{suffix}"]
        F_3[:, j] = arr[:, 0] - B2[:, 0]
        F_4[:, j] = arr[:, 1] - B2[:, 1]

    # Rotate by -theta into the animal's own frame.  The original  does:
    #   t = -1 * theta;
    #   parfor i = 1:N
    #       R = [cos(t(i)) -sin(t(i)); sin(t(i)) cos(t(i))];
    #       temp = vertcat(F_3(i,:), F_4(i,:));
    #       temp_new = R * temp;
    #       F_3(i,:) = temp_new(2, :);  % note row swap
    #       F_4(i,:) = temp_new(1, :);
    #   end
    #
    # The row swap is intentional: after the rotation the
    # *second* row of the rotated matrix becomes F_3 and the *first* row
    # becomes F_4.  We reproduce that here, vectorised over all frames.
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    # rotated row 0 = cos*F_3 - sin*F_4   -> goes to F_4
    # rotated row 1 = sin*F_3 + cos*F_4   -> goes to F_3
    new_row0 = cos_t[:, None] * F_3 - sin_t[:, None] * F_4
    new_row1 = sin_t[:, None] * F_3 + cos_t[:, None] * F_4
    F_3_out = new_row1
    F_4_out = new_row0

    return Egocentric(F_3=F_3_out, F_4=F_4_out, theta=theta, B2=B2)


def rotate_into_frame(
    F_3: np.ndarray,
    F_4: np.ndarray,
    anchor: np.ndarray,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generic rotation helper for the "other animal in white coordinates" math.

    Centers ``(F_3, F_4)`` on ``anchor`` (per-frame ``(n, 2)`` array) and
    rotates by ``-theta``, with the same row-swap convention used by the
     ``parfor`` blocks.  Useful for the ``F_3BlueR`` / ``F_4BlueR``
    style transforms in `the process stage`.
    """
    F_3c = F_3 - anchor[:, [0]]
    F_4c = F_4 - anchor[:, [1]]
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    new_row0 = cos_t[:, None] * F_3c - sin_t[:, None] * F_4c
    new_row1 = sin_t[:, None] * F_3c + cos_t[:, None] * F_4c
    return new_row1, new_row0
