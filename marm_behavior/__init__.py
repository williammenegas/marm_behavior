"""
marm_behavior
=============

Multi-animal marmoset behavioral analysis pipeline. Takes a video,
runs DeepLabCut pose estimation, extracts per-animal body-part tracks,
computes per-frame behavioral features, and projects the features into
a learned cluster space.

Quickstart
----------

>>> from marm_behavior import run
>>> result = run("my_video.avi")
>>> result["descriptions"]
{'w': Path('w_description_my_video.csv'), ...}

See :func:`marm_behavior.run.run` for all options.
"""

from __future__ import annotations

from .run import run

__all__ = ["run"]
