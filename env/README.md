env/
====

Conda environment specs for marm_behavior.

deep_learning.yml
-----------------
Cross-platform environment file for setting up the lab's reference
Python stack. Use this one for new installs:

    conda env create -f env/deep_learning.yml
    conda activate deep_learning
    pip install -e .

This installs Python 3.8.12 plus the exact ML stack the canonical
nn-stage outputs were produced with: TensorFlow 2.9.1, openTSNE 0.6.2,
scikit-learn 1.1.2, DeepLabCut 2.2.0.6, NumPy 1.22.4. The Windows-only
Conda packages (vc, vs2015_runtime, wincertstore) from the original
lab env have been stripped so the same file installs cleanly on
Linux/macOS too.

Do not bump TensorFlow, openTSNE, scikit-learn, DeepLabCut, or NumPy
without re-validating nn-stage outputs against your reference set —
small version bumps in any of these can shift hcoord/hlabel by a few
percent in some videos.

deep_learning_windows_original.yml
----------------------------------
Verbatim export of the original Windows env from the lab machine that
produced the canonical reference data. Kept here for reference and
for bit-exact reproducibility on Windows. New users should use
``deep_learning.yml`` instead — it's identical except for the
platform-specific Conda packages.
