nn_reference/
=============

This folder ships pre-packaged with marm_behavior and contains the
canonical reference data the `nn` stage uses to project new videos into
the lab's standard t-SNE / DBSCAN cluster space. Users normally do not
need to touch this folder — it's loaded automatically.

Override location
-----------------
If you have your own reference set (e.g. for a different cohort), pass
`--nn-reference-dir PATH` (or `reference_dir=...` from Python) to point
elsewhere.

Files
-----
Always required (small, ship with the package):

    out_inner_mean1.csv   (256,)     mean for per-run normalization
    out_inner_std1.csv    (256,)     std for per-run normalization
    tsne_temp1_1.csv      (N, 2)     reference 2D t-SNE coordinates
    dbscan_temp1_1.csv    (N,)       reference cluster labels

Native t-SNE cache (regenerated automatically if missing, if the raw
training data is available):

    embedding_train_coords.npy             (n_train, 2)  training 2D coords
    embedding_train_annoy.bin                            cached annoy k-NN index
    embedding_train_meta.json                            format/version metadata
    embedding_train_optimizer_gains.npy    (n_train, 2)  optimizer state for
                                                         deterministic transform()

Required only to (re)build the cache:

    out_inner1.csv        (N, 256)   raw latent features from training
                                     data. ~3 GB; NOT shipped in the
                                     package. Only needed on the first
                                     run, or if the cache is invalidated
                                     (e.g. after an openTSNE upgrade).
                                     Once the four `embedding_train_*`
                                     files exist, this can be deleted.

Bootstrap mode
--------------
If you don't have any reference data at all and want to experiment, pass
`--nn-bootstrap` to generate everything from the current video's own
description CSVs. Bootstrap cluster IDs are *not* comparable across
videos or to canonical references, so this is opt-in and intended for
initial exploration only. In bootstrap mode the generated files are
written next to the video's outputs, never into this packaged folder.

Packaging note
--------------
Files in this folder ARE shipped as Python package data so that
`pip install marm_behavior` gets the user a working `nn` stage out of
the box. The 3 GB `out_inner1.csv` is intentionally NOT included; it
isn't needed once the `embedding_train_*` cache files are present.
