Bundled data (lazy-downloaded from Hugging Face)
================================================

This folder is intentionally almost empty. The runtime data
marm_behavior needs — the trained DeepLabCut project (~128 MB), the
LSTM encoder (~2 MB), the canonical NN reference set (~190 MB), and
``ground_normalized.npz`` (~570 KB) — is hosted on the Hugging Face
Hub at:

    https://huggingface.co/datasets/williammenegas/data

The first time any pipeline stage needs one of these files, it
downloads it via :mod:`marm_behavior._data_files` and caches it
under ``~/.cache/huggingface/hub/``. Every subsequent run reuses the
cache, so you pay the download cost exactly once per machine.

Pre-fetching
------------
To download everything up front (e.g. before going offline, or when
priming a new install):

    python -m marm_behavior --prefetch-data

Or from Python:

    from marm_behavior._data_files import prefetch_all
    prefetch_all()

Custom mirror / shared cache
----------------------------
``MARM_BEHAVIOR_HF_REPO=user/mirror`` overrides the source repo
(useful if your lab maintains an internal mirror).

``MARM_BEHAVIOR_DATA_DIR=/shared/path`` makes downloads land in the
named directory (mirroring the layout below) instead of the per-user
HF cache. Useful for shared cluster installs where every user should
hit the same on-disk copy.

Files (listed for reference; downloaded from the Hub at runtime)
----------------------------------------------------------------

    ground_normalized.npz             body-part clouds for hull filter
    nn_model/
        pred_model_marm_encoderD.h5   LSTM encoder (Stage 5)
    nn_reference/
        out_inner_mean1.csv           normalization mean
        out_inner_std1.csv            normalization std
        tsne_temp1_1.csv              reference 2D t-SNE coords
        dbscan_temp1_1.csv            reference cluster labels
        embedding_train_coords.npy    training 2D coords
        embedding_train_annoy.bin     cached annoy k-NN index
        embedding_train_meta.json     cache metadata
        embedding_train_optimizer_gains.npy   optimizer state
    dlc_model/
        config.yaml
        dlc-models/iteration-0/marm_4Sep5-trainset99shuffle1/{train,test}/...

If you want a fully air-gapped install, manually download these into
this directory tree (or into ``$MARM_BEHAVIOR_DATA_DIR``); the helper
checks the local copy first and only hits the network on a miss.
