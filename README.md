# marm_behavior

A six-stage multi-animal marmoset behavioral analysis pipeline. Takes
overhead stereo video (720p side-by-side, RGB left + depth right, 60 fps)
of four differently-marked marmosets recorded with a [Stereolabs ZED
mini](https://www.stereolabs.com/store/products/zed-mini), runs
DeepLabCut pose estimation, extracts per-animal body-part tracks,
computes per-frame behavioral features, and projects the features into
a learned behavioral cluster space. See
[Input video requirements](#input-video-requirements) for the full
spec.

Everything is packaged so a single command runs the full pipeline:

```bash
python -m marm_behavior path/to/video.avi
```

For a detailed reference covering every command-line flag, see
[USER_GUIDE.md](USER_GUIDE.md).

## Install

The fastest way is to use the bundled conda environment file. It pins
the exact ML stack (TensorFlow 2.9.1, openTSNE 0.6.2, scikit-learn
1.1.2, DeepLabCut 2.2.0.6, NumPy 1.22.4) the canonical reference
outputs were produced with — important because the nn stage's t-SNE
output is sensitive to openTSNE's version. Full install takes ~10
minutes on a clean machine.

### 1. Create the conda environment

If you don't have conda, install
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) first.
Then from the repo root:

```bash
conda env create -f env/deep_learning.yml
conda activate deep_learning
```

This creates an env named `deep_learning` (matching the lab's
canonical name) with everything the pipeline needs.

### 2. (GPU users only) Install CUDA + cuDNN

For TensorFlow 2.9.1, CUDA 11.2 + cuDNN 8.1 is the supported
combination:

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1 -y
```

Verify the GPU is visible to TensorFlow:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

You should see at least one `PhysicalDevice(name='/physical_device:GPU:0', ...)`.
If the list is empty, the `dlc` stage will still run on CPU but will be
~50× slower.

### 3. Install the marm_behavior package

Either clone and run in-place:

```bash
git clone <repo-url> marm_behavior
cd marm_behavior
python -m marm_behavior path/to/video.avi
```

...or install it editable:

```bash
cd marm_behavior
pip install -e .
marm-behavior path/to/video.avi
```

On the first invocation, the bundled data (DLC model ~128 MB, NN
encoder ~2 MB, NN reference set ~3 GB) downloads from the Hugging
Face Hub and is cached under `~/.cache/huggingface/hub/`. Every
subsequent run reuses the cache.

### 4. (Optional) Override the NN reference data

The default reference set is fetched from
[`williammenegas/data`](https://huggingface.co/datasets/williammenegas/data)
on first run. You only need to override this if you want to use your
own reference set (e.g. for a different cohort) — in that case, pass
`--nn-reference-dir /path/to/your/reference/folder`. See the
[Reference files for the NN stage](#reference-files-for-the-nn-stage)
section below.

### Verify the install

```bash
python -m marm_behavior --help
```

Should print the full usage banner with all the per-colour and
per-stage flags.

## Input video requirements

The pipeline is built around overhead footage from a
[Stereolabs ZED mini](https://www.stereolabs.com/store/products/zed-mini)
camera looking straight down at the arena floor. Every input video is a
**side-by-side stereo composite**: each frame holds two horizontally
stacked 1280×720 images, so the full frame is 2560×720.

| | Left half | Right half |
|---|---|---|
| **Content** | RGB image of the arena (the camera's left eye) | Per-pixel depth map (depth-from-stereo, encoded as an image) |
| **Resolution** | 1280 × 720 | 1280 × 720 |
| **Used by** | `dlc`, `extract`, `process`, `labels` stages — DLC runs pose estimation on the RGB half and the body-part coordinates flow through the rest of the pipeline in this image space | `depths` stage — each tracked body-part `(x, y)` from the RGB half is looked up in the matching pixel of the depth half to attach a `z` value |

| Property | Required value |
|---|---|
| Camera | Stereolabs ZED mini |
| Mount | Overhead, looking straight down at the arena floor |
| Layout | Side-by-side stereo (RGB left, depth right) |
| Resolution per half | 1280 × 720 (HD720) |
| Full composite resolution | 2560 × 720 |
| Frame rate | 60 fps |
| Container | `.avi` |

These values are not just suggestions — the bundled DLC model was
trained on this exact geometry, the alpha-shape hull in
`ground_normalized.npz` was fit to this image space, and the `depths`
stage hard-codes the assumption that the depth map occupies the right
half of every frame. Footage at a different resolution, frame rate,
mounting angle, or stereo layout will run end-to-end without erroring
but will produce silently wrong tracks and depth lookups. Re-encode
or re-record before running the pipeline.

The ZED mini's stock recording tools (the ZED SDK's `ZED_Explorer` and
`ZED_SVO_Export` utilities, or the ZED-OBS plugin) can write the
side-by-side `.avi` directly. Set the camera to `HD720` mode at 60 fps
and choose the side-by-side / depth-map export option.

## The six stages

| Stage | Input | Output |
|---|---|---|
| **dlc** | video | `*DLC_..._el.picklesingle.csv`, `*DLC_..._el.picklemulti.csv` |
| **extract** | DLC CSVs | `tracks_<video>.mat` (per-animal body-part tracks) |
| **process** | `tracks_*.mat` | `edges_<video>.mat` (per-animal edge matrices) |
| **depths** | `edges_*.mat` + video | `depths_<video>.mat` (pixel-depth lookups) |
| **labels** | `edges_*.mat` + `depths_*.mat` | `{w,b,r,y}_description_<video>.csv` (30 behavioral features per frame) |
| **nn** | description CSVs | `hcoord_{Red,White,Blue,Yellow}_<video>.csv` (2D t-SNE coords) and `hlabel_*_<video>.csv` (cluster labels) |

## Common invocations

**Run everything with defaults:**
```bash
python -m marm_behavior path/to/video.avi
```

**Skip DLC if the CSVs already exist:**
```bash
python -m marm_behavior video.avi --stages extract process depths labels nn
```

**Re-run just the labels stage** against existing `edges_*.mat` / `depths_*.mat`:
```bash
python -m marm_behavior video.avi --stages labels
```

**Point the NN stage at a custom reference folder:**
```bash
python -m marm_behavior video.avi --nn-reference-dir /path/to/references
```

**Declare one colour absent:**
```bash
python -m marm_behavior video.avi --no-yellow
```

**Use your own DLC project** instead of the bundled one:
```bash
python -m marm_behavior video.avi --dlc-config /path/to/your/config.yaml
```

See `python -m marm_behavior --help` for every flag.

## Bundled data

The runtime data marm_behavior needs — the trained DeepLabCut project
(~128 MB), the LSTM encoder (~2 MB), the canonical NN reference set
(~190 MB), and a `ground_normalized.npz` body-part cloud (~570 KB) — is
**not shipped in the wheel**. It's hosted on the Hugging Face Hub at
[`williammenegas/data`](https://huggingface.co/datasets/williammenegas/data)
and downloaded lazily on first use, then cached under
`~/.cache/huggingface/hub/`. The first pipeline run on a fresh machine
incurs a one-time ~310 MB download; every run after that is instant.

To pre-warm the cache (e.g. before going offline):

```bash
python -m marm_behavior --prefetch-data
```

For shared cluster installs where every user should hit the same
on-disk copy, set `MARM_BEHAVIOR_DATA_DIR=/shared/path/to/data`. The
helper checks that path before falling back to the per-user HF cache.

For lab-internal mirrors of the data repo, set
`MARM_BEHAVIOR_HF_REPO=your-org/your-mirror` to override the source
repo.

The lookup logic lives in `marm_behavior/_data_files.py` if you want
to read it.

## Reference files for the NN stage

The NN stage projects video features into a stable behavioral cluster
space. The canonical reference data is **fetched from the Hugging Face
Hub** as part of the bundled-data download above, so the stage works
out of the box. You can ignore this section unless you want to use a
different reference set.

The bundled folder contains:

```
out_inner_mean1.csv                    (256,)        normalization mean
out_inner_std1.csv                     (256,)        normalization std
tsne_temp1_1.csv                       (N, 2)        reference 2D coords
dbscan_temp1_1.csv                     (N,)          reference cluster labels
embedding_train_coords.npy             (n_train, 2)  training 2D coords
embedding_train_annoy.bin              (~180 MB)     cached annoy k-NN index
embedding_train_meta.json                            cache metadata
embedding_train_optimizer_gains.npy    (n_train, 2)  optimizer state
```

The 3 GB `out_inner1.csv` (raw training latents) is **not** shipped —
it isn't needed at runtime because the cache files above already encode
everything `transform()` needs.

**Using your own reference set.** Pass `--nn-reference-dir /path/to/dir`.
The folder needs the four small CSVs at minimum. If the
`embedding_train_*` cache files are missing, the stage will fit
openTSNE from `out_inner1.csv` (which must be present in that case;
takes ~6 min) and write the cache for future runs.

**Bootstrap mode.** If you don't have any reference data at all and
want to experiment, pass `--nn-bootstrap` to generate everything from
the current video's own description CSVs. Bootstrap cluster IDs are
**not comparable** across videos or to canonical references, so this
mode is opt-in and intended for initial exploration only.

**Buddy chains.** The NN stage pairs each animal's behavioral features
with one other animal's features before encoding. The default pairings
are Red↔Yellow, White↔Blue, Blue↔White, and Yellow↔Red, with two
further fallbacks per animal so any animal can pair with any other if
the primary's description CSV isn't present.
Override per-animal with the `--<color>-buddy` flags — each takes one
or more short color keys (`r`, `w`, `b`, `y`) in preference order:

```bash
# Always pair Red with Blue instead of Yellow:
python -m marm_behavior video.avi --red-buddy b

# Pair Yellow with White first, falling back to Blue:
python -m marm_behavior video.avi --yellow-buddy w b

# Multiple overrides at once:
python -m marm_behavior video.avi --red-buddy b --blue-buddy y r
```

From Python, pass `nn_buddies={'r': ['b'], 'y': ['w', 'b']}` to
`marm_behavior.run()`.

## One-animal mode

When exactly one of the four animals is marked present (via three
`--no-<color>` flags), the pipeline automatically switches into
**one-animal mode**. Four stages change behaviour:

1. **extract stage** — every multi-CSV tracklet is assigned to the
   focal animal regardless of which colour DLC's head classifier
   predicted. This is the correct behaviour because DLC's head
   classifier was trained on four-animal data and routinely
   mislabels a single animal across colours; the four-animal
   proximity-based assignment would otherwise scatter the focal
   animal's tracklets across whichever absent colours happened to
   get mislabelled. For each frame, the highest-quality tracklet
   (most surviving body parts after the confidence threshold) is
   picked, with ties broken on the lower track id.
2. **process stage** — non-focal animals get a constant body-length
   `bh = 30` instead of the per-frame movmedian + clamp +
   forward-fill used in four-animal mode. The focal animal still
   gets the full computation. This avoids producing meaningless
   body-length estimates from F matrices that are all-NaN because
   the colour isn't actually in the video.
3. **depths stage** — only the focal animal's per-frame depth lookup
   runs. The other three colours' inner loops are skipped entirely,
   roughly 4× faster than four-animal mode for a typical video.
4. **nn stage** — skipped automatically. The NN encoder pairs each
   self animal with a buddy animal's behavioural features, and in
   one-animal mode there is no buddy. Pass `--force-nn` to override
   if you want to run the NN stage anyway (e.g. for bootstrapping
   a one-animal-specific reference space).

The mode is detected automatically — there's no separate flag to
enable it. The CLI prints a clear banner up front so you can see
what's about to happen:

```
$ python -m marm_behavior video.avi --no-white --no-blue --no-yellow
[marm_behavior] ONE-ANIMAL MODE: only Red present
[marm_behavior]   extract: all multi-CSV tracklets are assigned to the focal animal (no proximity matching to colour-classified heads)
[marm_behavior]   process: non-focal animals will use constant bh = 30
[marm_behavior]   depths:  per-frame lookup runs only for the focal animal
[marm_behavior]   nn:      stage will be skipped (no buddy animal available; pass force_nn=True / --force-nn to override)
[marm_behavior] dlc:     ...
```

When zero, two, three, or four animals are present, behaviour is
unchanged from the standard four-animal pipeline.



Everything is also available as a Python function:

```python
from marm_behavior import run

result = run("path/to/video.avi")

print(result["stages_run"])    # ['dlc', 'extract', 'process', 'depths', 'labels', 'nn']
print(result["descriptions"])  # {'w': Path(...), 'b': Path(...), ...}
print(result["nn"])            # {'Red': (hcoord_path, hlabel_path), ...}
```

See `help(marm_behavior.run)` for every parameter.

## Dependencies

**Runtime** (always required):
- numpy ≥ 1.22, scipy ≥ 1.9, h5py ≥ 3.6

**Per-stage** (install the ones you need):

| Stage | Needs |
|---|---|
| dlc | `deeplabcut[tf]` or `deeplabcut[pytorch]` |
| depths | `opencv-python-headless` or `imageio` + `imageio-ffmpeg` |
| nn | `tensorflow`, `openTSNE`, `scikit-learn` |

The `extract`, `process`, and `labels` stages need only the core runtime
deps.

## Layout

```
marm_behavior/              <- the Python package
├── run.py                  <- pipeline entry point
├── __main__.py             <- CLI
├── dlc_inference.py        <- DLC shell-out
├── el_to_csv.py            <- tracklet-pickle to CSV converter
├── nn_postprocess.py       <- NN stage
├── io/                     <- .mat and .csv I/O
├── numerics/               <- hull, rolling reductions, NaN helpers
├── extract/                <- body-part track extraction
├── process/                <- posture and edge computation
├── depths/                 <- per-frame depth lookup
├── features/               <- behavioral feature labelling
├── pipeline/               <- batch orchestrators
└── data/                   <- bundled models + canonical NN reference data
pyproject.toml
README.md
```
