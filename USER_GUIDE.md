# marm_behavior — User Guide

This guide explains every command-line flag in detail, when to use it,
and how it interacts with the rest of the pipeline. For an
installation walkthrough and the input video spec (overhead Stereolabs
ZED mini, 720p side-by-side stereo at 60 fps) see [README.md](README.md).

The CLI is invoked as either:

```bash
python -m marm_behavior [options] [video]
marm-behavior          [options] [video]   # if installed via pip
```

Both forms accept the same arguments.

---

## Quick reference

| Flag | Purpose | Default |
|---|---|---|
| `video` (positional) | What to process: a file, a directory, or nothing (CWD) | current working directory |
| `--single-csv` | Override the auto-discovered DLC single CSV | auto-discover |
| `--multi-csv` | Override the auto-discovered DLC multi CSV | auto-discover |
| `-o`, `--output-dir` | Where to write artifacts | next to each video |
| `--ground-normalized` | Override the bundled ground-normalized reference | bundled |
| `--dlc-config` | Use a custom DLC project instead of the bundled one | bundled model |
| `--nn-reference-dir` | Override folder with NN reference files | bundled |
| `--nn-bootstrap` | Generate missing NN references from this video | off |
| `--force-nn` | Run the NN stage even in one-animal mode | off |
| `--prefetch-data` | Download bundled HF data files up-front and exit | off |
| `--red-buddy` | Override the buddy chain for Red in the NN stage | Yellow → Blue |
| `--white-buddy` | Override the buddy chain for White | Blue → Red |
| `--blue-buddy` | Override the buddy chain for Blue | White → Red |
| `--yellow-buddy` | Override the buddy chain for Yellow | Red → Blue |
| `--no-red` | Mark Red as not present in the video | Red present |
| `--no-white` | Mark White as not present | White present |
| `--no-blue` | Mark Blue as not present | Blue present |
| `--no-yellow` | Mark Yellow as not present | Yellow present |
| `--c-thresh` | DLC confidence threshold for extract stage | 0.1 |
| `--target-coverage` | Alpha-shape hull coverage target | 1.0 |
| `--stages` | Which pipeline stages to run | all six |
| `-q`, `--quiet` | Suppress progress messages | verbose |
| `-h`, `--help` | Print the help banner and exit | — |

---

## Concept refresher

The pipeline runs six stages in sequence. Each stage reads the
output of the previous one(s) and writes its own artifacts:

| # | Stage | Reads | Writes |
|---|---|---|---|
| 1 | `dlc` | video.avi | `<stem>DLC_..._el.picklesingle.csv`, `<stem>DLC_..._el.picklemulti.csv` |
| 2 | `extract` | the two DLC CSVs | `tracks_<stem>.mat` |
| 3 | `process` | `tracks_<stem>.mat` | `edges_<stem>.mat` |
| 4 | `depths` | `edges_<stem>.mat` + video | `depths_<stem>.mat` |
| 5 | `labels` | `edges_<stem>.mat` + `depths_<stem>.mat` | `{w,b,r,y}_description_<stem>.csv` |
| 6 | `nn` | description CSVs | `hcoord_<Color>_<stem>.csv`, `hlabel_<Color>_<stem>.csv` |

Stages that have already produced their outputs are auto-skipped via
`--stages` (see below), so you can resume an interrupted run or
re-run just one stage in isolation.

---

## What to process

### `video` (positional, optional)

Tells the pipeline which video(s) to process. Three forms:

**1. Omitted** — process every `.avi` in the current working directory:

```bash
cd /path/to/videos
python -m marm_behavior
```

This is the most common form for routine analysis. It globs both
`*.avi` and `*.AVI` (case-insensitive on the extension) and processes
them one by one in **batch mode** (see below).

**2. A single file** — process just that file:

```bash
python -m marm_behavior /data/exp1/test_4.avi
```

Outputs land next to the video unless `--output-dir` is given.

**3. A directory** — process every `.avi` inside that directory:

```bash
python -m marm_behavior /data/exp1/
```

Same as form 1, but you can run it from anywhere. Useful for
scripted batches.

#### Batch mode

When more than one video is found (forms 1 and 3 with multiple
matches), the CLI enters batch mode:

- Prints the full list of videos up front
- Processes each video with a `[i/N] video.avi` banner
- If a video fails, logs the failure and continues with the next
  one — a single bad video doesn't kill the whole run
- Prints a final `batch done: N/M succeeded` summary listing any
  failures by name and reason
- Returns exit code `0` if everything succeeded, `1` otherwise
- `--single-csv` and `--multi-csv` are warned about and ignored
  in batch mode (they'd point at a single video's files)

### `-o PATH`, `--output-dir PATH`

Directory where the pipeline writes its artifacts. Default: each
video's parent folder.

```bash
python -m marm_behavior /data/exp1/test_4.avi -o /tmp/results
```

The output directory is created if it doesn't exist. If you process
multiple videos to the same output directory, their artifacts are
distinguished by filename (`tracks_test_4.mat`, `tracks_test_5.mat`,
etc.) so they don't collide.

---

## Stage selection

### `--stages STAGE [STAGE ...]`

Pick a subset of the six stages to run. By default all six run in
sequence. The valid stage names are
`dlc extract process depths labels nn`.

Common patterns:

```bash
# Skip DLC because the *single.csv / *multi.csv already exist:
python -m marm_behavior video.avi --stages extract process depths labels nn

# Re-run only the labels stage (e.g., after editing labels.py):
python -m marm_behavior video.avi --stages labels

# Re-run the labels and nn stages without re-doing depth lookup:
python -m marm_behavior video.avi --stages labels nn

# Stop after the descriptions are written:
python -m marm_behavior video.avi --stages dlc extract process depths labels
```

A stage that depends on a missing artifact will fail with a clear
error message telling you which prior stage to run. For example, if
you run `--stages labels` but `edges_video.mat` doesn't exist,
you'll see:

```
error: video.avi: edges file not found: /path/to/edges_video.mat;
run stage 'process' first or provide it manually
```

### `-q`, `--quiet`

Suppress the per-stage `[marm_behavior]` progress lines. The pipeline
still prints errors to stderr if anything fails, and exit codes still
reflect success or failure. Useful for batch runs where you want a
clean log.

---

## Marking which animals are present

### `--no-red`, `--no-white`, `--no-blue`, `--no-yellow`

Tell the pipeline that a given colour is *not* in the video. By
default all four are assumed present. Each `--no-<colour>` flag
disables one.

```bash
# Yellow isn't in this video:
python -m marm_behavior video.avi --no-yellow

# Only Red is in this video:
python -m marm_behavior video.avi --no-white --no-blue --no-yellow
```

Marking a colour absent has three effects:

1. The **labels stage** skips that colour (no `<colour>_description_<stem>.csv` is written).
2. The **depths stage** skips that colour's per-frame depth lookup, which is the slowest part of the depths stage. A four-animal depths run takes ~4× longer than a single-animal one.
3. The **nn stage** skips that colour entirely.

If exactly one of the four flags leaves only one colour present,
the pipeline automatically switches into **one-animal mode** —
see the dedicated section in [README.md](README.md). Briefly:

- The **extract stage** assigns every multi-CSV tracklet to the
  focal animal instead of proximity-matching to colour-classified
  heads (which is the wrong behaviour when DLC's head classifier
  was trained on four-animal data and may mislabel colours).
- The **process stage** uses a constant body-length `bh = 30` for
  the three absent animals.
- The **depths stage** runs only the focal animal's depth lookup.
- The **nn stage** is skipped (no buddy animal available); pass
  `--force-nn` to override.

The mode is detected automatically and announced with a banner at
the start of the run.

---

## DLC stage flags

### `--dlc-config PATH`

Use a custom DLC `config.yaml` instead of the bundled trained model.
Default: the package ships a complete DLC project (ResNet-50, 21
multi-animal + 5 unique body parts, snapshot 15000) at
`marm_behavior/data/dlc_model/`, so this flag is rarely needed.

When you do pass it, point at a real DLC project's `config.yaml`:

```bash
python -m marm_behavior video.avi \
    --dlc-config /path/to/my-dlc-project/config.yaml
```

The pipeline still rewrites the `project_path` field in a temporary
copy of your config so the DLC runtime can find its snapshots, so
you don't need to worry about the absolute paths baked into the
config at project-creation time.

This flag is only relevant if the `dlc` stage actually runs. If
you're skipping `dlc` via `--stages` (because you already have the
two CSVs), it's a no-op.

---

## Extract stage flags

### `--single-csv PATH`

Path to the DLC `*single.csv` file (the unique-body-parts CSV, one
row per frame, 20 columns). Default: auto-discovered next to the
video via the glob `<video_stem>*single.csv`.

DLC + the el-to-csv converter writes this file with a name like
`test_4DLC_resnet50_marm_4Sep5shuffle1_15000_el.picklesingle.csv`.
The auto-discovery glob handles whatever middle text DLC inserts.

You only need this flag if the file lives somewhere unusual or has
a non-standard name. Ignored in batch mode.

### `--multi-csv PATH`

Path to the DLC `*multi.csv` file (the multi-animal tracklet CSV,
one row per `(track_id, frame)`, 86 columns). Same auto-discovery
logic and same caveats as `--single-csv`. Ignored in batch mode.

### `--c-thresh FLOAT`

DLC confidence threshold for the extract stage. Default: `0.1`.
Body parts with a confidence below this threshold are NaN'd out of
both the single and multi CSVs before the rest of the extract
pipeline runs.

```bash
# Be more aggressive about dropping low-confidence detections:
python -m marm_behavior video.avi --c-thresh 0.3
```

Raising the threshold:
- **Pros**: cleaner trajectories, fewer spurious body parts in
  occluded frames
- **Cons**: more NaN body parts in the resulting `tracks_*.mat`,
  which propagates to more gaps that the downstream
  fillmissing/interpolation has to bridge

The default of 0.1 is conservative — it accepts almost everything
and lets the geometric filtering in `extract_3` (the alpha-shape
hull) and the NaN-fill logic in `process_3` clean up the rest.
Don't raise it without testing the downstream output, especially if
your animals spend time partially occluded.

### `--target-coverage FLOAT`

Alpha-shape hull coverage target for the extract stage's body-part
filter. Default: `1.0`. Range: 0.0 to 1.0.

The extract stage drops body parts whose normalized positions fall
outside an alpha-shape hull computed from the training data
(`ground_normalized.npz`). The hull's tightness is controlled by
this parameter:

- `1.0` (default) — the hull encloses 100% of training points.
  Maximally permissive; rejects only points that are clearly
  outside the cloud of training data.
- `0.99` — encloses 99% of training points. Slightly tighter;
  rejects ~1% of training points as outliers.
- Lower values — progressively tighter.

In practice the default works for any reasonable input. Lower it if
you're seeing too many spurious body parts survive the filter; raise
it (or leave it at 1.0) if you're losing legitimate detections.

---

## Process stage flags

### `--ground-normalized PATH`

Override the bundled `ground_normalized.npz` reference. Default:
the package ships a copy at
`marm_behavior/data/ground_normalized.npz` (~570 KB), so you don't
need this flag for normal use.

This file contains the training-set body-part position clouds used
by the alpha-shape hull filter in `extract_3`. If your DLC model
was trained on a different anatomy (different body-part labels,
different number of parts), you'd need to regenerate this file and
point at it with this flag. Both `.mat` and `.npz` formats are
accepted.

---

## NN stage flags

The NN stage projects each animal's behavioural features into a
2D t-SNE space and assigns cluster IDs. It's the most complex stage
and has the most knobs.

### `--nn-reference-dir PATH`

Override the folder containing the reference files used to project
this video's features into the canonical cluster space. Default: the
bundled `data/nn_reference/` folder inside the installed package,
which ships pre-populated with the canonical reference set — you do
not normally need this flag.

The bundled folder contains four small CSVs
(`out_inner_mean1.csv`, `out_inner_std1.csv`, `tsne_temp1_1.csv`,
`dbscan_temp1_1.csv`) plus the native t-SNE cache
(`embedding_train_coords.npy`, `embedding_train_annoy.bin`,
`embedding_train_meta.json`, `embedding_train_optimizer_gains.npy`).

**Using a custom reference folder.** The folder needs the four small
CSVs at minimum. If the `embedding_train_*` cache files are missing,
the stage refits openTSNE from `out_inner1.csv` (which must then be
present; ~6 min) and writes the cache for future runs.

**Cache invalidation.** If the cache fails to load (e.g. after an
openTSNE version upgrade), the stage either refits from
`out_inner1.csv` if you still have it, or raises a clear error
asking you to put it back or delete the broken cache files.

```bash
python -m marm_behavior video.avi \
    --nn-reference-dir /shared/lab/my_reference_set
```

### `--nn-bootstrap`

Generate the five reference files from the *current video's* own
description CSVs instead of failing when they're missing. Off by
default.

This is intended for the very first time you set up the pipeline
and don't yet have canonical reference files. It runs the encoder
on each present animal's features, fits openTSNE on the result,
and writes the five files into the reference folder so subsequent
runs use them.

```bash
python -m marm_behavior video.avi --nn-bootstrap
```

**Critical caveat**: bootstrap cluster IDs are *not* comparable
across videos or to canonical references. DBSCAN assigns cluster
numbers in discovery order inside a single-video sample, so a
cluster labelled "7" in one bootstrap run bears no relation to a
cluster labelled "7" in another bootstrap run. Use this mode only
for initial exploration; for cross-video comparison you need
canonical reference files.

### `--force-nn`

Run the NN stage even when one-animal mode would normally skip it.
Off by default.

In one-animal mode (exactly one `*_present` flag is true), the NN
stage is skipped automatically because the encoder pairs each self
animal with a buddy animal's behavioural features, and there's no
buddy when only one animal is in the video.

`--force-nn` overrides that skip and activates a **zero-buddy
fallback**: the focal animal's encoder input is constructed by
concatenating its own 30-feature description with a 30-feature
zero matrix, producing the 60-feature input the encoder expects.
This lets the existing model run end-to-end on a one-animal video
without retraining.

```bash
python -m marm_behavior video.avi --no-white --no-blue --no-yellow --force-nn
```

When the fallback fires, the stage prints a loud multi-line
warning with `!!!!!` separators:

```
[marm_behavior.nn]   Red: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
[marm_behavior.nn]   Red: ZERO-BUDDY FALLBACK
[marm_behavior.nn]   Red: no buddy description CSV available (tried ['y', 'b'])
[marm_behavior.nn]   Red: fabricating a zero matrix as the buddy half of the encoder input
[marm_behavior.nn]   Red: WARNING — the encoder was trained on real two-animal inputs; feeding it zeros for the buddy
[marm_behavior.nn]   Red: half is OUT OF DISTRIBUTION. The resulting hcoord and hlabel outputs are NOT comparable
[marm_behavior.nn]   Red: to canonical references or to four-animal-mode outputs from the same encoder. Use
[marm_behavior.nn]   Red: only for exploratory analysis of solo behaviour.
[marm_behavior.nn]   Red: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

**Critical caveat**: the bundled NN encoder was trained on real
two-animal inputs (60-feature pairs). Feeding it zeros for the
buddy half produces out-of-distribution latents — the resulting
points may land in unusual parts of the t-SNE space and get
assigned to nonsensical clusters. The output files are still
written (`hcoord_<Color>_<video>.csv` and
`hlabel_<Color>_<video>.csv`), but the cluster IDs are *not*
comparable to:

- canonical four-animal references
- other one-animal videos run with `--force-nn`
- the same animal's own four-animal-mode runs from a different video

Use `--force-nn` only for exploratory analysis of solo behaviour
where you understand the cluster IDs are essentially per-video.
For cross-video comparison of solo behaviour, the right answer is
a dedicated one-animal encoder, which would need new training data
and is not currently part of this pipeline.

`--force-nn` has no effect in four-animal mode.

### `--red-buddy COLOR [COLOR ...]`, `--white-buddy`, `--blue-buddy`, `--yellow-buddy`

Override which other animal's description CSV is concatenated with
each self animal's during NN encoding. The encoder takes 60-column
input (30 features for the self animal + 30 features for the
buddy), and the buddy choice affects which behaviours get
encoded as "joint" vs "solo".

Each flag takes one or more short colour keys (`r`, `w`, `b`, `y`)
in preference order. The NN stage walks the chain and uses the
first buddy whose description CSV is on disk.

**Default chains:**

| Self | Default chain |
|---|---|
| Red | Yellow → Blue |
| White | Blue → Red |
| Blue | White → Red |
| Yellow | Red → Blue |

**Override examples:**

```bash
# Always pair Red with Blue, never Yellow:
python -m marm_behavior video.avi --red-buddy b

# Pair Yellow with White first, fall back to Blue:
python -m marm_behavior video.avi --yellow-buddy w b

# Multiple overrides at once:
python -m marm_behavior video.avi \
    --red-buddy b \
    --blue-buddy y r \
    --yellow-buddy w
```

Validation rules:
- Invalid colours rejected by argparse: `--red-buddy x` → error
- Self-as-buddy silently dropped: `--red-buddy r b` becomes `[b]`
- Duplicates removed: `--red-buddy b b y` becomes `[b, y]`

Partial overrides are fine — flags you don't pass keep their
defaults. The NN stage logs every non-default override at the
start of the run:

```
[marm_behavior.nn] buddy override: Red -> ['Blue']
[marm_behavior.nn] buddy override: Yellow -> ['White', 'Blue']
```

And logs which buddy was actually used per animal:

```
[marm_behavior.nn]   Red: self=r_description_video.csv buddy=b_description_video.csv (Blue) → features (10799, 60)
```

---

## Worked examples

### Routine analysis of one new video

```bash
cd /data/today
python -m marm_behavior test_4.avi
```

Runs all six stages with default parameters. Outputs land in
`/data/today/`. The NN stage uses the canonical reference data
bundled inside the package, so it works end-to-end out of the box.

### Batch-process every video in a folder

```bash
cd /data/exp7
python -m marm_behavior
```

Globs `*.avi` and `*.AVI`, processes each one, prints a summary
at the end. One bad video doesn't kill the rest.

### Re-run only the NN stage on a video that already has descriptions

```bash
python -m marm_behavior /data/exp7/test_4.avi --stages nn
```

Useful when you've swapped in a custom reference set with
`--nn-reference-dir` — no need to re-do DLC inference or feature
extraction.

### Process a one-animal Red-only video

```bash
python -m marm_behavior /data/red_only/r_2024_01_15.avi \
    --no-white --no-blue --no-yellow
```

Auto-detects one-animal mode, takes the optimised paths through
extract, process, and depths, and skips the NN stage. Should run
roughly 4× faster than four-animal mode for the depths stage and
significantly faster for extract.

### Process a video with a custom DLC model and a non-default reference folder

```bash
python -m marm_behavior video.avi \
    --dlc-config /home/me/my-dlc/config.yaml \
    --nn-reference-dir /shared/lab/nn_v3
```

### Re-run the labels stage with a tighter confidence threshold

```bash
python -m marm_behavior video.avi --c-thresh 0.3 \
    --stages extract process depths labels
```

The `dlc` stage is skipped (the CSVs already exist), `extract`
re-runs with the new threshold, and so do all the downstream
stages that depend on it. The `nn` stage is skipped here because
re-running it without changing the reference files would just
reproduce the existing output.

### Bootstrap a brand-new lab installation

You have a fresh install, no reference files yet, and want to see
end-to-end output on one video so you can sanity-check the
pipeline:

```bash
python -m marm_behavior /data/test/test_4.avi --nn-bootstrap
```

This generates synthetic reference files from this video's own
description CSVs and runs the NN stage against them. The cluster
IDs aren't meaningful for cross-video comparison, but you'll get
output to verify nothing is broken.

---

## Troubleshooting

### "no .avi files found in <CWD>"

You ran `python -m marm_behavior` with no arguments and no `.avi`
files exist in the current working directory. Either `cd` to a
folder with videos, or pass an explicit path.

### "single CSV not found" / "multi CSV not found"

The extract stage couldn't auto-discover the DLC CSVs next to your
video. Either:
- Run the `dlc` stage to generate them: drop `--stages` or pass
  `--stages dlc extract process depths labels nn`
- Pass the paths explicitly with `--single-csv` / `--multi-csv`
- Check the filenames — the auto-discovery glob is
  `<video_stem>*single.csv`, so the file must start with the
  video's name (without extension)

### "tracks file not found" / "edges file not found" / "depths file not found"

You requested a downstream stage with `--stages` but the upstream
stage's output isn't on disk. The error message names which prior
stage to run. Either run that stage first, or include it in your
`--stages` list.

### "NN reference files missing"

Only happens if you passed `--nn-reference-dir` to point at a custom
folder that doesn't have all the required files. The default bundled
reference folder ships pre-populated, so this error never fires for
out-of-the-box use. Two cases when overriding:

1. **No native cache present** — the folder needs all four small
   CSVs (mean, std, tsne_temp, dbscan_temp) plus `out_inner1.csv`
   so the stage can fit openTSNE and write the cache. Or pass
   `--nn-bootstrap` to generate everything from the current video.

2. **Cache present** — only the four small CSVs are required.
   `out_inner1.csv` is optional once the `embedding_train_*` cache
   files exist. The error message lists which specific files are
   missing.

### "The 'nn' stage requires tensorflow"

The NN stage uses the LSTM encoder via `tensorflow.keras`, which
isn't installed in your environment. Install it with
`pip install 'tensorflow>=2.5'`, or skip the stage with
`--stages dlc extract process depths labels`.

### One-animal mode banner doesn't appear

The auto-detection requires exactly one `--no-<colour>` flag to be
absent. If you marked all four animals absent (`--no-red --no-white
--no-blue --no-yellow`), the count is zero, not one, and the
banner won't fire. Conversely, if you marked only two as absent,
you have two animals present and four-animal mode applies.

### NN stage skipped unexpectedly

Check the `[marm_behavior.nn]` log lines. The most common reasons
are:

1. **One-animal mode** — pass `--force-nn` to override (and
   probably also `--nn-bootstrap` to get meaningful output)
2. **Self animal has no description CSV** — the `labels` stage
   didn't produce a CSV for this colour, usually because it was
   marked absent
3. **No usable buddy** — every entry in the buddy chain points at
   an animal whose description CSV doesn't exist on disk; either
   adjust `--<colour>-buddy` or unmark one of the absent animals
