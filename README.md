# Modular vocal separation for pop and musical theatre

This project is now optimized for a very specific case that works well in practice:

- a normal mastered pop / musical-theatre song,
- instruments underneath the vocals,
- a backing choir or backing-vocal ensemble,
- **two foreground singers singing simultaneously**, often in harmony.

The pipeline is deliberately hierarchical:

```text
320 kb/s MP3 / normal music file
        |
        v
[Stage 1: Mel-Band RoFormer]
        |
        `---- ALL VOCALS
                 |
                 v
[Stage 2: MVSep Mega 53 / BS-RoFormer]
                 |
          +------+----------------+
          |                       |
          v                       v
   lead-vocal mixture          back-vocal
   (two principals)          (choir/backing)
          |
          v
[Stage 3: UNMIXX]
          |
     +----+----+
     |         |
     v         v
 singer_01   singer_02
```

This matches the observed musical-theatre case much better than trying to use one model to produce every singer directly.

## Why UNMIXX is the new Stage 3

UNMIXX is specifically a **multiple singing voices separation** model. The published checkpoint/config is a two-source model (`num_sources: 2`, `n_src: 2`) at **24 kHz**, and the training configuration includes same-song, same-singer and unison mixtures. That is much closer to two actors singing harmonies than a generic speech separator.

The official repository contains:

```text
inference.py
ckpt/conf.yml
ckpt/best.ckpt
```

The project uses the official UNMIXX model/checkpoint but runs it through the bundled `unmixx_chunked_inference.py` helper. The upstream `inference.py` forwards the entire WAV in one call, which can exhaust GPU memory on full songs. The helper loads the model once, runs short overlapping windows, skips windows with a peak at or below -80 dBFS, aligns the two output permutations between neighbouring chunks, and overlap-adds them into full-length stems. Set `--unmixx-silence-threshold-db -inf` (or `UNMIXX_SILENCE_THRESHOLD_DB=-inf` with `make`) to skip only exact digital silence.

By default, Stage 3 also downloads Sony CSL's BYOL singer-identity TorchScript checkpoint from Hugging Face on its first run. It maintains a conservative, normalized identity prototype for each output track. When overlap audio is silent or its keep/swap result is ambiguous, the chunker compares each new separated stem to those two prototypes and assigns the higher-scoring permutation. Only confidently assigned, sufficiently voiced stems update a prototype. Disable this with `--unmixx-identity-model none` (or `UNMIXX_IDENTITY_MODEL=none` with `make`).

## Important interpretation of Stage 2

For this pipeline we assume Mega53 behaves like it did on the successful test song:

```text
lead-vocal -> the two foreground/principal singers
back-vocal -> choir / ensemble backing vocals
```

That is a production-role classification, not a guarantee of physical singer identity. On another mix, an ad-lib or one foreground singer may occasionally leak into `back-vocal`, or a loud backing singer may leak into `lead-vocal`.

Always keep the intermediate stems so you can diagnose which stage caused an error.

# Installation

A CUDA-capable NVIDIA GPU is strongly recommended. Mega53 is particularly memory-intensive; its upstream release recommends around 16 GB VRAM or more. Every pipeline accepts `--device` (or Make's `DEVICE` / `CHOIR_DEVICE`) as `cuda`, `mps`, `cpu`, or `auto`; `auto` selects CUDA first, then MPS (Apple Metal), then CPU.

## 1. ffmpeg

Install `ffmpeg` and make sure it is on `PATH`. The internal WAV I/O uses `soundfile`, so it does not depend on TorchCodec finding FFmpeg's shared libraries.

## 2. Prepare the environment

The Makefile prefers the local `uv`-managed project environment, then falls
back to `pip` when `uv` is unavailable. It updates both upstream repositories
to their current default branches, removes UNMIXX's non-portable requirements
entry, and installs the required runtime dependencies:

```bash
make prepare
```

The default repository locations are `third_party/unmixx` and
`third_party/Music-Source-Separation-Training`.

The project's `requirements.txt` is deliberately small: it lists only the core
full-separation runtime. `requirements-choir-parts.txt`,
`requirements-polyphonic-choir-midi.txt`, and GAME's upstream requirements are
installed only by their respective optional pipelines. UNMIXX and
Music-Source-Separation-Training publish frozen development/workstation exports
that include incompatible CUDA builds, notebooks, GUI tools, and training-only
packages; `make prepare` skips those exports by default. Use
`UNMIXX_REQUIREMENTS_MODE=full` or `MSS_REQUIREMENTS_MODE=full` only when you
intentionally want to reproduce an upstream development environment. The local
`patch_unmixx_requirements.py` removes UNMIXX's editable path to the original
author’s `tssep` checkout and replaces its Asteroid-only padding helper with an
equivalent local implementation. If upstream removes or renames either target,
`make prepare` stops so the change can be reviewed.

### Colab

Run the repository from `/content`:

```bash
make prepare
```

The Makefile detects Colab from its `COLAB_RELEASE_TAG` environment marker.
When it is present, the default package manager is `pip` and the default
interpreter is `python`, so installs and pipeline subprocesses use Colab's
existing Python environment. Outside Colab, the defaults remain `auto` for
`PACKAGE_MANAGER` (prefer `uv`, otherwise `pip`) and `python3` for `PYTHON`.
Explicit `PACKAGE_MANAGER=...` and `PYTHON=...` values always take precedence.

Colab's preinstalled CUDA-enabled PyTorch is retained. At the end of `make
prepare`, the Makefile reads its CUDA version and force-installs TorchAudio
2.11.0 from the corresponding PyTorch CUDA index. TorchAudio 2.11 uses
PyTorch's stable ABI and supports PyTorch 2.11 and later, including Colab's
newer PyTorch builds. This prevents a PyPI TorchAudio wheel compiled for a
different CUDA version from being mixed with Colab's PyTorch. The first `make
run-*` after this change repairs an already mismatched runtime automatically.

The UNMIXX patch also skips an eager import of its training-only PyTorch
Lightning progress-bar utilities. They are not used by inference but otherwise
pull TorchMetrics and TorchVision into the model import path, where mismatched
TorchVision CUDA builds can fail before separation begins.

The supplied notebook uses these defaults and points UNMIXX at
`/content/third_party/unmixx`. A Colab GPU runtime is required for practical
performance. Every `make run-*` target runs `prepare` itself, so no additional
environment-variable arguments are needed in Colab.

## 3. Music-Source-Separation-Training

The script automatically downloads the public Mega53 assets from `oulianov/mvsep_mega_53`:

```text
mvsep_mega_model_bs_roformer_53_stems.yaml
mvsep_mega_model_bs_roformer_53_stems_v1.ckpt
```

The checkpoint is large (about 1.37 GB).

## 4. UNMIXX

If `import look2hear.models` reports another missing package, install that package specifically rather than recreating the author's entire development environment.

Verify these files exist:

```text
third_party/unmixx/inference.py
third_party/unmixx/ckpt/conf.yml
third_party/unmixx/ckpt/best.ckpt
```

# Run

The pipeline now has three modes. Stage 1 (instrument/vocal isolation) can also be bypassed independently with `--input-is-vocals`.

## A. Full hierarchy: duet + backing choir

Use this when the song has two foreground singers plus a backing choir/ensemble:

```bash
python pipeline.py song.mp3 \
  --mode full \
  --mss-repo third_party/Music-Source-Separation-Training \
  --unmixx-repo third_party/unmixx \
  --device auto \
  --output-dir run_song
```

Flow: `mix -> all vocals -> foreground/choir -> singer 1/singer 2`.

## B. Duet only: no backing choir

Use this when the vocal stem is essentially just two foreground singers. Mega53 is skipped completely:

```bash
make run-duet INPUT=duet_song.mp3 OUTPUT_DIR=run_duet DEVICE=auto
```

Flow: `mix -> all vocals -> singer 1/singer 2`. No `--mss-repo` is needed.

## C. Lead singer + backing choir: no duet

Use this when there is one principal singer plus backing/ensemble vocals. UNMIXX is skipped completely:

```bash
python pipeline.py lead_and_choir.mp3 \
  --mode lead-choir \
  --mss-repo third_party/Music-Source-Separation-Training \
  --device auto \
  --output-dir run_lead_choir
```

Flow: `mix -> all vocals -> lead vocal + choir/backing`. No `--unmixx-repo` is needed.

## D. Start from an already isolated vocal stem

Add `--input-is-vocals` to any mode to skip Mel-Band RoFormer. The input is decoded to a standard WAV and used directly as the all-vocals stem. For example:

```bash
python pipeline.py already_isolated_vocals.wav \
  --mode duet \
  --input-is-vocals \
  --unmixx-repo third_party/unmixx \
  --device auto \
  --output-dir run_existing_vocals
```

## Three simultaneous lead singers: Colab experiments

`three_singer_experiments.sh` runs two diagnostic experiments for a song
with three lead singers and no backing choir. It is intentionally not a
three-singer production pipeline: the available UNMIXX checkpoint has two
outputs, so the purpose is to test whether its two outputs form a stable
one-singer plus two-singer grouping.

In a CUDA Colab runtime, clone or upload this repository, place the song in
the runtime, then run:

```bash
make prepare
bash three_singer_experiments.sh \
  --input "songs/hadestown/Anaïs Mitchell - Hadestown - Original Broadway Cast Recording - 15 - When the Chips Are Down.flac" \
  --output-dir experiments/when_the_chips_are_down \
  --device cuda
```

The script runs Mel-Band RoFormer exactly once, then reuses
`mega53_split/final/00_all_vocals.wav` for every later experiment:

```text
Experiment 1: all vocals -> Mega53 lead-vocal / back-vocal
              back-vocal -> UNMIXX

Experiment 2: all vocals -> UNMIXX
```

For Experiment 1, audition `mega53_split/final/02_lead_vocal.wav`: it must
contain one singer only. Then confirm that the two outputs under
`mega53_backing_unmixx/final/` each contain one of the other two singers.

For Experiment 2, audition both direct UNMIXX outputs across the full song.
If one is consistently a two-singer remainder, run the optional recursive
test explicitly, substituting the confirmed output name:

```bash
bash three_singer_experiments.sh \
  --output-dir experiments/when_the_chips_are_down \
  --device cuda \
  --recursive-only \
  --recursive-remainder singer_01
```

The script never guesses that choice: UNMIXX output order is arbitrary and a
false recursive choice would make the result look plausible while splitting
the wrong mixture. Each UNMIXX run writes its chunk-level alignment artifacts
under `review/` for diagnosing identity swaps.

# Final outputs

Outputs depend on the selected mode:

```text
full:
  00_all_vocals.wav
  01_choir_backing.wav
  02_duet_mix.wav
  03_singer_01.wav
  04_singer_02.wav

duet:
  00_all_vocals.wav
  02_duet_mix.wav
  03_singer_01.wav
  04_singer_02.wav

lead-choir:
  00_all_vocals.wav
  01_choir_backing.wav
  02_lead_vocal.wav
```

`00_all_vocals.wav` is useful for checking Stage 1.

`01_choir_backing.wav` is Mega53's `back-vocal` output.

`02_duet_mix.wav` is Mega53's `lead-vocal` output before UNMIXX. Keep this file: it is the most important diagnostic input if Stage 3 fails.

`03_singer_01.wav` and `04_singer_02.wav` are UNMIXX's two source estimates.

## Sample rates

Stages 1 and 2 stay at 44.1 kHz stereo where possible. Before UNMIXX, `02_duet_mix.wav` is converted to a **24 kHz mono** working file because that matches the public UNMIXX checkpoint configuration.

Therefore the two final singer files are expected to be 24 kHz mono, while the choir and duet diagnostic stems retain the upstream resolution.

Do not resample the UNMIXX outputs back to 44.1 kHz just to make the number larger; that does not restore lost bandwidth. Resample only if a downstream DAW/workflow requires a common project rate.

# Identity and output ordering

UNMIXX solves blind two-source separation. It does not know character names or singer identities.

This means:

```text
singer_01 != guaranteed Alice
singer_02 != guaranteed Bob
```

The ordering can be permutation-ambiguous. If you process multiple songs or isolated sections, output 1 may correspond to a different person in another run.

If persistent identity matters, add a later singer-identification / embedding step using clean reference clips for each performer.

# What to do if the duet separation is imperfect

## 1. First listen to `02_duet_mix.wav`

If the choir is already mostly gone and both singers are clearly present, Stage 3 has a good input and UNMIXX is the model to tune/replace.

If one singer is missing in `02_duet_mix.wav`, UNMIXX cannot recover that singer. The problem is Stage 2, not Stage 3.

## 2. Check `01_choir_backing.wav`

If part of a principal singer appears there, Mega53 assigned that phrase as backing vocal. Possible future improvement: recombine selected regions or use a dedicated foreground-vs-ensemble vocal model.

## 3. Avoid processing between Stage 2 and Stage 3

Do not add aggressive:

- denoising,
- noise gates,
- dereverb,
- stereo widening,
- compression,
- clipping normalization.

Those operations can remove cues that help distinguish the two singers.

## 4. Try shorter sections manually

UNMIXX's published training config uses 4-second segments. The project now uses **4-second chunks with 1-second overlap by default**. This prevents full-song attention tensors from exhausting GPU memory and is closer to the model's training regime.

The chunker first compares the overlapping tails/heads of both estimated sources and chooses the source permutation that maximizes continuity. When that evidence is absent or ambiguous—most importantly after silence—it falls back to the BYOL singer-identity prototype tracker.

After the online pass, the default offline BYOL refinement evaluates every raw chunk against global identity prototypes. It uses a two-state sequence solver to invert only coherent, contiguous runs of online decisions: a persistent wrong assignment after silence or leakage can be corrected as one block, while an isolated weak embedding cannot cause a swap. Low-evidence chunks at the end of a strong run inherit that run until a confident contrary embedding is found. This post-pass uses temporary raw-chunk files (or reuses `UNMIXX_ALIGNMENT_REVIEW_DIR`), does not rerun UNMIXX, and deletes its temporary cache after writing the final stems.

It is enabled with `UNMIXX_IDENTITY_GLOBAL_REFINE=1` by default. Tune its per-chunk evidence gate and block-boundary penalty with `UNMIXX_IDENTITY_GLOBAL_MIN_MARGIN=0.02` and `UNMIXX_IDENTITY_GLOBAL_SWITCH_PENALTY=0.10`; set `UNMIXX_IDENTITY_GLOBAL_REFINE=0` to retain online-only behavior.

## Review and correct chunk alignment in a notebook

For a song where leakage or silence still produces a wrong identity switch, ask
Stage 3 to retain the raw estimates and its online decisions:

```bash
make run-duet INPUT=duet.wav OUTPUT_DIR=run_duet \
  UNMIXX_ALIGNMENT_REVIEW_DIR=run_duet/alignment_review
```

This adds two small float-WAV files per UNMIXX chunk under the review directory,
plus `alignment_manifest.json`. It is deliberately opt-in, because retaining
the raw chunks increases disk use. The normal final stems are still written as
usual.

Install the notebook control once with `uv sync --group alignment-review`
(or `pip install anywidget` in a notebook/Colab environment). The repository
is a collection of scripts, not an installed Python package, so in Colab add
the repository directory explicitly before importing the widget:

```python
from pathlib import Path
import sys

PROJECT_ROOT = Path("/content/mss_midi")
if not (PROJECT_ROOT / "alignment_review_widget.py").exists():
    raise FileNotFoundError(
        "alignment_review_widget.py is not in this Colab checkout; update or copy the project files first."
    )
sys.path.insert(0, str(PROJECT_ROOT))

# Required once per Colab runtime for the custom AnyWidget front end.
from google.colab import output
output.enable_custom_widget_manager()

from alignment_review_widget import open_alignment_review

open_alignment_review(PROJECT_ROOT / "run_duet/alignment_review")
```

The widget starts with one green box per chunk, representing the final
automatic assignment after online and offline alignment. The manifest also
retains the online decision and any offline BYOL correction for diagnosis.
Click a box to turn it red and invert only that automatic assignment. Select an identity and a time with the controls below the map to
audition 12 seconds, or play either selected or both complete re-rendered
stems. Every playback uses the complete current configuration without rerunning
UNMIXX. **Save corrected stems** writes `spk1_corrected.wav`,
`spk2_corrected.wav`, and the reusable `alignment_edits.json` under
`alignment_review/corrected_stems/`. **Finish alignment**, at the right of the
controls, also saves those files and replaces `final/03_singer_01.wav` and
`final/04_singer_02.wav` for that run. Opening the same review directory later
automatically restores that saved red/green map.

# Roadmap

## A. Tune chunked UNMIXX / permutation tracking

For long songs, a robust Stage 3 could use overlapping windows and decide whether each new pair should be assigned as:

```text
new A -> previous A
new B -> previous B
```

or swapped:

```text
new A -> previous B
new B -> previous A
```

using correlation, spectral similarity and/or singer embeddings in the overlap region.

This is now implemented. You can tune it with `--unmixx-chunk-seconds` and `--unmixx-overlap-seconds`. The defaults are `4.0` and `1.0` seconds. Longer chunks may improve context but raise VRAM sharply; shorter chunks reduce memory but can hurt separation and continuity.

## B. Mixture consistency

UNMIXX outputs do not necessarily sum exactly to the duet mixture. A post-processing projection can enforce approximately:

```text
singer_01 + singer_02 = duet_mix
```

The residual can be distributed equally or according to local source energy. This sometimes improves reconstruction but can also re-introduce leakage, so it should be benchmarked rather than enabled blindly.

## C. Singer-conditioned extraction

If you have clean reference clips for the two actors, conditioned extraction may eventually beat blind separation:

```text
duet + reference(A) -> A
duet + reference(B) -> B
```

This also solves the identity-labeling problem.

## D. Pitch-conditioned separation

Musical-theatre harmony parts often occupy different F0 trajectories. Multi-pitch estimation could be used as an additional conditioning signal, especially for thirds, sixths and contrary-motion harmonies.

It helps less when the singers are in unison or octaves.

## E. Stereo-aware duet separation

The current public UNMIXX checkpoint is integrated at 24 kHz mono. That throws away panning information from Stage 2. A future stereo duet separator could exploit the fact that commercial mixes often place the two principals differently in the stereo field or give them different reverbs.

## F. Zero-shot duet diffusion separator

The previously discussed zero-shot duet singing diffusion model is still an interesting alternate Stage 3, particularly if UNMIXX suffers from identity instability. Diffusion inference is typically much slower, so UNMIXX is the practical default.

## G. Dedicated theatre / pop-duet fine-tuning

The strongest long-term path is fine-tuning a two-singer separator on examples that resemble the actual target domain:

- Broadway / West End style belting,
- simultaneous lyrics,
- thirds/sixths/octaves,
- lead doubles,
- room/reverb tails,
- compressed/mastered stems,
- Stage-2 separation artifacts.

Crucially, training on **Mega53-produced duet stems** rather than pristine isolated vocals would teach Stage 3 the exact artifacts it will see in production.

## H. Three simultaneous lead singers

The published UNMIXX checkpoint is a **two-source** model. This project is
therefore also two-source by design: the chunked runner allocates two outputs,
tracks only a keep-or-swap permutation between chunks, and writes
`spk1.wav` and `spk2.wav`. Running the same checkpoint repeatedly is not a
reliable way to obtain a third singer; artifacts and leakage from the first
split are compounded by the second.

The most promising route for three foreground singers is a three-source
UNMIXX fine-tune. Keep Stages 1 and 2, then train Stage 3 on three-source
foreground-vocal mixtures that include close harmony, unison, same-singer
doubles, reverb, mastering, and—ideally—Mega53 artifacts. The model and
runner must be generalized together:

- change the UNMIXX source count from two to three and train a new checkpoint;
- use three-source permutation-invariant training (six possible assignments);
- emit `spk01.wav`, `spk02.wav`, and `spk03.wav`;
- replace the current two-way overlap keep-or-swap rule with a three-way
  maximum-similarity assignment between neighbouring chunks.

There is no known drop-in public checkpoint for three named lead singers from
a produced commercial mix. The useful research baseline is
[MedleyVox](https://github.com/jeonchangbin49/MedleyVox): its benchmark
includes an N-singing category and its work uses an iSRNet + Conv-TasNet
approach. The official repository does not publish pretrained weights, so it
is a training or fine-tuning starting point rather than an inference-only
replacement. The [MedleyVox paper](https://arxiv.org/abs/2211.07302) and its
[dataset](https://zenodo.org/records/7984549) are suitable references for
three-source evaluation.

[SepACap](https://openreview.net/pdf?id=oERJ6K8FIn) and score-informed choral
separation are useful only for a cappella material or when the intended outputs
are musical parts (such as SATB), rather than the identities of three
individual singers. They are not replacements for a three-lead Stage 3 in a
produced song.

## I. SATB decomposition for backing vocals

[SepACap](https://openreview.net/pdf?id=oERJ6K8FIn) and related SATB models
are worth evaluating as an **optional post-processing branch** for
`01_choir_backing.wav`, the backing-vocal stem produced by Mega53. They are
designed to separate a cappella mixtures into musical parts such as soprano,
alto, tenor, and bass; this is useful when the backing arrangement behaves like
a conventional choir and part-level rehearsal or editing stems are the goal.

They should not replace Mega53's lead-vocal/back-vocal stage. A produced
backing stem can contain reverb, instruments, doubles, ad-libs, and singers
outside conventional SATB ranges; a SATB model may then create artifacts or
assign material to the wrong part. It also outputs **parts**, not the identities
of individual backing singers.

Recommended evaluation path:

1. Run the existing full pipeline and keep `01_choir_backing.wav` unchanged.
2. Apply a SATB model only to that stem, not to the original full mix.
3. Compare the resulting parts by listening for lead-vocal leakage, missing
   backing phrases, and part-assignment errors.
4. Keep the SATB outputs only when they are more useful than the intact backing
   stem; do not feed them back into Stage 3.

This is most promising for relatively dry, clearly arranged choir passages. It
is less appropriate for dense pop backing stacks, small ensembles with similar
voice ranges, or any case where individual-singer identity is required.

### Optional implemented pipeline

`choir_parts_separation.py` is a separate pipeline that accepts an existing
`01_choir_backing.wav`; it does not alter the main pipeline or feed any result
into UNMIXX. It downloads the public
[jaCappella DPTNet checkpoint](https://huggingface.co/jaCappella/DPTNet_jaCappella_VES_48k)
on first use and produces these six estimates at 48 kHz mono:

```text
01_vocal_percussion.wav
02_bass.wav
03_alto.wav
04_tenor.wav
05_soprano.wav
06_lead_vocal.wav
```

Run it with:

```bash
make run-choir-parts \
  CHOIR_INPUT=run_song/final/01_choir_backing.wav \
  CHOIR_OUTPUT_DIR=run_song/choir_parts \
  CHOIR_DEVICE=auto
```

The checkpoint selection is deliberate. SepACap reports the highest published
JaCappella scores, but this project uses DPTNet because its authors publish a
downloadable checkpoint and six-source configuration. It is the best
publicly-downloadable, benchmarked choice identified for this implementation;
it is not a claim that DPTNet exceeds SepACap's reported score. See the
[SepACap benchmark](https://openreview.net/pdf?id=oERJ6K8FIn) and the
[DPTNet model card](https://huggingface.co/jaCappella/DPTNet_jaCappella_VES_48k).

This is an a-cappella model trained on Japanese vocal ensembles. Its outputs
are musical-part estimates, not reliable individual-singer identities; audition
them before using them in production. The jaCappella model card directs users
to the dataset license, so review it before use.

# MIDI transcription

MIDI transcription is a separate branch: it never runs, modifies, or feeds
back into any audio separator. Its larger GAME dependency set is installed only
by a GAME MIDI command; regular separation commands do not change a Colab
runtime's NumPy/TensorFlow stack. There are two deliberately different routes.

## Isolated lead and choir-part stems: GAME Large

Use [GAME](https://github.com/openvpi/GAME), the current successor to SOME, for
an individual lead singer or an already-separated musical choir part. GAME is
specifically designed for singing-to-MIDI extraction and its authors document
robustness to separated vocals, noise, reverb, and accompaniment. `make
run-midi` updates GAME from its default branch, installs its current upstream
requirements, and downloads the largest compatible model bundle from the newest
official release that supports GAME's Python `infer.py` route into
`third_party/GAME-model-large`. GAME's latest release may contain ONNX-only
models; upstream explicitly does not provide Python ONNX inference, so those
assets are deliberately not selected.

```text
03_singer_01.wav ─┐
04_singer_02.wav ─┼──> GAME Large ──> one MIDI file per input stem
SATB part stem  ──┘
```

For Falling Slowly, transcribe both isolated lead-singer files in one command:

```bash
make run-midi \
  MIDI_INPUT=run_falling_slowly/final \
  MIDI_GLOB='0[3-4]_singer_*.wav' \
  MIDI_OUTPUT_DIR=run_falling_slowly/midi \
  MIDI_LANGUAGE=en
```

For Bad Idea's separated choir parts, run GAME on each melodic file. Do not
transcribe `01_vocal_percussion.wav` as pitched MIDI:

```bash
make run-midi \
  MIDI_INPUT=songs/BadIdea/choir_parts \
  MIDI_GLOB='0[2-6]_*.wav' \
  MIDI_OUTPUT_DIR=songs/BadIdea/choir_midi
```

Each invocation writes a `manifest.json` with the current checkpoint path,
GAME repository, input, language, and resulting MIDI files. `MIDI_LANGUAGE`
is optional; GAME's released language-aware models list `en`, `ja`, `yue`, and
`zh`. The model downloader intentionally follows the newest compatible GAME
release, not a pinned revision, matching this project's update policy.

GAME's published model files use **CC BY-NC-SA 4.0**. Review that license
before any commercial use. The code follows the upstream repository's MIT
license, but that does not change the model-files license.

## Unseparated choir mix: Basic Pitch fallback

If you intentionally keep a choir as one mixed stem, use the separate
polyphonic Basic Pitch route:

```bash
make run-polyphonic-choir-midi \
  MIDI_INPUT=run_song/final/01_choir_backing.wav \
  MIDI_OUTPUT_DIR=run_song/choir_polyphonic_midi
```

[Basic Pitch](https://github.com/spotify/basic-pitch) supports polyphonic note
estimation and is the practical fallback for this case, but its MIDI is one
unassigned collection of notes: it cannot reliably turn a mixed choir into
separate SATB or singer tracks. Use DPTNet followed by GAME when separate
part-level MIDI is the goal. Basic Pitch currently documents Python 3.7–3.11
support; use Python 3.10 or 3.11 for this fallback route.

Both routes estimate notes and timing, not lyrics or singer identity. Treat the
output as editable MIDI; review sustained notes, re-attacks, vocal slides, and
closely voiced harmony in a DAW or notation editor.

# Troubleshooting

## Mega53 runs out of VRAM

Mega53 is the heaviest stage. Close other GPU applications. The upstream model notes recommend roughly 16 GB VRAM or more.

## UNMIXX says files are missing

Make sure your clone includes:

```text
ckpt/conf.yml
ckpt/best.ckpt
```

and pass the repository root, not the `ckpt` folder:

```bash
--unmixx-repo third_party/unmixx
```

## `spk1.wav` / `spk2.wav` cannot be found

The pipeline searches recursively because the official UNMIXX inference script writes to a nested directory derived from the model/checkpoint and input filename. If the upstream repository changes its naming convention, inspect `stage3_duet_unmixx/raw_unmixx/` and update `find_unique_stem()`.

## One singer is consistently much louder

That may be the actual mix balance. Do not normalize each final singer independently before judging separation: independent normalization can make tiny leakage sound like a major failure.

# Model references

- UNMIXX official repository: `jihoojung0106/unmixx`
- UNMIXX paper: *UNMIXX: Untangling Highly Correlated Singing Voices Mixtures* (ICASSP 2026)
- Music-Source-Separation-Training: `ZFTurbo/Music-Source-Separation-Training`
- MVSep Mega 53: public BS-RoFormer 53-stem checkpoint with `lead-vocal` and `back-vocal`

# Bottom line

For the successful musical-theatre scenario, the intended interpretation is now:

```text
all vocals
   -> Mega53
       -> choir/backing
       -> two-principal-singer mixture
            -> UNMIXX
                -> singer 1
                -> singer 2
```

That is the project's new default architecture.
