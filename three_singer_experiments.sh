#!/usr/bin/env bash
# Run the three-singer separation experiments without repeating vocal isolation.
#
# This is designed for a CUDA Colab runtime after `make prepare`.  It does not
# install packages or clone repositories; those setup actions remain explicit.
#
# Experiments:
#   1. Mel-Band RoFormer -> Mega53 -> UNMIXX on Mega53's back-vocal remainder.
#   2. Mel-Band RoFormer -> direct UNMIXX on all isolated vocals.
#   3. Optional: UNMIXX on a manually selected direct-UNMIXX remainder.
set -Eeuo pipefail

# Colab commonly invokes this file as `bash project/three_singer_experiments.sh`
# while its current directory is `/content`.  Run all relative paths from the
# repository containing this script, rather than from the caller's directory.
REPOSITORY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$REPOSITORY_DIR"

usage() {
    cat <<'EOF'
Usage:
  bash three_singer_experiments.sh --input SONG [options]

Required unless --recursive-only is used:
  --input PATH                 Full song audio; Mel-Band RoFormer is run once.

Options:
  --output-dir PATH            Experiment root (default: experiments/three_singer)
  --device NAME                cuda, cpu, mps, or auto (default: cuda)
  --python PATH                Python command to use (default: python)
  --mss-repo PATH              Mega53 repository (default: third_party/Music-Source-Separation-Training)
  --unmixx-repo PATH           UNMIXX repository (default: third_party/unmixx)
  --recursive-remainder VALUE  Run experiment 3 on direct UNMIXX's two-singer
                               remainder. Use singer_01, singer_02, or a WAV path.
  --recursive-only             Run only experiment 3 using existing direct-UNMIXX
                               output; requires --recursive-remainder.
  -h, --help                   Show this help.

The script deliberately does not guess the recursive remainder.  Audition
direct_unmixx/final/03_singer_01.wav and 04_singer_02.wav first, then rerun
with --recursive-remainder singer_01 or singer_02 only if that stem contains
the other two singers consistently.
EOF
}

INPUT=""
OUTPUT_DIR="experiments/three_singer"
DEVICE="cuda"
PYTHON_BIN="python"
MSS_REPO="third_party/Music-Source-Separation-Training"
UNMIXX_REPO="third_party/unmixx"
RECURSIVE_REMAINDER=""
RECURSIVE_ONLY=0

while (($#)); do
    case "$1" in
        --input) INPUT="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --mss-repo) MSS_REPO="$2"; shift 2 ;;
        --unmixx-repo) UNMIXX_REPO="$2"; shift 2 ;;
        --recursive-remainder) RECURSIVE_REMAINDER="$2"; shift 2 ;;
        --recursive-only) RECURSIVE_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$RECURSIVE_ONLY" == 0 && -z "$INPUT" ]]; then
    printf '%s\n' '--input is required.' >&2
    usage >&2
    exit 2
fi
if [[ "$RECURSIVE_ONLY" == 0 && ! -f "$INPUT" ]]; then
    printf 'Input does not exist: %s\n' "$INPUT" >&2
    exit 2
fi
if [[ "$RECURSIVE_ONLY" == 0 && ! -f "$MSS_REPO/inference.py" ]]; then
    printf 'Mega53 repository is unavailable: %s\n' "$MSS_REPO" >&2
    exit 2
fi
if [[ ! -f "$UNMIXX_REPO/inference.py" || ! -f "$UNMIXX_REPO/ckpt/conf.yml" || ! -f "$UNMIXX_REPO/ckpt/best.ckpt" ]]; then
    printf 'UNMIXX repository/checkpoint is unavailable: %s\n' "$UNMIXX_REPO" >&2
    exit 2
fi
if [[ "$RECURSIVE_ONLY" == 1 && -z "$RECURSIVE_REMAINDER" ]]; then
    printf '%s\n' '--recursive-only requires --recursive-remainder.' >&2
    exit 2
fi

MEGA_DIR="$OUTPUT_DIR/mega53_split"
MEGA_BACKING_DIR="$OUTPUT_DIR/mega53_backing_unmixx"
DIRECT_DIR="$OUTPUT_DIR/direct_unmixx"

VOCALS="$MEGA_DIR/final/00_all_vocals.wav"
MEGA_LEAD="$MEGA_DIR/final/02_lead_vocal.wav"
MEGA_BACKING="$MEGA_DIR/final/01_choir_backing.wav"

if [[ "$RECURSIVE_ONLY" == 0 ]]; then
    printf '%s\n' '=== Experiment 1: Mega53 one-versus-two hypothesis ==='
    "$PYTHON_BIN" pipeline.py "$INPUT" \
        --mode lead-choir \
        --mss-repo "$MSS_REPO" \
        --device "$DEVICE" \
        --output-dir "$MEGA_DIR"

    printf '%s\n' '=== Experiment 1b: UNMIXX on Mega53 back-vocal remainder ==='
    "$PYTHON_BIN" pipeline.py "$MEGA_BACKING" \
        --input-is-vocals \
        --mode duet \
        --unmixx-repo "$UNMIXX_REPO" \
        --device "$DEVICE" \
        --unmixx-alignment-review-dir "$MEGA_BACKING_DIR/review" \
        --output-dir "$MEGA_BACKING_DIR"

    printf '%s\n' '=== Experiment 2: Direct UNMIXX one-versus-two hypothesis ==='
    "$PYTHON_BIN" pipeline.py "$VOCALS" \
        --input-is-vocals \
        --mode duet \
        --unmixx-repo "$UNMIXX_REPO" \
        --device "$DEVICE" \
        --unmixx-alignment-review-dir "$DIRECT_DIR/review" \
        --output-dir "$DIRECT_DIR"
fi

if [[ -n "$RECURSIVE_REMAINDER" ]]; then
    case "$RECURSIVE_REMAINDER" in
        singer_01) REMAINDER="$DIRECT_DIR/final/03_singer_01.wav" ;;
        singer_02) REMAINDER="$DIRECT_DIR/final/04_singer_02.wav" ;;
        *) REMAINDER="$RECURSIVE_REMAINDER" ;;
    esac
    if [[ ! -f "$REMAINDER" ]]; then
        printf 'Recursive remainder does not exist: %s\n' "$REMAINDER" >&2
        exit 2
    fi

    printf '%s\n' '=== Experiment 3: Recursive UNMIXX on selected remainder ==='
    "$PYTHON_BIN" pipeline.py "$REMAINDER" \
        --input-is-vocals \
        --mode duet \
        --unmixx-repo "$UNMIXX_REPO" \
        --device "$DEVICE" \
        --unmixx-alignment-review-dir "$OUTPUT_DIR/recursive_unmixx/review" \
        --output-dir "$OUTPUT_DIR/recursive_unmixx"
fi

cat <<EOF

Completed experiment outputs:
  Mega53 lead candidate:        $MEGA_LEAD
  Mega53 back-vocal candidate:  $MEGA_BACKING
  Mega53-remainder UNMIXX:      $MEGA_BACKING_DIR/final/03_singer_01.wav
                                $MEGA_BACKING_DIR/final/04_singer_02.wav
  Direct UNMIXX:                $DIRECT_DIR/final/03_singer_01.wav
                                $DIRECT_DIR/final/04_singer_02.wav

Audition complete passages, not just isolated moments.  Direct UNMIXX source
order is arbitrary; only run recursion after confirming one output is the
same two-singer remainder across the song.
EOF
