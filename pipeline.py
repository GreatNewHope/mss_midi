#!/usr/bin/env python3
"""Modular hierarchical vocal separation for pop / musical-theatre mixes.

Modes
-----
full:
    mix -> all vocals -> foreground vocals vs choir/backing -> singer 1 + singer 2

duet:
    mix -> all vocals -> singer 1 + singer 2
    Use when there is no backing choir and the vocal stem is essentially a duet.

lead-choir:
    mix -> all vocals -> lead/foreground vocal vs choir/backing
    Use when there is one principal singer plus a backing choir and no duet split is needed.

You may additionally pass --input-is-vocals to bypass Stage 1 when the input file is
already a vocal-only stem.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download
import torch


MEGA53_REPO_ID = "oulianov/mvsep_mega_53"
MEGA53_CONFIG = "mvsep_mega_model_bs_roformer_53_stems.yaml"
MEGA53_CHECKPOINT = "mvsep_mega_model_bs_roformer_53_stems_v1.ckpt"
UNMIXX_EXPECTED_FILES = ("ckpt/conf.yml", "ckpt/best.ckpt", "inference.py")


@dataclass(frozen=True)
class PipelineOutputs:
    all_vocals: Path
    choir_backing: Path | None = None
    foreground_vocals: Path | None = None
    singer_01: Path | None = None
    singer_02: Path | None = None


def run(
    cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> None:
    print("+", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def resolve_device(name: str) -> str:
    """Select CUDA, then MPS, then CPU for automatic execution."""
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if mps_available():
            return "mps"
        return "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    if name == "mps" and not mps_available():
        raise RuntimeError("--device mps was requested, but Metal Performance Shaders is not available")
    return name


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required to decode/resample audio. Install ffmpeg and make sure it is on PATH."
        )


def decode_to_wav(input_audio: Path, wav_out: Path, sample_rate: int = 44_100) -> None:
    """Decode any ffmpeg-supported file to 44.1 kHz stereo float WAV."""
    require_ffmpeg()
    wav_out.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_audio),
        "-ar", str(sample_rate),
        "-ac", "2",
        "-c:a", "pcm_f32le",
        str(wav_out),
    ])


def convert_for_unmixx(input_wav: Path, output_wav: Path) -> None:
    """Prepare a two-singer mixture for the public UNMIXX checkpoint: 24 kHz mono."""
    require_ffmpeg()
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_wav),
        "-ar", "24000",
        "-ac", "1",
        "-c:a", "pcm_f32le",
        str(output_wav),
    ])


def separate_vocals_melband(
    input_wav: Path,
    output_dir: Path,
    model_slug: str,
    device: str | None,
) -> Path:
    """Run melband-roformer-infer and return the combined vocal stem."""
    exe = shutil.which("melband-roformer-infer")
    if exe is None:
        raise RuntimeError(
            "Missing melband-roformer-infer CLI. Install it with `pip install melband-roformer-infer`."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="melband_input_") as td:
        in_dir = Path(td)
        staged = in_dir / input_wav.name
        shutil.copy2(input_wav, staged)

        cmd = [
            exe,
            "--input_folder", str(in_dir),
            "--store_dir", str(output_dir),
            "--model", model_slug,
        ]
        if device and device != "auto":
            cmd += ["--device", device]
        run(cmd)

    return find_unique_stem(output_dir, "vocals", reject=("instrument", "no_vocal"))


def resolve_mega53_assets(cache_dir: Path) -> tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    config = Path(hf_hub_download(
        repo_id=MEGA53_REPO_ID,
        filename=MEGA53_CONFIG,
        cache_dir=str(cache_dir),
    ))
    checkpoint = Path(hf_hub_download(
        repo_id=MEGA53_REPO_ID,
        filename=MEGA53_CHECKPOINT,
        cache_dir=str(cache_dir),
    ))
    return config, checkpoint


def validate_mss_repo(mss_repo: Path | None) -> Path:
    if mss_repo is None:
        raise RuntimeError(
            "This mode needs Mega53. Pass --mss-repo /path/to/Music-Source-Separation-Training."
        )
    repo = mss_repo.expanduser().resolve()
    inference = repo / "inference.py"
    if not inference.exists():
        raise RuntimeError(
            f"Music-Source-Separation-Training was not found at {repo}. "
            "Clone https://github.com/ZFTurbo/Music-Source-Separation-Training and pass --mss-repo."
        )
    return repo


def separate_lead_and_backing_mega53(
    vocals_wav: Path,
    output_dir: Path,
    mss_repo: Path | None,
    cache_dir: Path,
    python_executable: str,
    device: str,
) -> tuple[Path, Path]:
    """Run Mega53 and return (lead-vocal/foreground, back-vocal/choir)."""
    repo = validate_mss_repo(mss_repo)
    config, checkpoint = resolve_mega53_assets(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mega53_input_") as td:
        in_dir = Path(td)
        staged = in_dir / "all_vocals.wav"
        shutil.copy2(vocals_wav, staged)

        cmd = [
            python_executable,
            str(repo / "inference.py"),
            "--model_type", "bs_roformer",
            "--config_path", str(config),
            "--start_check_point", str(checkpoint),
            "--input_folder", str(in_dir),
            "--store_dir", str(output_dir),
        ]
        if device == "cpu":
            cmd.append("--force_cpu")
        env = os.environ.copy()
        if device == "mps":
            # Mega53 selects CUDA before MPS; hide CUDA for this explicit MPS request.
            env["CUDA_VISIBLE_DEVICES"] = ""
        run(cmd, cwd=repo, env=env)

    lead = find_unique_stem(output_dir, "lead-vocal")
    backing = find_unique_stem(output_dir, "back-vocal")
    return lead, backing


def validate_unmixx_repo(unmixx_repo: Path | None) -> Path:
    if unmixx_repo is None:
        raise RuntimeError(
            "This mode needs UNMIXX. Pass --unmixx-repo /path/to/unmixx."
        )
    repo = unmixx_repo.expanduser().resolve()
    missing = [rel for rel in UNMIXX_EXPECTED_FILES if not (repo / rel).exists()]
    if missing:
        raise RuntimeError(
            f"UNMIXX repo at {repo} is missing {missing}. Clone the official repo "
            "https://github.com/jihoojung0106/unmixx and keep its ckpt/ directory intact."
        )
    return repo


def separate_duet_unmixx(
    duet_wav: Path,
    output_dir: Path,
    unmixx_repo: Path | None,
    python_executable: str,
    device: str,
    chunk_seconds: float,
    overlap_seconds: float,
) -> tuple[Path, Path]:
    """Run memory-safe chunked UNMIXX inference and return two singer WAVs.

    The official UNMIXX inference script forwards the whole file at once, which
    is impractical for full songs because its attention memory grows rapidly
    with sequence length. The bundled helper keeps the model loaded once and
    processes overlapping short windows instead.
    """
    repo = validate_unmixx_repo(unmixx_repo)
    output_dir.mkdir(parents=True, exist_ok=True)
    helper = Path(__file__).resolve().with_name("unmixx_chunked_inference.py")
    if not helper.exists():
        raise RuntimeError(f"Missing bundled UNMIXX helper: {helper}")

    run([
        python_executable,
        str(helper),
        "--unmixx-repo", str(repo),
        "--conf-path", str(repo / "ckpt" / "conf.yml"),
        "--ckpt-path", str(repo / "ckpt" / "best.ckpt"),
        "--audio-path", str(duet_wav),
        "--output-dir", str(output_dir),
        "--device", unmixx_device(device),
        "--chunk-seconds", str(chunk_seconds),
        "--overlap-seconds", str(overlap_seconds),
    ])

    spk1 = find_unique_stem(output_dir, "spk1")
    spk2 = find_unique_stem(output_dir, "spk2")
    return spk1, spk2


def unmixx_device(device: str) -> str:
    """Choose a UNMIXX-compatible device without changing other stages."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "mps":
        print("[UNMIXX] Warning: MPS is unsupported; falling back to CPU.", file=sys.stderr)
        return "cpu"
    if device in {"cpu", "cuda"}:
        return device
    raise ValueError("UNMIXX supports device auto, cpu, cuda, or mps.")


def find_unique_stem(root: Path, token: str, *, reject: tuple[str, ...] = ()) -> Path:
    token_l = token.lower()
    candidates = []
    for p in root.rglob("*.wav"):
        name = p.name.lower()
        if token_l in name and not any(r.lower() in name for r in reject):
            candidates.append(p)

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError(f"Could not find a WAV containing {token!r} under {root}")

    exactish = [p for p in candidates if p.stem.lower() in {token_l, token_l.replace("-", "_")}]
    if len(exactish) == 1:
        return exactish[0]

    raise RuntimeError(
        f"Could not uniquely identify {token!r} under {root}. "
        f"Candidates: {[str(p) for p in candidates]}"
    )


def copy_final(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def prepare_all_vocals(args: argparse.Namespace, stage1: Path) -> Path:
    """Return the all-vocals stem, optionally bypassing Stage 1."""
    if args.input_is_vocals:
        print("[Stage 1] SKIPPED: treating input as an existing vocal-only stem")
        vocals = stage1 / "all_vocals_input.wav"
        decode_to_wav(args.input.expanduser().resolve(), vocals)
        return vocals

    print(f"[Stage 1] Isolate all vocals with {args.vocal_model}")
    decoded = stage1 / "input_44k_stereo.wav"
    decode_to_wav(args.input.expanduser().resolve(), decoded)
    return separate_vocals_melband(decoded, stage1, args.vocal_model, args.device)


def run_pipeline(args: argparse.Namespace) -> PipelineOutputs:
    out = args.output_dir.expanduser().resolve()
    stage1 = out / "stage1_all_vocals"
    stage2 = out / "stage2_lead_choir"
    stage3 = out / "stage3_duet_unmixx"
    final = out / "final"
    cache = out / ".model_cache"

    for d in (stage1, stage2, stage3, final, cache):
        d.mkdir(parents=True, exist_ok=True)

    all_vocals = prepare_all_vocals(args, stage1)
    final_all = copy_final(all_vocals, final / "00_all_vocals.wav")

    choir_native: Path | None = None
    foreground_native: Path | None = None
    raw_singer_1: Path | None = None
    raw_singer_2: Path | None = None

    if args.mode in {"full", "lead-choir"}:
        print("[Stage 2] Split foreground vocals from choir/backing with Mega53")
        foreground_native, choir_native = separate_lead_and_backing_mega53(
            all_vocals,
            stage2,
            args.mss_repo,
            cache / "mega53",
            args.python,
            args.device,
        )
    else:
        print("[Stage 2] SKIPPED: no backing-choir split requested")
        foreground_native = all_vocals

    if args.mode in {"full", "duet"}:
        print("[Stage 3] Separate the two foreground singers with UNMIXX")
        duet_24k = stage3 / "duet_mix_24k_mono.wav"
        convert_for_unmixx(foreground_native, duet_24k)
        raw_singer_1, raw_singer_2 = separate_duet_unmixx(
            duet_24k,
            stage3 / "raw_unmixx",
            args.unmixx_repo,
            args.python,
            args.unmixx_device,
            args.unmixx_chunk_seconds,
            args.unmixx_overlap_seconds,
        )
    else:
        print("[Stage 3] SKIPPED: one foreground singer expected")

    final_choir = None
    final_foreground = None
    final_s1 = None
    final_s2 = None

    if choir_native is not None:
        final_choir = copy_final(choir_native, final / "01_choir_backing.wav")

    if foreground_native is not None:
        foreground_name = "02_duet_mix.wav" if args.mode in {"full", "duet"} else "02_lead_vocal.wav"
        final_foreground = copy_final(foreground_native, final / foreground_name)

    if raw_singer_1 is not None and raw_singer_2 is not None:
        final_s1 = copy_final(raw_singer_1, final / "03_singer_01.wav")
        final_s2 = copy_final(raw_singer_2, final / "04_singer_02.wav")

    return PipelineOutputs(
        all_vocals=final_all,
        choir_backing=final_choir,
        foreground_vocals=final_foreground,
        singer_01=final_s1,
        singer_02=final_s2,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Modular vocal separation: choose full, duet-only, or lead+choir processing. "
            "Use --input-is-vocals to bypass instrument/vocal separation."
        )
    )
    p.add_argument("input", type=Path, help="Input audio, e.g. a 320 kb/s MP3 or WAV")
    p.add_argument(
        "--mode",
        choices=("full", "duet", "lead-choir"),
        default="full",
        help=(
            "full = vocals -> lead/choir -> duet split; "
            "duet = vocals -> duet split (skip choir stage); "
            "lead-choir = vocals -> lead/choir (skip duet stage)"
        ),
    )
    p.add_argument(
        "--input-is-vocals",
        action="store_true",
        help="Skip Stage 1 and treat the input file as an already isolated vocal stem.",
    )
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument(
        "--mss-repo",
        type=Path,
        help="Local clone of ZFTurbo/Music-Source-Separation-Training; needed for full/lead-choir modes.",
    )
    p.add_argument(
        "--unmixx-repo",
        type=Path,
        help="Local clone of jihoojung0106/unmixx; needed for full/duet modes.",
    )
    p.add_argument(
        "--vocal-model",
        default="melband-roformer-kim-vocals",
        help="melband-roformer-infer model slug for Stage 1",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Execution device: auto selects CUDA, then MPS, then CPU.",
    )
    p.add_argument(
        "--unmixx-chunk-seconds",
        type=float,
        default=4.0,
        help="UNMIXX Stage-3 chunk duration. Default 4.0 s matches the public training config.",
    )
    p.add_argument(
        "--unmixx-overlap-seconds",
        type=float,
        default=1.0,
        help="Overlap between UNMIXX chunks for crossfade and singer-order tracking (default: 1.0 s).",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch MSS and UNMIXX inference scripts",
    )
    return p


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.input.exists():
        parser.error(f"Input does not exist: {args.input}")
    if args.mode in {"full", "lead-choir"} and args.mss_repo is None:
        parser.error(f"--mode {args.mode} requires --mss-repo")
    if args.mode in {"full", "duet"} and args.unmixx_repo is None:
        parser.error(f"--mode {args.mode} requires --unmixx-repo")
    if args.unmixx_chunk_seconds <= 0:
        parser.error("--unmixx-chunk-seconds must be > 0")
    if args.unmixx_overlap_seconds < 0 or args.unmixx_overlap_seconds >= args.unmixx_chunk_seconds:
        parser.error("--unmixx-overlap-seconds must be >= 0 and smaller than --unmixx-chunk-seconds")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    # Resolve this before the general device selection so UNMIXX can keep its
    # own CUDA-or-CPU auto policy while the rest of the pipeline can use MPS.
    args.unmixx_device = unmixx_device(args.device)
    args.device = resolve_device(args.device)

    outputs = run_pipeline(args)
    print("\nDone. Final outputs:")
    print(f"  all vocals       : {outputs.all_vocals}")
    if outputs.choir_backing:
        print(f"  choir / backing  : {outputs.choir_backing}")
    if outputs.foreground_vocals:
        label = "duet mixture" if outputs.singer_01 else "lead vocal"
        print(f"  {label:17s}: {outputs.foreground_vocals}")
    if outputs.singer_01 and outputs.singer_02:
        print(f"  singer 01        : {outputs.singer_01}")
        print(f"  singer 02        : {outputs.singer_02}")
        print(
            "\nNote: UNMIXX output order is permutation-ambiguous. "
            "singer_01 and singer_02 are not guaranteed person identities across songs."
        )


if __name__ == "__main__":
    main()
