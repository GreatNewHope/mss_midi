#!/usr/bin/env python3
"""Optional a-cappella choir-part separation for an existing backing-vocal stem.

The public jaCappella DPTNet checkpoint expects 48 kHz mono audio and emits
six vocal-ensemble parts: vocal percussion, bass, alto, tenor, soprano, and
lead vocal. It is intended for a backing-vocal stem (for example Mega53's
``01_choir_backing.wav``), not for a full instrumental mix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchaudio
import yaml
from asteroid.models import DPTNet
from huggingface_hub import hf_hub_download

from audio_io import load_audio, save_wav


MODEL_REPO = "jaCappella/DPTNet_jaCappella_VES_48k"
CHECKPOINT_NAME = "best_model.pth"
CONFIG_NAME = "conf.yml"


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    if name == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("--device mps was requested, but Metal Performance Shaders is not available")
    return torch.device(name)


def load_model(cache_dir: Path, device: torch.device) -> tuple[DPTNet, list[str], int, float]:
    checkpoint = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=CHECKPOINT_NAME,
        cache_dir=str(cache_dir),
    )
    config_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=CONFIG_NAME,
        cache_dir=str(cache_dir),
    )
    with Path(config_path).open() as fh:
        config = yaml.safe_load(fh)

    model = DPTNet.from_pretrained(checkpoint).to(device).eval()
    sources = list(config["data"]["sources"])
    sample_rate = int(config["data"]["sample_rate"])
    segment_seconds = float(config["data"]["seq_dur"])
    if len(sources) != 6:
        raise RuntimeError(f"Expected six jaCappella sources, got {sources!r}")
    return model, sources, sample_rate, segment_seconds


def fade_window(length: int, overlap: int, first: bool, last: bool) -> torch.Tensor:
    window = torch.ones(length, dtype=torch.float32)
    ramp = min(length, overlap)
    if ramp > 1 and not first:
        window[:ramp] = torch.linspace(0.0, 1.0, ramp)
    if ramp > 1 and not last:
        window[-ramp:] = torch.linspace(1.0, 0.0, ramp)
    return window


def separate_long_audio(
    waveform: torch.Tensor,
    model: DPTNet,
    source_count: int,
    device: torch.device,
    segment_seconds: float,
    overlap_seconds: float,
    sample_rate: int,
) -> torch.Tensor:
    """Run fixed-size, overlap-added inference to keep long songs practical."""
    total = waveform.shape[-1]
    segment = int(round(segment_seconds * sample_rate))
    overlap = int(round(overlap_seconds * sample_rate))
    hop = segment - overlap
    if segment <= 0 or hop <= 0:
        raise ValueError("segment duration must be positive and greater than overlap")

    starts = list(range(0, total, hop))
    estimates = torch.zeros((source_count, total), dtype=torch.float32)
    weights = torch.zeros(total, dtype=torch.float32)

    for index, start in enumerate(starts):
        end = min(start + segment, total)
        chunk = waveform[:, start:end]
        actual = chunk.shape[-1]
        if actual < segment:
            chunk = torch.nn.functional.pad(chunk, (0, segment - actual))

        with torch.inference_mode():
            output = model(chunk.unsqueeze(0).to(device, non_blocking=True))[0, :, :actual]
        output = output.float().cpu()
        if output.shape[0] != source_count:
            raise RuntimeError(
                f"Checkpoint returned {output.shape[0]} sources; expected {source_count}"
            )

        window = fade_window(actual, overlap, index == 0, index == len(starts) - 1)
        estimates[:, start:end] += output * window.unsqueeze(0)
        weights[start:end] += window
        print(f"  chunk {index + 1:03d}/{len(starts):03d}: {start / sample_rate:.2f}-{end / sample_rate:.2f}s")

    return estimates / weights.clamp_min(1e-6).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split an existing choir/backing-vocal stem into jaCappella vocal-ensemble parts."
    )
    parser.add_argument("input", type=Path, help="Backing-vocal WAV, e.g. 01_choir_backing.wav")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--cache-dir", type=Path, default=Path(".model_cache/jacappella"))
    parser.add_argument("--segment-seconds", type=float, default=None)
    parser.add_argument("--overlap-seconds", type=float, default=0.5)
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input does not exist: {args.input}")
    if args.overlap_seconds < 0:
        parser.error("--overlap-seconds must be >= 0")

    device = resolve_device(args.device)
    model, sources, target_rate, trained_segment_seconds = load_model(args.cache_dir, device)
    segment_seconds = args.segment_seconds or trained_segment_seconds
    if segment_seconds <= args.overlap_seconds:
        parser.error("--segment-seconds must be greater than --overlap-seconds")

    waveform, input_rate = load_audio(args.input)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if input_rate != target_rate:
        waveform = torchaudio.functional.resample(waveform, input_rate, target_rate)
    waveform = waveform.contiguous().cpu()

    print(
        f"[jaCappella DPTNet] device={device}, input={args.input}, "
        f"sr={target_rate}, duration={waveform.shape[-1] / target_rate:.2f}s"
    )
    estimates = separate_long_audio(
        waveform,
        model,
        len(sources),
        device,
        segment_seconds,
        args.overlap_seconds,
        target_rate,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(sources, start=1):
        destination = args.output_dir / f"{index:02d}_{source}.wav"
        save_wav(destination, estimates[index - 1 : index], target_rate)
        print(f"[Save] {destination}")

    manifest = {
        "input": str(args.input.resolve()),
        "model": MODEL_REPO,
        "checkpoint": CHECKPOINT_NAME,
        "sample_rate": target_rate,
        "sources": sources,
        "segment_seconds": segment_seconds,
        "overlap_seconds": args.overlap_seconds,
        "warning": (
            "The checkpoint was trained on Japanese a-cappella ensembles. "
            "Treat labels as musical-part estimates and audition every output."
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
