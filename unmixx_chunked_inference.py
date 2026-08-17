#!/usr/bin/env python3
"""Memory-safe long-form inference for the official UNMIXX checkpoint.

UNMIXX was trained on short segments (4 s in the public config), while the
reference inference script forwards the whole file at once. This runner keeps
one model loaded, processes overlapping chunks, tracks the two-source
permutation between neighbouring chunks, and overlap-adds the results.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torchaudio
import yaml

from audio_io import load_audio, save_wav


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    if name == "mps":
        print("[UNMIXX] Warning: MPS is unsupported; falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(name)


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def import_unmixx(repo: Path):
    sys.path.insert(0, str(repo))
    import look2hear.models  # type: ignore
    return look2hear.models


def build_model(models_module, cfg: dict, ckpt_path: Path, device: torch.device):
    net_name = cfg["audionet"]["audionet_name"]
    model_cls = getattr(models_module, net_name)
    sr = int(cfg["datamodule"]["data_config"]["sample_rate"])
    model = model_cls(sample_rate=sr, **cfg["audionet"]["audionet_config"])

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        converted = {}
        for key, value in state.items():
            converted[key[len("audio_model."):] if key.startswith("audio_model.") else key] = value
        model.load_state_dict(converted, strict=True)

    return model.to(device).eval(), sr


def normalize_model_output(outs: object) -> torch.Tensor:
    if isinstance(outs, tuple):
        if len(outs) == 2:
            outs = outs[1]
        else:
            outs = outs[0]
    if isinstance(outs, dict):
        for key in ("output_final", "audio_out_final", "output", "audio_out"):
            if key in outs:
                outs = outs[key]
                break
        else:
            outs = next(iter(outs.values()))
    if not isinstance(outs, torch.Tensor):
        raise TypeError(f"Unexpected UNMIXX output type: {type(outs)!r}")
    if outs.dim() == 2:
        outs = outs.unsqueeze(0)
    if outs.dim() != 3:
        raise RuntimeError(f"Expected UNMIXX output [B,S,T], got {tuple(outs.shape)}")
    return outs


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom) < eps:
        return 0.0
    return float(torch.dot(a, b) / denom)


def align_two_sources(prev_tail: torch.Tensor, cur_head: torch.Tensor) -> tuple[torch.Tensor, bool, float, float]:
    """Choose source order maximizing continuity across the overlap.

    Both tensors are [2, overlap_samples].
    """
    keep = cosine_similarity(prev_tail[0], cur_head[0]) + cosine_similarity(prev_tail[1], cur_head[1])
    swap = cosine_similarity(prev_tail[0], cur_head[1]) + cosine_similarity(prev_tail[1], cur_head[0])
    if swap > keep:
        return cur_head.flip(0), True, keep, swap
    return cur_head, False, keep, swap


def make_window(length: int, overlap: int, is_first: bool, is_last: bool) -> torch.Tensor:
    w = torch.ones(length, dtype=torch.float32)
    n = min(overlap, length)
    if n > 1 and not is_first:
        w[:n] = torch.linspace(0.0, 1.0, n)
    if n > 1 and not is_last:
        w[-n:] = torch.linspace(1.0, 0.0, n)
    return w


def main() -> None:
    p = argparse.ArgumentParser(description="Chunked long-form inference for the official UNMIXX model")
    p.add_argument("--unmixx-repo", type=Path, required=True)
    p.add_argument("--conf-path", type=Path, required=True)
    p.add_argument("--ckpt-path", type=Path, required=True)
    p.add_argument("--audio-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--chunk-seconds", type=float, default=4.0)
    p.add_argument("--overlap-seconds", type=float, default=1.0)
    args = p.parse_args()

    if args.chunk_seconds <= 0:
        p.error("--chunk-seconds must be > 0")
    if args.overlap_seconds < 0 or args.overlap_seconds >= args.chunk_seconds:
        p.error("--overlap-seconds must be >= 0 and < --chunk-seconds")

    device = resolve_device(args.device)

    cfg = load_config(args.conf_path)
    models_module = import_unmixx(args.unmixx_repo.resolve())
    model, target_sr = build_model(models_module, cfg, args.ckpt_path, device)

    waveform, sr = load_audio(args.audio_path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.contiguous().cpu()

    total_samples = waveform.shape[-1]
    chunk_samples = int(round(args.chunk_seconds * sr))
    overlap_samples = int(round(args.overlap_seconds * sr))
    hop = chunk_samples - overlap_samples
    if hop <= 0:
        raise RuntimeError("Chunk overlap leaves a non-positive hop size")

    starts = list(range(0, total_samples, hop))
    if starts and starts[-1] + overlap_samples >= total_samples and len(starts) > 1:
        starts.pop()

    print(f"[UNMIXX chunked] device={device}, sr={sr}, duration={total_samples/sr:.2f}s")
    print(
        f"[UNMIXX chunked] chunk={args.chunk_seconds:.2f}s, "
        f"overlap={args.overlap_seconds:.2f}s, chunks={len(starts)}"
    )

    accum = torch.zeros((2, total_samples), dtype=torch.float32)
    weights = torch.zeros(total_samples, dtype=torch.float32)
    prev_raw: torch.Tensor | None = None

    for idx, start in enumerate(starts):
        end = min(start + chunk_samples, total_samples)
        raw = waveform[:, start:end]
        actual_len = raw.shape[-1]

        # Pad only the final chunk to the normal inference size. Crop after inference.
        if actual_len < chunk_samples:
            raw_in = torch.nn.functional.pad(raw, (0, chunk_samples - actual_len))
        else:
            raw_in = raw

        x = raw_in.unsqueeze(0).to(device, non_blocking=True)  # [1,1,T]
        with torch.inference_mode():
            outs = model(x, istest=True)
            est = normalize_model_output(outs)[0, :2, :actual_len].float().cpu()
        del x, outs

        swapped = False
        if prev_raw is not None and overlap_samples > 0:
            ov = min(overlap_samples, prev_raw.shape[-1], est.shape[-1])
            if ov > 0:
                _, swapped, keep_score, swap_score = align_two_sources(prev_raw[:, -ov:], est[:, :ov])
                if swapped:
                    est = est.flip(0)
                print(
                    f"  chunk {idx+1:03d}/{len(starts):03d} "
                    f"{start/sr:7.2f}-{end/sr:7.2f}s "
                    f"perm={'swap' if swapped else 'keep'} "
                    f"scores={keep_score:+.3f}/{swap_score:+.3f}"
                )
            else:
                print(f"  chunk {idx+1:03d}/{len(starts):03d} {start/sr:7.2f}-{end/sr:7.2f}s")
        else:
            print(f"  chunk {idx+1:03d}/{len(starts):03d} {start/sr:7.2f}-{end/sr:7.2f}s")

        window = make_window(
            actual_len,
            overlap_samples,
            is_first=(idx == 0),
            is_last=(idx == len(starts) - 1),
        )
        accum[:, start:end] += est * window.unsqueeze(0)
        weights[start:end] += window
        prev_raw = est

        if device.type == "cuda" and (idx + 1) % 16 == 0:
            torch.cuda.empty_cache()

    result = accum / weights.clamp_min(1e-6).unsqueeze(0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    spk1 = args.output_dir / "spk1.wav"
    spk2 = args.output_dir / "spk2.wav"
    save_wav(spk1, result[0:1], sr)
    save_wav(spk2, result[1:2], sr)
    print(f"[Save] {spk1}")
    print(f"[Save] {spk2}")


if __name__ == "__main__":
    main()
