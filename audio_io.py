"""WAV I/O that does not depend on TorchCodec's FFmpeg shared libraries."""

from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch


def load_audio(path: Path) -> tuple[torch.Tensor, int]:
    """Load audio as a contiguous, channels-first float32 tensor."""
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(samples.T.copy()), sample_rate


def save_wav(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """Save a channels-first tensor as a float WAV file."""
    samples = waveform.detach().cpu().transpose(0, 1).contiguous().numpy()
    sf.write(path, samples, sample_rate, subtype="FLOAT")
