"""AnyWidget-based notebook UI for correcting UNMIXX chunk permutations."""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path
from typing import Any

import anywidget
import numpy as np
import soundfile as sf
import traitlets


def _window(length: int, overlap: int, first: bool, last: bool) -> np.ndarray:
    window = np.ones(length, dtype=np.float32)
    n = min(length, overlap)
    if n > 1 and not first:
        window[:n] = np.linspace(0.0, 1.0, n, dtype=np.float32)
    if n > 1 and not last:
        window[-n:] = np.linspace(1.0, 0.0, n, dtype=np.float32)
    return window


class ChunkAlignmentReview:
    """Loads a review bundle and renders user-selected chunk permutations."""

    def __init__(self, review_dir: str | Path, output_dir: str | Path | None = None) -> None:
        self.review_dir = Path(review_dir).expanduser().resolve()
        manifest_path = self.review_dir / "alignment_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No alignment manifest at {manifest_path}")
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format_version") != 1:
            raise ValueError("Unsupported alignment-review bundle format")
        self.chunks: list[dict[str, Any]] = self.manifest["chunks"]
        if not self.chunks:
            raise ValueError("The alignment-review bundle has no chunks")
        self.sample_rate = int(self.manifest["sample_rate"])
        self.total_samples = int(self.manifest["total_samples"])
        self.overlap_samples = int(self.manifest["overlap_samples"])
        self.output_dir = Path(output_dir or self.review_dir / "corrected_stems").expanduser().resolve()
        self.initial_swaps = [bool(chunk["initial_swap"]) for chunk in self.chunks]
        # False means retain the online decision (green); True means invert it (red).
        self.flips = [False] * len(self.chunks)
        self.restored_edits = self._load_saved_edits()

    def _load_saved_edits(self) -> bool:
        """Restore prior human decisions when they were saved for this output directory."""
        edits_path = self.output_dir / "alignment_edits.json"
        if not edits_path.exists():
            return False
        try:
            edits = json.loads(edits_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Saved alignment edits are not valid JSON: {edits_path}") from exc
        flips = edits.get("online_alignment_flips")
        if (
            edits.get("format_version") != 1
            or not isinstance(flips, list)
            or len(flips) != len(self.chunks)
            or not all(isinstance(value, bool) for value in flips)
        ):
            raise ValueError(f"Saved alignment edits are incompatible with this review bundle: {edits_path}")
        self.flips = flips.copy()
        return True

    def _source(self, index: int, source: int) -> np.ndarray:
        samples, sample_rate = sf.read(
            self.review_dir / self.chunks[index][f"source_{source:02d}"],
            dtype="float32",
            always_2d=True,
        )
        if sample_rate != self.sample_rate:
            raise ValueError(f"Chunk {index} has {sample_rate} Hz, expected {self.sample_rate} Hz")
        return samples[:, 0]

    def render(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the two tracks for the current online-relative flips."""
        tracks = np.zeros((2, self.total_samples), dtype=np.float32)
        weights = np.zeros(self.total_samples, dtype=np.float32)
        for index, chunk in enumerate(self.chunks):
            start, end = int(chunk["start_sample"]), int(chunk["end_sample"])
            window = _window(end - start, self.overlap_samples, index == 0, index == len(self.chunks) - 1)
            pair = np.stack((self._source(index, 1), self._source(index, 2)))
            if self.initial_swaps[index] ^ self.flips[index]:
                pair = pair[::-1]
            tracks[:, start:end] += pair * window
            weights[start:end] += window
        tracks /= np.maximum(weights, 1e-6)
        return tracks[0], tracks[1]

    def save(self) -> tuple[Path, Path]:
        """Write corrected WAVs and a compact record of the human decisions."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        track_1, track_2 = self.render()
        path_1 = self.output_dir / "spk1_corrected.wav"
        path_2 = self.output_dir / "spk2_corrected.wav"
        sf.write(path_1, track_1, self.sample_rate, subtype="FLOAT")
        sf.write(path_2, track_2, self.sample_rate, subtype="FLOAT")
        (self.output_dir / "alignment_edits.json").write_text(
            json.dumps({
                "format_version": 1,
                "manifest": str((self.review_dir / "alignment_manifest.json").resolve()),
                "online_alignment_flips": self.flips,
                "effective_raw_swaps": [a ^ b for a, b in zip(self.initial_swaps, self.flips)],
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        return path_1, path_2

    def finish(self) -> tuple[Path, Path]:
        """Save corrections and replace this run's final singer stems."""
        corrected_1, corrected_2 = self.save()
        final_dir = self.review_dir.parent / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        final_1 = final_dir / "03_singer_01.wav"
        final_2 = final_dir / "04_singer_02.wav"
        shutil.copy2(corrected_1, final_1)
        shutil.copy2(corrected_2, final_2)
        return final_1, final_2

    def widget(self) -> "AlignmentReviewWidget":
        return AlignmentReviewWidget(self)


class AlignmentReviewWidget(anywidget.AnyWidget):
    """Browser-owned reviewer UI; the kernel only renders and saves audio."""

    _esm = Path(__file__).with_name("alignment_review_widget.js")
    flips = traitlets.List(trait=traitlets.Bool(), default_value=[]).tag(sync=True)
    chunk_starts = traitlets.List(trait=traitlets.Float(), default_value=[]).tag(sync=True)
    duration = traitlets.Float(0.0).tag(sync=True)
    status = traitlets.Unicode("").tag(sync=True)

    def __init__(self, review: ChunkAlignmentReview) -> None:
        super().__init__()
        self.review = review
        self.flips = review.flips.copy()
        self.chunk_starts = [int(chunk["start_sample"]) / review.sample_rate for chunk in review.chunks]
        self.duration = review.total_samples / review.sample_rate
        if review.restored_edits:
            self.status = "Restored the previously saved alignment edits."
        self.on_msg(self._handle_message)

    def _apply_flips(self, values: object) -> None:
        if not isinstance(values, list) or len(values) != len(self.review.flips) or not all(isinstance(v, bool) for v in values):
            raise ValueError("Invalid alignment flip list received from widget")
        self.review.flips = values.copy()
        self.flips = values.copy()

    def _audio_wav_bytes(self, samples: np.ndarray) -> bytes:
        """Encode compact playback audio for an AnyWidget binary message."""
        buffer = io.BytesIO()
        # PCM16 is sufficient for alignment audition and halves the payload
        # compared with the float review files.  Binary widget buffers avoid
        # the additional base64 expansion that can freeze Colab.
        sf.write(buffer, samples, self.review.sample_rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue()

    def _render_audio(self, command: dict[str, object]) -> None:
        track_1, track_2 = self.review.render()
        if command["action"] == "play_selected":
            identity = int(command["identity"])
            if identity not in (0, 1):
                raise ValueError("Selected identity must be 0 or 1")
            buffers = [self._audio_wav_bytes((track_1, track_2)[identity])]
        else:
            buffers = [self._audio_wav_bytes(track_1), self._audio_wav_bytes(track_2)]
        self.send({"type": "audio", "command": command}, buffers=buffers)

    def _handle_message(self, _widget: object, content: object, _buffers: object) -> None:
        if not isinstance(content, dict):
            return
        try:
            self._apply_flips(content.get("flips", self.flips))
            action = content.get("action")
            if action in {"play_selected", "play_both"}:
                self._render_audio({
                    "action": action,
                    "second": float(content.get("second", 0.0)),
                    "identity": int(content.get("identity", 0)),
                })
                self.status = ""
            elif action == "save":
                path_1, path_2 = self.review.save()
                self.status = f"Saved {path_1.name}, {path_2.name}, and alignment_edits.json"
            elif action == "finish":
                path_1, path_2 = self.review.finish()
                self.status = f"Alignment finished: updated {path_1} and {path_2}"
        except (TypeError, ValueError) as exc:
            self.status = f"Widget request rejected: {exc}"


def open_alignment_review(review_dir: str | Path, output_dir: str | Path | None = None) -> AlignmentReviewWidget:
    """Create the interactive review widget for a review-bundle directory."""
    return ChunkAlignmentReview(review_dir, output_dir).widget()
