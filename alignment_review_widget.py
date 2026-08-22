"""Notebook UI for correcting UNMIXX chunk source permutations.

Create a review bundle during inference with ``--alignment-review-dir`` and
then call :func:`open_alignment_review` from a Jupyter notebook.  The bundle
contains raw model estimates rather than the already overlap-added output, so
each human change is losslessly re-rendered with the original crossfades.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def _window(length: int, overlap: int, first: bool, last: bool) -> np.ndarray:
    window = np.ones(length, dtype=np.float32)
    n = min(length, overlap)
    if n > 1 and not first:
        window[:n] = np.linspace(0.0, 1.0, n, dtype=np.float32)
    if n > 1 and not last:
        window[-n:] = np.linspace(1.0, 0.0, n, dtype=np.float32)
    return window


class ChunkAlignmentReview:
    """Stateful notebook reviewer for one ``alignment_manifest.json`` bundle."""

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
        # A selection is relative to the model's native source order, not to a
        # preceding chunk. This makes every correction independent and reversible.
        self.swaps = [bool(chunk["initial_swap"]) for chunk in self.chunks]

    def _source(self, index: int, source: int) -> np.ndarray:
        key = f"source_{source:02d}"
        samples, sample_rate = sf.read(self.review_dir / self.chunks[index][key], dtype="float32", always_2d=True)
        if sample_rate != self.sample_rate:
            raise ValueError(f"Chunk {index} has {sample_rate} Hz, expected {self.sample_rate} Hz")
        return samples[:, 0]

    def render(self) -> tuple[np.ndarray, np.ndarray]:
        """Return both currently selected, overlap-added tracks as mono arrays."""
        tracks = np.zeros((2, self.total_samples), dtype=np.float32)
        weights = np.zeros(self.total_samples, dtype=np.float32)
        for index, chunk in enumerate(self.chunks):
            start, end = int(chunk["start_sample"]), int(chunk["end_sample"])
            first, last = index == 0, index == len(self.chunks) - 1
            window = _window(end - start, self.overlap_samples, first, last)
            pair = np.stack((self._source(index, 1), self._source(index, 2)))
            if self.swaps[index]:
                pair = pair[::-1]
            tracks[:, start:end] += pair * window
            weights[start:end] += window
        tracks /= np.maximum(weights, 1e-6)
        return tracks[0], tracks[1]

    def save(self) -> tuple[Path, Path]:
        """Write corrected stems and the editable decisions beside them."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        track_1, track_2 = self.render()
        path_1, path_2 = self.output_dir / "spk1_corrected.wav", self.output_dir / "spk2_corrected.wav"
        sf.write(path_1, track_1, self.sample_rate, subtype="FLOAT")
        sf.write(path_2, track_2, self.sample_rate, subtype="FLOAT")
        decisions = {
            "format_version": 1,
            "manifest": str((self.review_dir / "alignment_manifest.json").resolve()),
            "swaps": self.swaps,
        }
        (self.output_dir / "alignment_edits.json").write_text(json.dumps(decisions, indent=2) + "\n", encoding="utf-8")
        return path_1, path_2

    def widget(self):
        """Build and return the interactive ipywidgets view."""
        try:
            import ipywidgets as widgets
            from IPython.display import Audio, display
        except ImportError as exc:  # pragma: no cover - depends on notebook environment
            raise ImportError("Install the alignment-review dependency group: uv sync --group alignment-review") from exc

        chunk = widgets.IntSlider(value=0, min=0, max=len(self.chunks) - 1, description="Chunk", continuous_update=False)
        decision = widgets.ToggleButtons(options=[("Keep: raw 1 → track 1", False), ("Swap: raw 2 → track 1", True)], description="Mapping")
        details = widgets.HTML()
        audio = widgets.Output()
        status = widgets.HTML()
        preview_chunk = widgets.Button(description="Play selected chunk", icon="play")
        preview_full = widgets.Button(description="Play current full tracks", icon="play")
        save = widgets.Button(description="Save corrected stems", button_style="success", icon="save")

        def refresh(*_ignored: object) -> None:
            item = self.chunks[chunk.value]
            decision.value = self.swaps[chunk.value]
            start, end = int(item["start_sample"]) / self.sample_rate, int(item["end_sample"]) / self.sample_rate
            scores = "—" if item["keep_score"] is None else f"keep {item['keep_score']:+.3f}; swap {item['swap_score']:+.3f}"
            details.value = (
                f"<b>{start:.2f}–{end:.2f}s</b> · online: <b>{'swap' if item['initial_swap'] else 'keep'}</b> "
                f"via {item['assignment']} · scores: {scores}"
            )

        def choose(change: dict[str, Any]) -> None:
            if change["name"] == "value":
                self.swaps[chunk.value] = bool(change["new"])
                status.value = "<i>Unsaved change.</i>"

        def play_selected(_button: object) -> None:
            pair = (self._source(chunk.value, 1), self._source(chunk.value, 2))
            if self.swaps[chunk.value]:
                pair = pair[::-1]
            with audio:
                audio.clear_output(wait=True)
                display(Audio(pair[0], rate=self.sample_rate), Audio(pair[1], rate=self.sample_rate))

        def play_full(_button: object) -> None:
            track_1, track_2 = self.render()
            with audio:
                audio.clear_output(wait=True)
                display(Audio(track_1, rate=self.sample_rate), Audio(track_2, rate=self.sample_rate))

        def save_stems(_button: object) -> None:
            path_1, path_2 = self.save()
            status.value = f"<b>Saved:</b> {path_1.name}, {path_2.name}, and alignment_edits.json"

        chunk.observe(refresh, names="value")
        decision.observe(choose, names="value")
        preview_chunk.on_click(play_selected)
        preview_full.on_click(play_full)
        save.on_click(save_stems)
        refresh()
        return widgets.VBox([
            widgets.HTML("<h3>UNMIXX chunk alignment review</h3><p>Each audio player is track 1 then track 2.</p>"),
            chunk,
            decision,
            details,
            widgets.HBox([preview_chunk, preview_full, save]),
            status,
            audio,
        ])


def open_alignment_review(review_dir: str | Path, output_dir: str | Path | None = None):
    """Return a notebook-ready chunk-alignment widget for ``review_dir``."""
    return ChunkAlignmentReview(review_dir, output_dir).widget()
