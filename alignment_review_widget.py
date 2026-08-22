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
        # The UI edits are relative to the online alignment, rather than the
        # model's native source order. Thus every box starts green even when
        # the online process itself used a raw-source swap.
        self.initial_swaps = [bool(chunk["initial_swap"]) for chunk in self.chunks]
        self.flips = [False] * len(self.chunks)

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
            if self.initial_swaps[index] ^ self.flips[index]:
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
            "online_alignment_flips": self.flips,
            "effective_raw_swaps": [initial ^ flip for initial, flip in zip(self.initial_swaps, self.flips)],
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

        duration = self.total_samples / self.sample_rate
        second = widgets.FloatSlider(
            value=0.0,
            min=0.0,
            max=duration,
            step=0.1,
            description="Second",
            continuous_update=False,
            readout_format=".1f",
        )
        identity = widgets.ToggleButtons(
            options=[("Identity 1 / stem 1", 0), ("Identity 2 / stem 2", 1)],
            description="Listen to",
        )
        details = widgets.HTML()
        audio = widgets.Output()
        status = widgets.HTML()
        preview_at_second = widgets.Button(description="Play 12 s at selected second", icon="play")
        preview_full = widgets.Button(description="Play complete selected stem", icon="play")
        preview_both = widgets.Button(description="Play both complete stems", icon="play")
        save = widgets.Button(description="Save corrected stems", button_style="success", icon="save")
        boxes: list[Any] = []

        def refresh() -> None:
            red = sum(self.flips)
            details.value = (
                f"<span style='color:#188038'>green</span> = online alignment; "
                f"<span style='color:#d93025'>red</span> = swap it. "
                f"{red} of {len(self.chunks)} chunks inverted."
            )
            for index, box in enumerate(boxes):
                box.button_style = "danger" if self.flips[index] else "success"
                box.tooltip = (
                    f"Chunk {index + 1}: {'swap online alignment' if self.flips[index] else 'keep online alignment'}"
                )

        def toggle(index: int) -> None:
            self.flips[index] = not self.flips[index]
            refresh()
            status.value = "<i>Unsaved change.</i>"

        for index in range(len(self.chunks)):
            box = widgets.Button(description=str(index + 1), layout=widgets.Layout(width="38px", height="30px"))
            box.on_click(lambda _button, index=index: toggle(index))
            boxes.append(box)
        alignment_map = widgets.GridBox(
            boxes,
            layout=widgets.Layout(grid_template_columns="repeat(auto-fill, 38px)", grid_gap="4px"),
        )

        def selected_track() -> np.ndarray:
            return self.render()[int(identity.value)]

        def play_at_second(_button: object) -> None:
            track = selected_track()
            start = int(second.value * self.sample_rate)
            end = min(len(track), start + 12 * self.sample_rate)
            status.value = f"Playing identity {int(identity.value) + 1}, {start / self.sample_rate:.1f}–{end / self.sample_rate:.1f}s."
            with audio:
                audio.clear_output(wait=True)
                display(Audio(track[start:end], rate=self.sample_rate))

        def play_full(_button: object) -> None:
            track = selected_track()
            status.value = f"Playing complete identity {int(identity.value) + 1}."
            with audio:
                audio.clear_output(wait=True)
                display(Audio(track, rate=self.sample_rate))

        def play_both(_button: object) -> None:
            track_1, track_2 = self.render()
            status.value = "Playing both complete stems as separate players."
            with audio:
                audio.clear_output(wait=True)
                display(Audio(track_1, rate=self.sample_rate), Audio(track_2, rate=self.sample_rate))

        def save_stems(_button: object) -> None:
            path_1, path_2 = self.save()
            status.value = f"<b>Saved:</b> {path_1.name}, {path_2.name}, and alignment_edits.json"

        preview_at_second.on_click(play_at_second)
        preview_full.on_click(play_full)
        preview_both.on_click(play_both)
        save.on_click(save_stems)
        refresh()
        return widgets.VBox([
            widgets.HTML("<h3>UNMIXX chunk alignment review</h3><p>Click a chunk box to invert its online assignment.</p>"),
            alignment_map,
            details,
            identity,
            second,
            widgets.HBox([preview_at_second, preview_full, preview_both, save]),
            status,
            audio,
        ])


def open_alignment_review(review_dir: str | Path, output_dir: str | Path | None = None):
    """Return a notebook-ready chunk-alignment widget for ``review_dir``."""
    return ChunkAlignmentReview(review_dir, output_dir).widget()
