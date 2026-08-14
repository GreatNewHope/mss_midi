#!/usr/bin/env python3
"""Transcribe a non-separated choir mix to polyphonic MIDI with Basic Pitch.

This is intentionally independent of the audio-separation pipelines. It is the
polyphonic fallback; use ``game_to_midi.py`` for isolated lead or SATB stems.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


SUPPORTED_AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".wav",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create polyphonic MIDI files from a non-separated choir mix."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Audio file(s) and/or directories to transcribe recursively.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated .mid files and manifest.json.",
    )
    parser.add_argument("--onset-threshold", type=float, default=0.5)
    parser.add_argument("--frame-threshold", type=float, default=0.3)
    parser.add_argument("--minimum-note-length-ms", type=float, default=127.7)
    parser.add_argument("--minimum-frequency", type=float)
    parser.add_argument("--maximum-frequency", type=float)
    parser.add_argument("--tempo", type=float, default=120.0)
    parser.add_argument(
        "--pitch-bends",
        action="store_true",
        help="Include Basic Pitch pitch-bend events in the MIDI output.",
    )
    parser.add_argument(
        "--no-melodia-trick",
        action="store_true",
        help="Disable Basic Pitch's additional note search after primary notes.",
    )
    return parser.parse_args()


def audio_files(path: Path) -> Iterable[tuple[Path, Path]]:
    """Yield ``(audio_file, relative_output_path)`` pairs for one input."""
    path = path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported audio file extension: {path}")
        yield path, Path(path.name)
        return

    if not path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {path}")

    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No supported audio files found in: {path}")
    for candidate in files:
        yield candidate, candidate.relative_to(path)


def main() -> None:
    args = parse_args()

    jobs: list[tuple[Path, Path]] = []
    for input_path in args.inputs:
        jobs.extend(audio_files(input_path))

    duplicate_outputs: set[Path] = set()
    seen_outputs: set[Path] = set()
    for _, relative_path in jobs:
        midi_path = relative_path.with_suffix(".mid")
        if midi_path in seen_outputs:
            duplicate_outputs.add(midi_path)
        seen_outputs.add(midi_path)
    if duplicate_outputs:
        names = ", ".join(str(path) for path in sorted(duplicate_outputs))
        raise ValueError(f"Inputs would overwrite the same MIDI output: {names}")

    # Import and load once so a directory of stems does not reload the model for
    # every file. Basic Pitch selects its available native runtime automatically.
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model, predict

    basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_outputs = []
    for audio_path, relative_path in jobs:
        midi_path = args.output_dir / relative_path.with_suffix(".mid")
        midi_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Transcribing {audio_path} -> {midi_path}")
        _, midi_data, note_events = predict(
            str(audio_path),
            basic_pitch_model,
            onset_threshold=args.onset_threshold,
            frame_threshold=args.frame_threshold,
            minimum_note_length=args.minimum_note_length_ms,
            minimum_frequency=args.minimum_frequency,
            maximum_frequency=args.maximum_frequency,
            multiple_pitch_bends=args.pitch_bends,
            melodia_trick=not args.no_melodia_trick,
            midi_tempo=args.tempo,
        )
        midi_data.write(str(midi_path))
        manifest_outputs.append(
            {
                "input": str(audio_path),
                "output": str(midi_path),
                "note_count": len(note_events),
            }
        )

    manifest = {
        "pipeline": "basic-pitch",
        "model": "ICASSP 2022 Basic Pitch model distributed with basic-pitch",
        "parameters": {
            "onset_threshold": args.onset_threshold,
            "frame_threshold": args.frame_threshold,
            "minimum_note_length_ms": args.minimum_note_length_ms,
            "minimum_frequency": args.minimum_frequency,
            "maximum_frequency": args.maximum_frequency,
            "pitch_bends": args.pitch_bends,
            "melodia_trick": not args.no_melodia_trick,
            "tempo": args.tempo,
        },
        "outputs": manifest_outputs,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest_outputs)} MIDI file(s) and {manifest_path}")


if __name__ == "__main__":
    main()
