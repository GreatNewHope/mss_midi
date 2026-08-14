#!/usr/bin/env python3
"""Run the official GAME singing-MIDI extractor as a standalone pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


MODEL_SUFFIXES = {".ckpt", ".pt", ".pth"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="An isolated singing stem or a directory of stems.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--game-repo", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--language", help="Optional GAME language code, e.g. en, ja, yue, zh.")
    parser.add_argument("--glob", help="Optional file glob when the input is a directory, e.g. 0[2-6]_*.wav.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tempo", type=float, default=120.0)
    return parser.parse_args()


def model_path(model_dir: Path) -> Path:
    recorded_path = model_dir / "model_path.txt"
    if recorded_path.is_file():
        candidate = Path(recorded_path.read_text(encoding="utf-8").strip())
        if candidate.is_file() and (candidate.parent / "config.yaml").is_file():
            return candidate
    candidates = sorted(
        path
        for path in model_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES and (path.parent / "config.yaml").is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No usable GAME checkpoint found in {model_dir}. Run `make prepare` to download one."
        )
    return candidates[0]


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    game_repo = args.game_repo.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    if not (game_repo / "infer.py").is_file():
        raise FileNotFoundError(f"GAME inference script not found in: {game_repo}")

    checkpoint = model_path(args.model_dir.expanduser().resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "infer.py",
        "extract",
        str(input_path),
        "--model",
        str(checkpoint),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--output-formats",
        "mid",
        "--batch-size",
        str(args.batch_size),
        "--tempo",
        str(args.tempo),
    ]
    if args.language:
        command.extend(["--language", args.language])
    if args.glob:
        command.extend(["--glob", args.glob])
    subprocess.run(command, cwd=game_repo, check=True)

    outputs = sorted(str(path) for path in args.output_dir.rglob("*.mid"))
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "pipeline": "GAME",
                "game_repository": str(game_repo),
                "checkpoint": str(checkpoint),
                "input": str(input_path),
                "language": args.language,
                "glob": args.glob,
                "tempo": args.tempo,
                "outputs": outputs,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(outputs)} GAME MIDI file(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
