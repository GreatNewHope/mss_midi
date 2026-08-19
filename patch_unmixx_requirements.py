#!/usr/bin/env python3
"""Apply the small local changes needed for UNMIXX inference."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


TSSEP_EDITABLE = re.compile(r"^\s*-e\s+\S*tssep\S*\s*$", re.IGNORECASE)
TSSEP_COMMENT = re.compile(r"^\s*#.*\btssep\b.*$", re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("unmixx_repo", type=Path)
    args = parser.parse_args()

    requirements = args.unmixx_repo / "requirements.txt"
    model_file = args.unmixx_repo / "look2hear" / "models" / "unmixx_model.py"
    utils_init_file = args.unmixx_repo / "look2hear" / "utils" / "__init__.py"
    if not requirements.is_file() or not model_file.is_file() or not utils_init_file.is_file():
        raise FileNotFoundError(f"Expected an UNMIXX repository, got: {args.unmixx_repo}")

    original = requirements.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    patched: list[str] = []
    removed = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.rstrip("\r\n")
        if TSSEP_EDITABLE.match(stripped):
            removed += 1
            index += 1
            continue
        # Remove only an adjacent explanatory comment; preserve other upstream
        # comments that happen to mention tssep.
        if TSSEP_COMMENT.match(stripped) and index + 1 < len(lines):
            next_line = lines[index + 1].rstrip("\r\n")
            if TSSEP_EDITABLE.match(next_line):
                removed += 1
                index += 2
                continue
        patched.append(line)
        index += 1

    if removed == 0:
        raise RuntimeError(
            f"No editable tssep requirement found in {requirements}. "
            "Review the current upstream requirements before changing this patcher."
        )
    requirements.write_text("".join(patched), encoding="utf-8")
    print(f"Removed UNMIXX's private editable tssep requirement from {requirements}.")

    original_model = model_file.read_text(encoding="utf-8")
    upstream_import = "from asteroid.utils.torch_utils import pad_x_to_y\n"
    local_implementation = '''def pad_x_to_y(x, y):
    """Match UNMIXX's final estimate length without importing Asteroid."""
    target_length = y.shape[-1]
    if x.shape[-1] > target_length:
        return x[..., :target_length]
    if x.shape[-1] < target_length:
        return F.pad(x, (0, target_length - x.shape[-1]))
    return x

'''
    if upstream_import in original_model:
        model_file.write_text(
            original_model.replace(upstream_import, local_implementation, 1),
            encoding="utf-8",
        )
        print(f"Replaced UNMIXX's Asteroid-only pad helper in {model_file}.")
    elif local_implementation in original_model:
        print("UNMIXX's local pad helper is already installed.")
    else:
        raise RuntimeError(
            f"UNMIXX no longer has the expected Asteroid pad helper import in {model_file}. "
            "Review this local inference patch."
        )

    upstream_lightning_import = (
        "from .lightning_utils import print_only, RichProgressBarTheme, "
        "MyRichProgressBar, BatchesProcessedColumn, MyMetricsTextColumn\n"
    )
    local_lightning_comment = (
        "# Training-only PyTorch Lightning utilities are intentionally not imported "
        "for inference.\n"
    )
    original_utils_init = utils_init_file.read_text(encoding="utf-8")
    if upstream_lightning_import in original_utils_init:
        utils_init_file.write_text(
            original_utils_init.replace(
                upstream_lightning_import, local_lightning_comment, 1
            ),
            encoding="utf-8",
        )
        print(f"Skipped UNMIXX's training-only Lightning utilities in {utils_init_file}.")
    elif local_lightning_comment in original_utils_init:
        print("UNMIXX's training-only Lightning utilities are already skipped.")
    else:
        raise RuntimeError(
            f"UNMIXX no longer has the expected Lightning import in {utils_init_file}. "
            "Review this local inference patch."
        )


if __name__ == "__main__":
    main()
