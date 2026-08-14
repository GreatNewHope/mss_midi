#!/usr/bin/env python3
"""Remove UNMIXX's private editable tssep checkout from requirements.txt."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


TSSEP_EDITABLE = re.compile(r"^\s*-e\s+\S*tssep\S*\s*$", re.IGNORECASE)
TSSEP_COMMENT = re.compile(r"^\s*#.*\btssep\b.*$", re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requirements", type=Path)
    args = parser.parse_args()

    original = args.requirements.read_text(encoding="utf-8")
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
            f"No editable tssep requirement found in {args.requirements}. "
            "Review the current upstream requirements before changing this patcher."
        )
    args.requirements.write_text("".join(patched), encoding="utf-8")
    print(f"Removed UNMIXX's private editable tssep requirement from {args.requirements}.")


if __name__ == "__main__":
    main()
