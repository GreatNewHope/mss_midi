#!/usr/bin/env python3
"""Download the newest Python-inference-compatible official GAME Large model."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


RELEASES_URL = "https://api.github.com/repos/openvpi/GAME/releases"
MODEL_SUFFIXES = {".ckpt", ".pt", ".pth"}
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")


def release_metadata() -> list[dict]:
    request = urllib.request.Request(
        RELEASES_URL,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "choir-separator-project"},
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def select_large_model_asset(releases: list[dict]) -> tuple[dict, dict]:
    for release in releases:
        candidates = []
        for asset in release.get("assets", []):
            name = asset["name"].lower()
            if name.startswith("source code") or not name.endswith(ARCHIVE_SUFFIXES + tuple(MODEL_SUFFIXES)):
                continue
            candidates.append(asset)
        if candidates:
            # The official release names model sizes; prefer Large, then the
            # largest bundle rather than depending on an upstream filename.
            return release, max(
                candidates,
                key=lambda asset: ("large" in asset["name"].lower(), asset.get("size", 0)),
            )
    raise RuntimeError(
        "No GAME release exposes a Python-inference-compatible model bundle. "
        "The current release may contain ONNX-only assets, which upstream does not support from infer.py."
    )


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "choir-separator-project"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def safe_extract(archive: Path, destination: Path) -> None:
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            if any(Path(member.filename).is_absolute() or ".." in Path(member.filename).parts for member in members):
                raise RuntimeError("GAME model archive contains an unsafe path.")
            bundle.extractall(destination)
        return

    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        if any(Path(member.name).is_absolute() or ".." in Path(member.name).parts for member in members):
            raise RuntimeError("GAME model archive contains an unsafe path.")
        bundle.extractall(destination, members=members)


def find_model(model_dir: Path) -> Path:
    candidates = sorted(
        path
        for path in model_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES and (path.parent / "config.yaml").is_file()
    )
    if not candidates:
        raise RuntimeError(
            "The downloaded GAME bundle did not contain a checkpoint with its required config.yaml. "
            "Inspect the current release and update this downloader."
        )
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--upgrade", action="store_true", help="Replace an older downloaded release.")
    args = parser.parse_args()

    release, asset = select_large_model_asset(release_metadata())
    tag = release["tag_name"]
    stamp_path = args.output_dir / "release.json"
    model_path_file = args.output_dir / "model_path.txt"
    if not args.upgrade and model_path_file.is_file():
        print(f"Using existing GAME model: {model_path_file.read_text(encoding='utf-8').strip()}")
        return
    if args.upgrade and stamp_path.is_file() and model_path_file.is_file():
        installed = json.loads(stamp_path.read_text(encoding="utf-8"))
        model_path = Path(model_path_file.read_text(encoding="utf-8").strip())
        if installed.get("tag_name") == tag and model_path.is_file():
            print(f"GAME model is already current ({tag}).")
            return

    with tempfile.TemporaryDirectory(prefix="game-model-") as temporary_directory:
        temporary_directory = Path(temporary_directory)
        archive = temporary_directory / asset["name"]
        print(f"Downloading GAME {tag} model asset: {asset['name']}")
        download(asset["browser_download_url"], archive)
        extracted = temporary_directory / "extracted"
        extracted.mkdir()
        if archive.name.lower().endswith(ARCHIVE_SUFFIXES):
            safe_extract(archive, extracted)
        else:
            shutil.copy2(archive, extracted / archive.name)

        new_model = find_model(extracted)
        if args.output_dir.exists():
            shutil.rmtree(args.output_dir)
        shutil.move(str(extracted), str(args.output_dir))
        installed_model = args.output_dir / new_model.relative_to(extracted)
        model_path_file.write_text(str(installed_model.resolve()) + "\n", encoding="utf-8")
        stamp_path.write_text(
            json.dumps(
                {
                    "tag_name": tag,
                    "asset_name": asset["name"],
                    "asset_url": asset["browser_download_url"],
                    "model_path": str(installed_model.resolve()),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"Installed GAME model: {installed_model}")


if __name__ == "__main__":
    main()
