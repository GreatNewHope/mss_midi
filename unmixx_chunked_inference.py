#!/usr/bin/env python3
"""Memory-safe long-form inference for the official UNMIXX checkpoint.

UNMIXX was trained on short segments (4 s in the public config), while the
reference inference script forwards the whole file at once. This runner keeps
one model loaded, processes overlapping chunks, tracks the two-source
permutation between neighbouring chunks, overlap-adds the results, and skips
model inference for chunks that are below a configurable silence threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import torch
import torchaudio
import yaml
from huggingface_hub import hf_hub_download

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


def is_silent(waveform: torch.Tensor, peak_threshold: float) -> bool:
    """Return whether every sample is at or below the configured peak limit."""
    return bool(waveform.abs().amax() <= peak_threshold)


class SingerIdentityTracker:
    """Assign UNMIXX's permutation-ambiguous outputs to persistent tracks.

    The BYOL encoder was trained for singer identity, independently from
    UNMIXX.  Its TorchScript export keeps this runner independent of the
    original research repository and its training-time dependencies.
    """

    REPO_ID = "BernardoTorres/singer-identity"

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        input_sr: int,
        min_peak: float,
        min_margin: float,
    ) -> None:
        model_path = hf_hub_download(
            repo_id=self.REPO_ID,
            filename=f"{model_name}/model.ts",
        )
        self.model = torch.jit.load(model_path, map_location=device).eval()
        self.device = device
        self.input_sr = input_sr
        self.min_peak = min_peak
        self.min_margin = min_margin
        self.prototypes: list[torch.Tensor | None] = [None, None]
        self.counts = [0, 0]

    def embeddings(self, estimates: torch.Tensor) -> list[torch.Tensor | None]:
        """Return one normalized singer embedding per sufficiently voiced stem."""
        valid = estimates.abs().amax(dim=1) > self.min_peak
        if not bool(valid.any()):
            return [None, None]

        audio = estimates[valid]
        if self.input_sr != 44_100:
            audio = torchaudio.functional.resample(audio, self.input_sr, 44_100)
        with torch.inference_mode():
            vectors = self.model(audio.to(self.device, non_blocking=True)).float().cpu()
        vectors = torch.nn.functional.normalize(vectors, dim=1)

        result: list[torch.Tensor | None] = [None, None]
        for source_index, vector in zip(valid.nonzero(as_tuple=False).flatten().tolist(), vectors):
            result[source_index] = vector
        return result

    def decide(self, vectors: list[torch.Tensor | None]) -> tuple[bool, float, float] | None:
        """Return (swap, keep score, swap score), if both tracks have evidence."""
        if any(vector is None for vector in vectors) or any(proto is None for proto in self.prototypes):
            return None
        assert vectors[0] is not None and vectors[1] is not None
        assert self.prototypes[0] is not None and self.prototypes[1] is not None
        keep = float(torch.dot(self.prototypes[0], vectors[0]) + torch.dot(self.prototypes[1], vectors[1]))
        swap = float(torch.dot(self.prototypes[0], vectors[1]) + torch.dot(self.prototypes[1], vectors[0]))
        if abs(swap - keep) < self.min_margin:
            return None
        return swap > keep, keep, swap

    def update(self, vectors: list[torch.Tensor | None]) -> None:
        """Update only the prototype belonging to a voiced, assigned stem."""
        for track, vector in enumerate(vectors):
            if vector is None:
                continue
            prototype = self.prototypes[track]
            # A capped running mean avoids one early chunk dominating forever,
            # while retaining enough inertia to bridge a long silence.
            weight = 1.0 / min(self.counts[track] + 1, 8)
            if prototype is None:
                updated = vector
            else:
                updated = torch.nn.functional.normalize((1.0 - weight) * prototype + weight * vector, dim=0)
            self.prototypes[track] = updated
            self.counts[track] += 1


def identity_prototypes(
    raw_swaps: list[bool],
    vectors_by_chunk: list[list[torch.Tensor | None]],
) -> list[torch.Tensor | None]:
    """Build one normalized identity reference per final output track."""
    assigned: list[list[torch.Tensor]] = [[], []]
    for raw_swap, vectors in zip(raw_swaps, vectors_by_chunk):
        for source, vector in enumerate(vectors):
            if vector is not None:
                assigned[source ^ raw_swap].append(vector)

    result: list[torch.Tensor | None] = []
    for track_vectors in assigned:
        if not track_vectors:
            result.append(None)
            continue
        result.append(torch.nn.functional.normalize(torch.stack(track_vectors).mean(dim=0), dim=0))
    return result


def identity_assignment_score(
    raw_swap: bool,
    vectors: list[torch.Tensor | None],
    prototypes: list[torch.Tensor | None],
) -> float | None:
    """Score assigning a raw source pair to the two persistent identities."""
    if any(vector is None for vector in vectors) or any(proto is None for proto in prototypes):
        return None
    assert vectors[0] is not None and vectors[1] is not None
    assert prototypes[0] is not None and prototypes[1] is not None
    return float(
        torch.dot(prototypes[0], vectors[int(raw_swap)])
        + torch.dot(prototypes[1], vectors[1 - int(raw_swap)])
    )


def find_offline_identity_flips(
    online_swaps: list[bool],
    vectors_by_chunk: list[list[torch.Tensor | None]],
    min_margin: float,
    switch_penalty: float,
) -> tuple[list[bool], list[float | None]]:
    """Find contiguous online-relative corrections favored by global identity.

    The online alignment's permutation can be consistently wrong after a
    difficult boundary.  Here state 0 preserves its decision and state 1
    inverts it.  A Viterbi-style transition cost favors long correction runs
    over isolated source-order changes; uncertain chunks contribute no vote
    but can bridge a well-supported run.
    """
    if len(online_swaps) != len(vectors_by_chunk):
        raise ValueError("Online assignments and identity vectors must have the same length")

    candidates = online_swaps.copy()
    margins: list[float | None] = [None] * len(online_swaps)
    flips = [False] * len(online_swaps)
    # Re-estimating once after a coherent block is removed prevents that block
    # from contaminating the global reference identities.
    for _ in range(2):
        prototypes = identity_prototypes(candidates, vectors_by_chunk)
        emissions: list[float] = []
        iteration_margins: list[float | None] = []
        for online_swap, vectors in zip(online_swaps, vectors_by_chunk):
            keep = identity_assignment_score(online_swap, vectors, prototypes)
            swap = identity_assignment_score(not online_swap, vectors, prototypes)
            if keep is None or swap is None:
                emissions.append(0.0)
                iteration_margins.append(None)
                continue
            margin = swap - keep
            iteration_margins.append(margin)
            emissions.append(margin if abs(margin) >= min_margin else 0.0)

        # State scores include the cost of entering an inverted block.  The
        # terminal adjustment charges leaving one too, including at EOF.
        # A negligible per-chunk cost resolves otherwise arbitrary ties in
        # silent/uncertain stretches: it starts a block at its first evidence
        # rather than extending it backward through zero-evidence chunks.
        inverted_chunk_penalty = 1e-6
        scores = [0.0, -switch_penalty]
        parents: list[tuple[int, int]] = []
        for emission in emissions:
            next_keep = max(scores[0], scores[1] - switch_penalty)
            next_swap = max(scores[1], scores[0] - switch_penalty) + emission - inverted_chunk_penalty
            parents.append((
                0 if scores[0] >= scores[1] - switch_penalty else 1,
                1 if scores[1] >= scores[0] - switch_penalty else 0,
            ))
            scores = [next_keep, next_swap]

        state = 0 if scores[0] >= scores[1] - switch_penalty else 1
        proposed = [False] * len(online_swaps)
        for index in range(len(online_swaps) - 1, -1, -1):
            proposed[index] = bool(state)
            state = parents[index][state]
        corrected = [online_swap ^ flip for online_swap, flip in zip(online_swaps, proposed)]
        flips, margins = proposed, iteration_margins
        if corrected == candidates:
            break
        candidates = corrected

    # An online failure normally continues after the boundary that triggered
    # it.  Carry a strong run forward through low-evidence chunks, but stop at
    # the first confident BYOL vote for the online assignment.  This repairs a
    # weak tail without broadening a block backward into an earlier silence.
    for index, flip in enumerate(flips):
        if not flip or (index + 1 < len(flips) and flips[index + 1]):
            continue
        next_index = index + 1
        while next_index < len(flips):
            margin = margins[next_index]
            if margin is not None and margin <= -min_margin:
                break
            if margin is not None and margin >= min_margin:
                break
            flips[next_index] = True
            next_index += 1
    return flips, margins


def main() -> None:
    p = argparse.ArgumentParser(description="Chunked long-form inference for the official UNMIXX model")
    p.add_argument("--unmixx-repo", type=Path, required=True)
    p.add_argument("--conf-path", type=Path, required=True)
    p.add_argument("--ckpt-path", type=Path, required=True)
    p.add_argument("--audio-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--alignment-review-dir",
        type=Path,
        help="Optional directory for raw chunks and online alignment decisions for notebook review.",
    )
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--chunk-seconds", type=float, default=4.0)
    p.add_argument("--overlap-seconds", type=float, default=1.0)
    p.add_argument(
        "--identity-model",
        choices=("none", "byol", "contrastive", "contrastive-vc", "uniformity", "vicreg"),
        default="byol",
        help="Singer-identity encoder used after ambiguous boundaries (default: byol).",
    )
    p.add_argument(
        "--identity-min-peak-db",
        type=float,
        default=-45.0,
        help="Do not learn identity from an estimated stem below this peak level (default: -45 dBFS).",
    )
    p.add_argument(
        "--identity-min-margin",
        type=float,
        default=0.05,
        help="Minimum singer-identity keep/swap score difference required to decide an ambiguous boundary.",
    )
    p.add_argument(
        "--identity-global-refine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After online alignment, use BYOL over all chunks to repair confident "
            "contiguous identity-inconsistent runs (default: enabled)."
        ),
    )
    p.add_argument(
        "--identity-global-min-margin",
        type=float,
        default=0.02,
        help="Minimum global BYOL swap/keep margin for a chunk to vote for an offline correction (default: 0.02).",
    )
    p.add_argument(
        "--identity-global-switch-penalty",
        type=float,
        default=0.10,
        help="Penalty for entering or leaving an offline correction run (default: 0.10).",
    )
    p.add_argument(
        "--silence-threshold-db",
        type=float,
        default=-80.0,
        help=(
            "Skip UNMIXX for chunks whose peak is at or below this dBFS value "
            "(default: -80). Use -inf to skip only exact digital silence."
        ),
    )
    args = p.parse_args()

    if args.chunk_seconds <= 0:
        p.error("--chunk-seconds must be > 0")
    if args.overlap_seconds < 0 or args.overlap_seconds >= args.chunk_seconds:
        p.error("--overlap-seconds must be >= 0 and < --chunk-seconds")
    if math.isnan(args.silence_threshold_db) or args.silence_threshold_db > 0:
        p.error("--silence-threshold-db must be <= 0 dBFS")
    if math.isnan(args.identity_min_peak_db) or args.identity_min_peak_db > 0:
        p.error("--identity-min-peak-db must be <= 0 dBFS")
    if args.identity_min_margin < 0:
        p.error("--identity-min-margin must be >= 0")
    if args.identity_global_min_margin < 0:
        p.error("--identity-global-min-margin must be >= 0")
    if args.identity_global_switch_penalty < 0:
        p.error("--identity-global-switch-penalty must be >= 0")

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
    silence_peak_threshold = 10.0 ** (args.silence_threshold_db / 20.0)
    identity_peak_threshold = 10.0 ** (args.identity_min_peak_db / 20.0)
    hop = chunk_samples - overlap_samples
    if hop <= 0:
        raise RuntimeError("Chunk overlap leaves a non-positive hop size")

    starts = list(range(0, total_samples, hop))
    if starts and starts[-1] + overlap_samples >= total_samples and len(starts) > 1:
        starts.pop()

    print(f"[UNMIXX chunked] device={device}, sr={sr}, duration={total_samples/sr:.2f}s")
    print(
        f"[UNMIXX chunked] chunk={args.chunk_seconds:.2f}s, "
        f"overlap={args.overlap_seconds:.2f}s, chunks={len(starts)}, "
        f"silence peak<={args.silence_threshold_db:g} dBFS"
    )
    tracker = None
    if args.identity_model != "none":
        tracker = SingerIdentityTracker(
            args.identity_model,
            device,
            sr,
            identity_peak_threshold,
            args.identity_min_margin,
        )
        print(f"[UNMIXX identity] model={args.identity_model}, min-peak={args.identity_min_peak_db:g} dBFS")

    accum = torch.zeros((2, total_samples), dtype=torch.float32)
    weights = torch.zeros(total_samples, dtype=torch.float32)
    prev_raw: torch.Tensor | None = None
    prev_was_voiced = False
    skipped_silent_chunks = 0
    review_records: list[dict[str, object]] = []
    review_dir = args.alignment_review_dir
    if review_dir is not None:
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / "chunks").mkdir(exist_ok=True)

    # The offline pass needs the raw UNMIXX pairs to re-render a corrected
    # block.  Reuse review artifacts when requested; otherwise retain only
    # temporary float WAVs so long recordings do not stay in RAM.
    offline_cache: tempfile.TemporaryDirectory[str] | None = None
    offline_chunk_dir: Path | None = None
    if tracker is not None and args.identity_global_refine:
        if review_dir is not None:
            offline_chunk_dir = review_dir / "chunks"
        else:
            offline_cache = tempfile.TemporaryDirectory(prefix="unmixx-offline-alignment-")
            offline_chunk_dir = Path(offline_cache.name)
    online_swaps: list[bool] = []
    raw_identity_vectors: list[list[torch.Tensor | None]] = []

    for idx, start in enumerate(starts):
        end = min(start + chunk_samples, total_samples)
        raw = waveform[:, start:end]
        actual_len = raw.shape[-1]

        if is_silent(raw, silence_peak_threshold):
            # Keeping both sources at zero preserves the original timeline and
            # lets overlap-add handle transitions to neighbouring voiced chunks.
            est = torch.zeros((2, actual_len), dtype=torch.float32)
            skipped_silent_chunks += 1
            skipped = True
        else:
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
            skipped = False

        # Keep native (pre-permutation) estimates so human edits can be
        # rendered without another model pass.
        raw_chunk = est.clone()
        if review_dir is not None:
            chunk_dir = review_dir / "chunks"
            save_wav(chunk_dir / f"chunk_{idx:04d}_source_01.wav", raw_chunk[0:1], sr)
            save_wav(chunk_dir / f"chunk_{idx:04d}_source_02.wav", raw_chunk[1:2], sr)
        if offline_chunk_dir is not None and (review_dir is None or offline_chunk_dir != review_dir / "chunks"):
            save_wav(offline_chunk_dir / f"chunk_{idx:04d}_source_01.wav", raw_chunk[0:1], sr)
            save_wav(offline_chunk_dir / f"chunk_{idx:04d}_source_02.wav", raw_chunk[1:2], sr)

        swapped = False
        assignment = "initial"
        keep_score: float | None = None
        swap_score: float | None = None
        identity_vectors: list[torch.Tensor | None] | None = None
        identity_assignment_trusted = False
        if skipped:
            print(
                f"  chunk {idx+1:03d}/{len(starts):03d} "
                f"{start/sr:7.2f}-{end/sr:7.2f}s silence=skip"
            )
        elif prev_raw is not None and prev_was_voiced and overlap_samples > 0:
            ov = min(overlap_samples, prev_raw.shape[-1], est.shape[-1])
            if ov > 0:
                previous, current = prev_raw[:, -ov:], est[:, :ov]
                _, overlap_swap, keep_score, swap_score = align_two_sources(previous, current)
                overlap_reliable = (
                    previous.abs().amax() > identity_peak_threshold
                    and current.abs().amax() > identity_peak_threshold
                    and abs(swap_score - keep_score) >= args.identity_min_margin
                )
                if tracker is None:
                    # Preserve the previous continuity-only behavior when
                    # identity tracking is explicitly disabled.
                    swapped = overlap_swap
                    assignment = "overlap"
                elif overlap_reliable:
                    swapped = overlap_swap
                    assignment = "overlap"
                    identity_assignment_trusted = True
                elif tracker is not None:
                    identity_vectors = tracker.embeddings(est)
                    identity_decision = tracker.decide(identity_vectors)
                    if identity_decision is not None:
                        swapped, keep_score, swap_score = identity_decision
                        assignment = "identity"
                        identity_assignment_trusted = True
                    else:
                        assignment = "ambiguous"
                else:
                    assignment = "ambiguous"
                if swapped:
                    est = est.flip(0)
                    if identity_vectors is not None:
                        identity_vectors.reverse()
                print(
                    f"  chunk {idx+1:03d}/{len(starts):03d} "
                    f"{start/sr:7.2f}-{end/sr:7.2f}s "
                    f"perm={'swap' if swapped else 'keep'} via={assignment} "
                    f"scores={keep_score:+.3f}/{swap_score:+.3f}"
                )
            else:
                print(f"  chunk {idx+1:03d}/{len(starts):03d} {start/sr:7.2f}-{end/sr:7.2f}s")
        else:
            if tracker is not None:
                identity_vectors = tracker.embeddings(est)
                identity_decision = tracker.decide(identity_vectors)
                if identity_decision is not None:
                    swapped, keep_score, swap_score = identity_decision
                    identity_assignment_trusted = True
                    if swapped:
                        est = est.flip(0)
                        identity_vectors.reverse()
                    print(
                        f"  chunk {idx+1:03d}/{len(starts):03d} "
                        f"{start/sr:7.2f}-{end/sr:7.2f}s "
                        f"perm={'swap' if swapped else 'keep'} via=identity "
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
        if tracker is not None and not skipped:
            if identity_vectors is None:
                identity_vectors = tracker.embeddings(est)
            if all(prototype is None for prototype in tracker.prototypes):
                # The first clearly voiced UNMIXX pair establishes arbitrary,
                # but thereafter persistent, track identities.
                identity_assignment_trusted = True
            if identity_assignment_trusted:
                tracker.update(identity_vectors)
        prev_raw = est
        prev_was_voiced = not skipped

        online_swaps.append(swapped)
        if tracker is None or skipped:
            raw_identity_vectors.append([None, None])
        else:
            # ``identity_vectors`` is ordered like the post-online-alignment
            # estimate at this point.  Return it to raw UNMIXX source order so
            # the offline solver can evaluate either assignment.
            assert identity_vectors is not None
            raw_vectors = identity_vectors.copy()
            if swapped:
                raw_vectors.reverse()
            raw_identity_vectors.append(raw_vectors)

        if review_dir is not None:
            review_records.append({
                "index": idx,
                "start_sample": start,
                "end_sample": end,
                "source_01": f"chunks/chunk_{idx:04d}_source_01.wav",
                "source_02": f"chunks/chunk_{idx:04d}_source_02.wav",
                "initial_swap": swapped,
                "online_swap": swapped,
                "assignment": assignment,
                "keep_score": keep_score,
                "swap_score": swap_score,
                "skipped_silence": skipped,
            })

        if device.type == "cuda" and (idx + 1) % 16 == 0:
            torch.cuda.empty_cache()

    offline_flips = [False] * len(online_swaps)
    offline_margins: list[float | None] = [None] * len(online_swaps)
    final_swaps = online_swaps.copy()
    if tracker is not None and args.identity_global_refine:
        offline_flips, offline_margins = find_offline_identity_flips(
            online_swaps,
            raw_identity_vectors,
            args.identity_global_min_margin,
            args.identity_global_switch_penalty,
        )
        if any(offline_flips):
            final_swaps = [online_swap ^ flip for online_swap, flip in zip(online_swaps, offline_flips)]
            corrected_accum = torch.zeros_like(accum)
            corrected_weights = torch.zeros_like(weights)
            assert offline_chunk_dir is not None
            for idx, (start, raw_swap) in enumerate(zip(starts, final_swaps)):
                source_1, source_sr = load_audio(offline_chunk_dir / f"chunk_{idx:04d}_source_01.wav")
                source_2, source_2_sr = load_audio(offline_chunk_dir / f"chunk_{idx:04d}_source_02.wav")
                if source_sr != sr or source_2_sr != sr:
                    raise RuntimeError(f"Offline alignment cache chunk {idx} has an unexpected sample rate")
                pair = torch.cat((source_1, source_2), dim=0)
                if raw_swap:
                    pair = pair.flip(0)
                end = start + pair.shape[-1]
                window = make_window(
                    pair.shape[-1],
                    overlap_samples,
                    is_first=(idx == 0),
                    is_last=(idx == len(starts) - 1),
                )
                corrected_accum[:, start:end] += pair * window.unsqueeze(0)
                corrected_weights[start:end] += window
            accum, weights = corrected_accum, corrected_weights
            corrected = [index + 1 for index, flip in enumerate(offline_flips) if flip]
            print(
                "[UNMIXX identity offline] corrected online-relative chunks "
                + ", ".join(map(str, corrected))
            )
        else:
            print("[UNMIXX identity offline] no confident contiguous correction run")

    result = accum / weights.clamp_min(1e-6).unsqueeze(0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    spk1 = args.output_dir / "spk1.wav"
    spk2 = args.output_dir / "spk2.wav"
    save_wav(spk1, result[0:1], sr)
    save_wav(spk2, result[1:2], sr)
    if review_dir is not None:
        for record, final_swap, offline_flip, offline_margin in zip(
            review_records, final_swaps, offline_flips, offline_margins
        ):
            record["initial_swap"] = final_swap
            record["offline_identity_flip"] = offline_flip
            record["offline_identity_margin"] = offline_margin
        (review_dir / "alignment_manifest.json").write_text(
            json.dumps({
                "format_version": 1,
                "sample_rate": sr,
                "total_samples": total_samples,
                "chunk_samples": chunk_samples,
                "overlap_samples": overlap_samples,
                "chunks": review_records,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[UNMIXX alignment review] {review_dir / 'alignment_manifest.json'}")
    if offline_cache is not None:
        offline_cache.cleanup()
    print(f"[UNMIXX chunked] skipped silent chunks: {skipped_silent_chunks}/{len(starts)}")
    print(f"[Save] {spk1}")
    print(f"[Save] {spk2}")


if __name__ == "__main__":
    main()
