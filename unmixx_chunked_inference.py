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
from dataclasses import dataclass
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


@dataclass
class ChunkEstimate:
    """One unpermuted UNMIXX estimate retained until global assignment."""

    start: int
    end: int
    estimates: torch.Tensor
    skipped: bool
    embeddings: list[torch.Tensor | None] | None = None


def pair_scores(
    first: list[torch.Tensor | None],
    second: list[torch.Tensor | None],
) -> tuple[float, float] | None:
    """Return keep/swap identity scores when both separated stems are voiced."""
    if any(vector is None for vector in first) or any(vector is None for vector in second):
        return None
    assert first[0] is not None and first[1] is not None
    assert second[0] is not None and second[1] is not None
    keep = float(torch.dot(first[0], second[0]) + torch.dot(first[1], second[1]))
    swap = float(torch.dot(first[0], second[1]) + torch.dot(first[1], second[0]))
    return keep, swap


class SingerIdentityTracker:
    """Generate singer-identity embeddings for separated UNMIXX stems.

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
    ) -> None:
        model_path = hf_hub_download(
            repo_id=self.REPO_ID,
            filename=f"{model_name}/model.ts",
        )
        self.model = torch.jit.load(model_path, map_location=device).eval()
        self.device = device
        self.input_sr = input_sr
        self.min_peak = min_peak

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


def mean_prototype(vectors: list[torch.Tensor]) -> torch.Tensor | None:
    if not vectors:
        return None
    return torch.nn.functional.normalize(torch.stack(vectors).mean(dim=0), dim=0)


def cluster_global_prototypes(chunks: list[ChunkEstimate]) -> list[torch.Tensor | None]:
    """Estimate two singer prototypes using all voiced, paired chunk embeddings.

    This is pair-constrained two-cluster refinement: each UNMIXX chunk must
    contribute one embedding to each identity, so leaky stems cannot be
    independently assigned to the same singer cluster.
    """
    pairs = [chunk.embeddings for chunk in chunks if chunk.embeddings is not None]
    pairs = [pair for pair in pairs if pair_scores(pair, pair) is not None]
    if not pairs:
        return [None, None]

    # Start from the chunk whose two estimated sources are most distinct.
    seed = min(pairs, key=lambda pair: float(torch.dot(pair[0], pair[1])))
    assert seed[0] is not None and seed[1] is not None
    prototypes: list[torch.Tensor | None] = [seed[0], seed[1]]
    for _ in range(12):
        assigned: list[list[torch.Tensor]] = [[], []]
        for pair in pairs:
            scores = pair_scores(prototypes, pair)
            assert scores is not None
            keep, swap = scores
            order = (1, 0) if swap > keep else (0, 1)
            for track, source in enumerate(order):
                vector = pair[source]
                assert vector is not None
                assigned[track].append(vector)
        updated = [mean_prototype(assigned[0]), mean_prototype(assigned[1])]
        if any(vector is None for vector in updated):
            break
        assert prototypes[0] is not None and prototypes[1] is not None
        shift = float(torch.dot(prototypes[0], updated[0]) + torch.dot(prototypes[1], updated[1]))
        prototypes = updated
        if shift > 1.9999:
            break
    return prototypes


def cluster_states(
    chunks: list[ChunkEstimate], prototypes: list[torch.Tensor | None]
) -> list[bool | None]:
    """Return the identity-only keep/swap choice before temporal smoothing."""
    states: list[bool | None] = []
    for chunk in chunks:
        scores = pair_scores(prototypes, chunk.embeddings or [None, None])
        states.append(None if scores is None else scores[1] > scores[0])
    return states


def global_assignments(
    chunks: list[ChunkEstimate],
    overlap_samples: int,
    min_peak: float,
    min_margin: float,
) -> tuple[list[bool], list[torch.Tensor | None], dict[str, object]]:
    """Globally choose one keep/swap state per chunk.

    Reliable overlaps are high-weight transitions. Identity scores are global
    emissions, so a voiced section after silence can inform its predecessor.
    The two-state Viterbi pass is repeated while robust identity prototypes are
    re-estimated from its assignments.
    """
    if not chunks:
        return [], [None, None], {"initial_cluster_states": [], "iterations": []}
    prototypes = cluster_global_prototypes(chunks)
    states = [False] * len(chunks)
    overlap_weight = 4.0
    trace: dict[str, object] = {
        "initial_cluster_states": cluster_states(chunks, prototypes),
        "iterations": [],
        "overlap_weight": overlap_weight,
    }

    for _ in range(8):
        scores = torch.full((len(chunks), 2), -float("inf"), dtype=torch.float64)
        back = torch.zeros((len(chunks), 2), dtype=torch.long)
        chunk_trace: list[dict[str, object]] = []
        for index, chunk in enumerate(chunks):
            emissions = (0.0, 0.0)
            identity_scores = pair_scores(prototypes, chunk.embeddings or [None, None])
            if identity_scores is not None:
                emissions = identity_scores
            if index == 0:
                # Global identity labels are arbitrary, so permit either
                # orientation instead of pinning the first raw UNMIXX source.
                scores[index, 0] = emissions[0]
                scores[index, 1] = emissions[1]
                chunk_trace.append(
                    {
                        "chunk": index,
                        "start": chunk.start,
                        "end": chunk.end,
                        "skipped": chunk.skipped,
                        "identity_keep": emissions[0],
                        "identity_swap": emissions[1],
                        "overlap_keep": 0.0,
                        "overlap_swap": 0.0,
                        "overlap_reliable": False,
                        "score_keep": float(scores[index, 0]),
                        "score_swap": float(scores[index, 1]),
                        "previous_for_keep": None,
                        "previous_for_swap": None,
                    }
                )
                continue

            previous = chunks[index - 1]
            transition = (0.0, 0.0)
            overlap_reliable = False
            if not previous.skipped and not chunk.skipped and overlap_samples > 0:
                overlap = min(overlap_samples, previous.estimates.shape[-1], chunk.estimates.shape[-1])
                if overlap > 0:
                    prev_tail = previous.estimates[:, -overlap:]
                    cur_head = chunk.estimates[:, :overlap]
                    _, _, keep, swap = align_two_sources(prev_tail, cur_head)
                    reliable = (
                        prev_tail.abs().amax() > min_peak
                        and cur_head.abs().amax() > min_peak
                        and abs(swap - keep) >= min_margin
                    )
                    if reliable:
                        transition = (overlap_weight * keep, overlap_weight * swap)
                        overlap_reliable = True

            for state in range(2):
                candidates = (
                    scores[index - 1, 0] + transition[state],
                    scores[index - 1, 1] + transition[1 - state],
                )
                predecessor = 0 if candidates[0] >= candidates[1] else 1
                back[index, state] = predecessor
                scores[index, state] = candidates[predecessor] + emissions[state]
            chunk_trace.append(
                {
                    "chunk": index,
                    "start": chunk.start,
                    "end": chunk.end,
                    "skipped": chunk.skipped,
                    "identity_keep": emissions[0],
                    "identity_swap": emissions[1],
                    "overlap_keep": transition[0],
                    "overlap_swap": transition[1],
                    "overlap_reliable": overlap_reliable,
                    "score_keep": float(scores[index, 0]),
                    "score_swap": float(scores[index, 1]),
                    "previous_for_keep": int(back[index, 0]),
                    "previous_for_swap": int(back[index, 1]),
                }
            )

        new_states = [False] * len(chunks)
        state = 0 if scores[-1, 0] >= scores[-1, 1] else 1
        for index in range(len(chunks) - 1, -1, -1):
            new_states[index] = bool(state)
            state = int(back[index, state])
        for item, state in zip(chunk_trace, new_states):
            item["state"] = state
        iterations = trace["iterations"]
        assert isinstance(iterations, list)
        iterations.append({"states": new_states, "chunks": chunk_trace})

        grouped: list[list[torch.Tensor]] = [[], []]
        for chunk, swapped in zip(chunks, new_states):
            vectors = chunk.embeddings
            if vectors is None or any(vector is None for vector in vectors):
                continue
            order = (1, 0) if swapped else (0, 1)
            for track, source in enumerate(order):
                vector = vectors[source]
                assert vector is not None
                grouped[track].append(vector)
        updated = [mean_prototype(grouped[0]), mean_prototype(grouped[1])]
        if updated[0] is None or updated[1] is None or new_states == states:
            states = new_states
            break
        states, prototypes = new_states, updated
    return states, prototypes, trace


def write_alignment_audit(output_dir: Path, trace: dict[str, object], sample_rate: int) -> None:
    """Write the score trace and an interactive timeline for alignment review."""
    trace["sample_rate"] = sample_rate
    json_path = output_dir / "alignment_audit.json"
    html_path = output_dir / "alignment_audit.html"
    json_path.write_text(json.dumps(trace, indent=2) + "\n")
    data = json.dumps(trace, separators=(",", ":"))
    html_path.write_text(
        """<meta charset="utf-8">
<title>UNMIXX alignment audit</title>
<style>
body{font:14px system-ui,sans-serif;margin:24px;color:#182230;background:#fbfcfe}h1{margin:0 0 6px}p{margin:0 0 16px;color:#526070}.legend{display:flex;gap:15px;margin:12px 0}.key{display:inline-block;width:12px;height:12px;margin-right:5px}.keep{background:#287f5b}.swap{background:#b94b61}.none{background:#aeb8c4}.grid{display:grid;grid-template-columns:150px repeat(var(--chunks),minmax(22px,1fr));gap:3px;align-items:center;overflow-x:auto;padding-bottom:8px}.label{font-weight:600}.cell{height:27px;min-width:22px;border:0;cursor:pointer;color:#fff;font-weight:700}.cell.keep{background:#287f5b}.cell.swap{background:#b94b61}.cell.none{background:#aeb8c4}.cell.selected{outline:3px solid #1c4f9c;outline-offset:1px}.detail{margin-top:18px;border:1px solid #ccd5df;padding:14px;background:white;max-width:840px}.detail table{border-collapse:collapse;width:100%}.detail th,.detail td{text-align:right;padding:5px 7px;border-bottom:1px solid #e5eaf0}.detail th:first-child,.detail td:first-child{text-align:left}.warning{color:#963545;font-weight:700}</style>
<main id="alignment-audit"><h1>Offline alignment audit</h1><p id="subtitle"></p><div class="legend"><span><i class="key keep"></i>keep</span><span><i class="key swap"></i>swap</span><span><i class="key none"></i>no identity evidence</span></div><div id="timeline" class="grid"></div><section class="detail" id="detail">Select a chunk.</section></main>
<script>
const trace=""" + data + """;
const root=document.getElementById('alignment-audit');const timeline=document.getElementById('timeline');const detail=document.getElementById('detail');const iterations=trace.iterations;const initial=trace.initial_cluster_states;const n=initial.length;timeline.style.setProperty('--chunks',n);document.getElementById('subtitle').textContent=`${n} chunks · overlap weight ${trace.overlap_weight} · click any state to inspect its evidence`;
function stateClass(v){return v===null?'none':v?'swap':'keep'}function label(v){return v===null?'—':v?'swap':'keep'}
function row(name,states,iteration){const head=document.createElement('div');head.className='label';head.textContent=name;timeline.append(head);states.forEach((value,index)=>{const b=document.createElement('button');b.type='button';b.className=`cell ${stateClass(value)}`;b.textContent=value===null?'—':value?'S':'K';b.title=`chunk ${index+1}: ${label(value)}`;b.onclick=()=>show(index,iteration,b);timeline.append(b)})}
row('Initial clustering',initial,-1);iterations.forEach((entry,index)=>row(`Viterbi iteration ${index+1}`,entry.states,index));
function num(value){return value===null?'—':Number(value).toFixed(3)}function show(index,iteration,button){root.querySelectorAll('.selected').forEach(x=>x.classList.remove('selected'));button.classList.add('selected');if(iteration<0){detail.innerHTML=`<strong>Chunk ${index+1}</strong><p>Initial identity-only cluster choice: <b>${label(initial[index])}</b>. It has not yet used overlap continuity or dynamic programming.</p>`;return}const item=iterations[iteration].chunks[index];const seconds=v=>v/trace.sample_rate;const margin=item.identity_swap-item.identity_keep;const jump=index&&iterations[iteration].states[index]!==iterations[iteration].states[index-1];detail.innerHTML=`<strong>Chunk ${index+1} · Viterbi iteration ${iteration+1}</strong><p>${seconds(item.start).toFixed(2)}–${seconds(item.end).toFixed(2)} s${item.skipped?' · <span class="warning">silent / skipped</span>':''}${jump?' · <span class="warning">state changed from prior chunk</span>':''}</p><table><tr><th></th><th>keep</th><th>swap</th></tr><tr><td>identity emission</td><td>${num(item.identity_keep)}</td><td>${num(item.identity_swap)}</td></tr><tr><td>weighted overlap transition</td><td>${num(item.overlap_keep)}</td><td>${num(item.overlap_swap)}</td></tr><tr><td>Viterbi total</td><td>${num(item.score_keep)}</td><td>${num(item.score_swap)}</td></tr><tr><td>backpointer</td><td>${item.previous_for_keep===null?'—':label(Boolean(item.previous_for_keep))}</td><td>${item.previous_for_swap===null?'—':label(Boolean(item.previous_for_swap))}</td></tr></table><p>Reliable overlap: <b>${item.overlap_reliable?'yes':'no'}</b>. Identity margin (swap − keep): <b>${num(margin)}</b>. Final state in this iteration: <b>${label(item.state)}</b>.</p>`}
</script>"""
    )
    print(f"[UNMIXX identity] audit: {html_path}")


def load_alignment_overrides(path: Path, chunk_count: int) -> list[bool]:
    """Read reviewed keep/swap choices exported by the alignment review page."""
    try:
        payload = json.loads(path.read_text())
        entries = payload["chunks"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Could not read alignment overrides from {path}: {exc}") from exc
    if not isinstance(entries, list) or len(entries) != chunk_count:
        raise RuntimeError(f"{path} must contain exactly {chunk_count} reviewed chunk choices")
    states: list[bool] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("index") != index + 1:
            raise RuntimeError(f"{path}: expected chunk index {index + 1}")
        choice = entry.get("state")
        if choice not in {"keep", "swap"}:
            raise RuntimeError(f"{path}: chunk {index + 1} must be keep or swap")
        states.append(choice == "swap")
    return states


def write_alignment_review(
    output_dir: Path, chunks: list[ChunkEstimate], model_states: list[bool], sample_rate: int
) -> None:
    """Write raw chunk clips and a local page for human keep/swap annotation."""
    review_dir = output_dir / "alignment_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, object]] = []
    for index, chunk in enumerate(chunks):
        clip_a = clip_b = None
        if not chunk.skipped:
            clip_a = f"chunk_{index + 1:03d}_a.wav"
            clip_b = f"chunk_{index + 1:03d}_b.wav"
            save_wav(review_dir / clip_a, chunk.estimates[0:1], sample_rate)
            save_wav(review_dir / clip_b, chunk.estimates[1:2], sample_rate)
        metadata.append(
            {
                "index": index + 1,
                "start": chunk.start / sample_rate,
                "end": chunk.end / sample_rate,
                "skipped": chunk.skipped,
                "model_state": "swap" if model_states[index] else "keep",
                "clip_a": clip_a,
                "clip_b": clip_b,
            }
        )
    data = json.dumps(metadata, separators=(",", ":"))
    (review_dir / "index.html").write_text(
        """<meta charset="utf-8"><title>UNMIXX manual alignment review</title>
<style>body{font:14px system-ui,sans-serif;max-width:1100px;margin:24px;color:#17212d}h1{margin-bottom:4px}.note{color:#566474}.chunk{display:grid;grid-template-columns:100px 1fr 1fr 180px;gap:12px;align-items:center;border-top:1px solid #dce3ea;padding:10px 0}.silent{opacity:.55}.choice label{display:block;margin:4px 0}audio{width:100%}button{padding:8px 12px;font:inherit}.missing{outline:2px solid #bd3d4b}@media(max-width:700px){.chunk{grid-template-columns:1fr}.chunk audio{max-width:350px}}</style>
<main id="review"><h1>Manual chunk alignment</h1><p class="note">For each voiced chunk, audition raw UNMIXX A and B. Choose <b>keep</b> when A belongs in Stem 1 and B in Stem 2; choose <b>swap</b> for the reverse. The model suggestion is shown only for comparison. Export the completed choices and provide that JSON file to a later run.</p><button id="export" type="button">Export reviewed alignment JSON</button><p id="status" class="note"></p><section id="chunks"></section></main>
<script>const chunks=""" + data + """;const host=document.getElementById('chunks');const status=document.getElementById('status');for(const c of chunks){const row=document.createElement('article');row.className='chunk'+(c.skipped?' silent':'');row.innerHTML=`<strong>Chunk ${c.index}<br>${c.start.toFixed(2)}–${c.end.toFixed(2)} s<br><small>model: ${c.model_state}</small></strong>${c.skipped?'<span>silent / skipped</span><span></span><span>keep is exported automatically</span>':`<audio controls preload="none" src="${c.clip_a}"></audio><audio controls preload="none" src="${c.clip_b}"></audio><span class="choice"><label><input type="radio" name="chunk-${c.index}" value="keep"> Keep: A → Stem 1</label><label><input type="radio" name="chunk-${c.index}" value="swap"> Swap: B → Stem 1</label></span>`}`;host.append(row)}document.getElementById('export').onclick=()=>{const reviewed=[];let missing=false;for(const c of chunks){const selected=document.querySelector(`input[name="chunk-${c.index}"]:checked`);const state=c.skipped?'keep':selected?.value;if(!state){missing=true;document.querySelector(`input[name="chunk-${c.index}"]`)?.closest('.chunk').classList.add('missing')}reviewed.push({index:c.index,state:state||'unassigned'})}if(missing){status.textContent='Choose keep or swap for every voiced chunk before exporting.';return}const blob=new Blob([JSON.stringify({format:'unmixx-alignment-overrides-v1',chunks:reviewed},null,2)+'\\n'],{type:'application/json'});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='alignment_overrides.json';link.click();URL.revokeObjectURL(link.href);status.textContent='Downloaded alignment_overrides.json.'}</script>"""
    )
    print(f"[UNMIXX identity] manual review: {review_dir / 'index.html'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Chunked long-form inference for the official UNMIXX model")
    p.add_argument("--unmixx-repo", type=Path, required=True)
    p.add_argument("--conf-path", type=Path, required=True)
    p.add_argument("--ckpt-path", type=Path, required=True)
    p.add_argument("--audio-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--chunk-seconds", type=float, default=4.0)
    p.add_argument("--overlap-seconds", type=float, default=1.0)
    p.add_argument(
        "--identity-model",
        choices=("none", "byol", "contrastive", "contrastive-vc", "uniformity", "vicreg"),
        default="byol",
        help="Singer-identity encoder used for global chunk assignment (default: byol).",
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
        "--alignment-review",
        action="store_true",
        help="Write raw chunk clips and a browser page for manual keep/swap annotation.",
    )
    p.add_argument(
        "--alignment-overrides",
        type=Path,
        help="Reviewed alignment_overrides.json exported by --alignment-review.",
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
        )
        print(f"[UNMIXX identity] model={args.identity_model}, min-peak={args.identity_min_peak_db:g} dBFS")

    chunks: list[ChunkEstimate] = []
    skipped_silent_chunks = 0

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

        embeddings = tracker.embeddings(est) if tracker is not None and not skipped else None
        chunks.append(ChunkEstimate(start, end, est, skipped, embeddings))
        if skipped:
            print(
                f"  chunk {idx+1:03d}/{len(starts):03d} "
                f"{start/sr:7.2f}-{end/sr:7.2f}s silence=skip"
            )
        else:
            print(f"  chunk {idx+1:03d}/{len(starts):03d} {start/sr:7.2f}-{end/sr:7.2f}s inferred")

        if device.type == "cuda" and (idx + 1) % 16 == 0:
            torch.cuda.empty_cache()

    if tracker is None:
        # Preserve continuity-only behavior when identity tracking is disabled.
        states = [False] if chunks else []
        for index in range(1, len(chunks)):
            previous, current = chunks[index - 1], chunks[index]
            swapped = False
            if not previous.skipped and not current.skipped and overlap_samples > 0:
                overlap = min(overlap_samples, previous.estimates.shape[-1], current.estimates.shape[-1])
                if overlap > 0:
                    _, swapped, _, _ = align_two_sources(
                        previous.estimates[:, -overlap:], current.estimates[:, :overlap]
                    )
            states.append(states[-1] ^ swapped)
        assignment_name = "overlap"
    else:
        print("[UNMIXX identity] globally assigning chunk permutations")
        states, _, trace = global_assignments(
            chunks, overlap_samples, identity_peak_threshold, args.identity_min_margin
        )
        assignment_name = "global"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.alignment_review:
        write_alignment_review(args.output_dir, chunks, states, sr)
    if args.alignment_overrides is not None:
        states = load_alignment_overrides(args.alignment_overrides, len(chunks))
        assignment_name = "human-review"
        if tracker is not None:
            trace["human_overrides"] = states
    if tracker is not None:
        write_alignment_audit(args.output_dir, trace, sr)

    accum = torch.zeros((2, total_samples), dtype=torch.float32)
    weights = torch.zeros(total_samples, dtype=torch.float32)
    for idx, (chunk, swapped) in enumerate(zip(chunks, states)):
        estimates = chunk.estimates.flip(0) if swapped else chunk.estimates
        window = make_window(
            chunk.end - chunk.start,
            overlap_samples,
            is_first=(idx == 0),
            is_last=(idx == len(chunks) - 1),
        )
        accum[:, chunk.start:chunk.end] += estimates * window.unsqueeze(0)
        weights[chunk.start:chunk.end] += window
        if not chunk.skipped:
            print(
                f"  chunk {idx+1:03d}/{len(chunks):03d} "
                f"{chunk.start/sr:7.2f}-{chunk.end/sr:7.2f}s "
                f"perm={'swap' if swapped else 'keep'} via={assignment_name}"
            )

    result = accum / weights.clamp_min(1e-6).unsqueeze(0)

    spk1 = args.output_dir / "spk1.wav"
    spk2 = args.output_dir / "spk2.wav"
    save_wav(spk1, result[0:1], sr)
    save_wav(spk2, result[1:2], sr)
    print(f"[UNMIXX chunked] skipped silent chunks: {skipped_silent_chunks}/{len(starts)}")
    print(f"[Save] {spk1}")
    print(f"[Save] {spk2}")


if __name__ == "__main__":
    main()
