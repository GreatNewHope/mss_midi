import torch

from unmixx_chunked_inference import ChunkEstimate, global_assignments


def unit(*values: float) -> torch.Tensor:
    return torch.nn.functional.normalize(torch.tensor(values, dtype=torch.float32), dim=0)


def chunk(first: torch.Tensor, second: torch.Tensor) -> ChunkEstimate:
    return ChunkEstimate(
        start=0,
        end=4,
        estimates=torch.zeros((2, 4)),
        skipped=False,
        embeddings=[first, second],
    )


def test_global_assignment_uses_later_identity_evidence_across_silence() -> None:
    singer_a = unit(1, 0, 0)
    singer_b = unit(0, 1, 0)
    silence = ChunkEstimate(4, 8, torch.zeros((2, 4)), True)

    states, _, trace = global_assignments(
        [chunk(singer_a, singer_b), silence, chunk(singer_b, singer_a)],
        overlap_samples=0,
        min_peak=0.01,
        min_margin=0.05,
    )

    assert states == [False, False, True]
    assert trace["initial_cluster_states"] == [False, None, True]
    assert [iteration["states"] for iteration in trace["iterations"]][-1] == states
