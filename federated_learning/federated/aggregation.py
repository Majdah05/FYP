from __future__ import annotations

from typing import Sequence

import torch


def fedavg_aggregate(model_states: Sequence[dict], sample_counts: Sequence[int]) -> dict:
    """Aggregate model state dictionaries using sample-size weighted averaging.

    Floating-point tensors are averaged proportionally to client sample counts.
    Non-floating-point buffers are retained from the first model to avoid invalid
    averaging on metadata such as integer buffer values.
    """
    if len(model_states) != len(sample_counts):
        raise ValueError('model_states and sample_counts must have the same length.')
    if not model_states:
        raise ValueError('At least one client model is required for aggregation.')

    total_samples = sum(sample_counts)
    if total_samples <= 0:
        raise ValueError('Sample counts must sum to a positive value.')

    aggregated = {}
    for key in model_states[0].keys():
        tensors = [state[key] for state in model_states]
        if not all(isinstance(t, torch.Tensor) for t in tensors):
            aggregated[key] = tensors[0]
            continue

        if tensors[0].is_floating_point():
            weights = torch.tensor(sample_counts, dtype=tensors[0].dtype, device=tensors[0].device)
            weighted = torch.stack([tensor * weight for tensor, weight in zip(tensors, weights)])
            aggregated[key] = weighted.sum(dim=0) / weights.sum()
        else:
            aggregated[key] = tensors[0]

    return aggregated
