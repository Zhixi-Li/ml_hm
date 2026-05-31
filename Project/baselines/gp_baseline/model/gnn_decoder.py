import numpy as np
import torch
from torch import Tensor
from typing import List
from ml4co_kit import tsp_greedy_decoder


class TSPHeatmapDecoder:
    """Greedy decoder for TSP heatmap using ml4co-kit's standard interface."""

    def __init__(self):
        pass

    def decode(self, logits: Tensor, coords: Tensor) -> List[np.ndarray]:
        """
        Args:
            logits: (B, V, V, 2) — 2-channel edge logits (channel 1 = in-tour prob)
            coords: (B, V, 2)   — node coordinates (unused here, kept for API compat)
        Returns:
            List of B tours, each a numpy array of shape (V+1,) starting and ending at 0
        """
        # 2-channel -> probability of being in tour (channel 1)
        if logits.dim() == 4:
            heatmap = torch.softmax(logits, dim=-1)[..., 1]  # (B, V, V)
        else:
            # Legacy single-channel support
            heatmap = torch.sigmoid(logits)                  # (B, V, V)

        # Symmetrize and clip to valid probability range
        heatmap = (heatmap + heatmap.transpose(1, 2)) / 2   # (B, V, V)
        heatmap = heatmap.detach().cpu().numpy()
        heatmap = np.clip(heatmap, a_min=1e-14, a_max=1 - 1e-14)

        tours = []
        for b in range(heatmap.shape[0]):
            tour = tsp_greedy_decoder(heatmap[b])
            tours.append(tour)

        return tours
