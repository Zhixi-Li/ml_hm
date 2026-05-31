import torch
import torch.nn as nn
from torch import Tensor
from typing import List
import numpy as np
from .gnn_encoder import GNNEncoder
from .gnn_decoder import TSPHeatmapDecoder


class GPModel(nn.Module):
    """
    Complete GP (Global Prediction) model for TSP
    Combines encoder and decoder
    """
    def __init__(
        self,
        num_layers: int = 6,
        hidden_dim: int = 128,
        aggregation: str = "sum"
    ):
        super(GPModel, self).__init__()
        self.encoder = GNNEncoder(num_layers, hidden_dim, aggregation)
        self.decoder = TSPHeatmapDecoder()

    def forward(self, coords: Tensor) -> Tensor:
        """
        Forward pass: encode coordinates to 2-channel edge logits
        Args:
            coords: Node coordinates (B, V, 2)
        Returns:
            Edge logits (B, V, V, 2)
        """
        return self.encoder(coords)

    def solve(self, coords: Tensor) -> List[np.ndarray]:
        """
        Solve TSP instances
        Args:
            coords: Node coordinates (B, V, 2)
        Returns:
            List of tours
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(coords)
            tours = self.decoder.decode(logits, coords)
        return tours

    def compute_loss(self, coords: Tensor, ground_truth: Tensor) -> Tensor:
        """
        Compute supervised learning loss (cross-entropy on 2-channel edge logits)
        Args:
            coords: Node coordinates (B, V, 2)
            ground_truth: Binary edge labels (B, V, V), values in {0, 1}
        Returns:
            Cross-entropy loss (scalar)
        """
        logits = self.forward(coords)  # (B, V, V, 2)
        B, V, _, _ = logits.shape

        # Reshape for cross_entropy: (B*V*V, 2) vs (B*V*V,)
        logits_flat = logits.view(B * V * V, 2)
        labels_flat = ground_truth.view(B * V * V).long()

        loss = nn.functional.cross_entropy(logits_flat, labels_flat)
        return loss
