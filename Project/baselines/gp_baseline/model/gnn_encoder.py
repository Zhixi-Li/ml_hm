import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


# ---------------------------------------------------------------------------
# Sinusoidal position / scalar embeddings (adapted from gnn4co reference)
# ---------------------------------------------------------------------------

class PositionEmbeddingSine(nn.Module):
    """2D sinusoidal position encoding for (x, y) node coordinates.

    Input : (B, N, 2)  — coordinates in [0, 1]^2
    Output: (B, N, 2*embedding_dim)
    """
    def __init__(self, embedding_dim: int, tau: int = 10000, normalize: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.tau = tau
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, coords: Tensor) -> Tensor:
        # coords: (B, N, 2)
        x = coords[..., 0]  # (B, N)
        y = coords[..., 1]  # (B, N)
        if self.normalize:
            x = x * self.scale
            y = y * self.scale

        dim_t = torch.arange(self.embedding_dim, dtype=torch.float32, device=coords.device)
        dim_t = 2.0 * torch.div(dim_t, 2, rounding_mode='trunc') / self.embedding_dim
        dim_t = self.tau ** dim_t  # (embedding_dim,)

        x_embed = x.unsqueeze(-1) / dim_t          # (B, N, embedding_dim)
        y_embed = y.unsqueeze(-1) / dim_t

        x_embed = torch.stack([x_embed[..., 0::2].sin(), x_embed[..., 1::2].cos()], dim=-1).flatten(-2)
        y_embed = torch.stack([y_embed[..., 0::2].sin(), y_embed[..., 1::2].cos()], dim=-1).flatten(-2)

        return torch.cat([x_embed, y_embed], dim=-1)  # (B, N, 2*embedding_dim)


class ScalarEmbeddingSine(nn.Module):
    """Sinusoidal embedding for a scalar field (e.g. distance matrix).

    Input : (B, V, V)
    Output: (B, V, V, embedding_dim)
    """
    def __init__(self, embedding_dim: int, tau: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.tau = tau

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, V, V)
        dim_t = torch.arange(self.embedding_dim, dtype=torch.float32, device=x.device)
        dim_t = 2.0 * torch.div(dim_t, 2, rounding_mode='trunc') / self.embedding_dim
        dim_t = self.tau ** dim_t  # (embedding_dim,)

        embed = x.unsqueeze(-1) / dim_t  # (B, V, V, embedding_dim)
        embed = torch.stack([embed[..., 0::2].sin(), embed[..., 1::2].cos()], dim=-1).flatten(-2)
        return embed  # (B, V, V, embedding_dim)


# ---------------------------------------------------------------------------
# GroupNorm helper (matches reference out-layer)
# ---------------------------------------------------------------------------

class GroupNorm32(nn.GroupNorm):
    """GroupNorm that keeps the input dtype (float16-safe)."""
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


# ---------------------------------------------------------------------------
# GNN Layer
# ---------------------------------------------------------------------------

class GNNLayer(nn.Module):
    """Gated Graph ConvNet layer (Bresson & Laurent, 2018).

    h_i^(l+1) = ReLU( U h_i  +  Aggr_j[ sigma_ij * V h_j ] )  + h_i
    e_ij^(l+1) = ReLU( A h_i + B h_j + C e_ij )               + e_ij  (residual via per_layer_out)
    sigma_ij   = sigmoid( e_ij^(l+1) )
    """
    def __init__(self, hidden_dim: int, aggregation: str = "sum"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation

        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)
        self.A = nn.Linear(hidden_dim, hidden_dim)
        self.B = nn.Linear(hidden_dim, hidden_dim)
        self.C = nn.Linear(hidden_dim, hidden_dim)

        self.norm_h = nn.LayerNorm(hidden_dim)
        self.norm_e = nn.LayerNorm(hidden_dim)

    def forward(self, h: Tensor, e: Tensor) -> Tuple[Tensor, Tensor]:
        """
        h: (B, V, H)
        e: (B, V, V, H)
        Returns: dh (B, V, H), de (B, V, V, H)  — deltas, NOT added to input here
        """
        B, V, H = h.shape

        Uh = self.U(h)                                            # (B, V, H)
        Vh = self.V(h).unsqueeze(1).expand(-1, V, -1, -1)        # (B, V, V, H)

        Ah = self.A(h).unsqueeze(2)   # (B, V, 1, H)  — row (source)
        Bh = self.B(h).unsqueeze(1)   # (B, 1, V, H)  — col (target)
        Ce = self.C(e)                # (B, V, V, H)

        e_new = Ah + Bh + Ce          # (B, V, V, H)
        gates = torch.sigmoid(e_new)  # (B, V, V, H)

        # Aggregate
        gated = gates * Vh            # (B, V, V, H)
        if self.aggregation == "mean":
            aggr = gated.mean(dim=2)
        else:  # sum
            aggr = gated.sum(dim=2)

        h_new = Uh + aggr             # (B, V, H)

        # Norm + activation
        h_new = F.relu(self.norm_h(h_new.view(B * V, H)).view(B, V, H))
        e_new = F.relu(self.norm_e(e_new.view(B * V * V, H)).view(B, V, V, H))

        return h_new, e_new


# ---------------------------------------------------------------------------
# GNN Encoder
# ---------------------------------------------------------------------------

class GNNEncoder(nn.Module):
    """GNN Encoder for TSP (dense, fully-connected graph).
    """

    def __init__(
        self,
        num_layers: int = 6,
        hidden_dim: int = 128,
        aggregation: str = "sum",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # --- Embedders ---
        self.pos_embed = PositionEmbeddingSine(hidden_dim // 2)   # -> (B, V, hidden_dim)
        self.node_embed = nn.Linear(hidden_dim, hidden_dim)

        self.edge_pos_embed = ScalarEmbeddingSine(hidden_dim)     # -> (B, V, V, hidden_dim)
        self.edge_embed = nn.Linear(hidden_dim, hidden_dim)

        # --- GNN layers + per-layer edge projections ---
        self.layers = nn.ModuleList([
            GNNLayer(hidden_dim, aggregation) for _ in range(num_layers)
        ])

        # Zero-init: each layer contributes e_delta = W * e_layer_out
        # Starting from zero means layer 0 is a skip connection initially.
        self.per_layer_out = nn.ModuleList([
            self._zero_linear(hidden_dim) for _ in range(num_layers)
        ])

        # --- Output head: GroupNorm + ReLU + Conv2d (1x1) ---
        self.out = nn.Sequential(
            GroupNorm32(32, hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 2, kernel_size=1, bias=True),
        )

    @staticmethod
    def _zero_linear(dim: int) -> nn.Sequential:
        linear = nn.Linear(dim, dim)
        nn.init.zeros_(linear.weight)
        nn.init.zeros_(linear.bias)
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            linear,
        )

    def forward(self, coords: Tensor) -> Tensor:
        """
        Args:
            coords: (B, V, 2)  — node coordinates in [0,1]^2
        Returns:
            logits: (B, V, V, 2)  — 2-channel edge logits
                    channel 0 = NOT in tour, channel 1 = IN tour
        """
        B, V, _ = coords.shape

        # --- Node embeddings ---
        h = self.node_embed(self.pos_embed(coords))   # (B, V, H)

        # --- Edge embeddings (distance matrix) ---
        dist = torch.cdist(coords, coords, p=2)        # (B, V, V)
        e = self.edge_embed(self.edge_pos_embed(dist)) # (B, V, V, H)

        # --- GNN layers with residual streams ---
        for layer, proj in zip(self.layers, self.per_layer_out):
            h_in, e_in = h, e
            dh, de = layer(h, e)
            h = h_in + dh          # node residual
            e = e_in + proj(de)    # edge residual via zero-init projection

        # --- Output head ---
        # (B, V, V, H) -> (B, H, V, V) -> Conv2d -> (B, 2, V, V) -> (B, V, V, 2)
        logits = self.out(e.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return logits  # (B, V, V, 2)
