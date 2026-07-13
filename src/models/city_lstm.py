"""
City LSTM - the single active forecasting model in the system.

The paper specifies one LSTM family used at all three tiers, sized differently
per tier (Algorithm 5 / Diagram 4 right column):
    City    : large DNN (>~100k params)   teacher
    District: mid   DNN                   relay
    Building: small DNN                   student / clipped round-update agent

This module exposes a single `CityLSTM` class with the architecture from the
paper diagram (encoder → learnable horizon-positional decoder → scaled
dot-product attention over encoder states → linear projection per step), plus
a `tier_size()` helper that returns the (hidden_dim, num_layers) recipe per
tier so the building/district/city modules don't carry magic numbers.

`mc_forward` is preserved for MC-Dropout uncertainty in evaluation.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from src.data.schema import INPUT_DIM


_TIER_SIZES = {
    "city":     {"hidden_dim": 64,  "num_layers": 2, "dropout": 0.2},
    "district": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.2},
    # Building and City participate in parameter-delta federation, so their
    # tensor schemas must match. The shared 145k-parameter model remains well
    # below the Building tier's <1M edge budget; dropout differs by role.
    "building": {"hidden_dim": 64,  "num_layers": 2, "dropout": 0.1},
}

Tier = Literal["city", "district", "building"]


def tier_size(tier: Tier) -> dict:
    if tier not in _TIER_SIZES:
        raise ValueError(f"Unknown tier '{tier}'. Use one of: {list(_TIER_SIZES)}.")
    return dict(_TIER_SIZES[tier])


class CityLSTM(nn.Module):
    """Encoder-decoder LSTM with learnable horizon embeddings + attention.

    Args:
        input_dim: number of input features (defaults to canonical schema)
        hidden_dim: hidden width of encoder + decoder
        num_layers: stacked LSTM depth
        pred_len: prediction horizon (number of future steps)
        dropout: dropout applied between LSTM layers and to positional emb
    """

    def __init__(
        self,
        pred_len: int,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        if pred_len < 1:
            raise ValueError("pred_len must be >= 1.")
        self.pred_len = int(pred_len)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)

        self.encoder = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.pos_embedding = nn.Parameter(torch.randn(self.pred_len, hidden_dim) * 0.02)

        self.decoder = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attn_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_scale = hidden_dim ** -0.5

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.peak_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.dropout = nn.Dropout(dropout)

    @classmethod
    def for_tier(cls, tier: Tier, pred_len: int, input_dim: int = INPUT_DIM) -> "CityLSTM":
        cfg = tier_size(tier)
        return cls(pred_len=pred_len, input_dim=input_dim, **cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, input_dim) → (B, pred_len)."""
        B = x.size(0)
        enc_out, (hn, cn) = self.encoder(x)

        dec_in = self.pos_embedding.unsqueeze(0).expand(B, -1, -1)
        dec_in = self.dropout(dec_in)
        dec_out, _ = self.decoder(dec_in, (hn, cn))

        q = self.attn_q(dec_out)
        k = self.attn_k(enc_out)
        v = self.attn_v(enc_out)
        attn = torch.softmax(torch.matmul(q, k.transpose(1, 2)) * self.attn_scale, dim=-1)
        ctx = torch.matmul(attn, v)

        fused = torch.cat([dec_out, ctx], dim=-1)
        return self.head(fused).squeeze(-1)

    def multitask_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, S, input_dim) -> ((B, pred_len), (B, pred_len))."""
        B = x.size(0)
        enc_out, (hn, cn) = self.encoder(x)

        dec_in = self.pos_embedding.unsqueeze(0).expand(B, -1, -1)
        dec_in = self.dropout(dec_in)
        dec_out, _ = self.decoder(dec_in, (hn, cn))

        q = self.attn_q(dec_out)
        k = self.attn_k(enc_out)
        v = self.attn_v(enc_out)
        attn = torch.softmax(torch.matmul(q, k.transpose(1, 2)) * self.attn_scale, dim=-1)
        ctx = torch.matmul(attn, v)

        fused = torch.cat([dec_out, ctx], dim=-1)
        reg = self.head(fused).squeeze(-1)
        peak_logits = self.peak_head(fused).squeeze(-1)
        return reg, peak_logits

    def mc_forward(self, x: torch.Tensor, n_samples: int = 10):
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1.")
        was_training = self.training
        self.train()
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(self.forward(x))
        if not was_training:
            self.eval()
        stacked = torch.stack(preds)
        return stacked.mean(dim=0), stacked.std(dim=0, unbiased=False)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
