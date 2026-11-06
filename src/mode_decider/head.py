"""The pointer readout: reads the backbone's hidden state at each question and option, lets the options of one
question attend to each other, and scores every option against its question."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from mode_decider.encoding import QUESTION_TYPES


class PointerHead(nn.Module):
    def __init__(self, hidden_size: int, width: int = 512, heads: int = 8, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.option_proj = nn.Linear(hidden_size, width)
        self.question_proj = nn.Linear(hidden_size, width)
        self.type_embedding = nn.Embedding(len(QUESTION_TYPES), width)
        self.slot_embedding = nn.Embedding(2, width)  # 0 = question, 1 = option
        self.mixer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, heads, 4 * width, dropout, activation="gelu", batch_first=True,
                                       norm_first=True), num_layers=layers, enable_nested_tensor=False)
        self.scorer = nn.Sequential(nn.LayerNorm(3 * width), nn.Linear(3 * width, width), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(width, 1))

    def forward(self, hidden: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        """Logits [questions, max options]; missing options are -inf."""
        flat = self.norm(hidden.reshape(-1, hidden.shape[-1]).to(self.norm.weight.dtype))
        n_q, k_max = batch["option_mask"].shape
        q = self.question_proj(flat[batch["q_flat"]]) + self.type_embedding(batch["q_type"])
        opts = torch.zeros((n_q, k_max, q.shape[-1]), dtype=q.dtype, device=q.device)
        opts[batch["slot_q"], batch["slot_k"]] = self.option_proj(flat[batch["slot_flat"]])
        tokens = torch.cat([(q + self.slot_embedding.weight[0]).unsqueeze(1), opts + self.slot_embedding.weight[1]], dim=1)
        pad = torch.cat([torch.zeros((n_q, 1), dtype=torch.bool, device=q.device), ~batch["option_mask"]], dim=1)
        mixed = self.mixer(tokens, src_key_padding_mask=pad)
        qv, ov = mixed[:, :1].expand(-1, k_max, -1), mixed[:, 1:]
        logits = self.scorer(torch.cat([ov, qv, ov * qv], dim=-1)).squeeze(-1)
        return logits.float().masked_fill(~batch["option_mask"], float("-inf"))
