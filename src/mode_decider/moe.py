"""A faster forward for MobileMoE's mixture-of-experts layers.

The released layer loops over its 60 experts in Python and syncs with the host for each one. This forward computes
the same result with two grouped matrix multiplies and no host syncs, so it can also be captured in a CUDA graph.
Routing is unchanged: fp32 router -> sigmoid -> + expert bias for selection -> top-k -> normalised scores.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F


def _route(moe, hidden: torch.Tensor):
    with torch.autocast("cuda", enabled=False):
        router_logits = F.linear(hidden.float(), moe.router.weight.float())
    scores = torch.sigmoid(router_logits)
    _, topk = torch.topk(scores + moe.expert_bias, moe.top_k, dim=-1)
    weights = scores.gather(1, topk)
    if moe.norm_topk_prob:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return router_logits, topk, weights * moe.routed_scaling_factor


def _gmm(layer, x: torch.Tensor, w: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    if layer._use_triton:
        from mode_decider.moe_triton import grouped_mm

        return grouped_mm(x, w, offsets)
    return torch._grouped_mm(x, w, offs=offsets)


def _grouped_forward(self, hidden: torch.Tensor):
    hidden = hidden.reshape(-1, self.hidden_dim)
    router_logits, topk, weights = _route(self, hidden)
    flat = topk.reshape(-1)
    order = torch.argsort(flat, stable=True)
    token = order // self.top_k
    counts = torch.zeros(self.num_experts, dtype=torch.int32, device=hidden.device)
    counts.index_add_(0, flat.int(), torch.ones_like(flat, dtype=torch.int32))
    offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
    x = hidden[token]
    gate, up = _gmm(self, x, self.experts.gate_up_proj, offsets).chunk(2, dim=-1)
    expert_out = _gmm(self, self.experts.act_fn(gate) * up, self.experts.down_proj, offsets)
    out = torch.zeros(hidden.shape, dtype=torch.float32, device=hidden.device)
    out.index_add_(0, token, expert_out.float() * weights.reshape(-1)[order].unsqueeze(-1))
    shared = self.shared_expert(hidden)
    return out.to(shared.dtype) + shared, router_logits


def use_fast_moe(model) -> int:
    """Swap in the grouped forward on every MoE layer (GPU only). Uses torch's grouped GEMM on H100/H200 and B200, and
    the Triton kernel elsewhere (e.g. RTX PRO 6000, A100), where torch's version syncs with the host. Returns the
    number of layers changed."""
    if not torch.cuda.is_available():
        return 0
    triton = torch.cuda.get_device_capability() not in ((9, 0), (10, 0))
    layers = [m for m in model.modules() if type(m).__name__ == "MobileMoETextMoe"]
    for layer in layers:
        layer.forward = types.MethodType(_grouped_forward, layer)
        layer._use_triton = triton
    return len(layers)
