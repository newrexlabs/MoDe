"""CUDA-graph replay of the backbone. At batch 1 a forward is a few thousand small kernel launches, so launch overhead
dominates; replaying a captured graph makes it one launch.

``precapture()`` captures every (batch bucket, length bucket) up to a token budget at load, so no request pays for a
capture. Inputs are right-padded to their bucket; shapes outside the captured grid run eagerly.
"""

from __future__ import annotations

import bisect

import torch

LENGTH_BUCKETS = (64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192)
BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64)


def _bucket(value: int, buckets) -> int | None:
    i = bisect.bisect_left(buckets, value)
    return buckets[i] if i < len(buckets) else None


class GraphedBackbone:
    def __init__(self, backbone, pad_id: int, device, warmup: int = 3):
        self.backbone, self.pad_id, self.device, self.warmup = backbone, pad_id, device, warmup
        self.graphs: dict[tuple[int, int], tuple] = {}
        self.pool = None
        self.sealed = False

    def _forward(self, ids, mask):
        # A 4D causal + padding mask built with tensor ops. A 2D mask makes transformers check it on the host,
        # which breaks capture.
        length = ids.shape[1]
        causal = torch.ones((length, length), dtype=torch.bool, device=ids.device).tril()
        mask4 = causal[None, None] & mask[:, None, None, :].bool()
        return self.backbone(input_ids=ids, attention_mask=mask4, use_cache=False).last_hidden_state

    def _capture(self, b: int, length: int):
        ids = torch.full((b, length), self.pad_id, dtype=torch.long, device=self.device)
        mask = torch.zeros((b, length), dtype=torch.long, device=self.device)
        mask[:, 0] = 1
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.inference_mode():
            for _ in range(self.warmup):
                self._forward(ids, mask)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph, pool=self.pool):
            out = self._forward(ids, mask)
        self.pool = graph.pool()
        self.graphs[(b, length)] = (graph, ids, mask, out)

    def precapture(self, token_budget: int = 32768) -> int:
        for length in LENGTH_BUCKETS:
            for b in BATCH_BUCKETS:
                if b == 1 or b * length <= token_budget:
                    self._capture(b, length)
        torch.cuda.synchronize()
        self.sealed = True
        return len(self.graphs)

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        b, length = input_ids.shape
        key = (_bucket(b, BATCH_BUCKETS), _bucket(length, LENGTH_BUCKETS))
        if None in key or (key not in self.graphs and self.sealed):
            with torch.inference_mode():
                return self._forward(input_ids, attention_mask)
        if key not in self.graphs:
            self._capture(*key)
        graph, ids, mask, out = self.graphs[key]
        ids.fill_(self.pad_id)
        mask.zero_()
        mask[:, 0] = 1  # padded rows keep one live token so attention never sees an all-masked row
        ids[:b, :length].copy_(input_ids)
        mask[:b, :length].copy_(attention_mask)
        graph.replay()
        return out[:b, :length]
