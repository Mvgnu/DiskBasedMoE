"""One byte budget shared by all routed experts in the model.

This is a single-consumer cache. A returned dictionary pins its tensors through
the caller's computation even if a later lookup evicts the dictionary. The
baseline evaluates experts serially and releases the caller's reference before
the next lookup. File decoding/staging and accelerator allocations are separate
from retained expert tensor bytes; allocator/OS caches are not measured here.
"""

from __future__ import annotations

from collections import OrderedDict

import torch

from .artifact import Artifact, tensor_bytes


class ExpertCache:
    def __init__(self, artifact: Artifact, budget_bytes: int, *, device: str = "cpu"):
        if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int) or budget_bytes < 0:
            raise ValueError("cache_budget_bytes must be a nonnegative integer")
        self.artifact = artifact
        self.budget_bytes = budget_bytes
        self.device = torch.device(device)
        self._cache: OrderedDict[tuple[int, int], tuple[dict[str, torch.Tensor], int]] = OrderedDict()
        self.retained_bytes = 0
        self.peak_retained_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.bypasses = 0
        self.logical_bytes_read = 0
        self.max_expert_tensor_bytes = 0

    def get(self, layer: int, expert: int) -> dict[str, torch.Tensor]:
        key = (layer, expert)
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key][0]
        entry = self.artifact.experts[key]
        size = sum(tensor_bytes(metadata) for metadata in entry["tensors"].values())
        self.misses += 1
        self.max_expert_tensor_bytes = max(self.max_expert_tensor_bytes, size)
        if size <= self.budget_bytes:
            while self._cache and self.retained_bytes + size > self.budget_bytes:
                _, (_, evicted_size) = self._cache.popitem(last=False)
                self.retained_bytes -= evicted_size
                self.evictions += 1
        tensors = self.artifact.read(entry)
        self.logical_bytes_read += entry["size_bytes"]
        tensors = {name: tensor.to(self.device) for name, tensor in tensors.items()}
        if size > self.budget_bytes:
            self.bypasses += 1
            return tensors
        self._cache[key] = (tensors, size)
        self.retained_bytes += size
        self.peak_retained_bytes = max(self.peak_retained_bytes, self.retained_bytes)
        return tensors

    def stats(self) -> dict[str, int | str]:
        return {
            "cache_budget_bytes": self.budget_bytes,
            "cache_retained_bytes": self.retained_bytes,
            "cache_peak_retained_bytes": self.peak_retained_bytes,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "cache_evictions": self.evictions,
            "cache_bypasses": self.bypasses,
            "logical_bytes_read": self.logical_bytes_read,
            "max_expert_tensor_bytes": self.max_expert_tensor_bytes,
            "io_measurement": "logical expert-file reads; physical storage/page cache not measured",
        }

    def clear(self) -> None:
        self._cache.clear()
        self.retained_bytes = 0
