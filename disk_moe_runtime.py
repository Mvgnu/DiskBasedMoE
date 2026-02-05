#!/usr/bin/env python3
"""
Disk-backed MoE inference runtime with oracle prefetch for OLMoE sharded experts.
Now with full autoregressive loop (KV cache), top-k routing, advanced oracle metrics, and a test suite.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from pathlib import Path
import asyncio, concurrent.futures
from collections import OrderedDict, defaultdict, deque
import json
from typing import List, Dict, Optional
import argparse
import time
import sys
import numpy as np
try:
    import wikitext_dataset
except ModuleNotFoundError:
    _HERE = Path(__file__).resolve().parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    import wikitext_dataset
from safetensors.torch import load_file as safe_load_file

# --- Utility: Load config.json for model dims ---
def load_config(config_path: Path):
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        return {
            'd_model': cfg.get('hidden_size', 2048),
            'd_ff': cfg.get('intermediate_size', 1024),
            'n_experts': cfg.get('num_experts', 64),
            'vocab_size': cfg.get('vocab_size', 50304)
        }
    except Exception as e:
        print(f"[WARN] Could not load config.json: {e}")
        return {'d_model': 2048, 'd_ff': 1024, 'n_experts': 64, 'vocab_size': 50304}

# --- Expert class (from distributed_moe_synthetic_gradients.py) ---
class Expert(nn.Module):
    """Feed-forward expert block."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wo(self.dropout(torch.nn.functional.silu(self.wi(x))))

class OLMoEExpert(nn.Module):
    """OLMoE/Mixtral-style expert block: gate_proj, up_proj, down_proj."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        gate = torch.sigmoid(self.gate_proj(x))
        up = torch.nn.functional.silu(self.up_proj(x))
        return self.down_proj(up * gate)

# --- RoutingLogger ---
class RoutingLogger:
    """Logs (chunk_emb, expert_ids) pairs for oracle training.
    If memmap=True, saves as .npz for memory-efficient loading.
    TODO: For extreme scale, implement incremental append-to-file logging (e.g., with np.memmap or HDF5).
    """
    def __init__(self, d_model=2048, n_experts=64, memmap=False):
        self.buffer = []
        self.d_model = d_model
        self.n_experts = n_experts
        self.memmap = memmap
    def log(self, chunk_emb: torch.Tensor, expert_ids: List[int]):
        y = torch.zeros(self.n_experts)
        y[expert_ids] = 1
        self.buffer.append((chunk_emb.cpu().numpy(), y.numpy()))
    def flush(self, path: Path):
        try:
            if self.memmap:
                # Save as .npz arrays
                arrs = list(zip(*self.buffer))
                np.savez(path.with_suffix('.npz'),
                         embeddings=np.stack(arrs[0]),
                         labels=np.stack(arrs[1]))
            else:
                torch.save(self.buffer, path)
            self.buffer.clear()
        except Exception as e:
            print(f"[ERROR] Failed to save routing log: {e}")

class RoutingLoggerMemmap:
    """Production-grade streaming logger using np.memmap for (chunk_emb, expert_ids)."""
    def __init__(self, d_model=2048, n_experts=64, max_samples=1000000, log_path="oracle_log_memmap", flush_every=1000):
        self.d_model = d_model
        self.n_experts = n_experts
        self.max_samples = max_samples
        self.embeddings = np.memmap(f"{log_path}_embeddings.dat", dtype="float32", mode="w+", shape=(max_samples, d_model))
        self.labels = np.memmap(f"{log_path}_labels.dat", dtype="float32", mode="w+", shape=(max_samples, n_experts))
        self.idx = 0
        self.flush_every = flush_every
        self.log_path = log_path
    def log(self, chunk_emb: torch.Tensor, expert_ids: list):
        if self.idx >= self.max_samples:
            raise RuntimeError("RoutingLoggerMemmap: Exceeded max_samples")
        self.embeddings[self.idx, :] = chunk_emb.cpu().numpy()
        y = np.zeros(self.n_experts, dtype="float32")
        y[expert_ids] = 1.0
        self.labels[self.idx, :] = y
        self.idx += 1
        if self.idx % self.flush_every == 0:
            self.embeddings.flush()
            self.labels.flush()
    def flush(self):
        self.embeddings.flush()
        self.labels.flush()
        # Save as .npz for easy loading
        np.savez(f"{self.log_path}.npz", embeddings=self.embeddings[:self.idx], labels=self.labels[:self.idx])
        print(f"[INFO] Routing log memmap flushed to {self.log_path}.npz with {self.idx} samples.")

# --- OracleDataset and WarmupOracle ---
class OracleDataset(Dataset):
    """Loads (chunk_emb, expert_ids) pairs from disk for oracle training.
    If memmap=True, loads from .npz with memory mapping.
    """
    def __init__(self, log_file: Path, memmap=False):
        self.memmap = memmap
        try:
            if memmap or str(log_file).endswith(".npz"):
                npz = np.load(log_file.with_suffix('.npz'), mmap_mode='r')
                self.embeddings = npz['embeddings']
                self.labels = npz['labels']
                self.samples = None
            else:
                self.samples = torch.load(log_file, weights_only=False)
                self.embeddings = self.labels = None
        except Exception as e:
            print(f"[ERROR] Failed to load oracle log: {e}")
            self.samples = []
            self.embeddings = self.labels = None
    def __len__(self):
        if self.memmap and self.embeddings is not None:
            return self.embeddings.shape[0]
        return len(self.samples)
    def __getitem__(self, idx):
        if self.memmap and self.embeddings is not None:
            x = torch.from_numpy(self.embeddings[idx])
            y = torch.from_numpy(self.labels[idx])
            return x, y
        x, y = self.samples[idx]
        return x, y

def compute_pos_weight(samples, n_experts):
    if isinstance(samples, list):
        ys = torch.stack([torch.from_numpy(y) if isinstance(y, np.ndarray) else y for _, y in samples])
    else:
        ys = torch.from_numpy(samples)
    pos = ys.sum(dim=0)
    neg = len(ys) - pos
    pos_weight = neg / (pos + 1e-6)
    return pos_weight

class WarmupOracle(nn.Module):
    """Tiny MLP for multi-label expert prediction. Outputs logits (no sigmoid)."""
    def __init__(self, d_in: int, d_hidden: int, n_experts: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_experts)
        )
    def forward(self, x):
        return self.net(x)  # logits, no sigmoid

def train_oracle(log_path: Path, d_in, d_hidden, n_experts, device, memmap=False, epochs=3, batch_size=512, lr=3e-4):
    """Train the oracle on the routing log. Handles both in-memory and memmap logs. Prints progress and timing."""
    import time
    ds = OracleDataset(log_path, memmap=memmap)
    print(f"[INFO] Routing log contains {len(ds)} samples at {log_path}")
    if len(ds) == 0:
        print(f"[ERROR] Routing log is empty at {log_path}. Exiting.")
        return None
    print("[INFO] Training oracle on real routing log...")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    pos_weight = compute_pos_weight(ds.samples if ds.samples is not None else ds.labels, n_experts).to(device)
    model = WarmupOracle(d_in, d_hidden, n_experts).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    start = time.time()
    for ep in range(epochs):
        running_loss = 0.0
        for i, (x, y) in enumerate(dl):
            x, y = x.to(device), y.to(device)
            x = x.float()  # Ensure float32 for oracle
            preds = model(x)
            loss = loss_fn(preds, y)
            loss.backward(); opt.step(); opt.zero_grad()
            running_loss += loss.item()
            if (i+1) % 10 == 0:
                print(f"[INFO] Epoch {ep+1} Batch {i+1} Loss {loss.item():.4f}")
        print(f"[INFO] Epoch {ep+1} Average Loss {running_loss/(i+1):.4f}")
    print(f"[INFO] Oracle training complete. Time: {time.time()-start:.2f}s")
    return model.cpu()

# --- Simple Attention with KV Cache (demo, single-head) ---
class SimpleKVCacheAttention(nn.Module):
    """Production-grade single-head attention with efficient pre-allocated KV cache."""
    def __init__(self, d_model, max_seq_len=128):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5
        self.max_seq_len = max_seq_len
    def forward(self, x, k_cache, v_cache, cache_idx):
        # x: (batch, d_model)
        q = self.q_proj(x).unsqueeze(1)  # (batch, 1, d_model)
        if cache_idx > 0:
            k = k_cache[:, :cache_idx, :]  # (batch, seq, d_model)
            v = v_cache[:, :cache_idx, :]
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, 1, seq)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)  # (batch, 1, d_model)
            attn_output = attn_output.squeeze(1)
        else:
            attn_output = torch.zeros_like(x)
        return attn_output

# --- DiskMoELoader ---
class DiskMoELoader:
    """
    Async LRU cache for experts with in-flight load tracking.
    All cache ops are thread-safe. On cache miss, forward() will await the load.
    TODO: Add quantization support for expert weights (e.g., bitsandbytes, GGUF).
    TODO: Switch to safetensors for expert weights/logs for faster and safer I/O.
    """
    def __init__(self, expert_dir: Path, d_model=2048, d_ff=1024, n_experts=64, device="cuda", cache_size=4, use_real_olmoe=False):
        self.expert_dir = expert_dir
        self.device = device
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.cache_size = cache_size
        self.cache: OrderedDict[int, nn.Module] = OrderedDict()
        self.cache_lock = asyncio.Lock()
        # self.loop = asyncio.get_event_loop()  # REMOVE
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.in_flight_loads: Dict[int, asyncio.Event] = {}
        self.use_real_olmoe = use_real_olmoe
        self.expert_dtype = torch.float32  # Default, will update after loading real expert
        try:
            with open(expert_dir / "expert_info.json") as f:
                self.expert_info = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load expert_info.json: {e}")
            self.expert_info = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.io_time = 0.0
        self.cache_miss_wait_time = 0.0
        self.cache_miss_count = 0
    def _filename(self, idx: int) -> Path:
        return self.expert_dir / self.expert_info[str(idx)]["filename"]
    def _load_expert_sync(self, idx: int) -> Optional[nn.Module]:
        try:
            if self.use_real_olmoe:
                model = OLMoEExpert(self.d_model, self.d_ff)
            else:
                model = Expert(self.d_model, self.d_ff)
            fname = self._filename(idx)
            if fname.suffix == ".safetensors":
                state = safe_load_file(str(fname), device=self.device)
            else:
                # Legacy fallback for .pt files
                state = torch.load(fname, map_location=self.device)
            model.load_state_dict(state, strict=False)  # Allow extra keys
            model.eval()
            # Set dtype after loading first expert
            if hasattr(self, 'expert_dtype') and self.expert_dtype == torch.float32:
                try:
                    self.expert_dtype = next(model.parameters()).dtype
                except Exception:
                    pass
            return model
        except Exception as e:
            print(f"[ERROR] Failed to load expert {idx}: {e}")
            return None
    async def _load_and_cache_expert(self, idx: int):
        async with self.cache_lock:
            if idx in self.cache:
                return
            if idx in self.in_flight_loads:
                event = self.in_flight_loads[idx]
                await event.wait()
                return
            event = asyncio.Event()
            self.in_flight_loads[idx] = event
        t0 = time.time()
        try:
            loop = asyncio.get_running_loop()
            model = await loop.run_in_executor(self.executor, self._load_expert_sync, idx)
        finally:
            t1 = time.time()
            self.io_time += t1 - t0
            async with self.cache_lock:
                if idx in self.in_flight_loads:
                    self.in_flight_loads[idx].set()
                    del self.in_flight_loads[idx]
        if model is not None:
            async with self.cache_lock:
                self.cache[idx] = model
                self.cache.move_to_end(idx)
                while len(self.cache) > self.cache_size:
                    _, evicted = self.cache.popitem(last=False)
                    del evicted  # Rely on PyTorch's allocator to reclaim memory. Only add torch.cuda.empty_cache() if OOMs are observed.
    async def get_expert(self, idx: int) -> nn.Module:
        async with self.cache_lock:
            if idx in self.cache:
                self.cache_hits += 1
                self.cache.move_to_end(idx)
                return self.cache[idx]
        self.cache_misses += 1
        t0 = time.time()
        await self._load_and_cache_expert(idx)
        t1 = time.time()
        self.cache_miss_wait_time += t1 - t0
        self.cache_miss_count += 1
        async with self.cache_lock:
            if idx in self.cache:
                return self.cache[idx]
            else:
                raise RuntimeError(f"Failed to load expert {idx} after cache miss.")
    async def forward(self, hidden: torch.Tensor, expert_idx: int) -> torch.Tensor:
        model = await self.get_expert(expert_idx)
        return model(hidden)
    async def prefetch(self, idxs: List[int]):
        for i in idxs:
            async with self.cache_lock:
                if i not in self.cache and i not in self.in_flight_loads:
                    asyncio.create_task(self._load_and_cache_expert(i))
    def cache_stats(self):
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        avg_wait = self.cache_miss_wait_time / self.cache_miss_count if self.cache_miss_count > 0 else 0.0
        return {"hits": self.cache_hits, "misses": self.cache_misses, "hit_rate": hit_rate, "io_time": self.io_time, "avg_miss_wait": avg_wait}

# --- OLMoE Loader ---
def load_olmoe_components(olmoe_dir, device):
    """Load embedding and LM head weights from moe_app/models/olmo_sharded/model_components.pt."""
    pt_path = Path(olmoe_dir) / "model_components.pt"
    components = torch.load(pt_path, map_location=device)
    # Find embedding and lm_head keys
    embed_key = next(k for k in components if 'embed' in k.lower())
    lm_head_key = next(k for k in components if 'lm_head' in k.lower())
    embedding_w = components[embed_key]['weight']
    lm_head_w = components[lm_head_key]['weight']
    lm_head_b = components[lm_head_key].get('bias', None)
    return embedding_w, lm_head_w, lm_head_b

def save_router(router, olmoe_dir, filename="router.pt"):
    """Save the trained router to disk."""
    router_path = Path(olmoe_dir) / filename
    torch.save(router.state_dict(), router_path)
    print(f"[INFO] Saved router to {router_path}")

def load_router(olmoe_dir, d_model, n_experts, device, filename="router.pt"):
    """Load a trained router from disk, or create a new one if not found."""
    router_path = Path(olmoe_dir) / filename
    router = nn.Linear(d_model, n_experts).to(device)

    if router_path.exists():
        try:
            state_dict = torch.load(router_path, map_location=device)
            router.load_state_dict(state_dict)
            print(f"[INFO] Loaded existing router from {router_path}")
        except Exception as e:
            print(f"[WARN] Failed to load router from {router_path}: {e}")
            print("[INFO] Using randomly initialized router")
    else:
        print(f"[INFO] No router found at {router_path}, using randomly initialized router")

    return router

# --- HybridMoERuntime ---
class HybridMoERuntime:
    """
    Disk-backed MoE inference runtime with streaming decode, top-k routing, and oracle prefetch.
    Supports both demo and real OLMoE model integration.

    Parameters:
        oracle: WarmupOracle
        moe_loader: DiskMoELoader
        tokenizer_name: str
        d_model: int
        vocab_size: int
        seq_len: int
        top_k: int
        device: str
        batch_size: int
        router_top_k: int (top-k routing)
        oracle_recall_window: int (recall@N window)
        use_real_olmoe: bool (use real OLMoE embedding/attention/lm_head)
        olmoe_dir: str (path to OLMoE sharded model)
    """
    def __init__(self, oracle: WarmupOracle, moe_loader: DiskMoELoader,
                 tokenizer_name, d_model, vocab_size, seq_len=32, top_k=1, device="cuda", batch_size=1, router_top_k=1, oracle_recall_window=8, use_real_olmoe=False, olmoe_dir=None):
        self.oracle = oracle.to(device).eval()
        self.moe_loader = moe_loader
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.seq_len = seq_len
        self.top_k = top_k
        self.device = device
        self.lm_head = nn.Linear(d_model, vocab_size).to(device)
        # Use shared router for consistent routing between training and inference
        if olmoe_dir is not None:
            self.router = load_router(olmoe_dir, d_model, moe_loader.n_experts, device)
        else:
            self.router = nn.Linear(d_model, moe_loader.n_experts).to(device)
        self.moe_layernorm = nn.LayerNorm(d_model).to(device)
        self.router_layernorm = nn.LayerNorm(d_model).to(device)
        self.attn_layernorm = nn.LayerNorm(d_model).to(device)
        self.use_real_olmoe = use_real_olmoe
        if use_real_olmoe and olmoe_dir is not None:
            embedding_w, lm_head_w, lm_head_b = load_olmoe_components(olmoe_dir, device)
            self.embedding = nn.Embedding.from_pretrained(embedding_w, freeze=False).to(device)
            self.lm_head = nn.Linear(d_model, vocab_size).to(device)
            self.lm_head.weight.data.copy_(lm_head_w)
            if lm_head_b is not None:
                self.lm_head.bias.data.copy_(lm_head_b)
            # Use a real multi-head attention block
            self.attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True).to(device)
            # TODO: Load transformer block weights if available
        else:
            self.embedding = None
            self.attn = SimpleKVCacheAttention(d_model).to(device)
        self.backbone = None  # Plug in your transformer here
        self.oracle_prefetch_hits = 0
        self.oracle_prefetch_total = 0
        self.batch_size = batch_size
        self.router_top_k = router_top_k
        self.oracle_recall_window = oracle_recall_window
        self.oracle_recall_buffer = deque(maxlen=oracle_recall_window)
        self.oracle_recall_hits = 0
        self.oracle_recall_total = 0
        self.debug_routes = False
        self.debug_routes_buffer = []

    def reset_debug_routes(self):
        self.debug_routes_buffer = []

    def get_debug_routes(self):
        return list(self.debug_routes_buffer)

    async def _generation_loop(self, prompts: List[str]):
        """
        Internal async generator for autoregressive MoE generation.
        Yields: (step, next_ids, tokens, hidden, oracle metrics dict)
        """
        batch_size = len(prompts)
        toks = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        ids = toks["input_ids"]  # (batch, seq)
        with torch.inference_mode():
            max_seq_len = self.seq_len
            d_model = self.moe_loader.d_model
            # Pre-allocate KV caches for all attention paths
            k_cache = torch.zeros(batch_size, max_seq_len, d_model, device=self.device, dtype=self.moe_loader.expert_dtype)
            v_cache = torch.zeros(batch_size, max_seq_len, d_model, device=self.device, dtype=self.moe_loader.expert_dtype)
            cache_idx = 0
            if self.use_real_olmoe and self.embedding is not None:
                input_embeds = self.embedding(ids)
                input_embeds = input_embeds.to(self.moe_loader.expert_dtype)
                hidden = input_embeds[:, -1, :]
            elif self.backbone is not None:
                outputs = self.backbone(input_ids=ids)
                hidden = outputs.last_hidden_state[:, -1, :].to(self.device)
                hidden = hidden.to(self.moe_loader.expert_dtype)
            else:
                hidden = torch.randn(batch_size, d_model, device=self.device)
                hidden = hidden.to(self.moe_loader.expert_dtype)
            output_texts = list(prompts)
            chunk_emb = hidden[0]
            oracle_probs = self.oracle(chunk_emb.to(torch.float32).unsqueeze(0))
            warm_idxs = oracle_probs.topk(self.top_k, dim=-1).indices[0].tolist()
            await self.moe_loader.prefetch(warm_idxs)
            oracle_window = set(warm_idxs)
            for step in range(self.seq_len):
                if self.use_real_olmoe and self.embedding is not None:
                    # --- OLMoE MultiheadAttention with pre-allocated KV cache ---
                    if step == 0:
                        token_embeds = input_embeds[:, -1, :].unsqueeze(1)  # (batch, 1, d_model)
                    else:
                        last_ids = torch.tensor(next_ids, device=self.device).unsqueeze(1)
                        token_embeds = self.embedding(last_ids)
                    token_embeds = token_embeds.to(self.moe_loader.expert_dtype)
                    # Compute K/V for new token and insert into cache
                    # For MultiheadAttention, use in_proj_weight to get Q/K/V
                    # But for production, we use the official API and manage the cache manually
                    # Compute K/V for new token
                    k_new, v_new = None, None
                    # Use the internal projections of nn.MultiheadAttention
                    # (Assume batch_first=True)
                    attn_mod = self.attn
                    # Project new token to K/V (simulate what MultiheadAttention does internally)
                    k_new = attn_mod.in_proj_weight[d_model:2*d_model, :] @ token_embeds.squeeze(1).T
                    v_new = attn_mod.in_proj_weight[2*d_model:, :] @ token_embeds.squeeze(1).T
                    k_cache[:, cache_idx, :] = k_new.T
                    v_cache[:, cache_idx, :] = v_new.T
                    # Use all cached K/V up to cache_idx+1
                    # For MultiheadAttention, pass the full sequence so far
                    attn_out, _ = attn_mod(token_embeds, k_cache[:, :cache_idx+1, :], v_cache[:, :cache_idx+1, :])
                    hidden = hidden + attn_out.squeeze(1)
                    cache_idx += 1
                else:
                    # --- Custom single-head attention with pre-allocated KV cache ---
                    # Compute K/V for new token and insert into cache
                    k_cache[:, cache_idx, :] = self.attn.k_proj(hidden)
                    v_cache[:, cache_idx, :] = self.attn.v_proj(hidden)
                    attn_input = self.attn_layernorm(hidden)
                    attn_out = self.attn(attn_input, k_cache, v_cache, cache_idx+1)
                    hidden = hidden + attn_out
                    cache_idx += 1
                # MoE, router, top-k logic as before
                normed_hidden = self.router_layernorm(hidden)
                router_logits = self.router(normed_hidden)
                if self.router_top_k == 1:
                    expert_idxs = torch.argmax(router_logits, dim=-1)
                    expert_to_indices = defaultdict(list)
                    for i, expert_id in enumerate(expert_idxs.tolist()):
                        expert_to_indices[expert_id].append(i)
                    moe_input = self.moe_layernorm(hidden)
                    moe_input = moe_input.to(self.moe_loader.expert_dtype)
                    tasks = []
                    expert_order = []
                    for expert_id, indices in expert_to_indices.items():
                        mini_batch = moe_input[indices]
                        tasks.append(self.moe_loader.forward(mini_batch, expert_id))
                        expert_order.append((expert_id, indices))
                    moe_outputs = await asyncio.gather(*tasks)
                    moe_output = torch.zeros_like(hidden)
                    for out, (expert_id, indices) in zip(moe_outputs, expert_order):
                        moe_output[indices] = out
                else:
                    topk_vals, topk_idxs = torch.topk(router_logits, self.router_top_k, dim=-1)
                    moe_input = self.moe_layernorm(hidden)
                    moe_input = moe_input.to(self.moe_loader.expert_dtype)
                    expert_to_token_and_weight = defaultdict(list)
                    for i in range(batch_size):
                        for k in range(self.router_top_k):
                            expert_id = topk_idxs[i, k].item()
                            gate = torch.softmax(topk_vals[i], dim=0)[k].item()
                            expert_to_token_and_weight[expert_id].append((i, gate))
                    tasks = []
                    expert_order = []
                    for expert_id, token_gate_list in expert_to_token_and_weight.items():
                        indices = [i for i, _ in token_gate_list]
                        mini_batch = moe_input[indices]
                        tasks.append(self.moe_loader.forward(mini_batch, expert_id))
                        expert_order.append((expert_id, token_gate_list))
                    moe_outputs = await asyncio.gather(*tasks)
                    moe_output = torch.zeros_like(hidden)
                    for out, (expert_id, token_gate_list) in zip(moe_outputs, expert_order):
                        for j, (i, gate) in enumerate(token_gate_list):
                            moe_output[i] += out[j] * gate
                hidden = hidden + moe_output
                logits = self.lm_head(hidden)
                next_ids = torch.argmax(logits, dim=-1).tolist()
                tokens = [self.tokenizer.decode([nid]) for nid in next_ids]
                for i in range(batch_size):
                    output_texts[i] += tokens[i]
                if self.debug_routes:
                    if self.router_top_k == 1:
                        route = [int(x) for x in expert_idxs.tolist()]
                    else:
                        route = [[int(x) for x in row.tolist()] for row in topk_idxs]
                    self.debug_routes_buffer.append(route)
                # Oracle metrics
                if self.router_top_k == 1:
                    used_experts = set([expert_idxs[0].item()])
                else:
                    used_experts = set(topk_idxs[0, :self.router_top_k].tolist())
                self.oracle_prefetch_total += 1
                if any(e in oracle_window for e in used_experts):
                    self.oracle_prefetch_hits += 1
                self.oracle_recall_buffer.append(used_experts)
                if len(self.oracle_recall_buffer) == self.oracle_recall_window:
                    experts_in_window = set().union(*self.oracle_recall_buffer)
                    recall = len(experts_in_window & oracle_window) / max(1, len(experts_in_window))
                    self.oracle_recall_hits += recall
                    self.oracle_recall_total += 1
                metrics = {
                    'oracle_prefetch_total': self.oracle_prefetch_total,
                    'oracle_prefetch_hits': self.oracle_prefetch_hits,
                    'oracle_recall_total': self.oracle_recall_total,
                    'oracle_recall_hits': self.oracle_recall_hits,
                }
                yield (step, next_ids, tokens, hidden, metrics)
                # Oracle prefetch refresh
                if (step+1) % 8 == 0:
                    chunk_emb = hidden[0]
                    # Ensure oracle input is float32
                    oracle_probs = self.oracle(chunk_emb.to(torch.float32).unsqueeze(0))
                    warm_idxs = oracle_probs.topk(self.top_k, dim=-1).indices[0].tolist()
                    await self.moe_loader.prefetch(warm_idxs)
                    oracle_window = set(warm_idxs)

    async def run(self, prompts: List[str]):
        """
        Run MoE inference and return the final output texts for each prompt.
        Uses the internal generation loop.
        """
        output_texts = list(prompts)
        async for step, next_ids, tokens, hidden, metrics in self._generation_loop(prompts):
            for i in range(len(prompts)):
                output_texts[i] += tokens[i]
        return output_texts

    async def stream_generate(self, prompts: List[str]):
        """
        Async generator: yields (step, [token for each prompt]) as soon as each token is ready.
        Uses the internal generation loop.
        """
        async for step, next_ids, tokens, hidden, metrics in self._generation_loop(prompts):
            yield (step, tokens)

# --- MODIFIED: MinimalBackbone class to correctly use OLMoE components ---
class MinimalBackbone(nn.Module):
    """
    A minimal backbone that uses the real OLMoE embedding and final layernorm.
    This produces much more representative hidden states than just embeddings alone.
    """
    def __init__(self, embedding_weight: torch.Tensor, norm_weight: torch.Tensor, norm_bias: Optional[torch.Tensor]):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=True)
        self.norm = nn.LayerNorm(embedding_weight.size(1))
        self.norm.weight.data.copy_(norm_weight)
        if norm_bias is not None:
            self.norm.bias.data.copy_(norm_bias)
        self.norm.eval()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Get embeddings, apply final model norm
        x = self.embedding(input_ids)
        x = self.norm(x)
        return x

# --- Demo main() ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_dir', type=str, default='moe_app/models/olmo_sharded/experts')
    parser.add_argument('--tokenizer', type=str, default='moe_app/models/olmo_sharded')
    parser.add_argument('--config', type=str, default='moe_app/models/olmo_sharded/config.json')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--cache_size', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=32)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=None)
    parser.add_argument('--d_ff', type=int, default=None)
    parser.add_argument('--n_experts', type=int, default=None)
    parser.add_argument('--vocab_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--oracle_memmap', action='store_true', help='Use memory-mapped oracle log for large logs')
    parser.add_argument('--router_top_k', type=int, default=1, help='Top-k routing for router (1=top-1, 2=top-2, etc)')
    parser.add_argument('--oracle_recall_window', type=int, default=8, help='Window size for oracle recall@N metric')
    parser.add_argument('--use_real_olmoe', action='store_true', help='Use real OLMoE embedding, attention, and LM head')
    parser.add_argument('--olmoe_dir', type=str, default='moe_app/models/olmo_sharded', help='Path to OLMoE sharded model directory')
    parser.add_argument('--streaming', action='store_true', help='Use streaming decode (async generator)')
    parser.add_argument('--generate_routing_log_from_wikitext', action='store_true', help='Generate routing log from WikiText-103 using real OLMoE weights and router')
    parser.add_argument('--use_backbone', action='store_true', help='Use OLMoE backbone for chunk embeddings and inference')
    parser.add_argument('--logger_memmap', action='store_true', help='Use np.memmap streaming logger for routing log')
    args = parser.parse_args()
    # Load config for model dims
    config = load_config(Path(args.config))
    d_model = args.d_model or config['d_model']
    d_ff = args.d_ff or config['d_ff']
    n_experts = args.n_experts or config['n_experts']
    vocab_size = args.vocab_size or config['vocab_size']
    device = args.device
    expert_dir = Path(args.expert_dir)
    tokenizer_name = args.tokenizer
    olmoe_dir = args.olmoe_dir
    seq_len = args.seq_len
    batch_size = args.batch_size
    # --- Step 1: Generate routing log from WikiText-103 ---
    if args.generate_routing_log_from_wikitext:
        print("[INFO] Generating routing log from WikiText-103 with real OLMoE weights and router...")

        # --- MODIFIED: Load all components and find the correct keys ---
        pt_path = Path(olmoe_dir) / "model_components.pt"
        components = torch.load(pt_path, map_location=device)

        # Correctly find embedding, final norm, and router (if present)
        embed_key = next(k for k in components if k == 'model.embed_tokens')
        norm_key = next(k for k in components if k == 'model.norm')
        # The router is part of the expert block, which is not in model_components.pt
        # So we initialize a new one.

        embedding_w = components[embed_key]['weight']
        norm_w = components[norm_key]['weight']
        norm_b = components[norm_key].get('bias') # Bias is optional

        router_dtype = embedding_w.dtype
        # Use shared router that can be saved/loaded
        router = load_router(olmoe_dir, d_model, n_experts, device).to(dtype=router_dtype)
        router.eval()

        train_loader = wikitext_dataset.WikiText103DataLoader(
            split="train",
            batch_size=batch_size,
            seq_len=seq_len,
            tokenizer_name=tokenizer_name,
            max_samples=10000,
            buffer_size=1000
        )
        if args.logger_memmap:
            logger = RoutingLoggerMemmap(d_model=d_model, n_experts=n_experts, max_samples=100000, log_path="oracle_log_memmap")
            log_path = Path("oracle_log_memmap.npz")
        else:
            logger = RoutingLogger(d_model=d_model, n_experts=n_experts, memmap=args.oracle_memmap)
            log_path = Path("oracle_log.pt")
        total = 0

        backbone = None
        if args.use_backbone:
            print("[INFO] Using OLMoE MinimalBackbone (embedding + final norm) for chunk embeddings.")
            backbone = MinimalBackbone(embedding_w, norm_w, norm_b).to(device)
            embedding = backbone.embedding # Use the same embedding module
        else:
            # Fallback to just using the embedding layer if not using the full backbone
            embedding = nn.Embedding.from_pretrained(embedding_w, freeze=True).to(device)

        import time
        start = time.time()
        for tokens, _ in train_loader:
            tokens = tokens.to(device)
            with torch.no_grad():
                if args.use_backbone and backbone is not None:
                    # Pass tokens through the minimal backbone to get normed hidden states
                    x = backbone(tokens)
                else:
                    # Original behavior: just use embeddings
                    x = embedding(tokens)

            # Use mean of sequence as the representative chunk embedding
            chunk_emb = x.mean(dim=1).to(router_dtype)
            router_logits = router(chunk_emb)

            # For logging, we assume top-1 routing determines the "true" expert
            expert_ids = torch.argmax(router_logits, dim=-1).tolist()

            for i in range(chunk_emb.size(0)):
                logger.log(chunk_emb[i].detach().cpu(), [expert_ids[i]])
            total += chunk_emb.size(0)
            if total % 1000 == 0:
                print(f"[INFO] Processed {total} samples...")
        logger.flush()
        print(f"[INFO] Routing log generated at {log_path} with {total} samples. Time: {time.time()-start:.2f}s")

        # Save the router for consistent use in inference
        save_router(router, olmoe_dir)
        return
    # --- Step 2: Train oracle on real routing log ---
    if args.logger_memmap:
        log_path = Path("oracle_log_memmap.npz")
    else:
        log_path = Path("oracle_log.pt")
    oracle = train_oracle(log_path, d_in=d_model, d_hidden=512, n_experts=n_experts, device=device, memmap=args.logger_memmap or args.oracle_memmap, epochs=3, batch_size=512)
    if oracle is None:
        return
    # --- Step 3: Run inference ---
    print("[INFO] Running MoE inference with trained oracle...")
    moe_loader = DiskMoELoader(expert_dir, d_model=d_model, d_ff=d_ff, n_experts=n_experts, device=device, cache_size=args.cache_size, use_real_olmoe=args.use_real_olmoe)
    runtime = HybridMoERuntime(
        oracle, moe_loader, tokenizer_name, d_model, vocab_size, seq_len=seq_len, top_k=args.top_k, device=device, batch_size=batch_size, router_top_k=args.router_top_k, oracle_recall_window=args.oracle_recall_window, use_real_olmoe=args.use_real_olmoe, olmoe_dir=olmoe_dir)
    prompts = ["Hello, world!", "Test prompt"][:batch_size]
    import time
    async def go():
        start = time.time()
        if args.streaming:
            async for step, tokens in runtime.stream_generate(prompts):
                print(f"Step {step}: {tokens}")
        else:
            outs = await runtime.run(prompts)
            print("\nSample outputs:")
            for i, out in enumerate(outs):
                print(f"[{i}] {out}")
            print("Cache stats:", moe_loader.cache_stats())
        print(f"[PROFILE] Inference time: {time.time()-start:.2f}s")
    asyncio.run(go())

if __name__ == "__main__":
    main()
