#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import List, Optional
import sys

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from disk_moe_runtime import (
    DiskMoELoader,
    HybridMoERuntime,
    WarmupOracle,
    load_config,
    train_oracle,
)
from glm4_disk_runtime import Glm4DiskRuntime

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "static" / "index.html"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    prompt: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    max_new_tokens: Optional[int] = Field(default=None, ge=1, le=2048)
    router_top_k: Optional[int] = Field(default=None, ge=1)
    oracle_top_k: Optional[int] = Field(default=None, ge=1)
    stream: bool = False
    debug: bool = False


def _pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_oracle_log(path_arg: Optional[str]) -> Path:
    if path_arg:
        return Path(path_arg)
    memmap_path = Path("oracle_log_memmap.npz")
    if memmap_path.exists():
        return memmap_path
    pt_path = Path("oracle_log.pt")
    if pt_path.exists():
        return pt_path
    raise FileNotFoundError(
        "No oracle log found. Provide --oracle_log or generate oracle_log_memmap.npz / oracle_log.pt."
    )


def _messages_to_prompt(messages: List[ChatMessage]) -> str:
    lines: List[str] = []
    for msg in messages:
        role = msg.role.lower()
        if role == "user":
            prefix = "User"
        elif role == "assistant":
            prefix = "Assistant"
        else:
            prefix = "System"
        lines.append(f"{prefix}: {msg.content}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _load_or_train_oracle(
    log_path: Path,
    d_model: int,
    n_experts: int,
    device: str,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    oracle_weights: Optional[Path],
    save_oracle: Optional[Path],
    force_memmap: bool,
) -> WarmupOracle:
    if oracle_weights and oracle_weights.exists():
        model = WarmupOracle(d_model, hidden_dim, n_experts)
        state = torch.load(oracle_weights, map_location="cpu")
        model.load_state_dict(state)
        return model

    memmap = force_memmap or log_path.suffix == ".npz"
    model = train_oracle(
        log_path=log_path,
        d_in=d_model,
        d_hidden=hidden_dim,
        n_experts=n_experts,
        device=device,
        memmap=memmap,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )
    if model is None:
        raise RuntimeError("Oracle training failed or log was empty.")
    if save_oracle:
        save_oracle.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_oracle)
    return model


def build_runtime(args: argparse.Namespace) -> dict:
    if args.backend == "glm4":
        model_dir = Path(args.model_dir)
        tokenizer_name = args.tokenizer
        if not tokenizer_name:
            raise ValueError("GLM4 backend requires --tokenizer pointing to the original HF model or tokenizer path.")
        device = _pick_device(args.device)
        oracle_weights = Path(args.oracle_weights) if args.oracle_weights else None
        runtime = Glm4DiskRuntime(
            backbone_dir=model_dir,
            tokenizer_name=tokenizer_name,
            device=device,
            cache_size=args.cache_size,
            router_top_k=args.router_top_k,
            max_new_tokens=args.seq_len,
            attn_impl=args.attn_impl,
            oracle_weights=oracle_weights,
            oracle_hidden=args.oracle_hidden,
            oracle_top_k=args.oracle_top_k,
            oracle_layer=args.oracle_layer,
            oracle_interval=args.oracle_interval,
        )
        return {
            "runtime": runtime,
            "runtime_type": "glm4",
            "device": device,
            "config": runtime.config,
            "model_dir": model_dir,
            "expert_dir": model_dir / "experts",
            "tokenizer": tokenizer_name,
            "oracle_log": None,
            "oracle_enabled": runtime.oracle is not None,
            "oracle_top_k": runtime.oracle_top_k,
        }

    model_dir = Path(args.model_dir)
    expert_dir = Path(args.expert_dir) if args.expert_dir else (model_dir / "experts")
    tokenizer_name = args.tokenizer or str(model_dir)
    config_path = Path(args.config) if args.config else (model_dir / "config.json")

    cfg = load_config(config_path)
    d_model = cfg["d_model"]
    d_ff = cfg["d_ff"]
    n_experts = cfg["n_experts"]
    vocab_size = cfg["vocab_size"]

    device = _pick_device(args.device)
    log_path = _resolve_oracle_log(args.oracle_log)
    oracle_weights = Path(args.oracle_weights) if args.oracle_weights else None
    save_oracle = Path(args.save_oracle) if args.save_oracle else None

    oracle = _load_or_train_oracle(
        log_path=log_path,
        d_model=d_model,
        n_experts=n_experts,
        device=device,
        hidden_dim=args.oracle_hidden,
        epochs=args.oracle_epochs,
        batch_size=args.oracle_batch_size,
        lr=args.oracle_lr,
        oracle_weights=oracle_weights,
        save_oracle=save_oracle,
        force_memmap=args.oracle_memmap,
    )

    moe_loader = DiskMoELoader(
        expert_dir,
        d_model=d_model,
        d_ff=d_ff,
        n_experts=n_experts,
        device=device,
        cache_size=args.cache_size,
        use_real_olmoe=args.use_real_olmoe,
    )

    runtime = HybridMoERuntime(
        oracle=oracle,
        moe_loader=moe_loader,
        tokenizer_name=tokenizer_name,
        d_model=d_model,
        vocab_size=vocab_size,
        seq_len=args.seq_len,
        top_k=args.oracle_top_k,
        device=device,
        batch_size=args.batch_size,
        router_top_k=args.router_top_k,
        oracle_recall_window=args.oracle_recall_window,
        use_real_olmoe=args.use_real_olmoe,
        olmoe_dir=str(model_dir),
    )

    return {
        "runtime": runtime,
        "runtime_type": "olmo",
        "device": device,
        "config": cfg,
        "model_dir": model_dir,
        "expert_dir": expert_dir,
        "tokenizer": tokenizer_name,
        "oracle_log": log_path,
        "oracle_enabled": True,
        "oracle_top_k": runtime.top_k,
    }


def create_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="Disk-Backed MoE Chat", version="0.1.0")

    @app.on_event("startup")
    async def _startup() -> None:
        state = build_runtime(args)
        app.state.runtime = state["runtime"]
        app.state.runtime_type = state["runtime_type"]
        app.state.device = state["device"]
        app.state.config = state["config"]
        app.state.model_dir = state["model_dir"]
        app.state.expert_dir = state["expert_dir"]
        app.state.tokenizer = state["tokenizer"]
        app.state.oracle_log = state["oracle_log"]
        app.state.oracle_enabled = state.get("oracle_enabled", False)
        app.state.oracle_top_k = state.get("oracle_top_k", None)
        app.state.lock = asyncio.Lock()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        if INDEX_PATH.exists():
            return HTMLResponse(INDEX_PATH.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>Disk-Backed MoE Chat</h1><p>UI not found.</p>")

    @app.get("/api/health")
    async def health() -> JSONResponse:
        cfg = getattr(app.state, "config", {})
        if hasattr(cfg, "to_dict"):
            cfg = cfg.to_dict()
        return JSONResponse(
            {
                "status": "ok",
                "backend": getattr(app.state, "runtime_type", None),
                "device": getattr(app.state, "device", None),
                "model_dir": str(getattr(app.state, "model_dir", "")),
                "expert_dir": str(getattr(app.state, "expert_dir", "")),
                "tokenizer": str(getattr(app.state, "tokenizer", "")),
                "oracle_log": str(getattr(app.state, "oracle_log", "")),
                "oracle_enabled": getattr(app.state, "oracle_enabled", False),
                "oracle_top_k": getattr(app.state, "oracle_top_k", None),
                "config": cfg,
            }
        )

    async def _run_chat(prompt: str, req: ChatRequest) -> dict:
        if app.state.runtime_type == "glm4":
            runtime: Glm4DiskRuntime = app.state.runtime
            if req.router_top_k and req.router_top_k > runtime.config.n_routed_experts:
                raise HTTPException(status_code=400, detail="router_top_k exceeds number of experts")
            if (
                req.oracle_top_k
                and runtime.oracle is not None
                and req.oracle_top_k > runtime.config.n_routed_experts
            ):
                raise HTTPException(status_code=400, detail="oracle_top_k exceeds number of experts")
            previous = (runtime.seq_len, runtime.router_top_k, runtime.oracle_top_k)
            runtime.seq_len = req.max_new_tokens or runtime.seq_len
            if req.router_top_k:
                runtime.set_router_top_k(req.router_top_k)
            if req.oracle_top_k and runtime.oracle is not None:
                runtime.set_oracle_top_k(req.oracle_top_k)
            try:
                outputs = await runtime.run([prompt])
                return {
                    "text": outputs[0],
                    "cache_stats": runtime.cache_stats(),
                    "oracle_metrics": {},
                }
            finally:
                runtime.seq_len, runtime.router_top_k, runtime.oracle_top_k = previous
        runtime: HybridMoERuntime = app.state.runtime
        if req.router_top_k and req.router_top_k > runtime.moe_loader.n_experts:
            raise HTTPException(status_code=400, detail="router_top_k exceeds number of experts")
        previous = (runtime.seq_len, runtime.router_top_k, runtime.top_k, runtime.debug_routes)
        runtime.seq_len = req.max_new_tokens or runtime.seq_len
        runtime.router_top_k = req.router_top_k or runtime.router_top_k
        runtime.top_k = req.oracle_top_k or runtime.top_k
        runtime.debug_routes = bool(req.debug)
        if req.debug:
            runtime.reset_debug_routes()
        try:
            outputs = await runtime.run([prompt])
            cache_stats = runtime.moe_loader.cache_stats()
            oracle_metrics = {
                "oracle_prefetch_total": runtime.oracle_prefetch_total,
                "oracle_prefetch_hits": runtime.oracle_prefetch_hits,
                "oracle_recall_total": runtime.oracle_recall_total,
                "oracle_recall_hits": runtime.oracle_recall_hits,
            }
            payload = {
                "text": outputs[0],
                "cache_stats": cache_stats,
                "oracle_metrics": oracle_metrics,
            }
            if req.debug:
                payload["routes"] = runtime.get_debug_routes()
            return payload
        finally:
            runtime.seq_len, runtime.router_top_k, runtime.top_k, runtime.debug_routes = previous

    @app.post("/api/chat")
    async def chat(req: ChatRequest) -> JSONResponse:
        if not req.prompt and not req.messages:
            raise HTTPException(status_code=400, detail="Provide prompt or messages")
        if req.prompt:
            prompt = req.prompt
        else:
            prompt = _messages_to_prompt(req.messages or [])
        async with app.state.lock:
            result = await _run_chat(prompt, req)
        payload = {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(app.state.model_dir),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["text"]},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
            "cache_stats": result["cache_stats"],
            "oracle_metrics": result["oracle_metrics"],
        }
        return JSONResponse(payload)

    @app.post("/api/chat/stream")
    async def chat_stream(req: ChatRequest) -> StreamingResponse:
        if not req.prompt and not req.messages:
            raise HTTPException(status_code=400, detail="Provide prompt or messages")
        if req.prompt:
            prompt = req.prompt
        else:
            prompt = _messages_to_prompt(req.messages or [])

        async def event_generator():
            previous = None
            try:
                async with app.state.lock:
                    if app.state.runtime_type == "glm4":
                        runtime: Glm4DiskRuntime = app.state.runtime
                        if req.router_top_k and req.router_top_k > runtime.config.n_routed_experts:
                            raise HTTPException(status_code=400, detail="router_top_k exceeds number of experts")
                        if (
                            req.oracle_top_k
                            and runtime.oracle is not None
                            and req.oracle_top_k > runtime.config.n_routed_experts
                        ):
                            raise HTTPException(status_code=400, detail="oracle_top_k exceeds number of experts")
                        previous = (runtime.seq_len, runtime.router_top_k, runtime.oracle_top_k)
                        runtime.seq_len = req.max_new_tokens or runtime.seq_len
                        if req.router_top_k:
                            runtime.set_router_top_k(req.router_top_k)
                        if req.oracle_top_k and runtime.oracle is not None:
                            runtime.set_oracle_top_k(req.oracle_top_k)
                        async for step, tokens in runtime.stream_generate([prompt]):
                            token = tokens[0] if tokens else ""
                            data = json.dumps({"step": step, "token": token})
                            yield f"data: {data}\n\n"
                    else:
                        runtime: HybridMoERuntime = app.state.runtime
                        if req.router_top_k and req.router_top_k > runtime.moe_loader.n_experts:
                            raise HTTPException(status_code=400, detail="router_top_k exceeds number of experts")
                        previous = (runtime.seq_len, runtime.router_top_k, runtime.top_k)
                        runtime.seq_len = req.max_new_tokens or runtime.seq_len
                        runtime.router_top_k = req.router_top_k or runtime.router_top_k
                        runtime.top_k = req.oracle_top_k or runtime.top_k
                        async for step, tokens in runtime.stream_generate([prompt]):
                            token = tokens[0] if tokens else ""
                            data = json.dumps({"step": step, "token": token})
                            yield f"data: {data}\n\n"
                yield "data: [DONE]\n\n"
            except asyncio.CancelledError:
                raise
            finally:
                if previous is not None:
                    if app.state.runtime_type == "glm4":
                        runtime.seq_len, runtime.router_top_k, runtime.oracle_top_k = previous
                    else:
                        runtime.seq_len, runtime.router_top_k, runtime.top_k = previous

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Disk-Backed MoE Chat Server")
    parser.add_argument("--backend", type=str, default="olmo", choices=["olmo", "glm4"])
    parser.add_argument("--model_dir", type=str, default="moe_app/models/olmo_sharded")
    parser.add_argument("--expert_dir", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--cache_size", type=int, default=4)
    parser.add_argument("--attn_impl", type=str, default="eager")
    parser.add_argument("--router_top_k", type=int, default=1)
    parser.add_argument("--oracle_top_k", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--oracle_log", type=str, default=None)
    parser.add_argument("--oracle_memmap", action="store_true")
    parser.add_argument("--oracle_weights", type=str, default=None)
    parser.add_argument("--save_oracle", type=str, default=None)
    parser.add_argument("--oracle_hidden", type=int, default=512)
    parser.add_argument("--oracle_epochs", type=int, default=3)
    parser.add_argument("--oracle_batch_size", type=int, default=512)
    parser.add_argument("--oracle_lr", type=float, default=3e-4)
    parser.add_argument("--oracle_recall_window", type=int, default=8)
    parser.add_argument("--oracle_layer", type=int, default=None)
    parser.add_argument("--oracle_interval", type=int, default=1)
    parser.add_argument("--use_real_olmoe", action="store_true")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import uvicorn

    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
