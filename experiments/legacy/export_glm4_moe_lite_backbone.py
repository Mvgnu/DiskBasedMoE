#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch
from safetensors import safe_open
from transformers import Glm4MoeLiteForCausalLM, Glm4MoeLiteConfig
from huggingface_hub import hf_hub_download, snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GLM4 MoE Lite weights for disk-backed backbone + experts.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--streaming", action="store_true", help="Stream weights from safetensors without loading full model.")
    return parser.parse_args()


def _dtype_from_arg(name: str):
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {name}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "layers").mkdir(parents=True, exist_ok=True)

    dtype = _dtype_from_arg(args.dtype)
    config = Glm4MoeLiteConfig.from_pretrained(args.model_name_or_path, revision=args.revision)
    config.save_pretrained(output_dir)
    with open(output_dir / "export_meta.json", "w") as f:
        json.dump(
            {
                "source": args.model_name_or_path,
                "dtype": args.dtype,
                "num_layers": config.num_hidden_layers,
                "num_experts": config.n_routed_experts,
                "num_shared_experts": config.n_shared_experts,
                "model_type": config.model_type,
                "streaming": args.streaming,
            },
            f,
            indent=2,
        )

    if args.streaming:
        model_path = Path(args.model_name_or_path)
        if model_path.exists():
            model_dir = model_path
        else:
            model_dir = Path(
                snapshot_download(
                    repo_id=args.model_name_or_path,
                    revision=args.revision,
                    allow_patterns=["*.safetensors", "*.json"],
                    resume_download=True,
                )
            )
        index_path = model_dir / "model.safetensors.index.json"
        if not index_path.exists():
            index_file = hf_hub_download(
                repo_id=args.model_name_or_path,
                filename="model.safetensors.index.json",
                revision=args.revision,
            )
            index_path = Path(index_file)
            model_dir = index_path.parent
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index["weight_map"]

        def file_path(fname: str) -> Path:
            return model_dir / fname

        def load_tensors(names: list[str]) -> dict:
            by_file = defaultdict(list)
            for name in names:
                by_file[weight_map[name]].append(name)
            out = {}
            for fname, keys in by_file.items():
                with safe_open(file_path(fname), framework="pt", device="cpu") as sf:
                    for key in keys:
                        out[key] = sf.get_tensor(key)
            return out

        # Embeddings / norm / lm_head
        embed_name = "model.embed_tokens.weight"
        norm_name = "model.norm.weight"
        lm_head_name = "lm_head.weight"
        root_tensors = load_tensors([embed_name, norm_name, lm_head_name])
        torch.save({"weight": root_tensors[embed_name]}, output_dir / "embeddings.pt")
        torch.save({"weight": root_tensors[norm_name]}, output_dir / "final_norm.pt")
        torch.save({"weight": root_tensors[lm_head_name]}, output_dir / "lm_head.pt")

        num_experts = getattr(config, "num_local_experts", None) or config.n_routed_experts
        mlp_layer_types = getattr(config, "mlp_layer_types", ["sparse"] * config.num_hidden_layers)

        for layer_idx in range(config.num_hidden_layers):
            layer_dir = output_dir / f"layer_{layer_idx}_experts"
            layer_dir.mkdir(parents=True, exist_ok=True)
            prefix = f"model.layers.{layer_idx}."

            attn_keys = [
                prefix + "self_attn.q_a_proj.weight",
                prefix + "self_attn.q_a_layernorm.weight",
                prefix + "self_attn.q_b_proj.weight",
                prefix + "self_attn.kv_a_proj_with_mqa.weight",
                prefix + "self_attn.kv_a_layernorm.weight",
                prefix + "self_attn.kv_b_proj.weight",
                prefix + "self_attn.o_proj.weight",
                prefix + "input_layernorm.weight",
                prefix + "post_attention_layernorm.weight",
            ]
            tensors = load_tensors(attn_keys)
            layer_state = {
                "self_attn": {
                    "q_a_proj.weight": tensors[prefix + "self_attn.q_a_proj.weight"],
                    "q_a_layernorm.weight": tensors[prefix + "self_attn.q_a_layernorm.weight"],
                    "q_b_proj.weight": tensors[prefix + "self_attn.q_b_proj.weight"],
                    "kv_a_proj_with_mqa.weight": tensors[prefix + "self_attn.kv_a_proj_with_mqa.weight"],
                    "kv_a_layernorm.weight": tensors[prefix + "self_attn.kv_a_layernorm.weight"],
                    "kv_b_proj.weight": tensors[prefix + "self_attn.kv_b_proj.weight"],
                    "o_proj.weight": tensors[prefix + "self_attn.o_proj.weight"],
                },
                "input_layernorm": {"weight": tensors[prefix + "input_layernorm.weight"]},
                "post_attention_layernorm": {"weight": tensors[prefix + "post_attention_layernorm.weight"]},
            }

            mlp_type = mlp_layer_types[layer_idx] if layer_idx < len(mlp_layer_types) else "sparse"
            layer_state["mlp_type"] = mlp_type

            sparse_gate_key = prefix + "mlp.experts.gate_up_proj"
            sparse_gate_sep_key = prefix + "mlp.experts.0.gate_proj.weight"
            dense_gate_key = prefix + "mlp.gate_proj.weight"
            has_sparse = sparse_gate_key in weight_map or sparse_gate_sep_key in weight_map

            if (mlp_type == "dense" and not has_sparse) or (not has_sparse and dense_gate_key in weight_map):
                mlp_keys = [
                    prefix + "mlp.gate_proj.weight",
                    prefix + "mlp.up_proj.weight",
                    prefix + "mlp.down_proj.weight",
                ]
                mlp_tensors = load_tensors(mlp_keys)
                layer_state["mlp"] = {
                    "gate_proj.weight": mlp_tensors[prefix + "mlp.gate_proj.weight"],
                    "up_proj.weight": mlp_tensors[prefix + "mlp.up_proj.weight"],
                    "down_proj.weight": mlp_tensors[prefix + "mlp.down_proj.weight"],
                }
                layer_state["mlp_type"] = "dense"
                torch.save(layer_state, output_dir / "layers" / f"layer_{layer_idx}.pt")
                print(f"[EXPORT] Layer {layer_idx}: dense MLP saved.")
                continue
            if not has_sparse and dense_gate_key not in weight_map:
                raise KeyError(
                    f"Missing MLP keys for layer {layer_idx}; expected sparse or dense keys under {prefix}mlp.*"
                )

            gate_name = prefix + "mlp.gate.weight"
            gate_bias_name = prefix + "mlp.gate.e_score_correction_bias"
            shared_keys = [
                prefix + "mlp.shared_experts.gate_proj.weight",
                prefix + "mlp.shared_experts.up_proj.weight",
                prefix + "mlp.shared_experts.down_proj.weight",
            ]
            gate_tensors = load_tensors([gate_name, gate_bias_name] + shared_keys)
            layer_state["gate"] = {
                "weight": gate_tensors[gate_name],
                "e_score_correction_bias": gate_tensors[gate_bias_name],
            }
            layer_state["shared_experts"] = {
                "gate_proj.weight": gate_tensors[shared_keys[0]],
                "up_proj.weight": gate_tensors[shared_keys[1]],
                "down_proj.weight": gate_tensors[shared_keys[2]],
            }
            torch.save(layer_state, output_dir / "layers" / f"layer_{layer_idx}.pt")

            expert_gate_name = prefix + "mlp.experts.0.gate_proj.weight"
            expert_up_name = prefix + "mlp.experts.0.up_proj.weight"
            expert_down_name = prefix + "mlp.experts.0.down_proj.weight"
            uses_separate = expert_gate_name in weight_map and expert_up_name in weight_map

            expert_info = {}
            if uses_separate:
                for expert_idx in range(num_experts):
                    gate_k = prefix + f"mlp.experts.{expert_idx}.gate_proj.weight"
                    up_k = prefix + f"mlp.experts.{expert_idx}.up_proj.weight"
                    down_k = prefix + f"mlp.experts.{expert_idx}.down_proj.weight"
                    tensors = load_tensors([gate_k, up_k, down_k])
                    fname = f"expert_{expert_idx}.pt"
                    torch.save(
                        {
                            "gate_proj": tensors[gate_k],
                            "up_proj": tensors[up_k],
                            "down_proj": tensors[down_k],
                        },
                        layer_dir / fname,
                    )
                    expert_info[str(expert_idx)] = {"filename": fname}
            else:
                gate_up_name = prefix + "mlp.experts.gate_up_proj"
                down_name = prefix + "mlp.experts.down_proj"
                gate_file = file_path(weight_map[gate_up_name])
                down_file = file_path(weight_map[down_name])
                if gate_file == down_file:
                    with safe_open(gate_file, framework="pt", device="cpu") as sf:
                        gate_view = sf.get_slice(gate_up_name)
                        down_view = sf.get_slice(down_name)
                        for expert_idx in range(num_experts):
                            fname = f"expert_{expert_idx}.pt"
                            torch.save(
                                {
                                    "gate_up_proj": gate_view[expert_idx].clone(),
                                    "down_proj": down_view[expert_idx].clone(),
                                },
                                layer_dir / fname,
                            )
                            expert_info[str(expert_idx)] = {"filename": fname}
                else:
                    with safe_open(gate_file, framework="pt", device="cpu") as sf_gate, safe_open(
                        down_file, framework="pt", device="cpu"
                    ) as sf_down:
                        gate_view = sf_gate.get_slice(gate_up_name)
                        down_view = sf_down.get_slice(down_name)
                        for expert_idx in range(num_experts):
                            fname = f"expert_{expert_idx}.pt"
                            torch.save(
                                {
                                    "gate_up_proj": gate_view[expert_idx].clone(),
                                    "down_proj": down_view[expert_idx].clone(),
                                },
                                layer_dir / fname,
                            )
                            expert_info[str(expert_idx)] = {"filename": fname}
            with open(layer_dir / "expert_info.json", "w") as f:
                json.dump(expert_info, f, indent=2)
            print(f"[EXPORT] Layer {layer_idx}: saved {num_experts} experts.")
        print(f"[DONE] Exported GLM4 MoE Lite backbone to {output_dir}")
        return

    model = Glm4MoeLiteForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
        revision=args.revision,
    )
    model.eval()

    torch.save(model.model.embed_tokens.state_dict(), output_dir / "embeddings.pt")
    torch.save(model.model.norm.state_dict(), output_dir / "final_norm.pt")
    torch.save(model.lm_head.state_dict(), output_dir / "lm_head.pt")

    for layer_idx, layer in enumerate(model.model.layers):
        layer_dir = output_dir / f"layer_{layer_idx}_experts"
        layer_dir.mkdir(parents=True, exist_ok=True)

        layer_state = {
            "self_attn": layer.self_attn.state_dict(),
            "input_layernorm": layer.input_layernorm.state_dict(),
            "post_attention_layernorm": layer.post_attention_layernorm.state_dict(),
        }
        is_sparse = hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "experts")
        layer_state["mlp_type"] = "sparse" if is_sparse else "dense"
        if is_sparse:
            layer_state["gate"] = layer.mlp.gate.state_dict()
            layer_state["shared_experts"] = layer.mlp.shared_experts.state_dict()
        else:
            layer_state["mlp"] = layer.mlp.state_dict()
        torch.save(layer_state, output_dir / "layers" / f"layer_{layer_idx}.pt")

        if is_sparse:
            experts = layer.mlp.experts
            num_experts = experts.gate_up_proj.shape[0]
            expert_info = {}
            for expert_idx in range(num_experts):
                fname = f"expert_{expert_idx}.pt"
                torch.save(
                    {
                        "gate_up_proj": experts.gate_up_proj[expert_idx].cpu(),
                        "down_proj": experts.down_proj[expert_idx].cpu(),
                    },
                    layer_dir / fname,
                )
                expert_info[str(expert_idx)] = {"filename": fname}
            with open(layer_dir / "expert_info.json", "w") as f:
                json.dump(expert_info, f, indent=2)
            print(f"[EXPORT] Layer {layer_idx}: saved {num_experts} experts.")
        else:
            print(f"[EXPORT] Layer {layer_idx}: dense MLP saved.")

    print(f"[DONE] Exported GLM4 MoE Lite backbone to {output_dir}")


if __name__ == "__main__":
    main()
