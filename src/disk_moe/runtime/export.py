"""Stream native GLM4 weights into independent expert safetensors shards.

Only one resident tensor or one expert is materialized by the exporter at once.
The caller of export_model already owns an in-memory model; export_checkpoint
reads tensor slices directly from local or revision-pinned Hub safetensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath

import torch
import transformers
from huggingface_hub import hf_hub_download, model_info
from huggingface_hub.errors import EntryNotFoundError
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import GenerationConfig, Glm4MoeLiteConfig, Glm4MoeLiteForCausalLM
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteRotaryEmbedding

from .artifact import DTYPES, FORMAT_VERSION, NATIVE_VERSION, Artifact, file_metadata
from .rotary import snapshot_static_rotary, validate_static_rope_config

EXPERT_KEY = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.")
MTP_PREFIX = "model.layers.47."
MTP_POLICY = "native-glm4-moe-lite-declared-layer47-mtp-v1"
# Exact separate-expert auxiliary layout in the official GLM-4.7-Flash index at
# 7dd20894a642a0aa287e9827cb1a1f7f91386b67. This is not a blanket prefix ignore.
MTP_NONEXPERT_SUFFIXES = {
    "embed_tokens.weight", "shared_head.head.weight", "shared_head.norm.weight",
    "eh_proj.weight", "enorm.weight", "hnorm.weight",
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "mlp.gate.e_score_correction_bias", "mlp.gate.weight",
    "mlp.shared_experts.down_proj.weight", "mlp.shared_experts.gate_proj.weight",
    "mlp.shared_experts.up_proj.weight", "self_attn.kv_a_layernorm.weight",
    "self_attn.kv_a_proj_with_mqa.weight", "self_attn.kv_b_proj.weight",
    "self_attn.o_proj.weight", "self_attn.q_a_layernorm.weight",
    "self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight",
}


def _native_version() -> None:
    if transformers.__version__ != NATIVE_VERSION:
        raise ValueError(f"This adapter is tested with transformers=={NATIVE_VERSION}, found {transformers.__version__}")


def _check_config(config: Glm4MoeLiteConfig) -> None:
    if not isinstance(config, Glm4MoeLiteConfig):
        raise ValueError("Only a native Glm4MoeLiteConfig is supported")
    if len(config.mlp_layer_types) != config.num_hidden_layers or any(kind not in {"dense", "sparse"} for kind in config.mlp_layer_types):
        raise ValueError("mlp_layer_types must describe every layer as dense or sparse")
    if not 1 <= config.num_experts_per_tok <= config.n_routed_experts:
        raise ValueError("Invalid native router cardinality")
    if config.n_group < 1 or config.n_routed_experts % config.n_group or config.n_routed_experts // config.n_group < 2:
        raise ValueError("Native grouped routing requires at least two experts per group")
    if not 1 <= config.topk_group <= config.n_group:
        raise ValueError("Invalid native topk_group")
    if config.num_experts_per_tok > config.topk_group * (config.n_routed_experts // config.n_group):
        raise ValueError("Router top-k exceeds the selected expert groups")


class _MemorySource:
    def __init__(self, model):
        self.state = model.state_dict()
        self.names = set(self.state)

    def get(self, name: str, index: int | None = None):
        value = self.state[name]
        return value if index is None else value[index]

    def shape(self, name):
        return tuple(self.state[name].shape)


class _CheckpointSource:
    def __init__(self, model_name_or_path: str | Path, revision: str | None):
        source = str(model_name_or_path)
        self.local = Path(source).is_dir()
        self.root = Path(source).resolve() if self.local else None
        self.repo_id = None if self.local else source
        self.revision = None if self.local else model_info(source, revision=revision).sha
        config_path = self._file("config.json")
        data = json.loads(config_path.read_text())
        if data.get("model_type") != "glm4_moe_lite":
            raise ValueError("Checkpoint is not a GLM4 MoE Lite model; no remote code is loaded")
        if data.get("quantization_config"):
            raise ValueError("Quantized checkpoints require a separate kernel adapter; export an unquantized source")
        self.config = Glm4MoeLiteConfig.from_dict(data)
        try:
            generation_path = self._file("generation_config.json")
        except (FileNotFoundError, EntryNotFoundError):
            self.generation_config = GenerationConfig.from_model_config(self.config)
        else:
            self.generation_config = GenerationConfig.from_dict(json.loads(generation_path.read_text()))
        try:
            index_path = self._file("model.safetensors.index.json")
        except (FileNotFoundError, EntryNotFoundError):
            if self.local:
                shard = self._file("model.safetensors")
                with safe_open(shard, framework="pt", device="cpu") as tensors:
                    self.weight_map = dict.fromkeys(tensors.keys(), "model.safetensors")
            else:
                raise ValueError("Remote checkpoint has no safetensors index; download it to a local directory first so tensor names can be checked before payload transfer") from None
        else:
            index = json.loads(index_path.read_text())
            self.weight_map = index.get("weight_map")
            if not isinstance(self.weight_map, dict) or not self.weight_map:
                raise ValueError("Invalid safetensors checkpoint index")
        if any(not isinstance(name, str) or not isinstance(shard, str) for name, shard in self.weight_map.items()):
            raise ValueError("Checkpoint index tensor names and shard paths must be strings")
        for shard in set(self.weight_map.values()):
            self._validate_shard_name(shard)
        self.names = set(self.weight_map)
        self.source = {"kind": "local" if self.local else "huggingface", "name": source, "revision": self.revision}

    def _file(self, name):
        self._validate_shard_name(name)
        if self.local:
            candidate = self.root / name
            if not candidate.resolve().is_relative_to(self.root):
                raise ValueError("Checkpoint shard escapes source directory")
            if not candidate.is_file():
                raise FileNotFoundError(candidate)
            return candidate
        return Path(hf_hub_download(self.repo_id, filename=name, revision=self.revision))

    @staticmethod
    def _validate_shard_name(name):
        path = PurePosixPath(name)
        if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError("Checkpoint index contains an unsafe path")

    def get(self, name: str, index: int | None = None):
        with safe_open(self._file(self.weight_map[name]), framework="pt", device="cpu") as shard:
            if index is None:
                return shard.get_tensor(name)
            return shard.get_slice(name)[index]

    def shape(self, name):
        with safe_open(self._file(self.weight_map[name]), framework="pt", device="cpu") as shard:
            return tuple(shard.get_slice(name).get_shape())


def _copy(tensor, dtype):
    # clone is essential: serializing a view must never retain the source bank.
    return tensor.detach().to(device="cpu", dtype=dtype).contiguous().clone()


def _write_tensor_file(root, name, tensors):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    return file_metadata(path, root, tensors)


def _preflight_layout(source_names, config, expected):
    """Validate the complete name layout without reading any source tensor.

Only a complete, explicitly declared auxiliary MTP layer at index 47 can be
excluded, and only under the pinned native model's matching unused-key policy.
No live decoder tensor, partial expert bank, or arbitrary extra name is ignored.
"""
    resident_names = sorted(name for name in expected if not EXPERT_KEY.match(name))
    resident_sources = {
        name: "model.embed_tokens.weight" if name == "lm_head.weight" and name not in source_names and config.tie_word_embeddings else name
        for name in resident_names
    }
    required = set(resident_sources.values())
    expert_layouts = {}
    for layer, kind in enumerate(config.mlp_layer_types):
        if kind != "sparse":
            continue
        prefix = f"model.layers.{layer}.mlp.experts."
        packed = {prefix + part for part in ("gate_up_proj", "down_proj")}
        separate = {prefix + f"{expert}.{part}.weight" for expert in range(config.n_routed_experts) for part in ("gate_proj", "up_proj", "down_proj")}
        layout = "packed" if packed & source_names else "separate"
        expert_layouts[layer] = layout
        required.update(packed if layout == "packed" else separate)
    missing = required - source_names
    if missing:
        raise ValueError(f"Checkpoint name preflight: missing required tensor(s): {', '.join(sorted(missing)[:5])}")
    unexpected = source_names - required
    ignored = set()
    if unexpected and config.num_hidden_layers == 47 and getattr(config, "num_nextn_predict_layers", None) == 1 and config.q_lora_rank is not None and not config.attention_bias:
        auxiliary = {MTP_PREFIX + suffix for suffix in MTP_NONEXPERT_SUFFIXES}
        auxiliary.update(MTP_PREFIX + f"mlp.experts.{expert}.{part}.weight" for expert in range(config.n_routed_experts) for part in ("gate_proj", "up_proj", "down_proj"))
        native_ignored = Glm4MoeLiteForCausalLM._keys_to_ignore_on_load_unexpected
        if unexpected == auxiliary and all(any(re.match(pattern, name) for pattern in native_ignored) for name in auxiliary):
            ignored = auxiliary
    if unexpected - ignored:
        raise ValueError(f"Checkpoint name preflight: unsupported tensor(s): {', '.join(sorted(unexpected - ignored)[:5])}")
    report = {
        "source_tensor_count": len(source_names),
        "consumed_source_tensor_count": len(required),
        "ignored_auxiliary_tensor_count": len(ignored),
        "ignored_auxiliary_tensors": sorted(ignored),
        "ignored_auxiliary_policy": MTP_POLICY if ignored else None,
        "ignored_auxiliary_reason": "Declared auxiliary MTP layer 47 is unused by the pinned native causal decoder" if ignored else None,
    }
    return resident_sources, expert_layouts, report


def _export(source, config, output_dir, *, dtype, tokenizer, provenance, generation_config, rotary):
    _native_version()
    _check_config(config)
    validate_static_rope_config(config)
    destination = Path(output_dir).absolute()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing export directory: {destination}")
    with torch.device("meta"):
        template = Glm4MoeLiteForCausalLM(config)
    expected = template.state_dict()
    resident_sources, expert_layouts, preflight = _preflight_layout(source.names, config, expected)
    rotary_buffers = snapshot_static_rotary(rotary, config)
    # Name validation above must precede even dtype inference: getting the
    # embedding tensor can otherwise download a large source shard too early.
    if dtype is None:
        dtype = source.get("model.embed_tokens.weight").dtype
    if isinstance(dtype, str):
        dtype = DTYPES.get(dtype)
    if dtype not in DTYPES.values():
        raise ValueError("Export dtype must be float32, float16 or bfloat16")
    dtype_name = str(dtype).removeprefix("torch.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.export-", dir=destination.parent))
    try:
        config_data = config.to_dict()
        config_data["dtype"] = dtype_name
        (stage / "config.json").write_text(json.dumps(config_data, indent=2) + "\n")
        (stage / "generation_config.json").write_text(json.dumps(generation_config.to_dict(), indent=2) + "\n")
        rotary_entry = _write_tensor_file(stage, "runtime/rotary.safetensors", rotary_buffers)
        resident = []
        for number, (name, source_name) in enumerate(resident_sources.items()):
            if source.shape(source_name) != tuple(expected[name].shape):
                raise ValueError(f"Resident tensor has wrong shape: {source_name}")
            # Upstream keeps the native router correction bias in FP32.
            tensor_dtype = torch.float32 if name.endswith("e_score_correction_bias") else dtype
            tensors = {name: _copy(source.get(source_name), tensor_dtype)}
            resident.append(_write_tensor_file(stage, f"resident/{number:05d}.safetensors", tensors))
            del tensors
        experts = []
        for layer, kind in enumerate(config.mlp_layer_types):
            if kind != "sparse":
                continue
            prefix = f"model.layers.{layer}.mlp.experts."
            fused = expert_layouts[layer] == "packed"
            if fused:
                keys = [prefix + "gate_up_proj", prefix + "down_proj"]
                for name in keys:
                    if source.shape(name) != tuple(expected[name].shape):
                        raise ValueError(f"Missing or incompatible fused expert tensor: {name}")
            for expert in range(config.n_routed_experts):
                if fused:
                    tensors = {
                        "gate_up_proj": _copy(source.get(keys[0], expert), dtype),
                        "down_proj": _copy(source.get(keys[1], expert), dtype),
                    }
                else:
                    keys = [prefix + f"{expert}.{part}.weight" for part in ("gate_proj", "up_proj", "down_proj")]
                    shapes = [(config.moe_intermediate_size, config.hidden_size)] * 2 + [(config.hidden_size, config.moe_intermediate_size)]
                    if any(source.shape(name) != shape for name, shape in zip(keys, shapes, strict=True)):
                        raise ValueError(f"Wrong expert tensor shape at layer {layer}, expert {expert}")
                    tensors = {
                        "gate_up_proj": torch.cat([_copy(source.get(keys[0]), dtype), _copy(source.get(keys[1]), dtype)]),
                        "down_proj": _copy(source.get(keys[2]), dtype),
                    }
                entry = _write_tensor_file(stage, f"experts/{layer:04d}/{expert:04d}.safetensors", tensors)
                entry.update(layer=layer, expert=expert)
                experts.append(entry)
                del tensors
        manifest = {
            "format_version": FORMAT_VERSION,
            "architecture": "glm4_moe_lite",
            "transformers_version": NATIVE_VERSION,
            "dtype": dtype_name,
            "source": provenance,
            "checkpoint_preflight": preflight,
            "config_sha256": hashlib.sha256((stage / "config.json").read_bytes()).hexdigest(),
            "generation_config_sha256": hashlib.sha256((stage / "generation_config.json").read_bytes()).hexdigest(),
            "resident": resident,
            "rotary": rotary_entry,
            "experts": experts,
            "tokenizer_saved": tokenizer is not None,
        }
        if tokenizer is not None:
            tokenizer.save_pretrained(stage / "tokenizer")
        # Written only after every weight file has been completed. The destination
        # is not visible until the final directory rename on this filesystem.
        with (stage / "manifest.json").open("w") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        Artifact(stage)
        if destination.exists():
            raise FileExistsError(destination)
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return destination


def export_model(model, output_dir, *, dtype=None, tokenizer=None, source=None) -> Path:
    """Export resident native weights plus their supported static rotary state.

The source is never mutated. Explicit dtype conversion applies to learned
weights; rotary buffer values and their precision are preserved from the source.
"""
    if not isinstance(model, Glm4MoeLiteForCausalLM):
        raise ValueError("export_model requires native Glm4MoeLiteForCausalLM")
    return _export(_MemorySource(model), model.config, output_dir, dtype=dtype, tokenizer=tokenizer, provenance=source or {"kind": "in_memory"}, generation_config=model.generation_config, rotary=model.model.rotary_emb)


def export_checkpoint(model_name_or_path, output_dir, *, dtype=None, revision=None) -> Path:
    """Incremental local/indexed-Hub export; no model-sized weight allocation.

Hub downloads are pinned to the resolved commit. Disk space must accommodate the
source checkpoint plus export. Remote sources require a safetensors index for
name preflight before payload transfer; unsharded sources must be local first.
Provide a tokenizer separately when using the completed artifact.
"""
    source = _CheckpointSource(model_name_or_path, revision)
    return _export(source, source.config, output_dir, dtype=dtype, tokenizer=None, provenance=source.source, generation_config=source.generation_config, rotary=Glm4MoeLiteRotaryEmbedding(source.config))


def export_summary(path) -> dict:
    """Compact CLI report; full ignored-name provenance stays in the manifest."""
    manifest = json.loads((Path(path) / "manifest.json").read_text())
    preflight = manifest["checkpoint_preflight"]
    return {
        "artifact_dir": str(path),
        "format_version": manifest["format_version"],
        "source_tensor_count": preflight["source_tensor_count"],
        "ignored_auxiliary_tensor_count": preflight["ignored_auxiliary_tensor_count"],
        "ignored_auxiliary_policy": preflight["ignored_auxiliary_policy"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Local safetensors checkpoint or indexed Hugging Face repository")
    parser.add_argument("output")
    parser.add_argument("--dtype", choices=list(DTYPES), default=None)
    parser.add_argument("--revision", default=None)
    args = parser.parse_args(argv)
    result = export_checkpoint(args.source, args.output, dtype=args.dtype, revision=args.revision)
    print(json.dumps(export_summary(result)))


if __name__ == "__main__":
    main()
