from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import Glm4MoeLiteForCausalLM

from disk_moe.runtime import export_checkpoint, load_model
from disk_moe.runtime.demo import tiny_config, tiny_model
from disk_moe.runtime.export import _CheckpointSource, export_summary


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _metadata_only_source(tmp_path, config, names):
    source = tmp_path / "source"
    source.mkdir()
    config.save_pretrained(source)
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": dict.fromkeys(names, "payload-must-not-be-read.safetensors")}))
    return source


@pytest.mark.parametrize("mutation", ["missing_resident", "missing_expert", "unknown", "auxiliary_unknown", "mixed_expert_layout"])
def test_invalid_name_layout_fails_before_any_tensor_or_shard_read(tmp_path, monkeypatch, mutation):
    native = tiny_model()
    names = set(native.state_dict())
    if mutation == "missing_resident":
        names.remove("model.norm.weight")
    elif mutation == "missing_expert":
        names.remove("model.layers.1.mlp.experts.down_proj")
    elif mutation == "unknown":
        names.add("arbitrary.unrecognized.weight")
    elif mutation == "auxiliary_unknown":
        names.add("model.layers.47.arbitrary.unrecognized.weight")
    else:
        names.add("model.layers.1.mlp.experts.0.gate_proj.weight")
    source = _metadata_only_source(tmp_path, native.config, names)

    def forbidden(*args, **kwargs):
        pytest.fail("Name preflight must precede tensor materialization and shard header reads")

    monkeypatch.setattr(_CheckpointSource, "get", forbidden)
    monkeypatch.setattr(_CheckpointSource, "shape", forbidden)
    with pytest.raises(ValueError, match="name preflight"):
        export_checkpoint(source, tmp_path / "artifact")
    assert not (tmp_path / "artifact").exists()


def _native_47_with_auxiliary():
    config = tiny_config()
    config.num_hidden_layers = 47
    config.mlp_layer_types = ["dense"] + ["sparse"] * 46
    config.num_nextn_predict_layers = 1
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    torch.manual_seed(173)
    native = Glm4MoeLiteForCausalLM(config).eval()
    state = native.state_dict()
    auxiliary = {}
    for name, tensor in state.items():
        if name.startswith("model.layers.46.") and ".experts." not in name:
            auxiliary[name.replace("model.layers.46.", "model.layers.47.")] = tensor.clone()
    prefix = "model.layers.47."
    auxiliary[prefix + "embed_tokens.weight"] = state["model.embed_tokens.weight"].clone()
    auxiliary[prefix + "shared_head.head.weight"] = state["lm_head.weight"].clone()
    for name in ("shared_head.norm.weight", "enorm.weight", "hnorm.weight"):
        auxiliary[prefix + name] = state["model.norm.weight"].clone()
    auxiliary[prefix + "eh_proj.weight"] = torch.zeros(config.hidden_size, 2 * config.hidden_size)
    for expert in range(config.n_routed_experts):
        gate, up = state["model.layers.46.mlp.experts.gate_up_proj"][expert].chunk(2, dim=0)
        auxiliary[prefix + f"mlp.experts.{expert}.gate_proj.weight"] = gate.clone()
        auxiliary[prefix + f"mlp.experts.{expert}.up_proj.weight"] = up.clone()
        auxiliary[prefix + f"mlp.experts.{expert}.down_proj.weight"] = state["model.layers.46.mlp.experts.down_proj"][expert].clone()
    return native, auxiliary


def test_declared_native_auxiliary_is_never_loaded_or_exported_and_logits_match(tmp_path, monkeypatch):
    native, auxiliary = _native_47_with_auxiliary()
    source = tmp_path / "source"
    source.mkdir()
    native.config.save_pretrained(source)
    native.generation_config.save_pretrained(source)
    state = native.state_dict()
    save_file(state, source / "decoder.safetensors")
    save_file(auxiliary, source / "unused-mtp.safetensors")
    weight_map = dict.fromkeys(state, "decoder.safetensors") | dict.fromkeys(auxiliary, "unused-mtp.safetensors")
    (source / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    # Compare to the pinned native loader's treatment of the same augmented
    # checkpoint, not to a hand-selected subset or the exporter's own account.
    reference = Glm4MoeLiteForCausalLM.from_pretrained(source, attn_implementation="eager", experts_implementation="eager").eval()
    original_file = _CheckpointSource._file
    original_get = _CheckpointSource.get
    original_shape = _CheckpointSource.shape
    visited = []

    def file(self, name):
        assert name != "unused-mtp.safetensors", "Auxiliary-only shards must not be fetched"
        return original_file(self, name)

    def get(self, name, index=None):
        assert not name.startswith("model.layers.47.")
        visited.append(name)
        return original_get(self, name, index)

    def shape(self, name):
        assert not name.startswith("model.layers.47.")
        return original_shape(self, name)

    monkeypatch.setattr(_CheckpointSource, "_file", file)
    monkeypatch.setattr(_CheckpointSource, "get", get)
    monkeypatch.setattr(_CheckpointSource, "shape", shape)
    path = export_checkpoint(source, tmp_path / "artifact")
    manifest = json.loads((path / "manifest.json").read_text())
    report = manifest["checkpoint_preflight"]
    assert report["ignored_auxiliary_tensors"] == sorted(auxiliary)
    assert report["ignored_auxiliary_tensor_count"] == len(auxiliary) == 44
    assert report["source_tensor_count"] == len(weight_map)
    assert report["consumed_source_tensor_count"] == len(state)
    assert all(not name.startswith("model.layers.47.") for name in visited)
    assert all(entry["layer"] < 47 for entry in manifest["experts"])
    assert all(not name.startswith("model.layers.47.") for entry in manifest["resident"] for name in entry["tensors"])
    assert export_summary(path)["ignored_auxiliary_tensor_count"] == 44
    disk = load_model(path, cache_budget_bytes=0)
    ids = torch.tensor([[3, 4, 5]])
    native_past = disk_past = None
    with torch.inference_mode():
        for _ in range(3):
            expected = reference(ids, use_cache=True, past_key_values=native_past)
            actual = disk(ids, use_cache=True, past_key_values=disk_past)
            # Use the existing FP32 parity bound: on this CPU, even cloning an
            # identical native F.linear weight changes its reduction rounding
            # through allocation alignment. Greedy token equality stays exact.
            torch.testing.assert_close(actual.logits, expected.logits, rtol=1e-6, atol=1e-6)
            ids = expected.logits[:, -1].argmax(-1, keepdim=True)
            assert torch.equal(ids, actual.logits[:, -1].argmax(-1, keepdim=True))
            native_past, disk_past = expected.past_key_values, actual.past_key_values


@pytest.mark.parametrize("mutation", ["undeclared", "incomplete_auxiliary", "arbitrary_auxiliary", "missing_live_decoder", "live_layer47"])
def test_auxiliary_exception_never_hides_bad_or_live_decoder_layout(tmp_path, monkeypatch, mutation):
    native, auxiliary = _native_47_with_auxiliary()
    config = native.config
    names = set(native.state_dict()) | set(auxiliary)
    if mutation == "undeclared":
        config.num_nextn_predict_layers = 0
    elif mutation == "incomplete_auxiliary":
        names.remove("model.layers.47.hnorm.weight")
    elif mutation == "arbitrary_auxiliary":
        names.add("model.layers.47.made_up.weight")
    elif mutation == "missing_live_decoder":
        names.remove("model.layers.46.input_layernorm.weight")
    else:
        config.num_hidden_layers = 48
        config.mlp_layer_types += ["sparse"]
        # Layer 47 is now live. Its auxiliary-only extras must not be ignored.
    source = _metadata_only_source(tmp_path, config, names)

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid names must fail before any payload or shard header read")

    monkeypatch.setattr(_CheckpointSource, "get", forbidden)
    monkeypatch.setattr(_CheckpointSource, "shape", forbidden)
    with pytest.raises(ValueError, match="name preflight"):
        export_checkpoint(source, tmp_path / "artifact")


def test_remote_unsharded_requires_local_source_before_payload_download(tmp_path, monkeypatch):
    import disk_moe.runtime.export as exporter

    native = tiny_model()
    native.config.save_pretrained(tmp_path)
    calls = []

    def download(repo_id, *, filename, revision):
        calls.append(filename)
        assert revision == "pinned-source-commit"
        if filename in {"generation_config.json", "model.safetensors.index.json"}:
            raise FileNotFoundError(filename)
        assert filename == "config.json", "No payload may be downloaded before complete name preflight"
        return str(tmp_path / filename)

    monkeypatch.setattr(exporter, "model_info", lambda *args, **kwargs: SimpleNamespace(sha="pinned-source-commit"))
    monkeypatch.setattr(exporter, "hf_hub_download", download)
    with pytest.raises(ValueError, match="local directory first"):
        export_checkpoint("example/model", tmp_path / "artifact")
    assert "model.safetensors" not in calls


def test_remote_sharded_invalid_names_fail_before_payload_download(tmp_path, monkeypatch):
    import disk_moe.runtime.export as exporter

    native = tiny_model()
    native.config.save_pretrained(tmp_path)
    names = set(native.state_dict()) | {"unknown.weight"}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": dict.fromkeys(names, "never-fetch.safetensors")}))
    calls = []

    def download(repo_id, *, filename, revision):
        calls.append(filename)
        assert revision == "pinned-source-commit"
        if filename == "generation_config.json":
            raise FileNotFoundError(filename)
        assert filename in {"config.json", "model.safetensors.index.json"}
        return str(tmp_path / filename)

    monkeypatch.setattr(exporter, "model_info", lambda *args, **kwargs: SimpleNamespace(sha="pinned-source-commit"))
    monkeypatch.setattr(exporter, "hf_hub_download", download)
    with pytest.raises(ValueError, match="name preflight"):
        export_checkpoint("example/model", tmp_path / "artifact")
    assert not any(filename.endswith(".safetensors") for filename in calls)
