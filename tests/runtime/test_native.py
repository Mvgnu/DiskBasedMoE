from __future__ import annotations

import hashlib
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from disk_moe.runtime import Glm4Engine, export_checkpoint, export_model, load_model
from disk_moe.runtime.artifact import Artifact
from disk_moe.runtime.cache import ExpertCache
from disk_moe.runtime.demo import tiny_model, tiny_tokenizer


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def exported(tmp_path):
    model = tiny_model()
    path = export_model(model, tmp_path / "artifact", tokenizer=tiny_tokenizer())
    return model, path


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("budget", [0, 1536, 10**6])
def test_native_prefill_cached_decode_and_greedy_parity(tmp_path, dtype, budget):
    native = tiny_model(dtype)
    path = export_model(native, tmp_path / "artifact")
    disk = load_model(path, cache_budget_bytes=budget)
    assert next(disk.parameters()).dtype == dtype
    assert all(p.device.type != "meta" for p in disk.parameters())
    ids = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]])
    native_cache = disk_cache = None
    tolerance = 2e-3 if dtype == torch.bfloat16 else 1e-6
    with torch.inference_mode():
        for _ in range(5):
            reference = native(ids, past_key_values=native_cache, use_cache=True)
            actual = disk(ids, past_key_values=disk_cache, use_cache=True)
            torch.testing.assert_close(actual.logits, reference.logits, rtol=tolerance, atol=tolerance)
            assert torch.equal(actual.logits[:, -1].argmax(-1), reference.logits[:, -1].argmax(-1))
            native_cache, disk_cache = reference.past_key_values, actual.past_key_values
            assert disk_cache.get_seq_length() == native_cache.get_seq_length()
            ids = reference.logits[:, -1].argmax(-1, keepdim=True)
    stats = disk.expert_cache.stats()
    assert stats["cache_retained_bytes"] <= budget
    assert stats["cache_peak_retained_bytes"] <= budget
    assert stats["logical_bytes_read"] > 0


def test_export_isolated_expert_storage_and_dtype(tmp_path):
    native = tiny_model()
    original_dtype = next(native.parameters()).dtype
    path = export_model(native, tmp_path / "artifact", dtype="bfloat16")
    artifact = Artifact(path)
    assert next(native.parameters()).dtype == original_dtype
    for entry in artifact.experts.values():
        tensors = artifact.read(entry)
        assert all(value.dtype == torch.bfloat16 for value in tensors.values())
        expected = 3 * native.config.hidden_size * native.config.moe_intermediate_size * 2
        assert entry["size_bytes"] < expected + 1024
        for value in tensors.values():
            assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
    loaded = load_model(path)
    assert next(loaded.parameters()).dtype == torch.bfloat16
    assert loaded.model.layers[1].mlp.gate.e_score_correction_bias.dtype == torch.float32


def test_global_cache_lru_and_bypass(exported):
    _, path = exported
    artifact = Artifact(path)
    cache = ExpertCache(artifact, 1536)
    first = cache.get(1, 0)
    assert cache.get(1, 0) is first
    cache.get(2, 0)
    assert cache.retained_bytes == 1536
    assert cache.evictions == 1
    assert cache.hits == 1
    assert cache.misses == 2
    # A caller reference pins evicted data through its computation.
    assert first["gate_up_proj"].shape == (16, 16)
    for budget in (0, 1):
        bypass = ExpertCache(artifact, budget)
        bypass.get(1, 0)
        bypass.get(1, 0)
        assert bypass.bypasses == 2
        assert bypass.hits == 0
        assert bypass.retained_bytes == 0
    with pytest.raises(ValueError):
        ExpertCache(artifact, -1)


def test_partial_missing_corrupt_artifacts_fail_closed(exported, tmp_path):
    _, path = exported
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(ValueError, match="completed artifact"):
        Artifact(incomplete)
    artifact = Artifact(path)
    entry = artifact.experts[(1, 0)]
    file = path / entry["path"]
    original = file.read_bytes()
    file.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="checksum"):
        ExpertCache(artifact, 0).get(1, 0)
    file.unlink()
    with pytest.raises(ValueError, match="Missing or truncated"):
        Artifact(path)


@pytest.mark.parametrize("path", ["../outside.safetensors", "/tmp/outside.safetensors", "..\\outside.safetensors"])
def test_manifest_path_escape_rejected(exported, path):
    _, directory = exported
    metadata = json.loads((directory / "manifest.json").read_text())
    metadata["experts"][0]["path"] = path
    (directory / "manifest.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Unsafe artifact path"):
        Artifact(directory)


def test_failed_export_leaves_no_complete_output(tmp_path, monkeypatch):
    import disk_moe.runtime.export as exporter
    original = exporter._write_tensor_file
    count = 0

    def fail(*args):
        nonlocal count
        count += 1
        if count == 3:
            raise OSError("simulated write failure")
        return original(*args)

    monkeypatch.setattr(exporter, "_write_tensor_file", fail)
    with pytest.raises(OSError, match="write failure"):
        export_model(tiny_model(), tmp_path / "failed")
    assert list(tmp_path.iterdir()) == []


def test_export_refuses_overwrite(exported):
    model, path = exported
    before = (path / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        export_model(model, path)
    assert (path / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("sharded", [False, True])
def test_incremental_local_checkpoint_roundtrip(tmp_path, sharded):
    native = tiny_model()
    source = tmp_path / "checkpoint"
    native.save_pretrained(source, max_shard_size="2KB" if sharded else "2GB")
    output = export_checkpoint(source, tmp_path / "export", dtype="float32")
    disk = load_model(output, cache_budget_bytes=0)
    with torch.inference_mode():
        ids = torch.tensor([[3, 4, 5]])
        torch.testing.assert_close(disk(ids).logits, native(ids).logits, rtol=1e-6, atol=1e-6)


def test_incremental_separate_experts_and_dtype(tmp_path):
    native = tiny_model()
    source = tmp_path / "checkpoint"
    source.mkdir()
    native.config.save_pretrained(source)
    state = dict(native.state_dict())
    for layer in (1, 2):
        prefix = f"model.layers.{layer}.mlp.experts."
        packed = state.pop(prefix + "gate_up_proj")
        down = state.pop(prefix + "down_proj")
        for expert in range(native.config.n_routed_experts):
            gate, up = packed[expert].chunk(2, dim=0)
            state[prefix + f"{expert}.gate_proj.weight"] = gate.clone()
            state[prefix + f"{expert}.up_proj.weight"] = up.clone()
            state[prefix + f"{expert}.down_proj.weight"] = down[expert].clone()
    save_file(state, source / "model.safetensors")
    path = export_checkpoint(source, tmp_path / "export", dtype="bfloat16")
    assert Artifact(path).manifest["dtype"] == "bfloat16"
    model = load_model(path)
    with torch.inference_mode():
        assert model(torch.tensor([[3, 4]])).logits.dtype == torch.bfloat16


def test_wrong_resident_shape_fails_even_with_consistent_hash(exported):
    _, path = exported
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = next(e for e in manifest["resident"] if "lm_head.weight" in e["tensors"])
    file = path / entry["path"]
    tensors = load_file(file)
    tensors["lm_head.weight"] = tensors["lm_head.weight"][:3].clone()
    save_file(tensors, file)
    entry["size_bytes"] = file.stat().st_size
    entry["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    entry["tensors"]["lm_head.weight"]["shape"] = [3, 16]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not expected"):
        load_model(path)


def test_engine_incremental_generation_and_request_isolation(exported):
    native, path = exported
    engine = Glm4Engine(path, cache_budget_bytes=1536)
    engine.model.generation_config.eos_token_id = None
    native.generation_config.eos_token_id = None
    sizes = []
    cache_lengths = []

    def inspect(module, args, kwargs):
        sizes.append(args[0].shape[1])
        past = kwargs.get("past_key_values")
        cache_lengths.append(0 if past is None else past.get_seq_length())

    hook = engine.model.register_forward_pre_hook(inspect, with_kwargs=True)
    events = list(engine.generate("hello disk world", 5))
    first_tokens = [event.token_id for event in events if event.token_id is not None]
    assert sizes == [3, 1, 1, 1, 1]
    assert cache_lengths == [0, 3, 4, 5, 6]
    expected = tiny_tokenizer().decode(first_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    assert "".join(event.delta for event in events) == expected
    assert events[-1].finish_reason == "length"
    metrics = events[-1].metrics
    assert metrics["prompt_tokens"] == 3
    assert metrics["generated_tokens"] == 5
    assert metrics["decode_tokens"] == 4
    assert metrics["decode_seconds"] > 0
    again = list(engine.generate("hello disk world", 5))
    assert [event.token_id for event in again if event.token_id is not None] == first_tokens
    assert cache_lengths[5] == 0
    hook.remove()


def test_engine_eos_limit_cancellation_and_validation(exported):
    _, path = exported
    engine = Glm4Engine(path, cache_budget_bytes=0)
    first = list(engine.generate("hello", 1))
    token = next(e.token_id for e in first if e.token_id is not None)
    engine.model.generation_config.eos_token_id = [token]
    events = list(engine.generate("hello", 10))
    assert events[-1].finish_reason == "stop"
    assert events[-1].metrics["generated_tokens"] == 1
    assert events[-1].metrics["decode_tokens"] == 0
    assert events[-1].metrics["decode_tokens_per_second"] == 0
    engine.model.generation_config.eos_token_id = None
    iterator = engine.generate("hello", 10)
    next(iterator)
    iterator.close()
    assert list(engine.generate("hello", 1))[-1].metrics["generated_tokens"] == 1
    for prompt, limit in [("", 1), ("hello", 0), ("hello", -1), ("hello", True), ("hello", 129)]:
        with pytest.raises(ValueError):
            engine.validate_request(prompt, limit)
    assert engine.format_messages([{"role": "user", "content": "hello"}]).endswith("assistant ")


def test_export_preserves_native_generation_eos_override(tmp_path):
    native = tiny_model()
    native.generation_config.eos_token_id = [7, 8]
    artifact = export_model(native, tmp_path / "export", tokenizer=tiny_tokenizer())
    assert load_model(artifact).generation_config.eos_token_id == [7, 8]
    checkpoint = tmp_path / "source"
    native.save_pretrained(checkpoint)
    artifact = export_checkpoint(checkpoint, tmp_path / "incremental")
    assert load_model(artifact).generation_config.eos_token_id == [7, 8]


def test_unicode_stream_buffers_incomplete_byte_fallback():
    from disk_moe.runtime.generation import _TextStream

    class ByteFallbackTokenizer:
        def decode(self, tokens, **kwargs):
            return {1: "\ufffd", 2: "\ufffd", 3: "€", 4: "€ ", 5: "€ value"}[len(tokens)]

    stream = _TextStream(ByteFallbackTokenizer())
    deltas = [stream.add(i) for i in range(5)]
    deltas.append(stream.finish())
    assert "".join(deltas) == "€ value"
    assert "\ufffd" not in "".join(deltas)


def _wider_bfloat16_native():
    from transformers import Glm4MoeLiteForCausalLM

    from disk_moe.runtime.demo import tiny_config

    config = tiny_config()
    config.hidden_size = 64
    config.intermediate_size = 96
    config.moe_intermediate_size = 32
    config.num_experts_per_tok = 5
    config.topk_group = 2
    config.mlp_layer_types = ["sparse"] * 3
    config._attn_implementation = "sdpa"
    config._experts_implementation = "eager"
    torch.manual_seed(173)
    native = Glm4MoeLiteForCausalLM(config).eval().to(torch.bfloat16)
    for layer in native.model.layers:
        layer.mlp.gate.e_score_correction_bias = torch.linspace(-0.005, 0.005, 8)
    return native


ROTARY_EOS_PREFIX = [20, 9, 20, 20, 21, 20, 19, 13, 25, 28, 15, 10, 10, 7, 18, 28, 6, 26, 14, 29, 1, 11]


def test_bfloat16_in_memory_rotary_roundtrip_preserves_native_eos_and_logits(tmp_path):
    native = _wider_bfloat16_native()
    path = export_model(native, tmp_path / "artifact")
    disk = load_model(path, cache_budget_bytes=0, attn_implementation="sdpa")
    torch.testing.assert_close(disk.model.rotary_emb.inv_freq, native.model.rotary_emb.inv_freq, rtol=0, atol=0)
    assert disk.model.rotary_emb.inv_freq.dtype == torch.bfloat16
    ids = torch.tensor([ROTARY_EOS_PREFIX])
    native_past = disk_past = None
    with torch.inference_mode():
        for step in range(4):
            expected = native(ids, past_key_values=native_past, use_cache=True)
            actual = disk(ids, past_key_values=disk_past, use_cache=True)
            torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
            if step == 0:
                assert expected.logits[:, -1].argmax().item() == native.generation_config.eos_token_id == 31
            ids = expected.logits[:, -1].argmax(-1, keepdim=True)
            native_past, disk_past = expected.past_key_values, actual.past_key_values

    class FixedPrefixTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([ROTARY_EOS_PREFIX])}

        def decode(self, tokens, **kwargs):
            return ""

    engine = Glm4Engine(path, tokenizer=FixedPrefixTokenizer(), attn_implementation="sdpa")
    events = list(engine.generate("the critic's fixed token prefix", 5))
    assert [e.token_id for e in events if e.token_id is not None] == [31]
    assert events[-1].finish_reason == "stop"
    assert events[-1].metrics["generated_tokens"] == 1


def test_bfloat16_checkpoint_uses_native_checkpoint_rotary_precision(tmp_path):
    from transformers import Glm4MoeLiteForCausalLM

    source = _wider_bfloat16_native()
    checkpoint = tmp_path / "source"
    source.save_pretrained(checkpoint)
    path = export_checkpoint(checkpoint, tmp_path / "artifact")
    native = Glm4MoeLiteForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16, attn_implementation="sdpa", experts_implementation="eager").eval()
    disk = load_model(path, cache_budget_bytes=0, attn_implementation="sdpa")
    assert native.model.rotary_emb.inv_freq.dtype == disk.model.rotary_emb.inv_freq.dtype == torch.float32
    assert not torch.equal(source.model.rotary_emb.inv_freq.float(), disk.model.rotary_emb.inv_freq)
    torch.testing.assert_close(native.model.rotary_emb.inv_freq, disk.model.rotary_emb.inv_freq, rtol=0, atol=0)
    ids = torch.tensor([ROTARY_EOS_PREFIX])
    native_past = disk_past = None
    with torch.inference_mode():
        for _ in range(4):
            expected = native(ids, past_key_values=native_past, use_cache=True)
            actual = disk(ids, past_key_values=disk_past, use_cache=True)
            torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
            assert torch.equal(actual.logits[:, -1].argmax(-1), expected.logits[:, -1].argmax(-1))
            ids = expected.logits[:, -1].argmax(-1, keepdim=True)
            native_past, disk_past = expected.past_key_values, actual.past_key_values


@pytest.mark.parametrize("mutation", ["frequency", "scaling", "sequence", "dynamic"])
def test_incompatible_runtime_rotary_state_is_rejected_before_export(tmp_path, mutation):
    native = tiny_model()
    rotary = native.model.rotary_emb
    if mutation == "frequency":
        rotary.inv_freq[1] += 0.001
    elif mutation == "scaling":
        rotary.attention_scaling = 1.5
    elif mutation == "sequence":
        rotary.max_seq_len_cached += 1
    else:
        native.config.rope_parameters["rope_type"] = "dynamic"
    with pytest.raises(ValueError, match="rotary"):
        export_model(native, tmp_path / "artifact")
    assert not (tmp_path / "artifact").exists()


def test_artifact_without_rotary_snapshot_requires_reexport(exported):
    _, path = exported
    file = path / "manifest.json"
    manifest = json.loads(file.read_text())
    del manifest["rotary"]
    file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="re-export"):
        load_model(path)
