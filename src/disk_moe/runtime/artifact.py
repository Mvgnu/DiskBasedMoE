"""Versioned, complete-only safetensors artifacts for the native GLM4 adapter.

The manifest is a completion marker, not a claim about physical disk traffic.
Headers and sizes are checked at open; content hashes are checked on every load.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

import torch
from safetensors import safe_open
from safetensors.torch import load
from transformers import GenerationConfig, Glm4MoeLiteConfig

FORMAT_VERSION = 1
NATIVE_VERSION = "5.20.2"
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
SAFE_DTYPES = {"F32": "float32", "F16": "float16", "BF16": "bfloat16"}


def tensor_metadata(tensor: torch.Tensor) -> dict:
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")}


def file_metadata(path: Path, root: Path, tensors: dict[str, torch.Tensor]) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "tensors": {name: tensor_metadata(tensor) for name, tensor in tensors.items()},
    }


def safe_path(root: Path, value: str) -> Path:
    if not isinstance(value, str):
        raise ValueError("Artifact paths must be strings")
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or any(p in {".", ".."} for p in relative.parts):
        raise ValueError(f"Unsafe artifact path: {value!r}")
    if "\\" in value:
        raise ValueError(f"Unsafe artifact path: {value!r}")
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Artifact path escapes its directory: {value!r}")
    for parent in [path, *path.parents]:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError(f"Artifact symlinks are unsupported: {value!r}")
    return path


def _positive_int(value, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if allow_zero else 'positive'} integer")
    return value


def tensor_bytes(metadata: dict) -> int:
    if not isinstance(metadata, dict) or set(metadata) != {"shape", "dtype"}:
        raise ValueError("Invalid tensor metadata")
    if metadata["dtype"] not in DTYPES or not isinstance(metadata["shape"], list):
        raise ValueError("Only explicit float32/float16/bfloat16 tensor metadata is supported")
    count = 1
    for dimension in metadata["shape"]:
        count *= _positive_int(dimension, "Tensor dimension", allow_zero=True)
    return count * (4 if metadata["dtype"] == "float32" else 2)


class Artifact:
    """Validated metadata; experts remain on disk until requested."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve()
        manifest_path = safe_path(self.directory, "manifest.json")
        try:
            self.manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f"Missing or invalid completed artifact: {manifest_path}") from exc
        manifest = self.manifest
        if not isinstance(manifest, dict) or manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported artifact format_version")
        if manifest.get("architecture") != "glm4_moe_lite":
            raise ValueError("Only glm4_moe_lite artifacts are supported")
        if manifest.get("transformers_version") != NATIVE_VERSION:
            raise ValueError(f"Artifact requires the tested Transformers {NATIVE_VERSION} exporter")
        if manifest.get("dtype") not in DTYPES:
            raise ValueError("Invalid artifact dtype")
        config_path = safe_path(self.directory, "config.json")
        try:
            config_bytes = config_path.read_bytes()
            if hashlib.sha256(config_bytes).hexdigest() != manifest.get("config_sha256"):
                raise ValueError("Configuration checksum mismatch")
            config_data = json.loads(config_bytes)
            if config_data.get("model_type") != "glm4_moe_lite":
                raise ValueError("Configuration model_type is not glm4_moe_lite")
            self.config = Glm4MoeLiteConfig.from_dict(config_data)
        except (OSError, TypeError, KeyError) as exc:
            raise ValueError("Invalid artifact configuration") from exc
        self.dtype = DTYPES[manifest["dtype"]]
        generation_path = safe_path(self.directory, "generation_config.json")
        try:
            generation_bytes = generation_path.read_bytes()
            if hashlib.sha256(generation_bytes).hexdigest() != manifest.get("generation_config_sha256"):
                raise ValueError("Generation configuration checksum mismatch")
            self.generation_config = GenerationConfig.from_dict(json.loads(generation_bytes))
        except (OSError, TypeError, KeyError) as exc:
            raise ValueError("Missing or invalid native generation configuration") from exc
        self.resident = manifest.get("resident")
        self.rotary = manifest.get("rotary")
        if not isinstance(self.rotary, dict):
            raise ValueError("Artifact is missing native rotary state; re-export it to preserve source-model parity")
        expert_entries = manifest.get("experts")
        if not isinstance(self.resident, list) or not self.resident:
            raise ValueError("Artifact has no resident weights")
        if not isinstance(expert_entries, list):
            raise ValueError("Artifact experts must be a list")
        self.experts: dict[tuple[int, int], dict] = {}
        paths = set()
        resident_names = set()
        for entry in self.resident + expert_entries + [self.rotary]:
            self._validate_entry(entry)
            if entry["path"] in paths:
                raise ValueError("Duplicate artifact file path")
            paths.add(entry["path"])
        for entry in self.resident:
            names = set(entry["tensors"])
            if names & resident_names:
                raise ValueError("Duplicate resident tensor")
            resident_names.update(names)
        for entry in expert_entries:
            layer = _positive_int(entry.get("layer"), "Expert layer", allow_zero=True)
            expert = _positive_int(entry.get("expert"), "Expert index", allow_zero=True)
            key = (layer, expert)
            if key in self.experts:
                raise ValueError("Duplicate expert")
            self.experts[key] = entry
        expected = {
            (layer, expert)
            for layer, kind in enumerate(self.config.mlp_layer_types)
            if kind == "sparse"
            for expert in range(self.config.n_routed_experts)
        }
        if set(self.experts) != expected:
            raise ValueError("Artifact does not contain exactly the configured routed experts")
        for entry in expert_entries:
            expected_tensors = {
                "gate_up_proj": {"shape": [2 * self.config.moe_intermediate_size, self.config.hidden_size], "dtype": manifest["dtype"]},
                "down_proj": {"shape": [self.config.hidden_size, self.config.moe_intermediate_size], "dtype": manifest["dtype"]},
            }
            if entry["tensors"] != expected_tensors:
                raise ValueError("Expert tensor shape/dtype does not match the model configuration")

    def _validate_entry(self, entry: dict) -> None:
        if not isinstance(entry, dict):
            raise ValueError("Invalid artifact file entry")
        path = safe_path(self.directory, entry.get("path"))
        size = _positive_int(entry.get("size_bytes"), "File size")
        digest = entry.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid file checksum")
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Missing or truncated artifact file: {path.name}")
        tensors = entry.get("tensors")
        if not isinstance(tensors, dict) or not tensors:
            raise ValueError("File has no tensor metadata")
        for metadata in tensors.values():
            tensor_bytes(metadata)
        try:
            with safe_open(path, framework="pt", device="cpu") as source:
                actual = {
                    name: {"shape": list(source.get_slice(name).get_shape()), "dtype": SAFE_DTYPES.get(source.get_slice(name).get_dtype())}
                    for name in source.keys()
                }
            if actual != tensors:
                raise ValueError(f"Tensor header mismatch: {path.name}")
        except Exception as exc:
            raise ValueError(f"Invalid safetensors artifact: {path.name}: {exc}") from exc

    def read(self, entry: dict) -> dict[str, torch.Tensor]:
        data = safe_path(self.directory, entry["path"]).read_bytes()
        if len(data) != entry["size_bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Artifact content checksum mismatch: {entry['path']}")
        tensors = load(data)
        if {name: tensor_metadata(tensor) for name, tensor in tensors.items()} != entry["tensors"]:
            raise ValueError(f"Artifact tensor mismatch: {entry['path']}")
        return tensors

    @property
    def resident_tensor_bytes(self) -> int:
        return sum(tensor_bytes(t) for entry in self.resident + [self.rotary] for t in entry["tensors"].values())
