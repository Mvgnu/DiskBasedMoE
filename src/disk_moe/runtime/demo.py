"""Tiny random native model for an offline correctness journey, not useful chat."""

from __future__ import annotations

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Glm4MoeLiteConfig, Glm4MoeLiteForCausalLM, PreTrainedTokenizerFast


def tiny_config():
    return Glm4MoeLiteConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24, moe_intermediate_size=8,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
        n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2,
        n_group=2, topk_group=1, q_lora_rank=8, kv_lora_rank=8,
        qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4,
        max_position_embeddings=128, eos_token_id=31, pad_token_id=0, bos_token_id=1,
        mlp_layer_types=["dense", "sparse", "sparse"],
    )


def tiny_model(dtype=torch.float32):
    torch.manual_seed(42)
    config = tiny_config()
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    model = Glm4MoeLiteForCausalLM(config).eval().to(dtype=dtype)
    # Native from_pretrained preserves this correction buffer in FP32.
    for layer in model.model.layers:
        if hasattr(layer.mlp, "gate"):
            layer.mlp.gate.e_score_correction_bias = layer.mlp.gate.e_score_correction_bias.float()
    return model


def tiny_tokenizer():
    words = ["<pad>", "<bos>", "<unk>", "hello", "disk", "world", "user", "assistant", "system"]
    words += [f"word{i}" for i in range(31 - len(words))]
    words += ["<eos>"]
    tokenizer = Tokenizer(WordLevel(dict(zip(words, range(32), strict=True)), unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<pad>", bos_token="<bos>", eos_token="<eos>")
    fast.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    return fast
