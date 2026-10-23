import hashlib

import pytest

from disk_moe.contracts import GenerationEvent
from disk_moe.evaluation.io_bench import benchmark_file
from disk_moe.evaluation.model_bench import benchmark_engine


@pytest.mark.parametrize("pattern", ["sequential", "random"])
def test_file_benchmark_is_read_only_bounded_and_honest(tmp_path, pattern):
    path = tmp_path / "existing.data"
    content = bytes(range(256)) * 4096 + b"tail"
    path.write_bytes(content)
    before = hashlib.sha256(content).hexdigest()
    result = benchmark_file(path, block_bytes=65536, max_bytes=150000, passes=2, pattern=pattern)
    assert [row["bytes_read"] for row in result["runs"]] == [150000, 150000]
    assert result["buffer_bytes"] <= 65536
    assert result["physical_disk_bytes"] is None
    assert "uncontrolled" in result["cache_state"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_file_benchmark_rejects_nonfiles_empty_and_huge_buffers(tmp_path):
    path = tmp_path / "empty"
    path.touch()
    with pytest.raises(ValueError):
        benchmark_file(path)
    with pytest.raises(ValueError):
        benchmark_file(path, block_bytes=2**30)
    with pytest.raises((ValueError, OSError)):
        benchmark_file(tmp_path)


class Engine:
    def describe(self):
        return {"fixture": True}

    def validate_request(self, prompt, max_new_tokens):
        if not prompt or max_new_tokens < 1:
            raise ValueError("bad request")

    def generate(self, prompt, max_new_tokens):
        yield GenerationEvent(delta="hello", token_id=1)
        yield GenerationEvent(delta=" world", token_id=2)
        yield GenerationEvent(finish_reason="length", metrics={"fixture": 1})


def test_benchmark_counts_tokens_not_characters_and_excludes_prefill_from_decode():
    report = benchmark_engine(Engine(), "PRIVATE PROMPT", repeats=2, include_text=True)
    assert "PRIVATE PROMPT" not in str(report)
    row = report["runs"][0]
    assert row["observed_generated_tokens"] == 2
    assert row["observed_decode_tokens"] == 1
    assert row["generated_text"] == "hello world"
    assert row["observed_ttft_seconds"] >= 0
    assert "N-1" in report["timing_convention"]


def test_benchmark_rejects_truncated_engine_stream():
    class Broken(Engine):
        def generate(self, prompt, max_new_tokens):
            yield GenerationEvent(delta="partial", token_id=1)

    with pytest.raises(RuntimeError, match="terminal"):
        benchmark_engine(Broken(), "hello")
