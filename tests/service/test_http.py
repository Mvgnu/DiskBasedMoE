"""Drive a real localhost server with deterministic, deliberately blocking engines.

These tests establish service behavior, not model quality or ML throughput.
"""

import concurrent.futures
import json
import socket
import threading
import time
from contextlib import contextmanager

import httpx
import pytest
import uvicorn

from disk_moe.contracts import GenerationEvent
from disk_moe.service import create_app


class DeterministicEngine:
    def __init__(self):
        self.calls = []
        self.formatted = []
        self.order = []
        self.active = 0
        self.max_active = 0
        self.guard = threading.Lock()
        self.blocked = threading.Event()
        self.release = threading.Event()

    def describe(self):
        return {"model": "deterministic-service-fixture", "native_top_k": 4}

    def format_messages(self, messages):
        self.formatted.append(messages)
        return "<native>" + "|".join(message["content"] for message in messages)

    def validate_request(self, prompt, max_new_tokens):
        if prompt == "invalid-context":
            raise ValueError("Prompt exceeds model context")

    def generate(self, prompt, max_new_tokens):
        with self.guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append((prompt, max_new_tokens))
            self.order.append(("start", prompt))
        try:
            if prompt == "fail":
                raise RuntimeError("Intentional fixture failure")
            yield GenerationEvent(delta="Hello", token_id=1)
            if prompt == "block":
                self.blocked.set()
                if not self.release.wait(5):
                    raise TimeoutError("Test did not release blocking engine")
            if prompt == "fail-after-token":
                raise RuntimeError("Intentional fixture failure after first token")
            time.sleep(0.01)
            yield GenerationEvent(delta=" world", token_id=2)
            yield GenerationEvent(finish_reason="length", metrics={"generated_tokens": 2})
        finally:
            with self.guard:
                self.active -= 1
                self.order.append(("close", prompt))


@contextmanager
def running_server(engine=None, **options):
    engine = engine or DeterministicEngine()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_app(engine, **options), log_level="critical", lifespan="on")
    )
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started, "Local HTTP server failed to start"
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5) as client:
            yield client, engine
    finally:
        engine.release.set()
        server.should_exit = True
        thread.join(6)
        sock.close()
        assert not thread.is_alive(), "HTTP worker did not stop"


def sse_payloads(response):
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def test_json_stream_equivalence_native_template_and_defaults():
    with running_server(default_max_new_tokens=7) as (client, engine):
        messages = [{"role": "system", "content": "Be precise"}, {"role": "user", "content": "Hi"}]
        normal = client.post("/api/chat", json={"messages": messages})
        streamed = client.post("/api/chat/stream", json={"messages": messages, "max_new_tokens": 3})
        again = client.post("/api/chat", json={"prompt": "Next"})
        assert normal.status_code == streamed.status_code == again.status_code == 200
        expected = {"text": "Hello world", "finish_reason": "length", "metrics": {"generated_tokens": 2}}
        assert normal.json() == expected
        payloads = sse_payloads(streamed)
        assert "".join(event["delta"] for event in payloads) == normal.json()["text"]
        assert payloads[-1]["metrics"] == expected["metrics"]
        assert payloads[-1]["finish_reason"] == "length"
        assert streamed.text.endswith("data: [DONE]\n\n")
        assert engine.formatted == [messages, messages]
        assert engine.calls == [("<native>Be precise|Hi", 7), ("<native>Be precise|Hi", 3), ("Next", 7)]
        assert engine.describe()["native_top_k"] == 4


@pytest.mark.parametrize("path", ["/api/chat", "/api/chat/stream"])
@pytest.mark.parametrize("body", [
    {},
    {"prompt": " "},
    {"prompt": "x", "messages": [{"role": "user", "content": "x"}]},
    {"messages": []},
    {"messages": [{"role": "tool", "content": "x"}]},
    {"messages": [{"role": "user", "content": "x", "unknown": True}]},
    {"prompt": "x", "max_new_tokens": 0},
    {"prompt": "x", "max_new_tokens": 2049},
    {"prompt": "x", "max_new_tokens": True},
    {"prompt": "x", "max_new_tokens": "4"},
    {"prompt": "x", "router_top_k": 1},
    {"prompt": "x", "oracle_top_k": 1},
    {"prompt": "x" * 32769},
    {"messages": [{"role": "user", "content": "x" * 20000}] * 2},
    {"prompt": "invalid-context"},
])
def test_invalid_requests_fail_before_sse_headers(path, body):
    with running_server() as (client, engine):
        response = client.post(path, json=body)
        assert response.status_code == 422
        assert response.headers["content-type"].startswith("application/json")
        assert not engine.calls


def test_requests_serialize_and_health_is_responsive_during_blocking_step():
    with running_server() as (client, engine):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(client.post, "/api/chat", json={"prompt": "block"})
            assert engine.blocked.wait(2)
            second = executor.submit(client.post, "/api/chat", json={"prompt": "second"})
            started = time.monotonic()
            health = client.get("/api/health")
            assert health.status_code == 200
            assert time.monotonic() - started < 1
            assert health.json()["engine"]["native_top_k"] == 4
            assert not second.done()
            engine.release.set()
            assert first.result(3).status_code == second.result(3).status_code == 200
        assert engine.max_active == 1
        assert engine.order == [("start", "block"), ("close", "block"), ("start", "second"), ("close", "second")]


def test_disconnect_does_not_release_model_before_blocking_step_finishes():
    with running_server() as (client, engine):
        with client.stream("POST", "/api/chat/stream", json={"prompt": "block"}) as response:
            assert response.status_code == 200
            assert '"delta": "Hello"' in next(response.iter_lines())
            assert engine.blocked.wait(2)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            next_request = executor.submit(client.post, "/api/chat", json={"prompt": "next"})
            assert client.get("/api/health").status_code == 200
            assert not next_request.done()
            with engine.guard:
                assert engine.order == [("start", "block")]
            engine.release.set()
            assert next_request.result(3).json()["text"] == "Hello world"
        assert engine.max_active == 1
        assert engine.order == [("start", "block"), ("close", "block"), ("start", "next"), ("close", "next")]


@pytest.mark.parametrize("path", ["/api/chat", "/api/chat/stream"])
def test_disconnected_queued_request_never_runs(path):
    with running_server() as (client, engine):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(client.post, "/api/chat", json={"prompt": "block"})
            assert engine.blocked.wait(2)
            with pytest.raises(httpx.ReadTimeout):
                client.post(path, json={"prompt": "abandoned"}, timeout=0.1)
            # Allow the server to observe the closed connection while still blocked.
            time.sleep(0.15)
            engine.release.set()
            assert first.result(3).status_code == 200
            assert client.post("/api/chat", json={"prompt": "next"}).status_code == 200
        assert [prompt for prompt, _ in engine.calls] == ["block", "next"]


def test_failures_have_explicit_terminal_errors_and_next_request_works():
    with running_server() as (client, _):
        failed = client.post("/api/chat", json={"prompt": "fail"})
        assert failed.status_code == 500
        streamed = client.post("/api/chat/stream", json={"prompt": "fail-after-token"})
        assert streamed.status_code == 200
        assert "event: error\n" in streamed.text
        assert "[DONE]" not in streamed.text
        assert "Generation failed" in sse_payloads(streamed)[-1]["error"]
        assert client.post("/api/chat", json={"prompt": "recovery"}).status_code == 200


def test_bounded_admission_rejects_excess_requests_without_waiting():
    with running_server(max_pending_requests=1) as (client, engine):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(client.post, "/api/chat", json={"prompt": "block"})
            assert engine.blocked.wait(2)
            busy = client.post("/api/chat/stream", json={"prompt": "extra"})
            assert busy.status_code == 503
            assert busy.headers["retry-after"] == "1"
            engine.release.set()
            assert first.result(3).status_code == 200


def test_ui_is_served_locally():
    with running_server() as (client, _):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "DiskBasedMoE" in response.text
        assert "https://" not in response.text
