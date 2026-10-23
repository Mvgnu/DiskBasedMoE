"""Local chat API with one serialized model owner."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from disk_moe.contracts import InferenceEngine

from .schemas import MAX_INPUT_CHARACTERS, MAX_NEW_TOKENS, ChatRequest
from .worker import GenerationFailure, GenerationJob, InferenceWorker, ServiceBusy


class _OwnedStreamResponse(StreamingResponse):
    def __init__(self, content, job: GenerationJob):
        super().__init__(
            content,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
        self.job = job

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.job.cancel()


@asynccontextmanager
async def _cancel_on_disconnect(request: Request):
    """Observe disconnects while waiting for admission, validation or JSON output.

StreamingResponse observes its own disconnect after headers. Before then, ASGI
does not automatically cancel the handler just because the client went away.
"""
    owner = asyncio.current_task()

    async def watch():
        while True:
            if await request.is_disconnected():
                owner.cancel()
                return
            await asyncio.sleep(0.05)

    watcher = asyncio.create_task(watch())
    try:
        yield
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)


def create_app(
    engine: InferenceEngine,
    *,
    default_max_new_tokens: int = 128,
    max_pending_requests: int = 8,
    stream_queue_size: int = 16,
) -> FastAPI:
    """Build a local app around an injected blocking inference engine.

Request limits are immutable; model routing is never an HTTP setting. The engine
is loaded by the CLI composition boundary, not while importing this module.
"""
    if not 1 <= default_max_new_tokens <= MAX_NEW_TOKENS:
        raise ValueError(f"default_max_new_tokens must be in 1..{MAX_NEW_TOKENS}")
    if max_pending_requests < 1 or stream_queue_size < 1:
        raise ValueError("Queue capacities must be positive")
    description = engine.describe()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        worker = InferenceWorker(
            engine, max_pending=max_pending_requests, queue_size=stream_queue_size
        )
        app.state.worker = worker
        try:
            yield
        finally:
            await worker.close()

    app = FastAPI(title="DiskBasedMoE local chat", version="0.2.0", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "engine": description,
            "limits": {
                "max_input_characters": MAX_INPUT_CHARACTERS,
                "max_new_tokens": MAX_NEW_TOKENS,
                "default_max_new_tokens": default_max_new_tokens,
                "max_pending_requests": max_pending_requests,
                "stream_queue_size": stream_queue_size,
            },
        }

    async def start(request: ChatRequest):
        try:
            return await app.state.worker.start(
                request,
                request.max_new_tokens if request.max_new_tokens is not None else default_max_new_tokens,
            )
        except ServiceBusy as error:
            raise HTTPException(503, str(error), headers={"Retry-After": "1"}) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/api/chat")
    async def chat(body: ChatRequest, request: Request):
        async with _cancel_on_disconnect(request):
            job = await start(body)
            text = []
            try:
                async for event in job.events():
                    if isinstance(event, GenerationFailure):
                        raise HTTPException(500, event.message)
                    text.append(event.delta)
                    if event.finish_reason is not None:
                        return {
                            "text": "".join(text),
                            "finish_reason": event.finish_reason,
                            "metrics": dict(event.metrics),
                        }
            finally:
                job.cancel()

    @app.post("/api/chat/stream")
    async def stream(body: ChatRequest, request: Request):
        import json

        async with _cancel_on_disconnect(request):
            job = await start(body)

        async def events():
            async for event in job.events():
                if isinstance(event, GenerationFailure):
                    yield f"event: error\ndata: {json.dumps({'error': event.message})}\n\n"
                    return
                payload = {"delta": event.delta, "token_id": event.token_id}
                if event.finish_reason is not None:
                    payload.update(finish_reason=event.finish_reason, metrics=dict(event.metrics))
                yield f"data: {json.dumps(payload)}\n\n"
                if event.finish_reason is not None:
                    yield "data: [DONE]\n\n"

        return _OwnedStreamResponse(events(), job)

    return app
