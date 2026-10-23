"""Single-owner blocking inference with bounded delivery and cooperative stop.

The HTTP request never owns the model lock's release. A separate monitor waits
for the actual worker to finish and close its iterator before releasing it.
This is important when a client disconnects during a non-interruptible forward.
"""

import asyncio
import concurrent.futures
import logging
import threading
from dataclasses import dataclass

from disk_moe.contracts import GenerationEvent, InferenceEngine

from .schemas import ChatRequest

logger = logging.getLogger(__name__)


class ServiceBusy(Exception):
    """The bounded request queue is full, or the service is closing."""


@dataclass(frozen=True)
class GenerationFailure:
    message: str = "Generation failed; see server logs for details."


class GenerationJob:
    def __init__(self, loop: asyncio.AbstractEventLoop, queue_size: int):
        self.loop = loop
        self.queue: asyncio.Queue[GenerationEvent | GenerationFailure] = asyncio.Queue(queue_size)
        self.ready: asyncio.Future[None] = loop.create_future()
        self.stop = threading.Event()

    def cancel(self) -> None:
        self.stop.set()

    def mark_ready(self, error: Exception | None = None) -> None:
        def resolve():
            if not self.ready.done():
                if error is None:
                    self.ready.set_result(None)
                else:
                    self.ready.set_exception(error)

        self.loop.call_soon_threadsafe(resolve)

    def deliver(self, item: GenerationEvent | GenerationFailure) -> bool:
        if self.stop.is_set():
            return False
        pending = asyncio.run_coroutine_threadsafe(self.queue.put(item), self.loop)
        while not self.stop.is_set():
            try:
                pending.result(timeout=0.05)
                return True
            except concurrent.futures.TimeoutError:
                continue
        pending.cancel()
        return False

    async def events(self):
        try:
            while True:
                item = await self.queue.get()
                yield item
                if isinstance(item, GenerationFailure) or item.finish_reason is not None:
                    return
        finally:
            self.cancel()


class InferenceWorker:
    def __init__(self, engine: InferenceEngine, *, max_pending: int, queue_size: int):
        self.engine = engine
        self.max_pending = max_pending
        self.queue_size = queue_size
        self._pending = 0
        self._closing = False
        self._gate = asyncio.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="disk-moe-inference"
        )
        self._jobs: set[GenerationJob] = set()
        self._monitors: set[asyncio.Task] = set()

    async def start(self, request: ChatRequest, max_new_tokens: int) -> GenerationJob:
        if self._closing or self._pending >= self.max_pending:
            raise ServiceBusy("The inference queue is full or the service is stopping")
        self._pending += 1
        acquired = False
        handed_off = False
        try:
            await self._gate.acquire()
            acquired = True
            if self._closing:
                raise ServiceBusy("The service is stopping")
            loop = asyncio.get_running_loop()
            job = GenerationJob(loop, self.queue_size)
            self._jobs.add(job)
            future = loop.run_in_executor(self._executor, self._run, job, request, max_new_tokens)
            monitor = asyncio.create_task(self._finish(job, future))
            self._monitors.add(monitor)
            monitor.add_done_callback(self._monitors.discard)
            handed_off = True
            try:
                await asyncio.shield(job.ready)
            except BaseException:
                job.cancel()
                raise
            return job
        finally:
            if not handed_off:
                self._pending -= 1
                if acquired:
                    self._gate.release()

    def _run(self, job: GenerationJob, request: ChatRequest, max_new_tokens: int) -> None:
        iterator = None
        prepared = False
        try:
            if job.stop.is_set():
                job.mark_ready()
                return
            prompt = request.prompt
            if prompt is None:
                prompt = self.engine.format_messages(
                    [message.model_dump() for message in request.messages or ()]
                )
            self.engine.validate_request(prompt, max_new_tokens)
            prepared = True
            job.mark_ready()
            if job.stop.is_set():
                return
            iterator = iter(self.engine.generate(prompt, max_new_tokens))
            terminal = False
            while not job.stop.is_set():
                try:
                    event = next(iterator)
                except StopIteration:
                    break
                if not isinstance(event, GenerationEvent):
                    raise TypeError("Engine returned an invalid generation event")
                if not job.deliver(event):
                    return
                if event.finish_reason is not None:
                    terminal = True
                    break
            if not terminal and not job.stop.is_set():
                raise RuntimeError("Engine ended without a terminal generation event")
        except Exception as error:
            if not prepared:
                job.mark_ready(error)
            else:
                logger.exception("Inference failed")
                job.deliver(GenerationFailure())
        finally:
            if iterator is not None:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()

    async def _finish(self, job: GenerationJob, future: asyncio.Future) -> None:
        try:
            await future
        except Exception:
            logger.exception("Inference worker cleanup failed")
        finally:
            # Retrieve errors even if a disconnected caller never awaited readiness.
            if job.ready.done() and not job.ready.cancelled():
                job.ready.exception()
            self._jobs.discard(job)
            self._pending -= 1
            self._gate.release()

    async def close(self) -> None:
        self._closing = True
        for job in tuple(self._jobs):
            job.cancel()
        if self._monitors:
            await asyncio.shield(asyncio.gather(*tuple(self._monitors)))
        await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)
