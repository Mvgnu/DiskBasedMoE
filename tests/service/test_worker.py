import asyncio
import threading

from disk_moe.contracts import GenerationEvent
from disk_moe.service.schemas import ChatRequest
from disk_moe.service.worker import InferenceWorker


def test_slow_consumer_bounds_production_and_cancellation_closes_iterator():
    class FastEngine:
        def __init__(self):
            self.produced = 0
            self.closed = threading.Event()

        def validate_request(self, prompt, max_new_tokens):
            pass

        def generate(self, prompt, max_new_tokens):
            try:
                for index in range(1000):
                    self.produced += 1
                    yield GenerationEvent(delta="x", token_id=index)
                yield GenerationEvent(finish_reason="length")
            finally:
                self.closed.set()

    async def scenario():
        engine = FastEngine()
        worker = InferenceWorker(engine, max_pending=1, queue_size=1)
        job = await worker.start(ChatRequest(prompt="slow"), 1000)
        # Deliberately consume nothing; one queued event plus one blocked producer.
        await asyncio.sleep(0.15)
        assert engine.produced == 2
        job.cancel()
        await asyncio.wait_for(worker.close(), timeout=2)
        assert engine.closed.is_set()

    asyncio.run(scenario())
