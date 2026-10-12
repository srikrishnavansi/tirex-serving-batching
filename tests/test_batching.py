"""Checks for the request batcher, with a fake model so they run in a second.

Run: .venv/bin/python -I -m pytest tests/test_batching.py -q
"""

import asyncio
import random
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "upstream" / "inference"))

from app import metrics  # noqa: E402
from app.batching import MicroBatcher  # noqa: E402


class FakeModel:
    """Echoes each series back tagged with the prediction_length, and logs every call."""

    def __init__(self, delay=0.005, fail_on=None):
        self.delay, self.fail_on = delay, fail_on
        self.calls = []  # (monotonic start, prediction_length, n_series)
        self.lock = threading.Lock()

    def __call__(self, contexts, prediction_length):
        with self.lock:
            self.calls.append((time.perf_counter(), prediction_length, len(contexts)))
        if self.fail_on is not None and self.fail_on in contexts:
            raise ValueError("bad series")
        time.sleep(self.delay)
        return [(series, prediction_length) for series in contexts]


def test_results_go_back_to_the_right_caller():
    model = FakeModel()
    batcher = MicroBatcher(model, max_batch_size=32, max_wait_ms=5)
    rng = random.Random(0)
    requests = [
        ([f"req{i}-s{j}" for j in range(rng.randint(1, 4))], rng.choice([8, 16, 32])) for i in range(300)
    ]

    async def client(contexts, prediction_length):
        await asyncio.sleep(rng.random() * 0.05)  # interleave arrivals
        return await batcher.predict(contexts, prediction_length)

    async def main():
        return await asyncio.gather(*(client(c, h) for c, h in requests))

    results = asyncio.run(main())

    for (contexts, h), got in zip(requests, results):
        assert got == [(series, h) for series in contexts]
    assert sum(n for _, _, n in model.calls) == sum(len(c) for c, _ in requests)
    assert len(model.calls) < len(requests) / 4, f"expected real batching, got {len(model.calls)} calls"


def test_batch_never_exceeds_max_size_and_requests_are_not_split():
    model = FakeModel(delay=0.02)
    batcher = MicroBatcher(model, max_batch_size=8, max_wait_ms=1000)
    sizes = [1, 3, 2, 5, 1, 1, 4, 3, 2, 2, 6, 1] * 4

    async def main():
        tasks = [batcher.predict([f"r{i}-s{j}" for j in range(n)], 16) for i, n in enumerate(sizes)]
        results = await asyncio.gather(*tasks)
        oversized = await batcher.predict([f"big-{j}" for j in range(12)], 16)  # bigger than max: runs alone
        return results, oversized

    start = time.perf_counter()
    results, oversized = asyncio.run(main())

    assert [len(r) for r in results] == sizes
    assert len(oversized) == 12
    batch_sizes = [n for _, _, n in model.calls]
    assert batch_sizes[-1] == 12
    assert all(n <= 8 for n in batch_sizes[:-1]), batch_sizes
    assert sum(batch_sizes[:-1]) == sum(sizes)
    # Batches that hit the size cap go out at once; only a final partial batch (< 8 series)
    # may sit out the 1 s window.
    early = sum(n for at, _, n in model.calls[:-1] if at - start < 0.9)
    assert early >= sum(sizes) - 7, early


def test_lone_request_waits_at_most_the_window():
    model = FakeModel(delay=0)
    batcher = MicroBatcher(model, max_batch_size=256, max_wait_ms=50)

    async def main():
        t0 = time.perf_counter()
        a = asyncio.create_task(batcher.predict(["a"], 8))
        await asyncio.sleep(0.02)
        b = asyncio.create_task(batcher.predict(["b"], 8))  # inside a's window: same batch
        await asyncio.sleep(0.09)
        c = asyncio.create_task(batcher.predict(["c"], 8))  # after the window closed: next batch
        await asyncio.gather(a, b, c)
        return t0

    t0 = asyncio.run(main())

    (first_at, _, first_n), (second_at, _, second_n) = model.calls
    assert (first_n, second_n) == (2, 1)
    assert 0.045 <= first_at - t0 <= 0.09, first_at - t0  # dispatched when a's 50 ms window ran out
    assert 0.11 + 0.045 <= second_at - t0 <= 0.11 + 0.09, second_at - t0


def test_one_bad_request_does_not_fail_its_batch_mates():
    model = FakeModel(fail_on="poison")
    batcher = MicroBatcher(model, max_batch_size=64, max_wait_ms=20)

    async def main():
        calls = asyncio.gather(
            batcher.predict(["ok1"], 8),
            batcher.predict(["poison"], 8),
            batcher.predict(["ok2", "ok3"], 8),
            return_exceptions=True,
        )
        return await asyncio.wait_for(calls, 5)  # a dropped future would otherwise hang forever

    ok1, bad, ok23 = asyncio.run(main())
    assert ok1 == [("ok1", 8)]
    assert isinstance(bad, ValueError)
    assert ok23 == [("ok2", 8), ("ok3", 8)]


def test_metrics_render_in_prometheus_text_format():
    batcher = MicroBatcher(FakeModel(), max_batch_size=8, max_wait_ms=1)
    asyncio.run(batcher.predict(["x", "y"], 8))
    text = metrics.render()
    assert "# TYPE tirex_batch_size histogram" in text
    assert 'tirex_batch_size_bucket{le="+Inf"}' in text
    assert "# TYPE tirex_queue_wait_seconds histogram" in text
    assert "# TYPE tirex_model_latency_seconds histogram" in text
    assert 'tirex_request_latency_seconds{quantile="0.95"}' in text
    count = next(line for line in text.splitlines() if line.startswith("tirex_batch_size_count"))
    assert int(count.split()[1]) == metrics.batch_size.count > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
