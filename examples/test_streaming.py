"""
Comprehensive streaming verification for TokenSense observe() wrapper.

Tests:
  1. Sync full stream — exact usage extraction (not estimation)
  2. Sync early break — partial event with nonzero input tokens
  3. Async full stream — exact usage extraction
  4. Async early break — partial event with nonzero input tokens
  5. Latency comparison — observe()-wrapped vs raw, confirm no added delay
"""

import time
import asyncio
from tokensense import observe
from tokensense.events import CallEvent

# ─── Mock Infrastructure ────────────────────────────────────────────────────────

EXACT_INPUT_TOKENS = 42
EXACT_OUTPUT_TOKENS = 73

class MockDelta:
    def __init__(self, content):
        self.content = content

class MockChoice:
    def __init__(self, content):
        self.delta = MockDelta(content)

class MockUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

class MockChunk:
    def __init__(self, content, usage=None):
        self.choices = [MockChoice(content)] if content else []
        self.usage = usage

STREAM_WORDS = ["The ", "quick ", "brown ", "fox ", "jumps ", "over ", "the ", "lazy ", "dog."]

def mock_openai_stream(messages, stream=False, stream_options=None, model="gpt-4o-mini"):
    for word in STREAM_WORDS:
        time.sleep(0.005)  # 5ms per chunk
        yield MockChunk(word)

    # Final usage chunk — only sent if stream_options was injected
    if stream_options and stream_options.get("include_usage"):
        yield MockChunk("", usage=MockUsage(EXACT_INPUT_TOKENS, EXACT_OUTPUT_TOKENS))

class MockOpenAIChatCompletions:
    def create(self, **kwargs):
        return mock_openai_stream(**kwargs)
    create.__module__ = "openai.resources.chat.completions"

class MockOpenAIChat:
    def __init__(self):
        self.completions = MockOpenAIChatCompletions()

class MockOpenAIClient:
    def __init__(self):
        self.chat = MockOpenAIChat()

async def mock_async_openai_stream(messages, stream=False, stream_options=None, model="gpt-4o-mini"):
    for word in STREAM_WORDS:
        await asyncio.sleep(0.005)
        yield MockChunk(word)

    if stream_options and stream_options.get("include_usage"):
        yield MockChunk("", usage=MockUsage(EXACT_INPUT_TOKENS, EXACT_OUTPUT_TOKENS))

class MockAsyncOpenAIChatCompletions:
    async def create(self, **kwargs):
        return mock_async_openai_stream(**kwargs)
    create.__module__ = "openai.resources.chat.completions"

class MockAsyncOpenAIChat:
    def __init__(self):
        self.completions = MockAsyncOpenAIChatCompletions()

class MockAsyncOpenAIClient:
    def __init__(self):
        self.chat = MockAsyncOpenAIChat()

# ─── Test Helpers ────────────────────────────────────────────────────────────────

captured_events: list[CallEvent] = []

def event_capture(event: CallEvent):
    captured_events.append(event)

def assert_eq(label, actual, expected):
    status = "✅" if actual == expected else "❌ FAIL"
    print(f"  {status} {label}: {actual} (expected {expected})")
    if actual != expected:
        raise AssertionError(f"{label}: got {actual}, expected {expected}")

def assert_gt(label, actual, threshold):
    status = "✅" if actual > threshold else "❌ FAIL"
    print(f"  {status} {label}: {actual} > {threshold}")
    if actual <= threshold:
        raise AssertionError(f"{label}: {actual} not > {threshold}")

def assert_lt(label, actual, threshold):
    status = "✅" if actual < threshold else "❌ FAIL"
    print(f"  {status} {label}: {actual} < {threshold}")
    if actual >= threshold:
        raise AssertionError(f"{label}: {actual} not < {threshold}")

# ─── Test 1: Sync Full Stream — Exact Usage ────────────────────────────────────

def test_sync_full_stream():
    print("\n═══ Test 1: Sync Full Stream — Exact Usage Extraction ═══")
    captured_events.clear()
    client = observe(MockOpenAIClient(), on_event=event_capture)

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello world"}],
        stream=True,
        model="gpt-4o-mini"
    )

    accumulated = ""
    for chunk in response:
        if chunk.choices:
            accumulated += chunk.choices[0].delta.content

    time.sleep(0.2)  # Let background thread emit

    assert_eq("Accumulated text", accumulated, "The quick brown fox jumps over the lazy dog.")
    assert len(captured_events) == 1, f"Expected 1 event, got {len(captured_events)}"

    ev = captured_events[0]
    # These MUST be the exact mock values, not estimation
    assert_eq("input_tokens (exact)", ev.input_tokens, EXACT_INPUT_TOKENS)
    assert_eq("output_tokens (exact)", ev.output_tokens, EXACT_OUTPUT_TOKENS)
    assert_gt("cost_usd", ev.cost_usd, 0.0)  # Should be nonzero with exact tokens
    print(f"  ℹ️  cost_usd = ${ev.cost_usd:.6f}")

# ─── Test 2: Sync Early Break — Partial Event ──────────────────────────────────

def test_sync_early_break():
    print("\n═══ Test 2: Sync Early Break — Partial Event with Input Tokens ═══")
    captured_events.clear()
    client = observe(MockOpenAIClient(), on_event=event_capture)

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello world"}],
        stream=True,
        model="gpt-4o-mini"
    )

    chunks_seen = 0
    for chunk in response:
        chunks_seen += 1
        if chunks_seen == 3:
            break  # Early termination

    # Explicitly close the generator to force immediate GeneratorExit cleanup.
    # Without this, CPython may defer cleanup to GC which races with assertions.
    response.close()

    time.sleep(0.2)

    assert len(captured_events) == 1, f"Expected 1 event, got {len(captured_events)}"

    ev = captured_events[0]
    # Input tokens MUST be nonzero — the prompt was sent regardless of when we broke
    assert_gt("input_tokens (partial)", ev.input_tokens, 0)
    # Output tokens should reflect partial accumulation via estimation
    assert_gt("output_tokens (partial)", ev.output_tokens, 0)
    print(f"  ℹ️  Partial event: in={ev.input_tokens} out={ev.output_tokens} | ${ev.cost_usd:.6f}")

# ─── Test 3: Async Full Stream — Exact Usage ───────────────────────────────────

async def test_async_full_stream():
    print("\n═══ Test 3: Async Full Stream — Exact Usage Extraction ═══")
    captured_events.clear()
    client = observe(MockAsyncOpenAIClient(), on_event=event_capture)

    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello world"}],
        stream=True,
        model="gpt-4o-mini"
    )

    accumulated = ""
    async for chunk in response:
        if chunk.choices:
            accumulated += chunk.choices[0].delta.content

    await asyncio.sleep(0.2)

    assert_eq("Accumulated text", accumulated, "The quick brown fox jumps over the lazy dog.")
    assert len(captured_events) == 1, f"Expected 1 event, got {len(captured_events)}"

    ev = captured_events[0]
    assert_eq("input_tokens (exact)", ev.input_tokens, EXACT_INPUT_TOKENS)
    assert_eq("output_tokens (exact)", ev.output_tokens, EXACT_OUTPUT_TOKENS)

# ─── Test 4: Async Early Break — Partial Event ─────────────────────────────────

async def test_async_early_break():
    print("\n═══ Test 4: Async Early Break — Partial Event with Input Tokens ═══")
    captured_events.clear()
    client = observe(MockAsyncOpenAIClient(), on_event=event_capture)

    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello world"}],
        stream=True,
        model="gpt-4o-mini"
    )

    chunks_seen = 0
    async for chunk in response:
        chunks_seen += 1
        if chunks_seen == 3:
            break

    # Explicitly close the async generator to force immediate cleanup.
    await response.aclose()

    await asyncio.sleep(0.2)

    assert len(captured_events) == 1, f"Expected 1 event, got {len(captured_events)}"

    ev = captured_events[0]
    assert_gt("input_tokens (partial)", ev.input_tokens, 0)
    assert_gt("output_tokens (partial)", ev.output_tokens, 0)
    print(f"  ℹ️  Partial event: in={ev.input_tokens} out={ev.output_tokens} | ${ev.cost_usd:.6f}")

# ─── Test 5: Latency Comparison — Raw vs Observed ──────────────────────────────

def test_latency_comparison():
    print("\n═══ Test 5: Latency Comparison — Raw vs observe() Wrapped ═══")

    # Baseline: raw stream, no wrapper
    raw_client = MockOpenAIClient()
    raw_start = time.time()
    raw_gen = raw_client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        model="gpt-4o-mini",
        stream_options={"include_usage": True}
    )
    raw_first_chunk_time = None
    for chunk in raw_gen:
        if raw_first_chunk_time is None:
            raw_first_chunk_time = (time.time() - raw_start) * 1000
    raw_total_ms = (time.time() - raw_start) * 1000

    # Observed: with wrapper
    captured_events.clear()
    obs_client = observe(MockOpenAIClient(), on_event=event_capture)
    obs_start = time.time()
    obs_gen = obs_client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        model="gpt-4o-mini"
    )
    obs_first_chunk_time = None
    for chunk in obs_gen:
        if obs_first_chunk_time is None:
            obs_first_chunk_time = (time.time() - obs_start) * 1000
    obs_total_ms = (time.time() - obs_start) * 1000

    time.sleep(0.2)

    first_chunk_overhead = obs_first_chunk_time - raw_first_chunk_time
    total_overhead = obs_total_ms - raw_total_ms

    print(f"  Raw  — first chunk: {raw_first_chunk_time:.2f}ms, total: {raw_total_ms:.2f}ms")
    print(f"  Obs  — first chunk: {obs_first_chunk_time:.2f}ms, total: {obs_total_ms:.2f}ms")
    print(f"  Overhead — first chunk: {first_chunk_overhead:+.2f}ms, total: {total_overhead:+.2f}ms")

    # Assert overhead is under 5ms (generous margin for Python GIL jitter)
    assert_lt("First chunk overhead (ms)", abs(first_chunk_overhead), 5.0)
    assert_lt("Total overhead (ms)", abs(total_overhead), 5.0)

# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    passed = 0
    failed = 0
    tests = [
        ("Sync Full Stream", test_sync_full_stream),
        ("Sync Early Break", test_sync_early_break),
        ("Latency Comparison", test_latency_comparison),
    ]

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ {name} FAILED: {e}")
            failed += 1

    # Async tests
    async def run_async_tests():
        nonlocal passed, failed
        async_tests = [
            ("Async Full Stream", test_async_full_stream),
            ("Async Early Break", test_async_early_break),
        ]
        for name, fn in async_tests:
            try:
                await fn()
                passed += 1
            except Exception as e:
                print(f"  ❌ {name} FAILED: {e}")
                failed += 1

    asyncio.run(run_async_tests())

    print(f"\n{'═' * 50}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    if failed == 0:
        print("🎉 All streaming tests passed!")
    else:
        print("⚠️  Some tests failed — review output above.")

if __name__ == "__main__":
    main()
