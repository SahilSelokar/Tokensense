import sys
import os
import unittest
import time
from typing import Dict, Any

# Ensure package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokensense import estimate_cost, observe, Router, Rule, Stdout, SQLite
from tokensense.outputs import BaseOutput
from tokensense.events import CallEvent

class ListOutput(BaseOutput):
    def __init__(self):
        self.logs = []

    def write(self, event: CallEvent) -> None:
        self.logs.append(event)

# Mock classes for testing
class MockUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

class MockChoice:
    def __init__(self, content):
        class MockMessage:
            def __init__(self, text):
                self.content = text
        self.message = MockMessage(content)

class MockOpenAIResponse:
    def __init__(self, model, content, prompt_tokens=100, completion_tokens=200):
        self.model = model
        self.choices = [MockChoice(content)]
        self.usage = MockUsage(prompt_tokens, completion_tokens)

class MockOpenAIChatCompletions:
    def create(self, model, messages, **kwargs):
        return MockOpenAIResponse(model, "mocked response content", 100, 200)

class MockOpenAIClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": MockOpenAIChatCompletions()})()

class TestTokenSense(unittest.TestCase):
    def test_estimate_cost(self):
        # gpt-4o input $5/M, output $15/M
        cost = estimate_cost("gpt-4o", 100_000, 200_000)
        # 100,000 * 5/1,000,000 + 200,000 * 15/1,000,000 = 0.5 + 3.0 = 3.5
        self.assertAlmostEqual(cost, 3.5, places=5)

        # fuzzy matching
        fuzzy_cost = estimate_cost("claude-sonnet-4-6-20250514", 1_000_000, 1_000_000)
        self.assertAlmostEqual(fuzzy_cost, 18.0, places=5) # 3.0 + 15.0 = 18.0

        # unknown fallback
        fallback_cost = estimate_cost("unknown-model", 1_000_000, 1_000_000)
        self.assertAlmostEqual(fallback_cost, 0.75, places=5) # 0.15 + 0.60 = 0.75

    def test_observe_calls(self):
        list_output = ListOutput()
        raw_client = MockOpenAIClient()
        
        # We need to capture events in a list, but background thread might race.
        # on_event runs in the same background thread.
        events_captured = []
        def on_event(ev):
            events_captured.append(ev)
            
        client = observe(
            raw_client, 
            output=list_output, 
            log_prompts=True, 
            log_responses=True,
            user_id="user_test",
            on_event=on_event
        )
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "test prompt"}]
        )
        
        # Give thread time to write
        time.sleep(0.1)
        
        self.assertEqual(len(list_output.logs), 1)
        self.assertEqual(len(events_captured), 1)
        
        event = list_output.logs[0]
        self.assertEqual(event.model, "gpt-4o")
        self.assertEqual(event.input_tokens, 100)
        self.assertEqual(event.output_tokens, 200)
        self.assertEqual(event.user_id, "user_test")
        self.assertEqual(event.prompt, "[{'role': 'user', 'content': 'test prompt'}]")
        self.assertEqual(event.response, "mocked response content")
        self.assertEqual(event.error, None)

    def test_router_model_selection(self):
        router = Router(
            tiers={
                "small": ["llama3-8b-8192", "claude-haiku-4-5"],
                "large": ["claude-sonnet-4-6", "gpt-4o"],
            },
            rules=[
                Rule(if_context_tokens_gt=4000, deny_tiers=["small"]),
                Rule(if_task="legal-review", pin_tier="large"),
            ]
        )

        # Short context, no task hint
        decision = router.route([{"content": "short text"}], task_hint=None)
        self.assertEqual(decision.model, "llama3-8b-8192")
        self.assertEqual(decision.tier, "small")
        
        # Long context > 4000
        long_messages = [{"content": "long text " * 5000}] # ~20000 chars = ~5000 tokens
        decision_long = router.route(long_messages, task_hint=None)
        self.assertEqual(decision_long.model, "claude-sonnet-4-6")
        self.assertEqual(decision_long.tier, "large")

        # Pinned task hint
        decision_task = router.route([{"content": "short text"}], task_hint="legal-review")
        self.assertEqual(decision_task.model, "claude-sonnet-4-6")
        self.assertEqual(decision_task.tier, "large")

if __name__ == "__main__":
    unittest.main()
