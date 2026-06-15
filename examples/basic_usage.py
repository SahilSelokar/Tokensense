import sys
import os
import time

# Ensure package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokensense import observe, Router, Rule
from tokensense.outputs import SQLite, Stdout, Multi

# Mock client to simulate LLM providers
class MockUsage:
    def __init__(self, prompt_tokens=15, completion_tokens=35, input_tokens=15, output_tokens=35):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

class MockChoice:
    def __init__(self, content):
        class MockMessage:
            def __init__(self, text):
                self.content = text
        self.message = MockMessage(content)

class MockOpenAIResponse:
    def __init__(self, model, content):
        self.model = model
        self.choices = [MockChoice(content)]
        self.usage = MockUsage()

class MockOpenAIChatCompletions:
    def create(self, model, messages, **kwargs):
        print(f"[Mock OpenAI API Call] Generating response using {model}...")
        # Simulate network latency
        time.sleep(0.3)
        return MockOpenAIResponse(model, f"This is a mock completion from {model}.")

class MockOpenAIClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": MockOpenAIChatCompletions()})()

# -----------------
# 1. Observe Client usage
# -----------------
print("=== 1. Observability Demo ===")
raw_client = MockOpenAIClient()

# Event hook
def on_expensive_call(event):
    if event.cost_usd > 0.005:
        print(f"  [ALERT] Expensive call detected: ${event.cost_usd:.4f}")

# Wrap with tokensense observe using multiple outputs
client = observe(
    raw_client, 
    output=Multi(Stdout(), SQLite("./examples_usage.db")),
    log_prompts=True,
    log_responses=True,
    user_id="user_123",
    session_id="session_abc",
    on_event=on_expensive_call
)

# Run a completions call
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me a story about coding."}]
)

# Give background thread a moment to print stdout
time.sleep(0.1)

# -----------------
# 2. Smart Router usage
# -----------------
print("\n=== 2. Smart Router Demo ===")
router = Router(
    tiers={
        "small": ["llama3-8b-8192", "claude-haiku-4-5"],
        "large": ["claude-sonnet-4-6", "gpt-4o"],
    },
    rules=[
        Rule(if_context_tokens_gt=4000, deny_tiers=["small"]),
        Rule(if_task="legal-review", pin_tier="large"),
    ],
    on_failure="escalate"
)

def run_routed_chat(messages, task):
    decision = router.route(messages=messages, task_hint=task)
    print(f"Router selected model: {decision.model}")
    print(f"Reason: {decision.reason}")
    print(f"Estimated Cost: ${decision.estimated_cost_usd:.4f}")
    
    # Execute the call
    res = client.chat.completions.create(
        model=decision.model,
        messages=messages
    )
    time.sleep(0.1) # give background thread a moment
    return res

print("\nRouting normal task hint 'code-review' (context length ~ short):")
run_routed_chat(
    messages=[{"role": "user", "content": "Review this python code."}],
    task="code-review"
)

print("\nRouting task hint 'legal-review' (pins to large tier):")
run_routed_chat(
    messages=[{"role": "user", "content": "Review this contract."}],
    task="legal-review"
)

# Show context bounds checking
print("\nRouting with huge context that exceeds small model limits:")
long_messages = [{"role": "user", "content": "long text " * 5000}] # ~20,000 characters = ~5,000 tokens
run_routed_chat(
    messages=long_messages,
    task="summarise"
)
