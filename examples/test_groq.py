import os
import sys
import time
import traceback

# Ensure package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokensense import observe, Router, Rule
from tokensense.outputs import Stdout, SQLite, Multi

try:
    import groq
except ImportError:
    print("Please install groq first: pip install groq")
    sys.exit(1)

# Check for API key
if not os.environ.get("GROQ_API_KEY"):
    print("WARNING: GROQ_API_KEY environment variable is not set.")
    print("The API call will likely fail with an AuthenticationError.")
    print("You can run this script like: GROQ_API_KEY=your_key python examples/test_groq.py")
    print("-" * 50)

print("=== TokenSense + Groq Test ===")

# 1. Initialize original Groq client
raw_client = groq.Groq()

# 2. Wrap it with TokenSense
client = observe(
    raw_client,
    output=Multi(Stdout(), SQLite("./examples_usage.db")),
    log_prompts=True,
    log_responses=True,
    user_id="groq_tester",
    tags=["groq", "test"]
)

# 3. Test a standard chat completion
print("\n[1] Making standard API call to llama-3.1-8b-instant...")
try:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a short haiku about coding."}
        ]
    )
    # Output should automatically appear due to Stdout() output!
    time.sleep(0.1) # allow background thread to print
    print(f"\nResponse Content:\n{response.choices[0].message.content}")
except Exception as e:
    print(f"Call failed: {e}")

print("\n" + "="*50 + "\n")

# 4. Test Router with Groq models
print("[2] Testing Router with Groq models...")
router = Router(
    tiers={
        "small": ["llama-3.1-8b-instant"],
        "large": ["llama-3.3-70b-versatile"]
    },
    rules=[
        Rule(if_task="complex-math", pin_tier="large"),
        Rule(if_context_tokens_gt=4000, deny_tiers=["small"])
    ]
)

messages = [{"role": "user", "content": "What is 2+2?"}]

decision = router.route(messages=messages, task_hint="complex-math")
print(f"Router Decision for 'complex-math' task:")
print(f" - Selected Model: {decision.model}")
print(f" - Reason: {decision.reason}")
print(f" - Estimated Cost: ${decision.estimated_cost_usd:.5f}")

try:
    res = client.chat.completions.create(
        model=decision.model,
        messages=messages
    )
    time.sleep(0.1)
    print(f"\nRouted Response Content:\n{res.choices[0].message.content}")
except Exception as e:
    print(f"Call failed: {e}")
