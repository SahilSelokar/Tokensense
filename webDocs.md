# TokenSense Documentation
**by Visual Vortex** · [GitHub](https://github.com/visualvortex/tokensense) · [PyPI](https://pypi.org/project/tokensense-ai)

> LLM cost observability and smart routing — stays in your code, works everywhere.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quickstart](#quickstart)
4. [Core Concepts](#core-concepts)
5. [observe()](#observe)
6. [Outputs](#outputs)
   - [Stdout](#stdout)
   - [SQLite](#sqlite)
   - [Logger](#logger)
   - [HTTP](#http)
   - [Multi](#multi)
   - [Auto Detection](#auto-detection)
7. [Router](#router)
   - [Tiers](#tiers)
   - [Rules](#rules)
   - [Context Budget](#context-budget)
   - [Per-Call Overrides](#per-call-overrides)
   - [Routing Decision](#routing-decision)
   - [on_failure](#on_failure)
8. [Cost Engine](#cost-engine)
9. [Privacy](#privacy)
10. [Environment Detection](#environment-detection)
11. [Event Callback](#event-callback)
12. [Supported Models](#supported-models)
13. [Production Guide](#production-guide)
14. [FAQ](#faq)
15. [Roadmap](#roadmap)

---

## Introduction

TokenSense is an open-source Python framework for LLM cost observability and smart model routing. It wraps your existing LLM clients with one line of code and captures metadata — tokens, cost, latency — after every call.

**It is not:**
- A proxy — your API keys and prompts never go through our servers
- A SaaS platform — no account, no sign-up, nothing to install except the package
- A dashboard — it outputs to wherever you point it

**It is:**
- A Python package that runs inside your process
- A transparent wrapper around OpenAI, Anthropic, and Groq clients
- A routing layer that is context-budget-aware, rule-based, and auditable
- Private by default — prompts are never captured unless you explicitly opt in

---

## Installation

```bash
pip install tokensense-ai
```

**Requirements:** Python 3.10+

TokenSense has zero required dependencies. The core package uses only Python's standard library — `sqlite3`, `logging`, `threading`, `urllib`.

**Optional — for development and testing:**
```bash
pip install tokensense-ai[dev]
# includes: pytest, pytest-asyncio, anthropic, openai, groq
```

---

## Quickstart

### Step 1 — Wrap your client

```python
from tokensense import observe
import anthropic

client = observe(anthropic.Anthropic())
```

### Step 2 — Use it exactly as before

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Explain async/await in Python"}]
)
print(response.content[0].text)
```

### Step 3 — See the output

```
→ ✓ model=claude-sonnet-4-6 | in=18 out=312 tokens | $0.0052 | 847ms
```

That's it. One import, one wrapper. Every call is now tracked. Your existing code doesn't change.

---

## Core Concepts

### How TokenSense intercepts calls

TokenSense wraps your LLM client using Python's `__getattr__` proxy pattern. When you call `client.messages.create(...)`, TokenSense:

1. Forwards the call to the original client — unchanged
2. Waits for the response
3. Extracts metadata from the response (tokens, model, cost)
4. Emits a `CallEvent` to a background thread
5. Returns the original response to your code

Your code receives the exact same response object as before. The background thread handles the event asynchronously — your call latency is not affected.

### CallEvent

Every intercepted call produces a `CallEvent`:

```python
@dataclass
class CallEvent:
    ts: str             # ISO 8601 UTC timestamp
    model: str          # model name from response
    provider: str       # anthropic | openai | groq | unknown
    input_tokens: int
    output_tokens: int
    cost_usd: float     # estimated cost
    latency_ms: int
    user_id: str | None
    session_id: str | None
    tags: list[str] | None
    routed_tier: str | None   # set by Router if routing was used
    error: str | None         # set if the call raised an exception
```

Prompt content and response content are **never** included by default.

---

## observe()

The core function. Wraps any supported LLM client and returns a drop-in replacement.

### Signature

```python
def observe(
    client: Any,
    output: BaseOutput | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    log_prompts: bool = False,
    log_responses: bool = False,
    on_event: Callable[[CallEvent], None] | None = None,
) -> ObservedClient
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `client` | Any | required | LLM client to wrap |
| `output` | BaseOutput | auto | Where events are sent. Auto-detects by ENV if not set |
| `user_id` | string | None | Identifier attached to every event from this client |
| `session_id` | string | None | Groups multiple calls into a session |
| `tags` | list[str] | None | Labels for filtering and segmentation |
| `log_prompts` | bool | False | Include prompt content in events (opt-in) |
| `log_responses` | bool | False | Include response content in events (opt-in) |
| `on_event` | callable | None | Function called after each event is written |

### Examples

**Minimal — just observe:**
```python
from tokensense import observe
client = observe(anthropic.Anthropic())
```

**With output:**
```python
from tokensense import observe
from tokensense.outputs import SQLite

client = observe(anthropic.Anthropic(), output=SQLite("./usage.db"))
```

**With user context:**
```python
client = observe(
    anthropic.Anthropic(),
    user_id="user_123",
    session_id="chat_session_456",
    tags=["production", "chat-feature"],
)
```

**Wrapping OpenAI:**
```python
import openai
client = observe(openai.OpenAI())
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)
```

**Wrapping Groq:**
```python
import groq
client = observe(groq.Groq())
response = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[{"role": "user", "content": "Hello"}]
)
```

**Async client:**
```python
import anthropic
client = observe(anthropic.AsyncAnthropic())
response = await client.messages.create(...)
```

**With explicit prompt logging:**
```python
# only do this when you specifically need prompt content in your logs
client = observe(
    anthropic.Anthropic(),
    log_prompts=True,
    log_responses=True,
)
```

---

## Outputs

Outputs define where captured events are sent. Pass any output class to `observe()` via the `output` parameter.

### Stdout

Prints a one-line summary to the terminal after every call.

```python
from tokensense.outputs import Stdout

client = observe(anthropic.Anthropic(), output=Stdout())
```

**Output format:**
```
→ ✓ model=claude-sonnet-4-6 | in=1204 out=387 tokens | $0.0061 | 934ms
→ ✗ model=gpt-4o-mini | in=44 out=0 tokens | $0.0000 | 203ms | error=rate limit exceeded
```

| | |
|---|---|
| Best for | Development, debugging |
| Infra required | None |
| Production safe | No |

---

### SQLite

Persists every event to a local SQLite `.db` file. File is created automatically if it doesn't exist.

```python
from tokensense.outputs import SQLite

client = observe(anthropic.Anthropic(), output=SQLite("./usage.db"))
client = observe(anthropic.Anthropic(), output=SQLite("/var/data/tokensense.db"))
```

**Query your data:**
```sql
-- total spend by model
SELECT model, COUNT(*) as calls, SUM(cost_usd) as total_cost
FROM calls
GROUP BY model
ORDER BY total_cost DESC;

-- calls in the last 24 hours
SELECT * FROM calls
WHERE ts > datetime('now', '-1 day')
ORDER BY ts DESC;

-- most expensive sessions
SELECT session_id, SUM(cost_usd) as total
FROM calls
GROUP BY session_id
ORDER BY total DESC
LIMIT 10;
```

**Schema:**
```sql
CREATE TABLE calls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    model         TEXT NOT NULL,
    provider      TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cost_usd      REAL,
    latency_ms    INTEGER,
    user_id       TEXT,
    session_id    TEXT,
    tags          TEXT,
    routed_tier   TEXT,
    error         TEXT
);
```

| | |
|---|---|
| Best for | Local dev, single-process staging |
| Infra required | None |
| Production safe | Single worker only |
| Multi-worker | Not recommended — use Logger or HTTP |

---

### Logger

Writes structured JSON to Python's standard `logging` module. In production this automatically routes to CloudWatch, Datadog, GCP Logging, Grafana Loki, or whatever log aggregator is already configured. Zero new infra.

```python
from tokensense.outputs import Logger

# default logger name: "tokensense"
client = observe(anthropic.Anthropic(), output=Logger())

# custom logger name
client = observe(anthropic.Anthropic(), output=Logger("myapp.llm"))
```

**Output format (one line per call):**
```json
{"ts": "2026-06-14T10:23:11Z", "model": "claude-sonnet-4-6", "provider": "anthropic", "input_tokens": 1204, "output_tokens": 387, "cost_usd": 0.0061, "latency_ms": 934, "user_id": "user_123", "session_id": null, "tags": ["production"], "routed_tier": null, "error": null}
```

**Works with your existing log setup:**
```python
import logging

# configure once in your app startup — tokensense uses it automatically
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()]
)

client = observe(anthropic.Anthropic(), output=Logger())
```

| | |
|---|---|
| Best for | Production |
| Infra required | None — uses existing log infra |
| Production safe | Yes — multi-worker safe |
| Works with | CloudWatch, Datadog, GCP, Loki, ELK, any log aggregator |

---

### HTTP

POSTs every event as JSON to any HTTP endpoint you control.

```python
from tokensense.outputs import HTTP

client = observe(
    anthropic.Anthropic(),
    output=HTTP("https://your-server.com/ingest")
)

# with custom headers
client = observe(
    anthropic.Anthropic(),
    output=HTTP(
        "https://your-server.com/ingest",
        headers={"Authorization": "Bearer your-token"}
    )
)
```

**Request format:**
```
POST https://your-server.com/ingest
Content-Type: application/json

{"ts": "...", "model": "...", "input_tokens": 1204, ...}
```

**Important:** HTTP output fires in a background thread and times out after 5 seconds. A failed POST is silently dropped — it never crashes your app or blocks your LLM call.

| | |
|---|---|
| Best for | Self-hosted dashboards, custom ingestion |
| Infra required | Your own endpoint |
| Production safe | Yes |
| Failure behaviour | Silent drop — never affects your app |

---

### Multi

Writes to multiple outputs at once.

```python
from tokensense.outputs import Multi, Stdout, SQLite, Logger

# see it in terminal AND save it locally
client = observe(
    anthropic.Anthropic(),
    output=Multi(Stdout(), SQLite("./usage.db"))
)

# production: log infra + your own endpoint
client = observe(
    anthropic.Anthropic(),
    output=Multi(Logger(), HTTP("https://your-server.com/ingest"))
)
```

If one output fails, the others continue. Failures are silently swallowed per output.

---

### Auto Detection

When no output is specified, TokenSense reads the environment and picks a sensible default:

```python
client = observe(anthropic.Anthropic())  # no output= argument
```

| `ENV` value | Output |
|---|---|
| `production` | `Logger("tokensense")` |
| `staging` | `Multi(Stdout(), SQLite())` |
| `development` | `Multi(Stdout(), SQLite())` |
| not set | `Multi(Stdout(), SQLite())` |

Reads from: `ENV`, `ENVIRONMENT`, `APP_ENV` — whichever is set first.

Override at any time by passing `output=` explicitly.

---

## Router

The routing layer. Selects the right model for each call based on rules, context budget, and cost constraints. Runs entirely in-process.

### Why routing matters

Naive routing — "short prompt = cheap model" — silently breaks things:
- A 150k token conversation sent to a model with an 8k context window gets truncated without warning
- Switching from Sonnet to Haiku for a complex multi-constraint prompt causes capability regression
- There is no way to know routing failed until a user reports a wrong answer

TokenSense's router checks the context budget before routing and escalates automatically on failure.

### Basic setup

```python
from tokensense.router import Router, Rule

router = Router(
    tiers={
        "small":  ["claude-haiku-4-5"],
        "large":  ["claude-sonnet-4-6"],
    },
    rules=[
        Rule(if_context_tokens_gt=4000, deny_tiers=["small"]),
        Rule(if_task="legal-review", pin_tier="large"),
    ],
    on_failure="escalate",
)
```

### Getting a routing decision

```python
messages = [{"role": "user", "content": "Summarise this contract..."}]

decision = router.route(
    messages=messages,
    task_hint="legal-review",
)

print(decision.model)   # claude-sonnet-4-6
print(decision.tier)    # large
print(decision.reason)  # pinned to large by rule
```

### Making the call with the routed model

```python
client = observe(anthropic.Anthropic())

decision = router.route(messages=messages, task_hint="summarise")

response = client.messages.create(
    model=decision.model,
    max_tokens=1024,
    messages=messages,
)
```

---

### Tiers

A tier is a named group of models. Models within a tier are tried in order — the first one whose context window fits the conversation is selected.

```python
tiers={
    "small":  ["claude-haiku-4-5", "groq/llama3-8b"],  # tried left to right
    "medium": ["gpt-4o-mini"],
    "large":  ["claude-sonnet-4-6", "gpt-4o"],
}
```

**Tier order matters.** The router tries tiers from first to last in the dict. Earlier tiers are preferred unless rules say otherwise.

---

### Rules

Rules are evaluated in order. The first matching rule that sets a `pin_tier` wins for that action. Deny rules are cumulative — multiple rules can deny multiple tiers.

#### deny_tiers
Excludes tiers when the condition is true.

```python
# never route to small models when the conversation is long
Rule(if_context_tokens_gt=4000, deny_tiers=["small"])

# never route to small or medium when cost is a concern
Rule(if_estimated_cost_gt=0.05, deny_tiers=["small", "medium"])
```

#### pin_tier
Forces a specific tier regardless of other rules. First pin wins.

```python
# always use the best model for sensitive tasks
Rule(if_task="legal-review", pin_tier="large")
Rule(if_task="medical-advice", pin_tier="large")
```

#### prefer_tier
Softly prefers a tier. Overridden by deny rules.

```python
# prefer cheap models when cost is low
Rule(prefer_tier="small", if_estimated_cost_gt=0.001)
```

#### escalate on error
Auto-retry on the next tier when a specific error code is received.

```python
Rule(if_error_code=429, escalate=True)   # rate limit → try next tier
Rule(if_error_code=529, escalate=True)   # overloaded → try next tier
```

#### Rule conditions reference

| Condition | Type | Description |
|---|---|---|
| `if_context_tokens_gt` | int | Fires when estimated token count exceeds this |
| `if_task` | string | Fires when `task_hint` matches exactly |
| `if_error_code` | int | Fires when previous call returned this HTTP status |
| `if_estimated_cost_gt` | float | Fires when estimated cost (USD) exceeds this |

All conditions on a single `Rule` are AND-ed. Use separate rules for OR logic.

---

### Context Budget

Before selecting a model, the router checks whether the conversation fits in the model's context window. Models that can't fit the conversation are automatically excluded.

```python
# llama3-8b has 8,192 token context
# claude-sonnet-4-6 has 200,000 token context

router = Router(tiers={
    "small": ["llama3-8b-8192"],
    "large": ["claude-sonnet-4-6"],
})

# a 10,000 token conversation
decision = router.route(messages=long_history)
# → small is excluded (10k > 8k context)
# → routed to large automatically
# → reason: "selected large — small excluded by context budget"
```

TokenSense applies a 10% safety margin — a model with a 100k context window is only used for conversations up to 90k tokens.

If `context_tokens` is not provided, TokenSense estimates it from message content using a 4-chars-per-token approximation. Pass the exact count if you have it:

```python
decision = router.route(messages=messages, context_tokens=8432)
```

---

### Per-Call Overrides

Override routing behaviour on a per-call basis.

```python
decision = router.route(
    messages=msgs,
    task_hint="code-review",   # matches if_task rules
    max_cost_usd=0.005,        # hard cost cap — tiers exceeding this are excluded
    min_tier="medium",         # floor — never route below this tier
    context_tokens=1842,       # skip estimation, use exact count
)
```

| Override | Type | Description |
|---|---|---|
| `task_hint` | string | Label passed to `if_task` rule conditions |
| `max_cost_usd` | float | Hard cost ceiling — expensive tiers excluded |
| `min_tier` | string | Minimum tier — never route below this |
| `context_tokens` | int | Exact token count — skips estimation |

---

### Routing Decision

`router.route()` always returns a `RoutingDecision` object.

```python
decision = router.route(messages=msgs)

decision.model              # "claude-haiku-4-5" — the selected model
decision.tier               # "small" — the selected tier
decision.reason             # "default tier small" — human-readable reason
decision.estimated_cost_usd # 0.000034 — pre-call cost estimate
decision.denied_tiers       # ["large"] — tiers that were excluded
```

Every routing decision is also included in the `CallEvent` as `routed_tier`, so your output captures which tier was used for each call.

---

### on_failure

Controls what happens when a model call fails.

```python
Router(on_failure="escalate")
# → if the selected model fails, try the next tier up automatically
# → if all tiers fail, raise the last exception

Router(on_failure="error")
# → surface the error immediately, no retry
# → useful when you want explicit control over fallback logic
```

---

## Cost Engine

TokenSense maintains a cost table for 20+ models and computes estimated cost for every call.

### estimate_cost()

```python
from tokensense.cost import estimate_cost

cost = estimate_cost("claude-sonnet-4-6", input_tokens=1000, output_tokens=500)
# → 0.0105 USD
```

### get_context_window()

```python
from tokensense.cost import get_context_window

window = get_context_window("claude-sonnet-4-6")
# → 200000
```

### Fuzzy model matching

Versioned model names are matched automatically:

```python
estimate_cost("claude-sonnet-4-6-20250514", 1000, 500)
# → matches "claude-sonnet-4-6" → $0.0105
```

### Updating the cost table

Token prices change. The cost table lives in `tokensense/cost.py` as a plain Python dict:

```python
MODEL_COSTS: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.00, 15.00),  # (input per 1M, output per 1M)
    ...
}
```

Update this file when providers change pricing. Pull requests to update prices are welcome.

---

## Privacy

### What is captured by default

Only metadata — never content:

| Field | Captured |
|---|---|
| Model name | ✅ |
| Token counts | ✅ |
| Cost estimate | ✅ |
| Latency | ✅ |
| Timestamp | ✅ |
| Routing decision | ✅ |
| Error message | ✅ |
| Prompt content | ❌ Never by default |
| Response content | ❌ Never by default |
| API keys | ❌ Never |
| User IP | ❌ Never |

### Opt-in content logging

```python
client = observe(
    anthropic.Anthropic(),
    log_prompts=True,     # include prompt messages in events
    log_responses=True,   # include response text in events
)
```

Even when enabled, content goes to your configured output only — never to us.

### No telemetry

TokenSense has no outbound network calls of its own. There is no analytics, no usage tracking, no background pings. The only network calls in the package are:
- Your LLM calls (forwarded to the provider you configured)
- `HTTP` output posts (if you use `HTTP` output — goes to your endpoint only)

---

## Environment Detection

TokenSense reads environment variables to pick sensible defaults without configuration.

### Variables read

```bash
ENV=production          # or
ENVIRONMENT=production  # or
APP_ENV=production
```

### Behaviour

| Value | Default output |
|---|---|
| `production` | `Logger("tokensense")` |
| `staging` | `Multi(Stdout(), SQLite())` |
| `development` | `Multi(Stdout(), SQLite())` |
| not set | `Multi(Stdout(), SQLite())` |

### Override

Environment detection only applies when you don't pass `output=`. Passing an explicit output always takes priority:

```python
# this ignores ENV — always uses SQLite
client = observe(anthropic.Anthropic(), output=SQLite("./usage.db"))
```

---

## Event Callback

Run any function after every event is written. Useful for alerting, custom metrics, or side effects.

```python
def alert_on_expensive_call(event: CallEvent) -> None:
    if event.cost_usd and event.cost_usd > 0.05:
        send_slack_alert(f"High cost call: ${event.cost_usd:.4f} on {event.model}")

client = observe(
    anthropic.Anthropic(),
    on_event=alert_on_expensive_call,
)
```

The callback fires in the same background thread as the output write. Keep it fast — expensive operations in the callback can cause the background queue to fall behind.

---

## Supported Models

### Anthropic

| Model | Context Window | Input (per 1M) | Output (per 1M) |
|---|---|---|---|
| claude-opus-4-8 | 200,000 | $15.00 | $75.00 |
| claude-opus-4-7 | 200,000 | $15.00 | $75.00 |
| claude-opus-4-6 | 200,000 | $15.00 | $75.00 |
| claude-sonnet-4-6 | 200,000 | $3.00 | $15.00 |
| claude-haiku-4-5 | 200,000 | $0.80 | $4.00 |

### OpenAI

| Model | Context Window | Input (per 1M) | Output (per 1M) |
|---|---|---|---|
| gpt-4o | 128,000 | $5.00 | $15.00 |
| gpt-4o-mini | 128,000 | $0.15 | $0.60 |
| gpt-4-turbo | 128,000 | $10.00 | $30.00 |
| gpt-3.5-turbo | 16,385 | $0.50 | $1.50 |
| o1 | 200,000 | $15.00 | $60.00 |
| o1-mini | 128,000 | $3.00 | $12.00 |
| o3-mini | 200,000 | $1.10 | $4.40 |

### Groq

| Model | Context Window | Input (per 1M) | Output (per 1M) |
|---|---|---|---|
| llama3-8b-8192 | 8,192 | $0.05 | $0.10 |
| llama3-70b-8192 | 8,192 | $0.59 | $0.79 |
| llama-3.1-8b-instant | 131,072 | $0.05 | $0.08 |
| llama-3.3-70b-versatile | 131,072 | $0.59 | $0.79 |
| mixtral-8x7b-32768 | 32,768 | $0.24 | $0.24 |
| gemma2-9b-it | 8,192 | $0.20 | $0.20 |

### Google

| Model | Context Window | Input (per 1M) | Output (per 1M) |
|---|---|---|---|
| gemini-1.5-pro | 2,097,152 | $3.50 | $10.50 |
| gemini-1.5-flash | 1,048,576 | $0.075 | $0.30 |
| gemini-2.0-flash | 1,048,576 | $0.10 | $0.40 |

---

## Production Guide

### Recommended setup

```python
import os
import anthropic
from tokensense import observe
from tokensense.outputs import Logger, HTTP, Multi

# production — log to existing infra + your own endpoint
client = observe(
    anthropic.Anthropic(),
    output=Multi(
        Logger("myapp.llm"),
        HTTP(os.environ["TOKENSENSE_INGEST_URL"])
    ),
    user_id=current_user.id,
    session_id=request.session_id,
    tags=["production", "chat"],
)
```

### Multi-worker deployments

Use `Logger` output — it is safe for Gunicorn, uWSGI, Celery, and any multi-process setup:

```python
client = observe(anthropic.Anthropic(), output=Logger())
```

Do not use `SQLite` output in multi-worker production — concurrent writes will cause lock contention.

### Serverless (Lambda, Cloud Run, Cloud Functions)

`Logger` output works in serverless environments — it writes to stdout which the platform captures automatically:

```python
# CloudWatch picks this up automatically on AWS Lambda
client = observe(anthropic.Anthropic(), output=Logger())
```

### Containerised deployments (Docker, Kubernetes)

```python
# logs go to container stdout → your log aggregator
client = observe(anthropic.Anthropic(), output=Logger())
```

### Using Router in production

```python
from tokensense.router import Router, Rule

router = Router(
    tiers={
        "small":  ["claude-haiku-4-5"],
        "large":  ["claude-sonnet-4-6"],
    },
    rules=[
        Rule(if_context_tokens_gt=8000, deny_tiers=["small"]),
        Rule(if_task="customer-facing", pin_tier="large"),
    ],
    on_failure="escalate",
    default_tier="small",
)

client = observe(anthropic.Anthropic(), output=Logger())

def handle_chat(messages, user_id, task):
    decision = router.route(messages=messages, task_hint=task)
    return client.messages.create(
        model=decision.model,
        max_tokens=1024,
        messages=messages,
    )
```

---

## FAQ

**Does TokenSense add latency to my LLM calls?**
No. The observer fires in a background thread after the response is returned to your code. Your call time is unchanged.

**Does TokenSense send my prompts anywhere?**
No. Prompts are never captured by default. Even if you enable `log_prompts=True`, content goes only to the output you configure — never to us.

**Does TokenSense call home?**
No. There is no telemetry, no usage tracking, no analytics. The package has no built-in outbound network calls other than the `HTTP` output which calls a URL you provide.

**Can I use TokenSense with LangChain?**
Wrap the underlying Anthropic or OpenAI client before passing it to LangChain. Full LangChain integration is on the roadmap.

**Can I use SQLite in production?**
For single-worker deployments yes. For multi-worker or containerised deployments use `Logger` output instead.

**Can I add a new model to the cost table?**
Yes — edit `tokensense/cost.py` and add the model to `MODEL_COSTS` and `CONTEXT_WINDOWS`. PRs welcome.

**What happens if my output fails?**
Failures in outputs are silently swallowed. A failed SQLite write or HTTP POST never crashes your app or affects your LLM call.

**What happens if the router has no eligible models?**
`router.route()` raises `RuntimeError` with a clear message. Check your rules — you may have denied all tiers simultaneously.

**Is TokenSense thread-safe?**
Yes. The background emit thread uses a daemon thread per event. `SQLite` output uses a threading lock for concurrent writes.

---

## Roadmap

### v0.1.0 — Current
- ✅ `observe()` — Anthropic, OpenAI, Groq
- ✅ Sync and async client support
- ✅ `Stdout`, `SQLite`, `Logger`, `HTTP`, `Multi` outputs
- ✅ Auto environment detection
- ✅ Cost table — 20+ models
- ✅ Context window table — 20+ models
- ✅ `Router` with context budget check
- ✅ Rule engine — deny, pin, prefer, escalate
- ✅ Per-call overrides
- ✅ `on_event` callback hook
- ✅ `PRIVACY.md`

### v0.2.0
- 🔜 Shadow testing — run two tiers in parallel before switching
- 🔜 `tokensense report` CLI — terminal spend summary
- 🔜 Budget alerts — projected monthly spend warnings
- 🔜 Google Gemini support
- 🔜 Streaming token count accuracy

### v0.3.0
- 🔜 `FileSink` — JSONL output
- 🔜 `OTelSink` — OpenTelemetry compatible export
- 🔜 LangChain integration
- 🔜 LlamaIndex integration

### Later
- 🔜 Postgres output
- 🔜 Prometheus metrics
- 🔜 TypeScript SDK
- 🔜 Visual Vortex dashboard sink (opt-in)

---

## Contributing

TokenSense is MIT licensed and built in public.

```bash
git clone https://github.com/visualvortex/tokensense
cd tokensense
pip install -e ".[dev]"
pytest
```

Issues, PRs, and feedback are welcome on [GitHub](https://github.com/visualvortex/tokensense).

---

## License

MIT — see [LICENSE](https://github.com/visualvortex/tokensense/blob/main/LICENSE)

---

<p align="center">Built by <a href="https://visualvortexcreatives.dev">Visual Vortex</a></p>