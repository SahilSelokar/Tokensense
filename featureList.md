# TokenSense — Complete Feature List
> by Visual Vortex | Open Source | MIT License

---

## Core Philosophy
- In-process — runs inside your code, never a proxy
- Privacy by default — prompts never leave your process unless you say so
- Zero latency — observer fires after response returns, your call time is unchanged
- Zero infra — no Docker, no server, no database required to get started
- Zero dependencies — core package has no required third-party dependencies
- Framework agnostic — wraps raw clients, not tied to LangChain or anything else
- You own the data — output goes wherever you point it

---

## 1. observe() — LLM Call Instrumentation

### What it does
Wraps any supported LLM client transparently. Every call you make is intercepted after it returns, metadata is captured, and the event is emitted to your configured output in a background thread.

### How it works
```python
from tokensense import observe

client = observe(anthropic.Anthropic())
# identical API — nothing else in your code changes
response = client.messages.create(...)
```

### What gets captured per call
| Field | Type | Description |
|---|---|---|
| `ts` | string | ISO 8601 UTC timestamp |
| `model` | string | Model name from the response |
| `provider` | string | anthropic / openai / groq / unknown |
| `input_tokens` | int | Tokens in the prompt |
| `output_tokens` | int | Tokens in the response |
| `cost_usd` | float | Estimated cost in USD |
| `latency_ms` | int | End-to-end call latency |
| `user_id` | string | Optional — set by caller |
| `session_id` | string | Optional — set by caller |
| `tags` | list | Optional — set by caller |
| `routed_tier` | string | Set by Router if routing was used |
| `error` | string | Error message if call failed |

### What never gets captured by default
- Prompt content
- Response content
- System prompts
- API keys
- Any PII

### Parameters
```python
client = observe(
    client,                    # any supported LLM client
    output=SQLite("./usage.db"), # where events go — default: auto-detect by ENV
    user_id="user_123",        # attach to every event from this client
    session_id="sess_abc",     # group calls into sessions
    tags=["production","chat"],# filter and segment events
    log_prompts=False,         # opt-in: include prompt content in events
    log_responses=False,       # opt-in: include response content in events
    on_event=my_callback,      # optional callback after each event is written
)
```

### Supported call types
| Call type | Supported |
|---|---|
| Sync | ✅ |
| Async | ✅ |
| Streaming | ✅ |
| Batch | 🔜 |

### Supported providers
| Provider | Status |
|---|---|
| Anthropic | ✅ |
| OpenAI | ✅ |
| Groq | ✅ |
| Google Gemini | 🔜 |
| Mistral | 🔜 |
| Cohere | 🔜 |

---

## 2. Outputs — Where Your Data Goes

### Stdout
Prints a one-line summary to the terminal after every call. Default in development.
```python
from tokensense.outputs import Stdout
client = observe(anthropic.Anthropic(), output=Stdout())
# → ✓ model=claude-sonnet-4-6 | in=1204 out=387 tokens | $0.0061 | 934ms
```
- Best for: development, debugging
- Infra required: none
- Production safe: no

---

### SQLite
Persists every call to a local `.db` file. Created automatically if it doesn't exist.
```python
from tokensense.outputs import SQLite
client = observe(anthropic.Anthropic(), output=SQLite("./usage.db"))
```
- Best for: local persistence, dev analytics, single-process staging
- Infra required: none
- Production safe: single worker only
- Schema: `calls` table with all event fields
- Queryable with any SQLite viewer or raw SQL

---

### Logger
Writes structured JSON to Python's standard `logging` module. In production this routes to CloudWatch, Datadog, GCP Logging, or whatever log infra is already configured. Zero new infra.
```python
from tokensense.outputs import Logger
client = observe(anthropic.Anthropic(), output=Logger("tokensense"))
```
- Best for: production deployments
- Infra required: none — uses your existing log infra
- Production safe: yes — multi-worker safe
- Works with: CloudWatch, Datadog, GCP Logging, Grafana Loki, ELK, any log aggregator

---

### HTTP
POSTs every event as JSON to any endpoint you control.
```python
from tokensense.outputs import HTTP
client = observe(
    anthropic.Anthropic(),
    output=HTTP(
        "https://your-server.com/ingest",
        headers={"X-API-Key": "your-key"}
    )
)
```
- Best for: self-hosted dashboards, custom ingestion pipelines
- Infra required: your own endpoint
- Production safe: yes
- Fires in background thread — never blocks the LLM call
- Timeouts handled silently — a failed POST never crashes your app

---

### Multi
Writes to multiple outputs simultaneously.
```python
from tokensense.outputs import Multi, Stdout, SQLite
client = observe(
    anthropic.Anthropic(),
    output=Multi(Stdout(), SQLite("./usage.db"))
)
```
- Best for: dev (see it + save it), custom pipelines
- Any number of outputs can be combined
- One output failing does not affect others

---

### Auto-detection (default)
When no output is specified, TokenSense auto-detects the environment:
```python
client = observe(anthropic.Anthropic())  # no output specified
# ENV=production → Logger("tokensense")
# everything else → Multi(Stdout(), SQLite("./tokensense.db"))
```
Reads: `ENV`, `ENVIRONMENT`, `APP_ENV` environment variables.

---

## 3. Router — Smart Model Routing

### What it does
Selects the best model for each call based on rules, context budget, and cost constraints. Runs entirely in-process — no proxy, no network call for the routing decision itself.

### The problem it solves
- "I switched to Haiku but it couldn't handle my long history" — context budget check prevents this
- "Smart routing is a black box" — every routing decision is logged with a reason
- "Haiku did badly with my existing information" — auto-escalation on failure recovers automatically

### Basic usage
```python
from tokensense.router import Router, Rule

router = Router(
    tiers={
        "small":  ["claude-haiku-4-5", "groq/llama3-8b"],
        "large":  ["claude-sonnet-4-6", "gpt-4o"],
    },
    rules=[
        Rule(if_context_tokens_gt=4000, deny_tiers=["small"]),
        Rule(if_task="legal-review", pin_tier="large"),
    ],
    on_failure="escalate",
)

decision = router.route(messages=msgs, task_hint="code-review")
print(decision.model)    # claude-haiku-4-5
print(decision.reason)   # default tier small
```

### Tier configuration
```python
tiers={
    "small":    ["groq/llama3-8b", "claude-haiku-4-5"],   # tried in order
    "medium":   ["gpt-4o-mini"],
    "large":    ["claude-sonnet-4-6", "gpt-4o"],
}
```
- Any number of tiers
- Models within a tier are tried in order until one fits the context window
- Tier names are yours to define

---

### Rules

#### Rule: deny_tiers
Excludes one or more tiers when the condition is true.
```python
# never use small models for long conversations
Rule(if_context_tokens_gt=4000, deny_tiers=["small"])
```

#### Rule: pin_tier
Forces routing to a specific tier when the condition is true. Takes priority over everything else.
```python
# legal review always goes to the best model
Rule(if_task="legal-review", pin_tier="large")
```

#### Rule: prefer_tier
Softly prefers a tier when the condition is true. Overridden by deny rules.
```python
# prefer small models on cheap tasks
Rule(prefer_tier="small", if_estimated_cost_gt=0.005)
```

#### Rule: escalate on error code
```python
# retry on next tier up when rate-limited
Rule(if_error_code=429, escalate=True)
```

#### Rule conditions
| Condition | Type | Description |
|---|---|---|
| `if_context_tokens_gt` | int | Fires when estimated context exceeds this value |
| `if_task` | string | Fires when task_hint matches this string |
| `if_error_code` | int | Fires when the previous call returned this HTTP error |
| `if_estimated_cost_gt` | float | Fires when estimated cost exceeds this USD value |

All conditions on a single Rule are ANDed. First matching rule wins.

---

### Context budget check
Before routing to any model, the router checks whether the conversation will fit in the model's context window. If it won't, that model is automatically excluded.
```python
# this never sends 150k tokens to a model with an 8k context window
# it escalates to the next tier automatically
router = Router(
    tiers={
        "small": ["llama3-8b-8192"],   # 8,192 token context
        "large": ["claude-sonnet-4-6"], # 200,000 token context
    }
)
decision = router.route(messages=very_long_history)
# → routed to large automatically, small was context-excluded
```

---

### Per-call overrides
```python
decision = router.route(
    messages=msgs,
    task_hint="code-review",      # matches if_task rules
    max_cost_usd=0.005,           # hard cost cap — expensive tiers excluded
    min_tier="medium",            # floor — never go below this tier
    context_tokens=1842,          # skip estimation, use this count directly
)
```

---

### on_failure behaviour
```python
Router(on_failure="escalate")  # if model call fails, try next tier up automatically
Router(on_failure="error")     # surface the error, don't silently degrade
```

---

### RoutingDecision object
```python
decision.model              # "claude-haiku-4-5"
decision.tier               # "small"
decision.reason             # "selected small within cost cap $0.005"
decision.estimated_cost_usd # 0.000034
decision.denied_tiers       # ["large"]
```

---

## 4. Cost Engine

### What it does
Estimates cost before and after every call. Used internally by the router and included in every event.

### estimate_cost()
```python
from tokensense.cost import estimate_cost
cost = estimate_cost("claude-sonnet-4-6", input_tokens=1000, output_tokens=500)
# → 0.0105 (USD)
```

### get_context_window()
```python
from tokensense.cost import get_context_window
window = get_context_window("claude-sonnet-4-6")
# → 200000
```

### Fuzzy model matching
Handles versioned model names automatically:
```python
estimate_cost("claude-sonnet-4-6-20250514", 1000, 500)
# → matches "claude-sonnet-4-6" pricing
```

### Supported models — cost table
| Model | Input (per 1M) | Output (per 1M) |
|---|---|---|
| claude-opus-4-6/4-7/4-8 | $15.00 | $75.00 |
| claude-sonnet-4-6 | $3.00 | $15.00 |
| claude-haiku-4-5 | $0.80 | $4.00 |
| gpt-4o | $5.00 | $15.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| o1 | $15.00 | $60.00 |
| o3-mini | $1.10 | $4.40 |
| llama3-8b-8192 (Groq) | $0.05 | $0.10 |
| llama3-70b-8192 (Groq) | $0.59 | $0.79 |
| gemini-1.5-pro | $3.50 | $10.50 |
| gemini-1.5-flash | $0.075 | $0.30 |
| gemini-2.0-flash | $0.10 | $0.40 |

---

## 5. Privacy Controls

### Default behaviour
- Prompts: never captured
- Responses: never captured
- API keys: never read or stored by the framework
- Network: no outbound calls from the framework itself

### Explicit opt-in for content logging
```python
client = observe(
    anthropic.Anthropic(),
    log_prompts=True,      # include prompt content in events
    log_responses=True,    # include response content in events
)
```
Even when enabled, content goes only to your configured output — never to us.

---

## 6. Event Callback Hook

Run any function after an event is written. Use for alerting, custom metrics, or integrations.
```python
def on_expensive_call(event):
    if event.cost_usd > 0.05:
        send_slack_alert(f"Expensive call: ${event.cost_usd:.4f}")

client = observe(anthropic.Anthropic(), on_event=on_expensive_call)
```

---

## 7. Environment Auto-Detection

TokenSense reads `ENV`, `ENVIRONMENT`, or `APP_ENV` to set defaults automatically.

| Environment | Default output |
|---|---|
| `production` | `Logger("tokensense")` |
| anything else | `Multi(Stdout(), SQLite("./tokensense.db"))` |

Override at any time by passing `output=` explicitly.

---

## 8. Roadmap

### v0.1.0 — Current
- [x] `observe()` wrapping Anthropic, OpenAI, Groq
- [x] `Stdout`, `SQLite`, `Logger`, `HTTP`, `Multi` outputs
- [x] Auto environment detection
- [x] Cost table — 20+ models
- [x] Context window table — 20+ models
- [x] Router with context budget check
- [x] Rule engine — deny, pin, prefer, escalate
- [x] Per-call overrides
- [x] `on_event` callback hook
- [x] Full test suite

### v0.2.0
- [ ] Shadow testing — run two tiers in parallel before switching
- [ ] `tokensense report ./usage.db` — spend summary in terminal
- [ ] Budget alerts — warn when projected spend crosses threshold
- [ ] Streaming token count accuracy improvements
- [ ] Google Gemini support

### v0.3.0
- [ ] `FileSink` — JSONL output for log pipelines
- [ ] `OTelSink` — OpenTelemetry compatible export
- [ ] Confidence scoring on router output
- [ ] LangChain integration
- [ ] LlamaIndex integration

### Later (demand-driven)
- [ ] Postgres output for high-throughput production
- [ ] Prometheus metrics endpoint
- [ ] Custom rule functions (lambda-based)
- [ ] Visual Vortex dashboard sink (opt-in cloud)
- [ ] Multi-language: TypeScript SDK

---

## 9. What TokenSense Does Not Do

Being clear about scope is part of the product.

| Feature | Our answer |
|---|---|
| Prompt management | Use Langfuse |
| LLM-as-judge evals | Use Phoenix / OpenLit |
| GPU monitoring | Use OpenLit |
| Proxy / gateway | By design — we don't do this |
| Storing your prompts | Never |
| Billing you | It's free and open source |

---

## 10. Installation

```bash
pip install tokensense-ai
```

No required dependencies. Python 3.10+.

Optional dev dependencies:
```bash
pip install tokensense-ai[dev]
# includes: pytest, anthropic, openai, groq
```

---

Built by [Visual Vortex](https://visualvortexcreatives.dev) · [GitHub](https://github.com/visualvortex/tokensense) · MIT License