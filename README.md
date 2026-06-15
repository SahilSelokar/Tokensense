<h1 align="center">
  <img src="https://visualvortexcreatives.dev/_next/image?url=%2Fvisual-vortex-logo.png&w=96&q=75" alt="Visual Vortex Creatives Logo" width="60" align="absmiddle" />
  <span style="font-family: 'Montserrat', sans-serif; font-weight: 900; font-size: 1.2em;">Visual Vortex</span>
</h1>

# TokenSense
### by [Visual Vortex](https://visualvortexcreatives.dev)

**LLM cost observability and smart routing — stays in your code, works everywhere.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/tokensense-ai.svg)](https://pypi.org/project/tokensense-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/tokensense-ai.svg)](https://pypi.org/project/tokensense-ai/)

---

```bash
pip install tokensense-ai
```

```python
from tokensense import observe
import anthropic

client = observe(anthropic.Anthropic())
response = client.messages.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Hello"}]
)
# → model=claude-sonnet-4-6 | in=12 out=24 tokens | $0.0003 | 847ms
```

That's it. One import. One wrapper. Every LLM call you make is now tracked.

---

## Why TokenSense?

Other observability tools make you choose between easy and private. A proxy is easy but your API keys and prompts go through someone else's server. A self-hosted platform is private but takes 30 minutes to set up before you see anything.

TokenSense does neither. It runs inside your process. Nothing leaves unless you tell it to.

- **Privacy by default** — prompts never leave your process. Ever. Not to us, not to anyone.
- **Zero latency** — the observer fires after the response returns. Your call time is unchanged.
- **Works everywhere** — dev, prod, Docker, Lambda, K8s. No infra required.
- **Smart routing** — context-budget-aware model switching with automatic escalation on failure.
- **Framework agnostic** — wraps OpenAI, Anthropic, Groq directly. Not tied to LangChain or anything else.
- **You own the data** — output to your terminal, SQLite, your existing logger, or any HTTP endpoint.

---

## Quickstart

### Observe any LLM client

```python
from tokensense import observe
import openai

# OpenAI
client = observe(openai.OpenAI())

# Anthropic
client = observe(anthropic.Anthropic())

# Groq
client = observe(groq.Groq())

# your existing code stays exactly the same
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain async/await"}]
)
# → model=gpt-4o-mini | in=18 out=312 tokens | $0.0001 | 623ms
```

### Change where data goes

```python
from tokensense import observe
from tokensense.outputs import SQLite, Logger, HTTP

# save locally — great for dev
client = observe(anthropic.Anthropic(), output=SQLite("./usage.db"))

# write to Python logger — goes to CloudWatch, Datadog, whatever you already use
client = observe(anthropic.Anthropic(), output=Logger("tokensense"))

# post to your own endpoint
client = observe(anthropic.Anthropic(), output=HTTP("https://your-server.com/ingest"))
```

Default output when you don't specify anything:
- **In development** → prints to terminal + writes to `./tokensense.db`
- **In production** (`ENV=production`) → writes to Python logger, goes to your existing log infra

### Add metadata per call

```python
client = observe(
    anthropic.Anthropic(),
    user_id="user_123",
    session_id="sess_abc",
    tags=["production", "chat"],
)
```

### Smart routing

```python
from tokensense import Router, Rule

router = Router(
    tiers={
        "small":  ["llama-3.1-8b-instant", "claude-haiku-4-5"],
        "large":  ["claude-sonnet-4-6", "gpt-4o"],
    },
    rules=[
        # never route to small model if history is long
        Rule(if_context_tokens_gt=4000, deny_tiers=["small"]),

        # pin sensitive tasks to large model always
        Rule(if_task="legal-review", pin_tier="large"),
    ]
)

decision = router.route(
    messages=[{"role": "user", "content": "Review this contract"}],
    task_hint="legal-review"
)

print(decision.model) # → 'claude-sonnet-4-6'
print(decision.reason) # → 'pinned to large by rule'
```

---

## What gets logged

By default, TokenSense logs only metadata — never your prompt content or response text.

| Field | Logged by default | Opt-in |
|---|---|---|
| Model name | ✅ | |
| Input tokens | ✅ | |
| Output tokens | ✅ | |
| Cost (USD) | ✅ | |
| Latency (ms) | ✅ | |
| Routing decision | ✅ | |
| Status (success/error) | ✅ | |
| Prompt content | ❌ | `log_prompts=True` |
| Response content | ❌ | `log_responses=True` |
| User ID / tags | ❌ | pass as kwargs |

To enable prompt logging explicitly:
```python
client = observe(anthropic.Anthropic(), log_prompts=True, log_responses=True)
```

---

## Output options

| Output | Best for | Setup |
|---|---|---|
| `Stdout()` | Development, debugging | None |
| `SQLite(path)` | Local persistence, single process | None |
| `Logger(name)` | Production — any log infra | None |
| `HTTP(url)` | Custom server, self-hosted dashboard | Your endpoint |

```python
from tokensense.outputs import Stdout, SQLite, Logger, HTTP

# chain multiple outputs
from tokensense.outputs import Multi

client = observe(
    anthropic.Anthropic(),
    output=Multi(Stdout(), SQLite("./usage.db"))
)
```

---

## Supported providers

| Provider | Sync | Async | Streaming |
|---|---|---|---|
| Anthropic | ✅ | ✅ | ✅ |
| OpenAI | ✅ | ✅ | ✅ |
| Groq | ✅ | ✅ | ✅ |
| Google Gemini | 🔜 | 🔜 | 🔜 |

---

## No telemetry. Ever.

TokenSense does not call home. There is no usage tracking, no anonymous analytics, no background pings to any server. The framework has no network dependency of its own.

Read [PRIVACY.md](PRIVACY.md) for exactly what is and isn't captured.

---

## Roadmap

- [x] `observe()` wrapper — OpenAI, Anthropic, Groq
- [x] `Stdout`, `SQLite`, `Logger`, `HTTP` outputs
- [x] Auto environment detection (dev vs prod)
- [x] `Router` with context budget check
- [ ] Shadow testing before switching tiers
- [ ] `tokensense report` CLI — spend summary in terminal
- [ ] Google Gemini support
- [ ] LangChain integration
- [ ] LlamaIndex integration

---

## Contributing

TokenSense is MIT licensed and built in public. Issues, PRs, and feedback are welcome.

```bash
git clone https://github.com/SahilSelokar/Tokensense
cd Tokensense
pip install -e ".[dev]"
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT — see [LICENSE](LICENSE)

---

<p align="center">
  Built by <a href="https://visualvortexcreatives.dev">Visual Vortex</a>
</p>