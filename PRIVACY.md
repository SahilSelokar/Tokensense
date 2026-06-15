# Privacy Policy

TokenSense is designed with privacy as a first-class citizen. 

## No Telemetry
TokenSense does **not** collect, store, or transmit any telemetry, usage data, or logs to any external server managed by Visual Vortex or third parties. All operations are run locally within your own infrastructure and process.

## Data Collection
By default, TokenSense only collects metadata about your LLM calls:
- Model name
- Input tokens
- Output tokens
- Calculated cost (USD)
- Latency (ms)
- Routing decisions
- Status (success or error)

### Prompts and Responses
By default, the actual **prompts and responses are NOT logged** or processed in any way. You must explicitly opt-in to prompt and response logging:
```python
client = observe(client, log_prompts=True, log_responses=True)
```

## Data Storage
You have complete control over where the observed data is sent. It is only sent to the outputs you configure:
- **Stdout / Console**: Printed directly to your terminal standard output.
- **SQLite**: Stored in a local database file (e.g., `./tokensense.db`) on your disk.
- **Logger**: Directed to standard Python `logging`, which follows your existing log collection and routing rules.
- **HTTP**: Sent to the custom endpoint URL you configure.
