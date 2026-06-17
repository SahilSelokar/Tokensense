# Changelog

All notable changes to the TokenSense AI Framework will be documented in this file.

## [0.2.1] - 2026-06-17

### Added
- **Local Semantic Caching**: Added `SQLiteVectorCache` powered by `sqlite-vec` to automatically intercept duplicate prompts locally. Drastically reduces latency (~1ms) and brings duplicate LLM request costs down to $0.00.
- **OpenTelemetry (OTEL) Export**: Added `OTEL` to `tokensense.outputs` to natively export traces and telemetry data directly to Datadog, Grafana, Jaeger, or any OTEL collector.
- **Framework Integrations**: Added native callback handlers for seamless integration with complex workflows:
  - `TokenSenseCallbackHandler` for LangChain.
  - `TokenSenseLlamaIndexCallback` for LlamaIndex.
- **CLI Tools**: Added built-in CLI commands to manage pricing and view reports locally:
  - `tokensense report`: View aggregated token usage and USD spend per model in the terminal.
  - `tokensense update-prices`: Download the latest live pricing database from the open-source LiteLLM project.
- **New Provider Support**: 
  - Added support for Google Gemini (`google-genai`).
  - Added support for observing the universal `litellm.completion()` handler.

## [0.1.1] - 2026-06-14

### Added
- Initial Release
- Observability wrappers for OpenAI, Anthropic, and Groq.
- Outputs for Stdout, SQLite, Logger, HTTP, and Multi.
- Smart routing capabilities with cost and token constraints.
- Privacy-first local logging.
