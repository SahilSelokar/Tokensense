# Changelog

All notable changes to the TokenSense AI Framework will be documented in this file.

## [0.2.3] - 2026-06-23

### Added
- **ShadowTest Harness**: Added a local evaluation harness that runs the same prompt against multiple models (current vs candidate) in parallel, scoring outputs for format, exact match, similarity, and LLM-judged quality.
- **Automatic Provider Detection**: The `observe()` wrapper automatically detects the underlying client type (OpenAI, Anthropic, Groq, LiteLLM) and applies appropriate patching without requiring manual provider specification.
- **OpenRouter Support**: Added native integration and pricing support for OpenRouter, enabling cost-aware routing to models on the OpenRouter platform.
- **Automatic Stream Options**: The `observe()` wrapper automatically injects `stream_options={"include_usage": True}` for OpenAI/Groq to ensure exact usage tokens are captured from streaming responses.

### Changed
- **Cost Estimation Accuracy**: Improved cost estimation to use the latest models from the LiteLLM pricing database, with hardcoded fallbacks for newly released models (e.g., Gemini 2.5 Flash).
- **Model Naming Normalization**: Implemented fuzzy matching and prefix stripping (e.g., `groq/llama-3-8b-8192` -> `llama-3-8b-8192`) to improve model name resolution against the pricing database.
- **Cost & Token Functions**: Refactored `calculate_cost()` and `estimate_tokens()` to use the unified `_load_litellm_prices()` helper, ensuring consistent pricing data across the framework.
- **Internal Module Aliases**: Added backward-compatible aliases for `calculate_cost` and `estimate_tokens` to prevent breaking internal imports while transitioning to the unified pricing system.

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
