import asyncio
import os
import time
import threading
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable

from tokensense.cost import estimate_cost
from tokensense.outputs import BaseOutput, Stdout, SQLite, Logger, Multi
from tokensense.events import CallEvent

def get_default_output() -> BaseOutput:
    env = os.environ.get("ENV") or os.environ.get("ENVIRONMENT") or os.environ.get("APP_ENV")
    if env == "production":
        return Logger("tokensense")
    else:
        return Multi(Stdout(), SQLite("./tokensense.db"))

def _emit_event_background(output: BaseOutput, event: CallEvent, on_event: Optional[Callable[[CallEvent], None]]):
    def worker():
        try:
            output.write(event)
            if on_event:
                on_event(event)
        except Exception:
            pass
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

@dataclass
class CallContext:
    start_time: float
    ts: str
    prompt_content: Optional[str]
    response_content: Optional[str]
    response_object: Any
    error_msg: Optional[str]
    provider: str
    model: str
    is_cached_hit: bool
    partial: bool = False

class ObservedMethodWrapper:
    def __init__(
        self, 
        original_method: Any, 
        output: BaseOutput, 
        log_prompts: bool, 
        log_responses: bool, 
        on_event: Optional[Callable[[CallEvent], None]],
        meta_kwargs: Dict[str, Any],
        cache: Optional[Any] = None
    ):
        self.original_method = original_method
        self.output = output
        self.log_prompts = log_prompts
        self.log_responses = log_responses
        self.on_event = on_event
        self.meta_kwargs = meta_kwargs
        self.cache = cache

    def _extract_chunk_text(self, chunk: Any) -> str:
        try:
            if hasattr(chunk, "choices") and chunk.choices:
                if hasattr(chunk.choices[0], "delta") and hasattr(chunk.choices[0].delta, "content"):
                    return chunk.choices[0].delta.content or ""
            elif hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
                return chunk.delta.text or ""
            elif hasattr(chunk, "text"):
                return chunk.text or ""
        except Exception:
            pass
        return ""

    def _extract_chunk_usage(self, chunk: Any, ctx: CallContext, input_tokens: int, output_tokens: int) -> tuple[int, int]:
        try:
            if hasattr(chunk, "usage") and chunk.usage:
                usage = chunk.usage
                if hasattr(usage, "prompt_tokens"):
                    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    ctx.provider = "openai"
                elif hasattr(usage, "input_tokens"):
                    input_tokens = getattr(usage, "input_tokens", 0) or 0
                    ctx.provider = "anthropic"
                
                if hasattr(usage, "completion_tokens"):
                    output_tokens = getattr(usage, "completion_tokens", 0) or 0
                elif hasattr(usage, "output_tokens"):
                    output_tokens = getattr(usage, "output_tokens", 0) or 0
            
            # Anthropic puts usage inside message_start events
            if hasattr(chunk, "message") and hasattr(chunk.message, "usage"):
                usage = chunk.message.usage
                if hasattr(usage, "input_tokens"):
                    input_tokens = getattr(usage, "input_tokens", 0) or 0
                    ctx.provider = "anthropic"
        except Exception:
            pass
        return input_tokens, output_tokens

    def _finalize_and_emit(self, ctx: CallContext):
        latency_ms = int((time.time() - ctx.start_time) * 1000.0)
        
        input_tokens = 0
        output_tokens = 0
        
        response = ctx.response_object
        
        # Try to pull model name from bound object (Gemini does this)
        if ctx.model == "unknown" and hasattr(self.original_method, "__self__"):
            bound_obj = self.original_method.__self__
            if hasattr(bound_obj, "model_name"):
                ctx.model = bound_obj.model_name.replace("models/", "")
                
        if response is not None and not isinstance(response, dict):
            if hasattr(response, "model") and getattr(response, "model"):
                ctx.model = response.model
            
            # Pull token counts from the response
            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                if hasattr(usage, "prompt_tokens"):
                    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    ctx.provider = "openai"
                elif hasattr(usage, "input_tokens"):
                    input_tokens = getattr(usage, "input_tokens", 0) or 0
                    ctx.provider = "anthropic"
                
                if hasattr(usage, "completion_tokens"):
                    output_tokens = getattr(usage, "completion_tokens", 0) or 0
                elif hasattr(usage, "output_tokens"):
                    output_tokens = getattr(usage, "output_tokens", 0) or 0
            elif hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = response.usage_metadata
                input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                output_tokens = getattr(usage, "candidates_token_count", 0) or 0
                ctx.provider = "gemini"

            # Grab response text for logging if enabled
            if not ctx.response_content:
                if hasattr(response, "choices") and response.choices:
                    try:
                        choice = response.choices[0]
                        if hasattr(choice, "message") and hasattr(choice.message, "content"):
                            ctx.response_content = choice.message.content
                    except Exception:
                        pass
                elif hasattr(response, "content") and response.content:
                    if isinstance(response.content, list):
                        ctx.response_content = "".join([getattr(c, "text", "") for c in response.content])
                    else:
                        ctx.response_content = str(response.content)

        # Streaming path — usage comes from accumulated chunks
        if isinstance(response, dict) and "streaming_usage" in response:
            usage_data = response["streaming_usage"]
            input_tokens = usage_data.get("input_tokens", 0)
            output_tokens = usage_data.get("output_tokens", 0)
            
            # No exact usage from chunks? Estimate from the text we have
            from tokensense.router import estimate_tokens
            if input_tokens == 0 and ctx.prompt_content:
                input_tokens = estimate_tokens([{"content": ctx.prompt_content}])
            if output_tokens == 0 and ctx.response_content:
                output_tokens = estimate_tokens([{"content": ctx.response_content}])

        cost = estimate_cost(ctx.model, input_tokens, output_tokens)

        event = CallEvent(
            ts=ctx.ts,
            model=ctx.model,
            provider="cache" if ctx.is_cached_hit else ctx.provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=0.0 if ctx.is_cached_hit else cost,
            latency_ms=latency_ms,
            user_id=self.meta_kwargs.get("user_id"),
            session_id=self.meta_kwargs.get("session_id"),
            tags=self.meta_kwargs.get("tags"),
            routed_tier=self.meta_kwargs.get("routed_tier"),
            error=ctx.error_msg,
            prompt=ctx.prompt_content if self.log_prompts else None,
            response=ctx.response_content if self.log_responses else None
        )

        if self.cache and not ctx.is_cached_hit and not ctx.error_msg and ctx.prompt_content and ctx.response_content and not ctx.partial:
            self.cache.set(ctx.prompt_content, ctx.response_content)

        _emit_event_background(self.output, event, self.on_event)

    def _stream_generator(self, original_generator, ctx: CallContext):
        accumulated = ""
        input_tokens = 0
        output_tokens = 0
        try:
            for chunk in original_generator:
                accumulated += self._extract_chunk_text(chunk)
                input_tokens, output_tokens = self._extract_chunk_usage(chunk, ctx, input_tokens, output_tokens)
                yield chunk
        except GeneratorExit:
            ctx.partial = True
            # Let finally handle cleanup — Python closes the generator for us
        except Exception as e:
            ctx.error_msg = str(e)
            ctx.partial = True
            raise
        else:
            ctx.partial = False
        finally:
            ctx.response_content = accumulated
            ctx.response_object = {"streaming_usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}}
            self._finalize_and_emit(ctx)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        ts = datetime.now(timezone.utc).isoformat()
        
        prompt_content = None
        messages = kwargs.get("messages") or (args[0] if args and isinstance(args[0], list) else None)
        if messages:
            prompt_content = str(messages)
        else:
            prompt_content = kwargs.get("prompt") or (args[0] if args and isinstance(args[0], str) else None)
            
        model = kwargs.get("model", "unknown")
        
        is_stream = kwargs.get("stream") is True

        # Streaming needs prompt for input token estimation on early break.
        # Actual logging is still gated by log_prompts in _finalize_and_emit.
        if not is_stream and not self.log_prompts and not self.cache:
            prompt_content = None
        
        # Don't check cache for streams
        is_cached_hit = False
        if not is_stream and self.cache and prompt_content:
            cached_text = self.cache.get(prompt_content)
            if cached_text:
                is_cached_hit = True
                class MockMsg: content = cached_text
                class MockChoice: message = MockMsg()
                class MockResp: 
                    content = cached_text
                    text = cached_text
                    choices = [MockChoice()]
                    model = "semantic-cache"
                    usage = None
                
                ctx = CallContext(
                    start_time=start_time, ts=ts, prompt_content=prompt_content,
                    response_content=None, response_object=MockResp(), error_msg=None,
                    provider="unknown", model=model, is_cached_hit=True
                )
                self._finalize_and_emit(ctx)
                return MockResp()

        # Sneak in stream_options so we get real usage from the final chunk
        mod = getattr(self.original_method, "__module__", "")
        if is_stream and "stream_options" not in kwargs and any(x in mod for x in ["openai", "groq", "litellm"]):
            kwargs["stream_options"] = {"include_usage": True}

        ctx = CallContext(
            start_time=start_time, ts=ts, prompt_content=prompt_content,
            response_content=None, response_object=None, error_msg=None,
            provider="unknown", model=model, is_cached_hit=False
        )

        try:
            response = self.original_method(*args, **kwargs)
            if is_stream:
                return self._stream_generator(response, ctx)
            ctx.response_object = response
        except Exception as e:
            ctx.error_msg = str(e)
            self._finalize_and_emit(ctx)
            raise

        self._finalize_and_emit(ctx)
        return response

class ObservedAsyncMethodWrapper(ObservedMethodWrapper):
    async def _async_stream_generator(self, original_generator, ctx: CallContext):
        accumulated = ""
        input_tokens = 0
        output_tokens = 0
        try:
            async for chunk in original_generator:
                accumulated += self._extract_chunk_text(chunk)
                input_tokens, output_tokens = self._extract_chunk_usage(chunk, ctx, input_tokens, output_tokens)
                yield chunk
        except GeneratorExit:
            ctx.partial = True
        except asyncio.CancelledError:
            # FastAPI/Starlette cancels the task on client disconnect
            ctx.partial = True
            raise
        except Exception as e:
            ctx.error_msg = str(e)
            ctx.partial = True
            raise
        else:
            ctx.partial = False
        finally:
            ctx.response_content = accumulated
            ctx.response_object = {"streaming_usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}}
            self._finalize_and_emit(ctx)

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        ts = datetime.now(timezone.utc).isoformat()
        
        prompt_content = None
        messages = kwargs.get("messages") or (args[0] if args and isinstance(args[0], list) else None)
        if messages:
            prompt_content = str(messages)
        else:
            prompt_content = kwargs.get("prompt") or (args[0] if args and isinstance(args[0], str) else None)
            
        model = kwargs.get("model", "unknown")
        
        is_stream = kwargs.get("stream") is True

        # Same as sync: keep prompt around for streaming token estimation
        if not is_stream and not self.log_prompts and not self.cache:
            prompt_content = None
        
        is_cached_hit = False
        if not is_stream and self.cache and prompt_content:
            cached_text = self.cache.get(prompt_content)
            if cached_text:
                is_cached_hit = True
                class MockMsg: content = cached_text
                class MockChoice: message = MockMsg()
                class MockResp: 
                    content = cached_text
                    text = cached_text
                    choices = [MockChoice()]
                    model = "semantic-cache"
                    usage = None
                
                ctx = CallContext(
                    start_time=start_time, ts=ts, prompt_content=prompt_content,
                    response_content=None, response_object=MockResp(), error_msg=None,
                    provider="unknown", model=model, is_cached_hit=True
                )
                self._finalize_and_emit(ctx)
                return MockResp()

        mod = getattr(self.original_method, "__module__", "")
        if is_stream and "stream_options" not in kwargs and any(x in mod for x in ["openai", "groq", "litellm"]):
            kwargs["stream_options"] = {"include_usage": True}

        ctx = CallContext(
            start_time=start_time, ts=ts, prompt_content=prompt_content,
            response_content=None, response_object=None, error_msg=None,
            provider="unknown", model=model, is_cached_hit=False
        )

        try:
            response = await self.original_method(*args, **kwargs)
            if is_stream:
                return self._async_stream_generator(response, ctx)
            ctx.response_object = response
        except Exception as e:
            ctx.error_msg = str(e)
            self._finalize_and_emit(ctx)
            raise

        self._finalize_and_emit(ctx)
        return response

class ObjectProxy:
    def __init__(
        self, 
        obj: Any, 
        output: BaseOutput, 
        log_prompts: bool, 
        log_responses: bool, 
        on_event: Optional[Callable[[CallEvent], None]],
        meta_kwargs: Dict[str, Any]
    ):
        self._obj = obj
        self._output = output
        self._log_prompts = log_prompts
        self._log_responses = log_responses
        self._on_event = on_event
        self._meta_kwargs = meta_kwargs
        self._cache = meta_kwargs.pop("cache", None)

    _PROXY_NAMESPACES = frozenset({
        "chat", "completions", "messages", "models", "embeddings",
        "images", "audio", "files", "fine_tuning", "beta",
        "generative_model", "generate_content",
    })

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._obj, name)
        
        if callable(attr):
            if name in ("create", "completion", "acompletion", "generate_content"):
                import inspect
                if inspect.iscoroutinefunction(attr) or name == "acompletion":
                    return ObservedAsyncMethodWrapper(
                        attr, self._output, self._log_prompts, self._log_responses, self._on_event, self._meta_kwargs, self._cache
                    )
                else:
                    return ObservedMethodWrapper(
                        attr, self._output, self._log_prompts, self._log_responses, self._on_event, self._meta_kwargs, self._cache
                    )
            return attr
        
        if name in self._PROXY_NAMESPACES:
            return ObjectProxy(
                attr, self._output, self._log_prompts, self._log_responses, self._on_event, {**self._meta_kwargs, "cache": self._cache}
            )
        return attr

def observe(
    client: Any,
    output: Optional[BaseOutput] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    log_prompts: bool = False,
    log_responses: bool = False,
    on_event: Optional[Callable[[CallEvent], None]] = None,
    **kwargs: Any
) -> Any:
    """Wrap any LLM client to track tokens, cost, and latency."""
    if output is None:
        output = get_default_output()
        
    meta_kwargs = {
        "user_id": user_id,
        "session_id": session_id,
        "tags": tags,
    }
    meta_kwargs.update(kwargs)
    return ObjectProxy(client, output, log_prompts, log_responses, on_event, meta_kwargs)
