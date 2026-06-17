import os
import time
import threading
from datetime import datetime, timezone
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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        ts = datetime.now(timezone.utc).isoformat()
        error_msg = None
        response = None
        
        prompt_content = None
        # Always extract prompt if caching is enabled or logging is enabled
        messages = kwargs.get("messages") or (args[0] if args and isinstance(args[0], list) else None)
        if messages:
            prompt_content = str(messages)
        else:
            prompt_content = kwargs.get("prompt") or (args[0] if args and isinstance(args[0], str) else None)
            
        if not self.log_prompts and not self.cache:
            prompt_content = None

        is_cached_hit = False

        try:
            if self.cache and prompt_content:
                cached_text = self.cache.get(prompt_content)
                if cached_text:
                    is_cached_hit = True
                    # Duck-typed mock response
                    class MockMsg: content = cached_text
                    class MockChoice: message = MockMsg()
                    class MockResp: 
                        content = cached_text
                        text = cached_text
                        choices = [MockChoice()]
                        model = "semantic-cache"
                        usage = None
                    response = MockResp()
            
            if not is_cached_hit:
                response = self.original_method(*args, **kwargs)
            
            # If the response is streaming, we cannot immediately extract tokens
            # For v0.1 we might need to rely on usage chunks if available, but
            # if stream=True is detected, we'll try to process it.
            if kwargs.get("stream") is True:
                # We need to wrap the generator/stream to capture chunks.
                # To simplify for v0.1, we'll return a wrapper generator if needed,
                # but if we don't, we will log what we have.
                pass
            
            return response
        except Exception as e:
            error_msg = str(e)
            raise e
        finally:
            if kwargs.get("stream") is True:
                # Basic stream detection, we will not block the generator here,
                # just log the initial invocation. Accurate streaming token counting
                # is complex and scheduled for later v0.2.0 updates.
                pass
            
            latency_ms = int((time.time() - start_time) * 1000.0)
            
            model = kwargs.get("model", "unknown")
            if model == "unknown" and hasattr(self.original_method, "__self__"):
                bound_obj = self.original_method.__self__
                if hasattr(bound_obj, "model_name"):
                    model = bound_obj.model_name.replace("models/", "")
                    
            input_tokens = 0
            output_tokens = 0
            response_content = None
            provider = "unknown"

            if response is not None:
                if hasattr(response, "model") and getattr(response, "model"):
                    model = response.model
                
                # Usage extraction
                if hasattr(response, "usage") and response.usage:
                    usage = response.usage
                    if hasattr(usage, "prompt_tokens"):
                        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                        provider = "openai"  # or groq / litellm
                    elif hasattr(usage, "input_tokens"):
                        input_tokens = getattr(usage, "input_tokens", 0) or 0
                        provider = "anthropic"
                    
                    if hasattr(usage, "completion_tokens"):
                        output_tokens = getattr(usage, "completion_tokens", 0) or 0
                    elif hasattr(usage, "output_tokens"):
                        output_tokens = getattr(usage, "output_tokens", 0) or 0
                elif hasattr(response, "usage_metadata") and response.usage_metadata:
                    usage = response.usage_metadata
                    input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                    output_tokens = getattr(usage, "candidates_token_count", 0) or 0
                    provider = "gemini"

                if self.log_responses:
                    if hasattr(response, "choices") and response.choices:
                        try:
                            choice = response.choices[0]
                            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                                response_content = choice.message.content
                        except Exception:
                            pass
                    elif hasattr(response, "content") and response.content:
                        if isinstance(response.content, list):
                            response_content = "".join([getattr(c, "text", "") for c in response.content])
                        else:
                            response_content = str(response.content)

            cost = estimate_cost(model, input_tokens, output_tokens)

            event = CallEvent(
                ts=ts,
                model=model,
                provider="cache" if is_cached_hit else provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=0.0 if is_cached_hit else cost,
                latency_ms=latency_ms,
                user_id=self.meta_kwargs.get("user_id"),
                session_id=self.meta_kwargs.get("session_id"),
                tags=self.meta_kwargs.get("tags"),
                routed_tier=self.meta_kwargs.get("routed_tier"),
                error=error_msg,
                prompt=prompt_content if self.log_prompts else None,
                response=response_content if self.log_responses else None
            )

            if self.cache and not is_cached_hit and not error_msg and prompt_content and response_content:
                self.cache.set(prompt_content, response_content)

            # Avoid blocking async streaming responses
            _emit_event_background(self.output, event, self.on_event)

class ObservedAsyncMethodWrapper(ObservedMethodWrapper):
    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        ts = datetime.now(timezone.utc).isoformat()
        error_msg = None
        response = None
        
        prompt_content = None
        messages = kwargs.get("messages") or (args[0] if args and isinstance(args[0], list) else None)
        if messages:
            prompt_content = str(messages)
        else:
            prompt_content = kwargs.get("prompt") or (args[0] if args and isinstance(args[0], str) else None)
            
        if not self.log_prompts and not self.cache:
            prompt_content = None

        is_cached_hit = False

        try:
            if self.cache and prompt_content:
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
                    response = MockResp()
            
            if not is_cached_hit:
                response = await self.original_method(*args, **kwargs)
            return response
        except Exception as e:
            error_msg = str(e)
            raise e
        finally:
            latency_ms = int((time.time() - start_time) * 1000.0)
            
            model = kwargs.get("model", "unknown")
            if model == "unknown" and hasattr(self.original_method, "__self__"):
                bound_obj = self.original_method.__self__
                if hasattr(bound_obj, "model_name"):
                    model = bound_obj.model_name.replace("models/", "")
                    
            input_tokens = 0
            output_tokens = 0
            response_content = None
            provider = "unknown"

            if response is not None:
                if hasattr(response, "model") and getattr(response, "model"):
                    model = response.model
                
                if hasattr(response, "usage") and response.usage:
                    usage = response.usage
                    if hasattr(usage, "prompt_tokens"):
                        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                        provider = "openai"
                    elif hasattr(usage, "input_tokens"):
                        input_tokens = getattr(usage, "input_tokens", 0) or 0
                        provider = "anthropic"
                    
                    if hasattr(usage, "completion_tokens"):
                        output_tokens = getattr(usage, "completion_tokens", 0) or 0
                    elif hasattr(usage, "output_tokens"):
                        output_tokens = getattr(usage, "output_tokens", 0) or 0
                elif hasattr(response, "usage_metadata") and response.usage_metadata:
                    usage = response.usage_metadata
                    input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                    output_tokens = getattr(usage, "candidates_token_count", 0) or 0
                    provider = "gemini"

                if self.log_responses:
                    if hasattr(response, "choices") and response.choices:
                        try:
                            choice = response.choices[0]
                            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                                response_content = choice.message.content
                        except Exception:
                            pass
                    elif hasattr(response, "content") and response.content:
                        if isinstance(response.content, list):
                            response_content = "".join([getattr(c, "text", "") for c in response.content])
                        else:
                            response_content = str(response.content)

            cost = estimate_cost(model, input_tokens, output_tokens)

            event = CallEvent(
                ts=ts,
                model=model,
                provider="cache" if is_cached_hit else provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=0.0 if is_cached_hit else cost,
                latency_ms=latency_ms,
                user_id=self.meta_kwargs.get("user_id"),
                session_id=self.meta_kwargs.get("session_id"),
                tags=self.meta_kwargs.get("tags"),
                routed_tier=self.meta_kwargs.get("routed_tier"),
                error=error_msg,
                prompt=prompt_content if self.log_prompts else None,
                response=response_content if self.log_responses else None
            )

            if self.cache and not is_cached_hit and not error_msg and prompt_content and response_content:
                self.cache.set(prompt_content, response_content)

            _emit_event_background(self.output, event, self.on_event)


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

    # Known SDK namespace attributes that need recursive proxying
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
        
        # Only proxy known SDK namespace objects, return everything else as-is
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
    """
    Wrap an LLM client to observe, record, and cost-track call invocations.
    """
    if output is None:
        output = get_default_output()
        
    meta_kwargs = {
        "user_id": user_id,
        "session_id": session_id,
        "tags": tags,
    }
    meta_kwargs.update(kwargs)
    return ObjectProxy(client, output, log_prompts, log_responses, on_event, meta_kwargs)
