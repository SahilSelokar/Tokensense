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
        meta_kwargs: Dict[str, Any]
    ):
        self.original_method = original_method
        self.output = output
        self.log_prompts = log_prompts
        self.log_responses = log_responses
        self.on_event = on_event
        self.meta_kwargs = meta_kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        ts = datetime.now(timezone.utc).isoformat()
        error_msg = None
        response = None
        
        prompt_content = None
        if self.log_prompts:
            messages = kwargs.get("messages") or (args[0] if args and isinstance(args[0], list) else None)
            if messages:
                prompt_content = str(messages)
            else:
                prompt_content = kwargs.get("prompt") or (args[0] if args and isinstance(args[0], str) else None)

        try:
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
            input_tokens = 0
            output_tokens = 0
            response_content = None
            provider = "unknown"

            if response is not None:
                if hasattr(response, "model"):
                    model = response.model
                
                # Usage extraction
                if hasattr(response, "usage") and response.usage:
                    usage = response.usage
                    if hasattr(usage, "prompt_tokens"):
                        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                        provider = "openai"  # or groq
                    elif hasattr(usage, "input_tokens"):
                        input_tokens = getattr(usage, "input_tokens", 0) or 0
                        provider = "anthropic"
                    
                    if hasattr(usage, "completion_tokens"):
                        output_tokens = getattr(usage, "completion_tokens", 0) or 0
                    elif hasattr(usage, "output_tokens"):
                        output_tokens = getattr(usage, "output_tokens", 0) or 0

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
                provider=provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
                user_id=self.meta_kwargs.get("user_id"),
                session_id=self.meta_kwargs.get("session_id"),
                tags=self.meta_kwargs.get("tags"),
                routed_tier=self.meta_kwargs.get("routed_tier"),
                error=error_msg,
                prompt=prompt_content if self.log_prompts else None,
                response=response_content if self.log_responses else None
            )

            # Avoid blocking async streaming responses
            _emit_event_background(self.output, event, self.on_event)

class ObservedAsyncMethodWrapper(ObservedMethodWrapper):
    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        ts = datetime.now(timezone.utc).isoformat()
        error_msg = None
        response = None
        
        prompt_content = None
        if self.log_prompts:
            messages = kwargs.get("messages") or (args[0] if args and isinstance(args[0], list) else None)
            if messages:
                prompt_content = str(messages)
            else:
                prompt_content = kwargs.get("prompt") or (args[0] if args and isinstance(args[0], str) else None)

        try:
            response = await self.original_method(*args, **kwargs)
            return response
        except Exception as e:
            error_msg = str(e)
            raise e
        finally:
            latency_ms = int((time.time() - start_time) * 1000.0)
            
            model = kwargs.get("model", "unknown")
            input_tokens = 0
            output_tokens = 0
            response_content = None
            provider = "unknown"

            if response is not None:
                if hasattr(response, "model"):
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
                provider=provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
                user_id=self.meta_kwargs.get("user_id"),
                session_id=self.meta_kwargs.get("session_id"),
                tags=self.meta_kwargs.get("tags"),
                routed_tier=self.meta_kwargs.get("routed_tier"),
                error=error_msg,
                prompt=prompt_content if self.log_prompts else None,
                response=response_content if self.log_responses else None
            )

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

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._obj, name)
        
        if callable(attr):
            if name == "create":
                import inspect
                if inspect.iscoroutinefunction(attr):
                    return ObservedAsyncMethodWrapper(
                        attr, self._output, self._log_prompts, self._log_responses, self._on_event, self._meta_kwargs
                    )
                else:
                    return ObservedMethodWrapper(
                        attr, self._output, self._log_prompts, self._log_responses, self._on_event, self._meta_kwargs
                    )
            return attr
        
        if hasattr(attr, "__dict__") or isinstance(attr, object):
            return ObjectProxy(
                attr, self._output, self._log_prompts, self._log_responses, self._on_event, self._meta_kwargs
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
