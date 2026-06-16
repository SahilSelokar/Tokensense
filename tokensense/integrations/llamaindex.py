import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, cast

try:
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler
    from llama_index.core.callbacks.schema import CBEventType, EventPayload
except ImportError:
    class BaseCallbackHandler:
        def __init__(self, *args, **kwargs): pass
        def on_event_start(self, *args, **kwargs): pass
        def on_event_end(self, *args, **kwargs): pass
        def start_trace(self, *args, **kwargs): pass
        def end_trace(self, *args, **kwargs): pass
    CBEventType = Any
    EventPayload = Any

from tokensense.cost import estimate_cost
from tokensense.events import CallEvent
from tokensense.outputs import BaseOutput
from tokensense.observe import get_default_output, _emit_event_background

class TokenSenseLlamaIndexCallback(BaseCallbackHandler):
    """
    A LlamaIndex callback handler that logs LLM token usage, cost, and latency using TokenSense.
    """
    def __init__(
        self,
        output: Optional[BaseOutput] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        log_prompts: bool = False,
        log_responses: bool = False,
        on_event: Optional[Any] = None
    ):
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.output = output or get_default_output()
        self.user_id = user_id
        self.session_id = session_id
        self.tags = tags
        self.log_prompts = log_prompts
        self.log_responses = log_responses
        self.on_event = on_event

        self._runs: Dict[str, Dict[str, Any]] = {}

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        pass

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        if event_type == CBEventType.LLM:
            prompt_content = None
            if self.log_prompts and payload:
                messages = payload.get(EventPayload.MESSAGES)
                if messages:
                    prompt_content = str(messages)
                else:
                    prompt_content = payload.get(EventPayload.PROMPT)
                    
            # In LlamaIndex, the model name is usually on the LLM class itself
            # We will attempt to extract it in on_event_end where the response obj lives
            self._runs[event_id] = {
                "start_time": time.time(),
                "prompt": prompt_content,
                "model": kwargs.get("model_name", "unknown")
            }
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        if event_type == CBEventType.LLM:
            run_data = self._runs.pop(event_id, None)
            if not run_data:
                return
                
            latency_ms = int((time.time() - run_data["start_time"]) * 1000.0)
            ts = datetime.now(timezone.utc).isoformat()
            
            input_tokens = 0
            output_tokens = 0
            model = run_data["model"]
            response_content = None
            
            if payload:
                response_obj = payload.get(EventPayload.RESPONSE)
                
                if response_obj:
                    # Attempt to extract model name from the raw object
                    if hasattr(response_obj, "raw"):
                        model_from_raw = getattr(response_obj.raw, "model", None)
                        if model_from_raw:
                            model = model_from_raw
                            
                    # Attempt to extract token counts
                    additional_kwargs = getattr(response_obj, "additional_kwargs", {})
                    
                    if "prompt_tokens" in additional_kwargs:
                        input_tokens = additional_kwargs.get("prompt_tokens", 0)
                        output_tokens = additional_kwargs.get("completion_tokens", 0)
                    elif "token_usage" in additional_kwargs:
                        usage = additional_kwargs["token_usage"]
                        input_tokens = usage.get("prompt_tokens", 0)
                        output_tokens = usage.get("completion_tokens", 0)
                    elif hasattr(response_obj, "raw"):
                        usage = getattr(response_obj.raw, "usage", None)
                        if usage:
                            input_tokens = getattr(usage, "prompt_tokens", 0)
                            output_tokens = getattr(usage, "completion_tokens", 0)

                    if self.log_responses:
                        response_content = getattr(response_obj, "message", getattr(response_obj, "text", None))
                        if response_content and hasattr(response_content, "content"):
                            response_content = response_content.content
                        elif response_content is not None:
                            response_content = str(response_content)

            cost = estimate_cost(model, input_tokens, output_tokens)

            event = CallEvent(
                ts=ts,
                model=model,
                provider="llamaindex",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
                user_id=self.user_id,
                session_id=self.session_id,
                tags=self.tags,
                routed_tier=None,
                error=None,
                prompt=run_data["prompt"],
                response=response_content
            )

            _emit_event_background(self.output, event, self.on_event)
