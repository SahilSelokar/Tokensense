import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
except ImportError:
    class BaseCallbackHandler:
        pass
    LLMResult = Any

from tokensense.cost import estimate_cost
from tokensense.events import CallEvent
from tokensense.outputs import BaseOutput
from tokensense.observe import get_default_output, _emit_event_background

class TokenSenseCallbackHandler(BaseCallbackHandler):
    """
    A LangChain callback handler that logs LLM token usage, cost, and latency using TokenSense.
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
        self.output = output or get_default_output()
        self.user_id = user_id
        self.session_id = session_id
        self.tags = tags
        self.log_prompts = log_prompts
        self.log_responses = log_responses
        self.on_event = on_event
        
        # Track active runs
        self._runs: Dict[str, Dict[str, Any]] = {}

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], *, run_id: str, parent_run_id: Optional[str] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Any:
        # Extract model name if possible
        model = kwargs.get("invocation_params", {}).get("model") or kwargs.get("invocation_params", {}).get("model_name") or "unknown"
        
        prompt_content = None
        if self.log_prompts and prompts:
            prompt_content = "\\n".join(prompts)
            
        self._runs[str(run_id)] = {
            "start_time": time.time(),
            "model": model,
            "prompt": prompt_content,
        }

    def on_llm_end(self, response: LLMResult, *, run_id: str, parent_run_id: Optional[str] = None, **kwargs: Any) -> Any:
        run_data = self._runs.pop(str(run_id), None)
        if not run_data:
            return

        latency_ms = int((time.time() - run_data["start_time"]) * 1000.0)
        ts = datetime.now(timezone.utc).isoformat()
        
        model = run_data["model"]
        # Some LLMs return model in llm_output
        if response.llm_output and "model_name" in response.llm_output:
            model = response.llm_output["model_name"]
            
        input_tokens = 0
        output_tokens = 0
        provider = "langchain"
        
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

        response_content = None
        if self.log_responses and response.generations:
            # Flatten generations
            texts = []
            for gens in response.generations:
                for gen in gens:
                    texts.append(gen.text)
            response_content = "\\n".join(texts)

        cost = estimate_cost(model, input_tokens, output_tokens)

        event = CallEvent(
            ts=ts,
            model=model,
            provider=provider,
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

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], *, run_id: str, parent_run_id: Optional[str] = None, **kwargs: Any) -> Any:
        run_data = self._runs.pop(str(run_id), None)
        if not run_data:
            return

        latency_ms = int((time.time() - run_data["start_time"]) * 1000.0)
        ts = datetime.now(timezone.utc).isoformat()
        model = run_data["model"]
        
        event = CallEvent(
            ts=ts,
            model=model,
            provider="langchain",
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            latency_ms=latency_ms,
            user_id=self.user_id,
            session_id=self.session_id,
            tags=self.tags,
            routed_tier=None,
            error=str(error),
            prompt=run_data["prompt"],
            response=None
        )

        _emit_event_background(self.output, event, self.on_event)
