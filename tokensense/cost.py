from typing import Tuple, Optional

import json
import os
from typing import Tuple, Optional, Dict

MODEL_COSTS: Dict[str, Tuple[float, float]] = {}
CONTEXT_WINDOWS: Dict[str, int] = {}

def _load_litellm_prices():
    global MODEL_COSTS, CONTEXT_WINDOWS
    
    # Prefer user's cached prices, fall back to bundled copy
    user_cache = os.path.join(os.path.expanduser("~"), ".tokensense", "model_prices.json")
    bundled = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_prices.json")
    json_path = user_cache if os.path.exists(user_cache) else bundled
    
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                
            for model_name, info in data.items():
                if model_name == "sample_spec":
                    continue
                
                # LiteLLM stores cost-per-token, we store cost-per-million
                in_cost = info.get("input_cost_per_token", 0.0) * 1_000_000
                out_cost = info.get("output_cost_per_token", 0.0) * 1_000_000
                MODEL_COSTS[model_name] = (in_cost, out_cost)
                
                context_window = info.get("max_input_tokens") or info.get("max_tokens")
                if context_window:
                    CONTEXT_WINDOWS[model_name] = int(context_window)
                    
            # Hardcoded prices for models LiteLLM doesn't know about yet
            if "gemini-2.5-flash" not in MODEL_COSTS:
                MODEL_COSTS["gemini-2.5-flash"] = (0.15, 1.25)
                CONTEXT_WINDOWS["gemini-2.5-flash"] = 1_048_576
            if "gemini-2.5-pro" not in MODEL_COSTS:
                MODEL_COSTS["gemini-2.5-pro"] = (0.625, 5.00)
                CONTEXT_WINDOWS["gemini-2.5-pro"] = 2_097_152
            if "gemini-3.1-flash-lite" not in MODEL_COSTS:
                MODEL_COSTS["gemini-3.1-flash-lite"] = (0.125, 0.75)
                CONTEXT_WINDOWS["gemini-3.1-flash-lite"] = 1_048_576
            if "gemini-3.1-pro-preview" not in MODEL_COSTS:
                MODEL_COSTS["gemini-3.1-pro-preview"] = (1.00, 6.00)
                CONTEXT_WINDOWS["gemini-3.1-pro-preview"] = 2_097_152
            if "gemini-3.5-flash" not in MODEL_COSTS:
                MODEL_COSTS["gemini-3.5-flash"] = (0.75, 4.50)
                CONTEXT_WINDOWS["gemini-3.5-flash"] = 1_048_576
                
        except Exception as e:
            print(f"Warning: Failed to load TokenSense model prices: {e}")

_load_litellm_prices()

def _fuzzy_match_model(model_name: str, target_dict: dict) -> Optional[str]:
    # Exact match first
    if model_name in target_dict:
        return model_name
        
    # Try stripping provider prefix (e.g. groq/llama3-8b-8192 -> llama3-8b-8192)
    clean_model = model_name
    if "/" in clean_model:
        clean_model = clean_model.split("/", 1)[1]
        if clean_model in target_dict:
            return clean_model
            
    # Longest prefix match as last resort
    candidates = []
    for k in target_dict.keys():
        if clean_model.startswith(k):
            candidates.append(k)
    
    if candidates:
        return max(candidates, key=len)
        
    return None

def estimate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost of a call given model + token counts."""
    matched_key = _fuzzy_match_model(model_name, MODEL_COSTS)
    
    if matched_key:
        input_price, output_price = MODEL_COSTS[matched_key]
    else:
        # Unknown model — use a safe middle-ground price
        input_price, output_price = 0.15, 0.60
        
    input_cost = (input_tokens / 1_000_000.0) * input_price
    output_cost = (output_tokens / 1_000_000.0) * output_price
    return input_cost + output_cost

# Legacy alias — some internal modules still reference this
calculate_cost = estimate_cost

def get_context_window(model_name: str) -> Optional[int]:
    """
    Get the context window (in tokens) for a given model.
    Returns None if unknown.
    """
    matched_key = _fuzzy_match_model(model_name, CONTEXT_WINDOWS)
    if matched_key:
        return CONTEXT_WINDOWS[matched_key]
    return None
