import os
from tokensense import TokenSenseLlamaIndexCallback
import sqlite3

def run_llamaindex_example():
    try:
        from llama_index.llms.groq import Groq
        from llama_index.core.callbacks import CallbackManager
        from llama_index.core import Settings
    except ImportError:
        print("Please install LlamaIndex to run this example: pip install llama-index-core llama-index-llms-groq")
        return

    # 1. Initialize TokenSense's LlamaIndex Callback
    tokensense_cb = TokenSenseLlamaIndexCallback()
    callback_manager = CallbackManager([tokensense_cb])
    Settings.callback_manager = callback_manager

    # 2. Initialize the LlamaIndex LLM
    llm = Groq(
        model="llama-3.1-8b-instant",
        api_key=os.environ.get("GROQ_API_KEY", "gsk_dummy"),
    )
    Settings.llm = llm

    print("Running LlamaIndex Groq with TokenSense Callback...")
    try:
        # 3. Use LlamaIndex natively!
        response = llm.complete("What is the capital of Spain? Reply in one word.")
        print(f"Response: {response.text}")
        
    except Exception as e:
        print(f"API Error: {e}")

if __name__ == "__main__":
    run_llamaindex_example()
