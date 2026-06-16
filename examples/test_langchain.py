import os
from langchain_core.messages import HumanMessage
from tokensense import TokenSenseCallbackHandler
import sqlite3

# This is an example of how a user would use TokenSense with LangChain.
# To run this, you must install: pip install langchain-groq
def run_langchain_example():
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        print("Please install langchain-groq to run this example: pip install langchain-groq")
        return

    # 1. Initialize TokenSense's Callback Handler
    tokensense_cb = TokenSenseCallbackHandler()

    # 2. Initialize your LangChain model
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.environ.get("GROQ_API_KEY", "gsk_dummy")
    )

    print("Running LangChain ChatGroq with TokenSense Callback...")
    try:
        # 3. Pass the callback handler directly to the invoke method
        response = llm.invoke(
            [HumanMessage(content="What is the capital of France? Reply in one word.")],
            config={"callbacks": [tokensense_cb]}
        )
        print(f"Response: {response.content}")
        
    except Exception as e:
        print(f"API Error: {e}")

if __name__ == "__main__":
    run_langchain_example()
