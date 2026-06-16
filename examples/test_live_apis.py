import os
import time

try:
    from groq import Groq
except ImportError:
    print("Please install groq: pip install groq")
    exit(1)

try:
    import google.generativeai as genai
except ImportError:
    print("Please install google-generativeai: pip install google-generativeai")
    exit(1)

from tokensense import observe

def run_tests():
    groq_api_key = os.environ.get("GROQ_API_KEY")
    gemini_api_key = os.environ.get("GEMINI_API_KEY")

    if not groq_api_key:
        print("Error: Please set the GROQ_API_KEY environment variable.")
        exit(1)
        
    if not gemini_api_key:
        print("Error: Please set the GEMINI_API_KEY environment variable.")
        exit(1)

    print("========================================")
    print("Testing Groq...")
    print("========================================")
    # 1. Initialize and wrap Groq client
    groq_client = Groq(api_key=groq_api_key)
    observed_groq = observe(groq_client, log_responses=True, user_id="user_123")
    
    # 2. Make an API call
    response = observed_groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "What is the capital of France? Reply in one word."}]
    )
    print(f"Groq Response: {response.choices[0].message.content}")
    time.sleep(1) # Wait for background thread to write to SQLite

    print("\n========================================")
    print("Testing Gemini...")
    print("========================================")
    # 1. Initialize Gemini
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    # 2. Wrap the model
    observed_model = observe(model, log_responses=True, user_id="user_123")
    
    # 3. Make an API call
    response = observed_model.generate_content("What is the capital of Japan? Reply in one word.")
    print(f"Gemini Response: {response.text}")
    time.sleep(1) # Wait for background thread to write to SQLite

    print("\n========================================")
    print("Done! Check your TokenSense CLI report.")
    print("========================================")

if __name__ == "__main__":
    run_tests()
