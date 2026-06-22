import json
from typing import Any, Dict

from tokensense.harness import ShadowTest
from tokensense.observe import observe

# We will create a mock client to simulate an SDK since we don't assume the user has OpenAI/Anthropic installed for this test script

class MockClient:
    def __init__(self, model_responses: Dict[str, str]):
        self.model_responses = model_responses
        self.chat = self.Chat(self)
        
    class Chat:
        def __init__(self, parent):
            self.completions = self.Completions(parent)
            
        class Completions:
            def __init__(self, parent):
                self.parent = parent
                
            def create(self, **kwargs) -> Any:
                model = kwargs.get("model", "unknown")
                text = self.parent.model_responses.get(model, '{"status": "ok"}')
                
                class MockChoice:
                    class MockMessage:
                        content = text
                    message = MockMessage()
                    
                class MockResponse:
                    choices = [MockChoice()]
                    
                return MockResponse()

def main():
    # Setup mock clients to simulate our current and candidate models
    mock_current = MockClient({
        "claude-3-5-sonnet-20240620": "42",
        "gpt-4o": '{"answer": 42}'
    })
    
    mock_candidate = MockClient({
        "claude-3-haiku-20240307": "42", 
        "gpt-4o-mini": '{"answer": 42, "extra": "invalid json format actually' # simulate failure
    })

    # You would normally wrap your real clients like this:
    # client_current = observe(anthropic.Anthropic())
    # But for the example we wrap the mock:
    client_current = observe(mock_current, user_id="tester-1")
    client_candidate = observe(mock_candidate, user_id="tester-2")

    print("Running ShadowTest with Exact Match Scoring...")
    
    test = ShadowTest(
        clients={
            "current": client_current,
            "candidate": client_candidate
        },
        prompts=[
            {
                "messages": [{"role": "user", "content": "What is the answer to life?"}],
                "model_current": "claude-3-5-sonnet-20240620",
                "model_candidate": "claude-3-haiku-20240307",
                "expected_output": "42"  # used for exact-match scoring
            },
            {
                "messages": [{"role": "user", "content": "Return the answer as JSON."}],
                "model_current": "gpt-4o",
                "model_candidate": "gpt-4o-mini",
                "expected_format": "json" # handled in format-check if scoring="format-check"
            }
        ],
        scoring="exact-match"
    )

    report = test.run()
    print(report.summary())
    
    print("\n------------------\n")
    print("Running ShadowTest with Format Check Scoring...")
    
    test_format = ShadowTest(
        clients={
            "current": client_current,
            "candidate": client_candidate
        },
        prompts=[
            {
                "messages": [{"role": "user", "content": "Return JSON"}],
                "model_current": "gpt-4o",
                "model_candidate": "gpt-4o-mini",
                "expected_format": "json"
            }
        ],
        scoring="format-check"
    )
    
    report_format = test_format.run()
    print(report_format.summary())

    print("\n------------------\n")
    print("Running ShadowTest with LLM Judge Scoring...")
    
    # Mocking the judge to always return PASS
    mock_judge = MockClient({
        "gpt-4o-mini": "PASS",
        "claude-3-haiku-20240307": "PASS"
    })
    
    test_judge = ShadowTest(
        clients={
            "current": client_current,
            "candidate": client_candidate
        },
        prompts=[
            {
                "messages": [{"role": "user", "content": "Explain relativity briefly."}],
                "model_current": "gpt-4o",
                "model_candidate": "gpt-4o-mini",
            }
        ],
        scoring="llm-judge",
        judge=mock_judge,
        judge_rubric="The response must be factually accurate and mention Einstein."
    )
    
    report_judge = test_judge.run()
    print(report_judge.summary())

    print("\n------------------\n")
    print("Running ShadowTest with Similarity Scoring...")
    
    # Simple mock embedding function
    def mock_embed(text: str) -> list[float]:
        # just returns a dummy vector
        return [1.0, 0.0, 0.0]
        
    test_sim = ShadowTest(
        clients={
            "current": client_current,
            "candidate": client_candidate
        },
        prompts=[
            {
                "messages": [{"role": "user", "content": "Summarize this text."}],
                "model_current": "gpt-4o",
                "model_candidate": "gpt-4o-mini",
            }
        ],
        scoring="similarity",
        embedding_fn=mock_embed,
        similarity_threshold=0.9
    )
    
    report_sim = test_sim.run()
    print(report_sim.summary())

if __name__ == "__main__":
    main()
