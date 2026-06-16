import unittest
from unittest.mock import MagicMock, patch
from tokensense import observe
from tokensense.events import CallEvent
from tokensense.outputs import BaseOutput

from types import SimpleNamespace

class MockOutput(BaseOutput):
    def __init__(self):
        self.events = []
    def write(self, event: CallEvent) -> None:
        print("MockOutput.write called with event:", event.provider)
        self.events.append(event)

class TestProviders(unittest.TestCase):
    def setUp(self):
        self.output = MockOutput()

    def test_openai(self):
        mock_response = SimpleNamespace(
            model="gpt-4o",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
            choices=[SimpleNamespace(message=SimpleNamespace(content="Hello OpenAI"))]
        )
        mock_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=MagicMock(return_value=mock_response)
                )
            )
        )

        client = observe(mock_client, output=self.output, log_responses=True)
        resp = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
        
        self.assertEqual(resp.choices[0].message.content, "Hello OpenAI")
        
        self.assertEqual(len(self.output.events), 1)
        event = self.output.events[0]
        self.assertEqual(event.model, "gpt-4o")
        self.assertEqual(event.provider, "openai")
        self.assertEqual(event.input_tokens, 10)
        self.assertEqual(event.output_tokens, 20)

    def test_anthropic(self):
        mock_response = SimpleNamespace(
            model="claude-sonnet-4-6",
            usage=SimpleNamespace(input_tokens=15, output_tokens=25),
            content=[SimpleNamespace(text="Hello Anthropic")]
        )
        mock_client = SimpleNamespace(
            messages=SimpleNamespace(
                create=MagicMock(return_value=mock_response)
            )
        )

        client = observe(mock_client, output=self.output, log_responses=True)
        resp = client.messages.create(model="claude-sonnet-4-6", messages=[{"role": "user", "content": "hi"}])
        
        self.assertEqual(len(self.output.events), 1)
        event = self.output.events[0]
        self.assertEqual(event.model, "claude-sonnet-4-6")
        self.assertEqual(event.provider, "anthropic")
        self.assertEqual(event.input_tokens, 15)
        self.assertEqual(event.output_tokens, 25)

    def test_litellm(self):
        mock_litellm = MagicMock()
        mock_response = SimpleNamespace(
            model="cohere/command-r",
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=10),
            choices=[SimpleNamespace(message=SimpleNamespace(content="Hello LiteLLM"))]
        )
        mock_litellm.completion.return_value = mock_response

        litellm = observe(mock_litellm, output=self.output, log_responses=True)
        resp = litellm.completion(model="cohere/command-r", messages=[{"role": "user", "content": "hi"}])
        
        self.assertEqual(len(self.output.events), 1)
        event = self.output.events[0]
        self.assertEqual(event.model, "cohere/command-r")
        self.assertEqual(event.provider, "openai") 
        self.assertEqual(event.input_tokens, 5)
        self.assertEqual(event.output_tokens, 10)

    def test_gemini(self):
        mock_model = MagicMock()
        mock_response = SimpleNamespace(
            model="gemini-1.5-flash",
            usage_metadata=SimpleNamespace(prompt_token_count=8, candidates_token_count=12),
            text="Hello Gemini"
        )
        mock_model.generate_content.return_value = mock_response

        model = observe(mock_model, output=self.output, log_responses=True)
        resp = model.generate_content("hi")
        
        self.assertEqual(len(self.output.events), 1)
        event = self.output.events[0]
        self.assertEqual(event.model, "gemini-1.5-flash")
        self.assertEqual(event.provider, "gemini")
        self.assertEqual(event.input_tokens, 8)
        self.assertEqual(event.output_tokens, 12)

if __name__ == "__main__":
    unittest.main()
