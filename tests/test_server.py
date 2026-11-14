import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

import server


class FakeModel:
    def create_chat_completion(self, messages, **options):
        if options.get("stream"):
            return iter(
                [
                    {"choices": [{"delta": {"content": "Hello "}, "finish_reason": None}]},
                    {"choices": [{"delta": {"content": "from Converse."}, "finish_reason": None}]},
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                ]
            )
        return {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from Converse."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
        }


class FakeRequest:
    def __init__(self, body):
        self.json = AsyncMock(return_value=body)


class ApiTests(unittest.TestCase):
    def test_health_reports_model_loading_phase(self):
        with patch.object(server, "model_status", "loading"), patch.object(server, "model_phase", "downloading"):
            response = server.health()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.body)["phase"], "downloading")

    def collect(self, response):
        async def read_body():
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
            return "".join(chunks)

        return asyncio.run(read_body())

    def test_normalizes_text_blocks_and_developer_role(self):
        messages = server.normalize_messages(
            [
                {"role": "developer", "content": "Be clear."},
                {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            ]
        )
        self.assertEqual(
            messages,
            [{"role": "system", "content": "Be clear."}, {"role": "user", "content": "Hi"}],
        )

    def test_rejects_non_text_content(self):
        with self.assertRaisesRegex(ValueError, "Only text"):
            server.normalize_messages([{"role": "user", "content": [{"type": "image"}]}])

    def test_rejects_generation_limits_outside_supported_range(self):
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            server.generation_options({"max_tokens": 2049})

    def test_openai_response_shape(self):
        with patch.object(server, "model_status", "ready"), patch.object(server, "model", FakeModel()):
            response = asyncio.run(
                server.chat_completions(
                    FakeRequest(
                        {
                            "model": "converse-1.3",
                            "messages": [{"role": "user", "content": "Hi"}],
                        }
                    )
                )
            )
        payload = json.loads(response.body)
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["choices"][0]["message"]["content"], "Hello from Converse.")

    def test_anthropic_response_shape(self):
        with patch.object(server, "model_status", "ready"), patch.object(server, "model", FakeModel()):
            response = asyncio.run(
                server.anthropic_messages(
                    FakeRequest(
                        {
                            "model": "converse-1.3",
                            "max_tokens": 32,
                            "messages": [{"role": "user", "content": "Hi"}],
                        }
                    )
                )
            )
        payload = json.loads(response.body)
        self.assertEqual(payload["type"], "message")
        self.assertEqual(payload["content"], [{"type": "text", "text": "Hello from Converse."}])
        self.assertEqual(payload["usage"]["input_tokens"], 4)

    def test_openai_stream_uses_sse_chunks(self):
        with patch.object(server, "model_status", "ready"), patch.object(server, "model", FakeModel()):
            response = asyncio.run(
                server.chat_completions(
                    FakeRequest(
                        {
                            "messages": [{"role": "user", "content": "Hi"}],
                            "stream": True,
                        }
                    )
                )
            )
            body = self.collect(response)
        self.assertIn('"object": "chat.completion.chunk"', body)
        self.assertTrue(body.endswith("data: [DONE]\n\n"))

    def test_anthropic_stream_uses_named_events(self):
        with patch.object(server, "model_status", "ready"), patch.object(server, "model", FakeModel()):
            response = asyncio.run(
                server.anthropic_messages(
                    FakeRequest(
                        {
                            "max_tokens": 32,
                            "messages": [{"role": "user", "content": "Hi"}],
                            "stream": True,
                        }
                    )
                )
            )
            body = self.collect(response)
        self.assertIn("event: message_start\n", body)
        self.assertIn("event: content_block_delta\n", body)
        self.assertIn("event: message_stop\n", body)

    def test_unready_model_returns_api_error(self):
        with patch.object(server, "model_status", "loading"):
            response = asyncio.run(
                server.chat_completions(
                    FakeRequest({"messages": [{"role": "user", "content": "Hi"}]})
                )
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("error", json.loads(response.body))


if __name__ == "__main__":
    unittest.main()
