import asyncio
import json
import os
import queue
import tempfile
import unittest
from pathlib import Path
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch

import server
from starlette.requests import Request


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


class FakeTensor:
    def __init__(self, token_count):
        self.shape = (1, token_count)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self
        start = key.start or 0
        return FakeTensor(max(0, self.shape[-1] - start))


class FakeStreamer:
    def __init__(self, *args, **kwargs):
        self.values = queue.Queue()

    def __iter__(self):
        while True:
            value = self.values.get()
            if value is None:
                return
            yield value

    def put_text(self, text):
        self.values.put(text)

    def end(self):
        self.values.put(None)


class FakeTokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return FakeTensor(4)

    def decode(self, tokens, **kwargs):
        return "Test response"


class FakeRuntime:
    def generate(self, *, streamer=None, **kwargs):
        if streamer:
            streamer.put_text("Test ")
            streamer.put_text("response")
            streamer.end()
        return FakeTensor(7)


class ApiTests(unittest.TestCase):
    def fake_torch(self):
        return SimpleNamespace(inference_mode=nullcontext)

    def test_transformers_adapter_generates_completion_and_usage(self):
        adapter = server.TransformersChatModel(FakeRuntime(), FakeTokenizer(), self.fake_torch())
        response = adapter.create_chat_completion(
            [{"role": "user", "content": "Hi"}],
            max_tokens=8,
            temperature=0,
            top_p=0.8,
        )
        self.assertEqual(response["choices"][0]["message"]["content"], "Test response")
        self.assertEqual(response["usage"]["prompt_tokens"], 4)
        self.assertEqual(response["usage"]["completion_tokens"], 3)
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")

    def test_transformers_adapter_enforces_configured_context(self):
        adapter = server.TransformersChatModel(FakeRuntime(), FakeTokenizer(), self.fake_torch())
        with patch.object(server, "MODEL_CONTEXT", 10):
            with self.assertRaisesRegex(ValueError, "configured context limit of 10"):
                adapter.create_chat_completion(
                    [{"role": "user", "content": "Hi"}],
                    max_tokens=8,
                    temperature=0,
                    top_p=0.8,
                )

    def test_transformers_adapter_streams_text_and_finish_reason(self):
        transformers = ModuleType("transformers")
        transformers.TextIteratorStreamer = FakeStreamer
        adapter = server.TransformersChatModel(FakeRuntime(), FakeTokenizer(), self.fake_torch())
        with patch.dict("sys.modules", {"transformers": transformers}):
            chunks = list(
                adapter.create_chat_completion(
                    [{"role": "user", "content": "Hi"}],
                    stream=True,
                    max_tokens=8,
                    temperature=0,
                    top_p=0.8,
                )
            )
        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "Test ")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_model_loader_uses_official_transformers_repository(self):
        runtime = SimpleNamespace(eval=lambda: None)
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.dict(os.environ, {"HF_HOME": temp_dir}),
                patch.object(server, "MODEL_REPO", "dgAI202/Converse-1.3"),
                patch.object(server, "MODEL_REVISION", "official-revision"),
                patch.object(server, "model_status", "loading"),
                patch("huggingface_hub.snapshot_download", return_value=temp_dir) as download,
                patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()) as tokenizer,
                patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=runtime) as model_loader,
            ):
                server.load_model()
                download.assert_called_once_with(
                    repo_id="dgAI202/Converse-1.3",
                    revision="official-revision",
                    cache_dir=Path(temp_dir) / "hub",
                )
                tokenizer.assert_called_once_with(temp_dir, local_files_only=True)
                self.assertEqual(model_loader.call_args.args, (temp_dir,))
                self.assertEqual(model_loader.call_args.kwargs["device_map"], "cpu")
                self.assertEqual(server.model_status, "ready")
                self.assertIsInstance(server.model, server.TransformersChatModel)

    def health_request(self, accept, query_string=b""):
        return Request(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": "/health",
                "raw_path": b"/health",
                "query_string": query_string,
                "root_path": "",
                "headers": [(b"accept", accept.encode())],
                "client": ("127.0.0.1", 1234),
                "server": ("testserver", 80),
            }
        )

    def test_health_reports_model_loading_phase(self):
        with (
            patch.object(server, "model_status", "loading"),
            patch.object(server, "model_phase", "downloading"),
            patch.object(server, "model_bytes_downloaded", 250),
            patch.object(server, "MODEL_SIZE_BYTES", 1000),
        ):
            response = server.health(self.health_request("application/json"))
        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        self.assertEqual(payload["phase"], "downloading")
        self.assertEqual(payload["progress_percent"], 25)
        self.assertEqual(payload["downloaded_bytes"], 250)

    def test_browser_health_request_returns_status_page(self):
        response = server.health(self.health_request("text/html"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.media_type, "text/html")
        self.assertTrue(str(response.path).endswith("static/health.html"))

    def test_health_format_json_overrides_html_accept(self):
        response = server.health(self.health_request("text/html", b"format=json"))
        self.assertEqual(response.media_type, "application/json")

    def test_partial_download_size_is_read_from_huggingface_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_cache = Path(temp_dir) / "models--dgAI--Converse-1.3" / "blobs"
            repo_cache.mkdir(parents=True)
            (repo_cache / "model.incomplete").write_bytes(b"0123456789")
            self.assertEqual(server.partial_download_bytes(repo_cache), 10)

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
