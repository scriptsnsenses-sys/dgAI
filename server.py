import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

ROOT = Path(__file__).parent
MODEL_REPO = os.getenv("MODEL_REPO", "dgAI202/Converse-1.3")
MODEL_FILE = os.getenv("MODEL_FILE", "model.safetensors")
MODEL_REVISION = os.getenv("MODEL_REVISION", "29efcd5a8fd2f55b727f06a8b636fb825bcbd4b3")
MODEL_CONTEXT = int(os.getenv("MODEL_CONTEXT", "32768"))
MODEL_THREADS = int(os.getenv("MODEL_THREADS", "2"))
MODEL_SIZE_BYTES = max(1, int(os.getenv("MODEL_SIZE_BYTES", "3098907121")))
MODEL_NAME = "converse-1.3"
MAX_GENERATION_TOKENS = 2048
PROCESS_STARTED_AT = time.time()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("dgai")
app = FastAPI(title="dgAI Converse 1.3", docs_url=None, redoc_url=None)
model_lock = threading.Lock()
health_lock = threading.Lock()
model: Any = None
model_status = "loading"
model_phase = "downloading"
model_error: str | None = None
model_bytes_downloaded = 0
phase_started_at = PROCESS_STARTED_AT


def update_model_state(
    *,
    status: str | None = None,
    phase: str | None = None,
    downloaded_bytes: int | None = None,
    error: str | None = None,
) -> None:
    global model_status, model_phase, model_bytes_downloaded, model_error, phase_started_at
    with health_lock:
        if status is not None:
            model_status = status
        if phase is not None:
            if phase != model_phase:
                phase_started_at = time.time()
            model_phase = phase
        if downloaded_bytes is not None:
            model_bytes_downloaded = max(0, min(downloaded_bytes, MODEL_SIZE_BYTES))
        if error is not None:
            model_error = error


def watch_model_download(hub_cache: Path, stop: threading.Event) -> None:
    repo_cache = hub_cache / f"models--{MODEL_REPO.replace('/', '--')}" / "blobs"
    while not stop.wait(1):
        update_model_state(downloaded_bytes=partial_download_bytes(repo_cache))


def partial_download_bytes(repo_cache: Path) -> int:
    downloaded = 0
    if repo_cache.exists():
        for partial_file in repo_cache.glob("*.incomplete"):
            try:
                downloaded += partial_file.stat().st_size
            except FileNotFoundError:
                continue
    return downloaded


class TransformersChatModel:
    def __init__(self, runtime: Any, tokenizer: Any, torch_module: Any) -> None:
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.torch = torch_module

    def _generation_inputs(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[dict[str, Any], int, dict[str, Any]]:
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(input_ids, "items"):
            inputs = dict(input_ids.items())
            input_ids = inputs["input_ids"]
        else:
            inputs = {"input_ids": input_ids}
        prompt_tokens = int(input_ids.shape[-1])
        if prompt_tokens + max_tokens > MODEL_CONTEXT:
            raise ValueError(
                f"Prompt ({prompt_tokens} tokens) and max_tokens ({max_tokens}) exceed the "
                f"configured context limit of {MODEL_CONTEXT} tokens."
            )
        options = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "top_p": top_p,
            "top_k": 20,
            "repetition_penalty": 1.1,
        }
        if temperature > 0:
            options["temperature"] = temperature
        return inputs, prompt_tokens, options

    def create_chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.8,
        **_: Any,
    ) -> Any:
        inputs, prompt_tokens, options = self._generation_inputs(
            messages,
            max_tokens,
            temperature,
            top_p,
        )
        if stream:
            return self._stream_completion(inputs, prompt_tokens, options, max_tokens)

        with self.torch.inference_mode():
            output = self.runtime.generate(**inputs, **options)
        generated_ids = output[0][prompt_tokens:]
        content = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        completion_tokens = int(generated_ids.shape[-1])
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "length" if completion_tokens >= max_tokens else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def _stream_completion(
        self,
        inputs: dict[str, Any],
        prompt_tokens: int,
        options: dict[str, Any],
        max_tokens: int,
    ) -> Iterator[dict[str, Any]]:
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        result: list[Any] = []
        failure: list[BaseException] = []

        def generate() -> None:
            try:
                with self.torch.inference_mode():
                    result.append(self.runtime.generate(**inputs, streamer=streamer, **options))
            except BaseException as exc:
                failure.append(exc)
                streamer.end()

        generation_thread = threading.Thread(target=generate, name="model-generation", daemon=True)
        generation_thread.start()
        for text in streamer:
            if text:
                yield {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
        generation_thread.join()
        if failure:
            raise failure[0]
        generated_tokens = int(result[0][0][prompt_tokens:].shape[-1]) if result else 0
        yield {
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "length" if generated_tokens >= max_tokens else "stop",
                }
            ]
        }


def load_model() -> None:
    global model
    try:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Downloading or locating model %s/%s", MODEL_REPO, MODEL_FILE)
        torch.set_num_threads(MODEL_THREADS)
        hf_home = os.getenv("HF_HOME")
        hub_cache = Path(hf_home) / "hub" if hf_home else Path(
            os.getenv("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub")
        )
        stop_download_watcher = threading.Event()
        download_watcher = threading.Thread(
            target=watch_model_download,
            args=(hub_cache, stop_download_watcher),
            name="model-download-progress",
            daemon=True,
        )
        download_watcher.start()
        try:
            model_path = snapshot_download(
                repo_id=MODEL_REPO,
                revision=MODEL_REVISION,
                cache_dir=hub_cache,
            )
        finally:
            stop_download_watcher.set()
            download_watcher.join()
        update_model_state(downloaded_bytes=MODEL_SIZE_BYTES, phase="initializing")
        logger.info("Initializing model from %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        runtime = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="cpu",
            local_files_only=True,
        )
        runtime.eval()
        loaded_model = TransformersChatModel(runtime, tokenizer, torch)
        with model_lock:
            model = loaded_model
        update_model_state(status="ready", phase="ready", downloaded_bytes=MODEL_SIZE_BYTES)
        logger.info("Loaded %s/%s", MODEL_REPO, MODEL_FILE)
    except Exception:
        logger.exception("Could not load the Converse 1.3 model")
        update_model_state(status="error", phase="error", error="Model loading failed. Check the service logs and model settings.")


@app.on_event("startup")
def start_model_loader() -> None:
    threading.Thread(target=load_model, name="model-loader", daemon=True).start()


def api_error(message: str, status: int, error_type: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": error_type, "param": None, "code": None}},
    )


def anthropic_error(message: str, status: int, error_type: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"type": "error", "error": {"type": error_type, "message": message}},
    )


def text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") not in ("text", "input_text"):
                raise ValueError("Only text message content is supported.")
            value = block.get("text", "")
            if not isinstance(value, str):
                raise ValueError("Text content blocks must contain a string.")
            parts.append(value)
        return "\n".join(parts)
    raise ValueError("Message content must be a string or an array of text blocks.")


def normalize_messages(messages: Any, anthropic: bool = False) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty array.")
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be an object.")
        role = message.get("role")
        allowed = ("system", "user", "assistant") if anthropic else ("system", "developer", "user", "assistant")
        if role not in allowed:
            raise ValueError(f"Unsupported message role: {role}.")
        if role == "developer":
            role = "system"
        normalized.append({"role": role, "content": text_content(message.get("content"))})
    return normalized


def generation_options(body: dict[str, Any]) -> dict[str, Any]:
    max_tokens = body.get("max_tokens", body.get("max_completion_tokens", 512))
    temperature = body.get("temperature", 0.7)
    top_p = body.get("top_p", 0.8)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= MAX_GENERATION_TOKENS:
        raise ValueError(f"max_tokens must be an integer from 1 to {MAX_GENERATION_TOKENS}.")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2.")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
        raise ValueError("top_p must be greater than 0 and at most 1.")
    return {"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p}


def unavailable_error(anthropic: bool) -> JSONResponse:
    if model_status == "error":
        message = model_error or "Model loading failed."
        return anthropic_error(message, 503, "api_error") if anthropic else api_error(message, 503, "server_error")
    message = "Converse 1.3 is still loading. Please retry shortly."
    return anthropic_error(message, 503, "api_error") if anthropic else api_error(message, 503, "server_error")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/docs", include_in_schema=False)
def usage_docs() -> FileResponse:
    return FileResponse(ROOT / "static" / "docs.html")


def health_details() -> dict[str, Any]:
    with health_lock:
        status = model_status
        phase = model_phase
        downloaded_bytes = model_bytes_downloaded
        error = model_error
        phase_started = phase_started_at
    progress = 100 if phase == "ready" else min(100, round(downloaded_bytes * 100 / MODEL_SIZE_BYTES)) if phase == "downloading" else None
    return {
        "status": status,
        "phase": phase,
        "progress_percent": progress,
        "downloaded_bytes": downloaded_bytes,
        "total_bytes": MODEL_SIZE_BYTES,
        "model": MODEL_NAME,
        "model_repo": MODEL_REPO,
        "model_file": MODEL_FILE,
        "revision": MODEL_REVISION,
        "context_tokens": MODEL_CONTEXT,
        "threads": MODEL_THREADS,
        "uptime_seconds": max(0, int(time.time() - PROCESS_STARTED_AT)),
        "phase_seconds": max(0, int(time.time() - phase_started)),
        "error": error,
    }


@app.get("/health", include_in_schema=False)
def health(request: Request) -> Response:
    details = health_details()
    status_code = 503 if details["status"] == "error" else 200
    if request.query_params.get("format") != "json" and "text/html" in request.headers.get("accept", ""):
        return FileResponse(
            ROOT / "static" / "health.html",
            status_code=status_code,
            headers={"Cache-Control": "no-store"},
        )
    return JSONResponse(
        status_code=status_code,
        content=details,
        headers={"Cache-Control": "no-store"},
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("Request body must be a JSON object.")
        messages = normalize_messages(body.get("messages"))
        options = generation_options(body)
        if body.get("stream", False):
            if not isinstance(body["stream"], bool):
                raise ValueError("stream must be a boolean.")
        if body.get("stream") and model_status == "ready":
            return StreamingResponse(
                stream_openai(messages, options),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
    except (ValueError, json.JSONDecodeError) as exc:
        return api_error(str(exc), 400)
    if model_status != "ready":
        return unavailable_error(False)

    try:
        with model_lock:
            result = model.create_chat_completion(messages=messages, **options)
        return JSONResponse(
            content={
                "id": result.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_NAME,
                "choices": result["choices"],
                "usage": result.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
            }
        )
    except ValueError as exc:
        return api_error(str(exc), 400)
    except Exception:
        logger.exception("OpenAI-compatible completion failed")
        return api_error("The model could not generate a completion.", 500, "server_error")


def stream_openai(messages: list[dict[str, str]], options: dict[str, Any]) -> Iterator[str]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    try:
        with model_lock:
            for result in model.create_chat_completion(messages=messages, stream=True, **options):
                choice = result["choices"][0]
                chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_NAME,
                    "choices": [
                        {
                            "index": 0,
                            "delta": choice.get("delta", {}),
                            "finish_reason": choice.get("finish_reason"),
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception:
        logger.exception("OpenAI-compatible stream failed")
        error = {"error": {"message": "The model stream failed.", "type": "server_error"}}
        yield f"data: {json.dumps(error)}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/v1/messages")
async def anthropic_messages(request: Request) -> Response:
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("Request body must be a JSON object.")
        messages = normalize_messages(body.get("messages"), anthropic=True)
        system = body.get("system")
        if system is not None:
            if not isinstance(system, str):
                raise ValueError("system must be a string.")
            messages.insert(0, {"role": "system", "content": system})
        if body.get("stream", False):
            if not isinstance(body["stream"], bool):
                raise ValueError("stream must be a boolean.")
        options = generation_options(
            {
                **body,
                "max_tokens": body.get("max_tokens", 512),
            }
        )
        if body.get("stream") and model_status == "ready":
            return StreamingResponse(
                stream_anthropic(messages, options),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
    except (ValueError, json.JSONDecodeError) as exc:
        return anthropic_error(str(exc), 400)
    if model_status != "ready":
        return unavailable_error(True)

    try:
        with model_lock:
            result = model.create_chat_completion(messages=messages, **options)
        choice = result["choices"][0]
        usage = result.get("usage", {})
        return JSONResponse(
            content={
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": choice["message"].get("content") or ""}],
                "model": MODEL_NAME,
                "stop_reason": "max_tokens" if choice.get("finish_reason") == "length" else "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                },
            }
        )
    except ValueError as exc:
        return anthropic_error(str(exc), 400)
    except Exception:
        logger.exception("Anthropic-compatible message generation failed")
        return anthropic_error("The model could not generate a response.", 500, "api_error")


def anthropic_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def stream_anthropic(messages: list[dict[str, str]], options: dict[str, Any]) -> Iterator[str]:
    message_id = f"msg_{uuid.uuid4().hex}"
    yield anthropic_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": MODEL_NAME,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )
    yield anthropic_event("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
    finish_reason = "end_turn"
    try:
        with model_lock:
            for result in model.create_chat_completion(messages=messages, stream=True, **options):
                choice = result["choices"][0]
                if choice.get("finish_reason") == "length":
                    finish_reason = "max_tokens"
                text = choice.get("delta", {}).get("content")
                if text:
                    yield anthropic_event(
                        "content_block_delta",
                        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
                    )
        yield anthropic_event("content_block_stop", {"type": "content_block_stop", "index": 0})
        yield anthropic_event(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": finish_reason, "stop_sequence": None}, "usage": {"output_tokens": 0}},
        )
        yield anthropic_event("message_stop", {"type": "message_stop"})
    except Exception:
        logger.exception("Anthropic-compatible stream failed")
        yield anthropic_event(
            "error",
            {"type": "error", "error": {"type": "api_error", "message": "The model stream failed."}},
        )
