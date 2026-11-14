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
MODEL_REPO = os.getenv("MODEL_REPO", "mradermacher/Converse-1.3-GGUF")
MODEL_FILE = os.getenv("MODEL_FILE", "Converse-1.3.Q4_K_M.gguf")
MODEL_REVISION = os.getenv("MODEL_REVISION", "3403f2f837186d6b755e2e71d2d9f25fff6a6f05")
MODEL_CONTEXT = int(os.getenv("MODEL_CONTEXT", "4096"))
MODEL_THREADS = int(os.getenv("MODEL_THREADS", str(max(1, (os.cpu_count() or 2) - 1))))
MODEL_NAME = "converse-1.3"
MAX_GENERATION_TOKENS = 2048

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("dgai")
app = FastAPI(title="dgAI Converse 1.3", docs_url=None, redoc_url=None)
model_lock = threading.Lock()
model: Any = None
model_status = "loading"
model_error: str | None = None


def load_model() -> None:
    global model, model_status, model_error
    try:
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            revision=MODEL_REVISION,
        )
        loaded_model = Llama(
            model_path=model_path,
            n_ctx=MODEL_CONTEXT,
            n_threads=MODEL_THREADS,
            n_batch=256,
            n_gpu_layers=0,
            verbose=False,
        )
        with model_lock:
            model = loaded_model
            model_status = "ready"
        logger.info("Loaded %s/%s", MODEL_REPO, MODEL_FILE)
    except Exception:
        logger.exception("Could not load the Converse 1.3 model")
        model_error = "Model loading failed. Check the service logs and model settings."
        model_status = "error"


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


@app.get("/health", include_in_schema=False)
def health() -> JSONResponse:
    status = 200 if model_status != "error" else 503
    return JSONResponse(
        status_code=status,
        content={"status": "ok" if model_status == "ready" else model_status, "model": MODEL_NAME},
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
