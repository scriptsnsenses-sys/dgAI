# Converse 1.3

Converse 1.3 is a public beta release of the Converse family of lightweight conversational models by dgAI.

## Overview

- **Model family:** Converse
- **Version:** 1.3
- **Status:** Public Beta
- **Architecture:** Qwen2 (causal LM)
- **Parameters:** ~1.5B
- **Precision:** bfloat16
- **Max context length:** 32768 tokens
- **Vocabulary size:** 151936

## Files

| File | Description |
|------|-------------|
| `config.json` | Model configuration |
| `generation_config.json` | Default generation parameters |
| `model.safetensors` | Model weights (bfloat16), published at [dgAI202/Converse-1.3](https://huggingface.co/dgAI202/Converse-1.3); weights are not stored in this Git repository |
| `tokenizer.json` | Tokenizer vocabulary and merges |
| `tokenizer_config.json` | Tokenizer configuration and chat template |
| `chat_template.jinja` | Chat template (Jinja) |

## Generation defaults

- Temperature: 0.7
- Top-p: 0.8
- Top-k: 20
- Repetition penalty: 1.1
- EOS token ids: 151645, 151643

## Usage

Load with Hugging Face `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "dgAI202/Converse-1.3"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

## Deploy the Converse 1.3 service

To deploy the chat app and APIs, use the `converse-1.3` branch of this repository:

1. In Render, choose **New → Blueprint**, connect `scriptsnsenses-sys/dgAI`, and select `converse-1.3`. Use the included root-level `render.yaml`; do not choose Docker. The Blueprint installs Gunicorn and runs it with Uvicorn's ASGI worker.
2. Choose a region close to your users or colocated with services that need private networking. No region is pinned in the Blueprint.
3. To preserve the model cache between deploys, attach a **2 GB persistent disk** under the service's **Disk** settings with mount path `/var/data`, and set the service environment variable `HF_HOME=/var/data/huggingface`. The disk is a separate Render resource and may incur an extra charge. The initial download of the roughly 986 MB model still takes however long the connection requires, so a sub-minute first download is not guaranteed.
4. Deploy. `/health` reports `downloading` and then `initializing`; the service is ready after initialization. The default context is 2,048 tokens to reduce memory use and speed model initialization. For an existing manual Web Service, set `MODEL_CONTEXT=2048`, `MODEL_THREADS=1`, configure the disk and `HF_HOME` above, and use start command `gunicorn server:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --workers 1 --timeout 120`; then redeploy.

The Blueprint configures a paid Standard instance (at least 2 GB RAM) and CPU inference. Expect slower generation than on a GPU. The weights are external to this Git repository. Render's default disk is ephemeral, so attach persistent storage as described above to avoid downloading the model again after restarts.

The service provides the chat UI at `/`, usage documentation at `/docs`, OpenAI-compatible `POST /v1/chat/completions`, and Anthropic-compatible `POST /v1/messages`.

## License

MIT — see repository LICENSE.
