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
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "dgAI202/Converse-1.3"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="cpu",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

## Deploy the Converse 1.3 service

To deploy the chat app and APIs, use the `converse-1.3` branch of this repository:

1. In Render, choose **New → Blueprint**, connect `scriptsnsenses-sys/dgAI`, and select `converse-1.3`. Use the included root-level `render.yaml`; do not choose Docker. The Blueprint installs Gunicorn and runs it with Uvicorn's ASGI worker.
2. Choose a region close to your users or colocated with services that need private networking. No region is pinned in the Blueprint.
3. The Blueprint uses an 8 GB RAM instance and a 5 GB persistent disk mounted at `/var/data`, with `HF_HOME=/var/data/huggingface`, so restarts can reuse the official approximately 3.1 GB snapshot. These paid resources may increase your Render bill. The first download still takes as long as the network requires; under a minute is not guaranteed.
4. Deploy. `/health` reports `downloading` and then `initializing`; the service is ready after initialization. The default context is 32,768 tokens. For a manual Web Service, use an 8 GB RAM instance, attach a 5 GB disk mounted at `/var/data`, and set `MODEL_REPO=dgAI202/Converse-1.3`, `MODEL_FILE=model.safetensors`, `MODEL_REVISION=29efcd5a8fd2f55b727f06a8b636fb825bcbd4b3`, `MODEL_SIZE_BYTES=3098907121`, `HF_HOME=/var/data/huggingface`, `HF_HUB_DISABLE_XET=1`, `MODEL_CONTEXT=32768`, and `MODEL_THREADS=2`. Use build command `pip install --upgrade pip && pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt` and start command `gunicorn server:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --workers 1 --timeout 120`; then redeploy.

The Blueprint configures a paid 8 GB RAM instance and persistent model-cache disk, and loads the official bfloat16 Transformers weights on CPU. Expect slower generation than on a GPU. The weights are external to this Git repository. The persistent disk preserves the cache across restarts.

The service provides the chat UI at `/`, usage documentation at `/docs`, OpenAI-compatible `POST /v1/chat/completions`, and Anthropic-compatible `POST /v1/messages`.

## License

MIT — see repository LICENSE.
