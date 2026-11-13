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
| `model.safetensors` | Model weights (bfloat16) |
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

model = AutoModelForCausalLM.from_pretrained("Converse 1.3")
tokenizer = AutoTokenizer.from_pretrained("Converse 1.3")
```

## License

MIT — see repository LICENSE.
