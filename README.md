# dgAI
![Official GitHub Repository](https://img.shields.io/badge/Official-GitHub%20Repository-D4AF37?style=for-the-badge&labelColor=000000&logo=github&logoColor=white)

### Building AI. Making it accessible.

**dgAI** is an independent AI development initiative by DG focused on building accessible AI models, developer tools, and services for students, developers, and the wider community.

Our goal is simple: make AI more accessible to build with, experiment with, and use.

## 🚀 What We're Building

- **AI Models** — Developing models such as Converse, with a focus on natural dialogue and practical usability.
- **AI APIs** — Creating developer-friendly ways to integrate AI into applications and projects.
- **Developer Tools** — Building utilities and experiments that make working with AI easier.
- **Open Development** — Sharing projects, updates, and resources with the community wherever possible.

## 🎯 Our Philosophy

- **Accessibility:** AI should be easier to experiment with and build upon.
- **Practicality:** Build things people can actually use.
- **Transparency:** Communicate capabilities and limitations clearly.
- **Community:** Learn, build, experiment, and improve together.
- **Independent development:** Explore ideas beyond the boundaries of a single project.

## 🛠️ Development Status

dgAI is an evolving ecosystem, we are currently on our Converse series and are actually shipping things as fast as possible.

## What this repo contains
This repo has the code for our open weights which you can self host. Till now, there is only one model family available: Converse Family

Check out the blog about it: https://dg-ai-blue.vercel.app/introducing-converse

## Converse 1.3 service

The `converse-1.3` branch includes a Render Blueprint for a chat UI and OpenAI-/Anthropic-compatible text APIs.

### Deploy to Render

1. In Render, choose **New → Blueprint** (not Docker), connect this repository, and select the `converse-1.3` branch.
2. Render reads the root-level `render.yaml` and creates the service with its Python runtime, build/start commands, an 8 GB RAM instance, and a 5 GB persistent model-cache disk. These paid resources cost more than the smaller/ephemeral setup. Keep the root directory empty if configuring a Web Service manually.
3. Choose a region near your users, or the same region as services that need to communicate privately. The Blueprint does not pin a region.
4. Deploy and check `/health`. It reports whether the service is `downloading` or `initializing`. The first boot downloads the official `dgAI202/Converse-1.3` snapshot (about 3.1 GB, including the bfloat16 `model.safetensors` weights); a sub-minute first download cannot be guaranteed. Later restarts reuse the persistent cache.

If configuring a Web Service manually instead, select **Python**, not Docker. Use build command `pip install --upgrade pip && pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt` and start command `gunicorn server:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --workers 1 --timeout 120`. Set `MODEL_REPO=dgAI202/Converse-1.3`, `MODEL_FILE=model.safetensors`, `MODEL_REVISION=29efcd5a8fd2f55b727f06a8b636fb825bcbd4b3`, `MODEL_SIZE_BYTES=3098907121`, `MODEL_CONTEXT=32768`, `MODEL_THREADS=2`, `HF_HOME=/var/data/huggingface`, and `HF_HUB_DISABLE_XET=1`. Existing manual services must update any old GGUF-related environment values.

Inference uses the official bfloat16 Transformers weights on CPU and may be slow. For an existing **manual** Web Service, select an 8 GB RAM instance, add a 5 GB persistent disk mounted at `/var/data`, and set `MODEL_REPO=dgAI202/Converse-1.3`, `MODEL_CONTEXT=32768`, `MODEL_THREADS=2`, and `HF_HOME=/var/data/huggingface`. These are paid resources and may increase your bill.

- `/` — Converse chat
- `/docs` — API and deployment guide
- `POST /v1/chat/completions` — OpenAI-compatible chat completions
- `POST /v1/messages` — Anthropic-compatible messages
- `/health` — model loading and readiness

See the [Converse 1.3 usage guide](./static/docs.html) for request examples, limits, and deployment notes. The model weights are not stored in this Git repository.

## 🤝 Contributing

Interested in dgAI? You can help by:

- Exploring our public projects.
- Reporting bugs and suggesting improvements.
- Sharing feedback and ideas.
- Contributing code or documentation where contributions are accepted.

Please check the relevant repository's contribution guidelines before submitting changes.

## 📄 Licensing

The repo is licensed as MIT because we want that people can self host our models and even fine tune them.

## 🌐 Connect

- **Website & Blog:** https://dg-ai-blue.vercel.app/
  *More projects and updates will be added as development continues.*
  
- **Discord**: https://discord.gg/5hC8TAAzQz

## Support Us

Star this repo and show your support.

---

**dgAI.**
*Making AI more accessible*
