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

The `converse-1.3` branch includes a Render Blueprint for a chat UI and OpenAI-/Anthropic-compatible text APIs. Connect the branch to Render and deploy `render.yaml`; the service downloads a Q4_K_M GGUF community conversion from Hugging Face on first boot. Allow several minutes for the initial download and model load. Use a paid instance with at least 2 GB RAM; CPU inference can be slow, and persistent storage is recommended to avoid downloading the model again after a restart.

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
