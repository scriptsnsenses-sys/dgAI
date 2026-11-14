# Converse
![Open Source](https://img.shields.io/badge/Open-Source-00C853?style=for-the-badge&labelColor=000000&logo=opensourceinitiative&logoColor=white)

Converse is a lightwieght open family of models developed by dgAI. The goal of these models is to deliver high quality conversations with users at low cost. Converse models use DG Dual Engine (Signature Version).

## Goal
To deliver high quality conversations at lowest costs possible.

## Line Up
* Converse 1 (LABS Only)
* Converse 1.2 (LABS ONLY)
* Converse 1.3 (PUBLIC BETA)

Converse family till 1.3 was a Labs only alpha family but now we are making it public with the launch of Converse 1.3.

Check out the official blog by DG (scriptsnsenses-sys):https://dg-ai-blue.vercel.app/introducing-converse

## Deploy Converse 1.3

The `converse-1.3` branch provides a Render Blueprint for the Converse chat app and compatible text APIs.

1. In Render, choose **New → Blueprint**, connect `scriptsnsenses-sys/dgAI`, and select the `converse-1.3` branch. Do not select Docker; the branch deploys with Python using the root-level `render.yaml`, which installs and runs Gunicorn with an ASGI worker.
2. Choose a region close to your users, or the same region as services that need private networking. The Blueprint does not pin a region.
3. The Blueprint uses an 8 GB RAM instance and a 5 GB persistent disk at `/var/data`, with `HF_HOME=/var/data/huggingface`, to cache the official approximately 3.1 GB snapshot from `dgAI202/Converse-1.3`. These paid resources may increase your Render bill. The first boot still downloads the files, so a sub-minute first download cannot be guaranteed.
4. Deploy; `/health` shows `downloading`, then `initializing`, before ready. For a manual Web Service, configure an 8 GB RAM instance, a 5 GB persistent disk mounted at `/var/data`, and set `MODEL_REPO=dgAI202/Converse-1.3`, `MODEL_FILE=model.safetensors`, `MODEL_REVISION=29efcd5a8fd2f55b727f06a8b636fb825bcbd4b3`, `MODEL_SIZE_BYTES=3098907121`, `MODEL_CONTEXT=2048`, `MODEL_THREADS=2`, `HF_HOME=/var/data/huggingface`, and `HF_HUB_DISABLE_XET=1`. Remove old GGUF values and redeploy.

The Blueprint uses a paid 8 GB RAM instance and persistent model-cache disk. The service runs the official bfloat16 Transformers weights on CPU, so generation may be slow. Model weights are downloaded from Hugging Face and are not stored in Git.

The deployed routes are `/` (chat), `/docs` (usage guide), `POST /v1/chat/completions` (OpenAI-compatible), and `POST /v1/messages` (Anthropic-compatible).
