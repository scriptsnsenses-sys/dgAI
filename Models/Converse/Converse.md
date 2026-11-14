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
3. To preserve model weights between deploys, add a **2 GB persistent disk** in Render's service **Disk** settings at `/var/data`, then set `HF_HOME=/var/data/huggingface`. The disk is a separate resource and may add a charge. Its first boot still downloads about 986 MB, so a sub-minute first download cannot be guaranteed.
4. Deploy; `/health` shows `downloading`, then `initializing`, before ready. The default context is 2,048 tokens to reduce memory use and initialization time. If you already created a manual Web Service, set `MODEL_CONTEXT=2048`, `MODEL_THREADS=1`, configure `HF_HOME` as above, and use start command `gunicorn server:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --workers 1 --timeout 120`; then redeploy.

The Blueprint uses a paid Standard instance with at least 2 GB RAM. The service runs inference on CPU, so generation may be slow. Model weights are downloaded from Hugging Face at startup and are not stored in Git; because Render disks are ephemeral by default, attach persistent storage if you want to avoid downloading them again after a restart.

The deployed routes are `/` (chat), `/docs` (usage guide), `POST /v1/chat/completions` (OpenAI-compatible), and `POST /v1/messages` (Anthropic-compatible).
