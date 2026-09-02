# Deploy SeoulDoc to Hugging Face Spaces

This repository runs the Next.js frontend and the FastAPI API in one Docker Space on port 7860.

## Create the Space

1. Create a Hugging Face Space.
2. Select **Docker** as the SDK.
3. Clone the Space repository.
4. Copy the tracked files from this repository into the Space repository.
5. Push the files to the Space `main` branch.

The YAML block at the top of `README.md` tells Spaces to build `Dockerfile` and expose port 7860.

## Add secrets

Open the Space **Settings** page. Add these values under **Secrets**:

- `OPENROUTER_API_KEY` is required for chat, extraction, and iterative retrieval.
- `OPENAI_API_KEY` is required for `text-embedding-3-small`.
- `GOOGLE_MAPS_API_KEY` is recommended for location lookup.
- `KAKAO_REST_API_KEY` is an optional location fallback.
- `HF_TOKEN` is required only if the medical dataset is private or gated.

Do not add API keys as public Variables. Do not commit a `.env` file.

## Set optional variables

The image sets the production defaults below. Add Variables only if you need to override them.

```text
LLM_PROVIDER=openrouter
OPENROUTER_CHAT_MODEL=openai/gpt-oss-120b
OPENROUTER_AGENT_MODEL=openai/gpt-oss-120b
ENABLE_RAW_REVIEWS=true
CHAT_RATE_LIMIT_REQUESTS=12
CHAT_RATE_LIMIT_WINDOW_SECONDS=60
```

## Protect provider credits

Start with a private Space, or a protected Space if your Hugging Face plan supports it. The app limits each client to 12 chat requests per 60 seconds by default, but this is not a substitute for access control. Set hard spending limits in OpenRouter and OpenAI before making the Space public. Configure OpenRouter to deny providers that retain or train on prompts when your data policy requires it.

## Persist the indexes

The app downloads the facility data and a 224 MiB raw-review snapshot on its first start. Building the 8,484-document semantic index can take more than five minutes and consumes OpenAI embedding quota.

Default Space disk is ephemeral. Attach a read-write Hugging Face Storage Bucket at `/data` so `/data/seouldoc` and the Hugging Face cache survive restarts. See the [Hugging Face Spaces storage guide](https://huggingface.co/docs/hub/main/spaces-storage).

## Verify the deployment

Build and smoke-test the image locally without loading real credentials:

```bash
docker build --tag seouldoc-hf-eval:latest .
scripts/smoke_hf_image.sh
```

The smoke command uses dummy provider values, disables container networking, mounts the local source data read-only, and removes its temporary container on success or failure. It copies Chroma into the container's temporary writable storage because Chroma performs SQLite housekeeping during startup; the source index remains read-only.

Wait for the Space status to become **Running**. Check these paths:

- `/` loads the SeoulDoc interface.
- `/health` returns `status: ok`, `model_provider: openrouter`, and `model: openai/gpt-oss-120b`.
- `/docs` loads the FastAPI API reference.
- `/chat` accepts the request body that the frontend sends.

Run the bilingual evaluation against the deployed Space:

```bash
backend/venv/bin/python backend/tests/run_openrouter_bilingual_eval.py \
  --endpoint https://OWNER-SPACE.hf.space/chat
```

The runner checkpoints complete English and Korean conversations under `backend/tests/evaluation_runs/`. Those files are ignored by Git and created with owner-only permissions. The runner sends selected facility fields and review excerpts to Qwen Max through OpenRouter. Run it only with authorization for that external processing and local transcript storage.
