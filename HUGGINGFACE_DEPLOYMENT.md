# Deploy SeoulDoc to Hugging Face Spaces

This repository runs the Next.js frontend and the FastAPI API in one Docker Space on port 7860. The committed review-retrieval design adds a second private GPU Space for BGE-M3 learned-sparse and dense search.

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
SEARCH_INDEX_ROOT=/data/seouldoc/search_indexes
SEARCH_INDEX_REQUIRED=false
```

## Two-Space retrieval topology

The target production topology separates the public application from the large semantic review index:

```text
Application Space
  |-- Next.js frontend and FastAPI
  |-- geo and hard eligibility
  |-- facility retrieval
  |-- local review BM25 and original review text
  |-- RRF, evidence ownership checks, coverage, and response generation
  |
  '-- private HTTPS request with query cells and facility IDs
          |
          v
Private BGE-M3 retrieval Space
  |-- BGE-M3 query encoder on GPU
  |-- learned-sparse review index
  |-- dense review index
  '-- evidence IDs and ranks only
```

The private semantic service never controls review text, citations, or facility ownership in the application response. The application resolves returned evidence IDs against its local immutable evidence store.

The two-Space design avoids adding roughly 5 GB of BGE-M3 dense vectors, sparse weights, row maps, and model cache to every application deployment. It also lets the semantic service and application roll back independently. If the private Space sleeps or fails, local BM25 remains available.

The app-side HTTP client, configuration, live three-channel RRF fusion, BM25 fallback, and scope-bound evidence-ID resolver are implemented and unit-tested. The production private-Space runtime and Docker image also exist in `services/retriever/`. They have not yet passed dependency installation or runtime tests and have not been deployed. The corpus encoder and builder, actual 1.79-million-review release, deployment, and live evaluation remain pending. Do not enable the remote URL until its release passes the component gates in `docs/RAG_UPSTREAM_ARCHITECTURE_2026-09-04.md`.

Configure the implemented application client with:

```text
BGE_M3_RETRIEVER_URL=https://OWNER-RETRIEVER.hf.space
BGE_M3_RETRIEVER_API_TOKEN=<read token for the private Space>
BGE_M3_RETRIEVER_RELEASE_ID=<pinned semantic release>
BGE_M3_RETRIEVER_TIMEOUT_SECONDS=8
```

Leave both `BGE_M3_RETRIEVER_URL` and `BGE_M3_RETRIEVER_RELEASE_ID` empty to disable semantic retrieval. Set them together to enable it. `BGE_M3_RETRIEVER_API_TOKEN` defaults to `HF_TOKEN` when omitted. A sleeping, unavailable, invalid, or release-mismatched service degrades to local BM25 rather than failing the doctor search.

The implemented production runtime exposes:

- `GET /health` with release, model revision, review-source digest, and review count
- `POST /v1/retrieve` for one batched facility-scoped request

There is no `/info` endpoint in the production contract.

Runtime requests send user query cells and shortlisted facility IDs. They do not send stored review text. The response returns evidence IDs, facility IDs, channels, ranks, and finite scores. The application rejects the response if the semantic release and local BM25 release do not share the same raw-review SHA-256 digest.

The production runtime validates manifest schema `seouldoc.bge-m3-review-manifest/v2`, dense vectors, sparse CSR arrays, artifact hashes, model revision, facility ranges, evidence-ID uniqueness, and full expected-release request identity. It searches only the rows owned by the requested facilities and returns exact per-channel top-k results.

Configure the private runtime with:

```text
BGE_M3_MODEL_REVISION=<pinned Hugging Face commit SHA>
RETRIEVER_RELEASE_DIR=/data/release
```

If the release is not mounted, the runtime can download a pinned private Dataset revision:

```text
RETRIEVER_DATASET_REPO=OWNER/seouldoc-bge-m3-reviews
RETRIEVER_DATASET_REVISION=<pinned Dataset commit SHA>
HF_TOKEN=<read-only token>
```

The service source and image still require dependency and runtime testing. The full corpus encoder and builder do not exist yet, and no 1.79-million-review release has been produced or deployed. Build that release as an offline administrative job. Do not build or mutate the corpus index inside Space startup or a chat request.
The existing cross-encoder described below is a separate optional service. Enabling it produces a three-service deployment: the application Space, the BGE-M3 retrieval Space, and the reranker Space. BGE-M3 sparse and dense candidate generation does not require the cross-encoder to stay online.

## Optional private GPU evidence reranker

The project can call a separate private Hugging Face Space between broad
retrieval and evidence attachment. The deployed test service is
[ValerianFourel/SeoulDoctor-Reranker](https://huggingface.co/spaces/ValerianFourel/SeoulDoctor-Reranker).
Its source lives in `services/reranker/`.

The service runs `BAAI/bge-reranker-v2-m3` through Hugging Face Text
Embeddings Inference. It accepts a bilingual query and review texts at
`POST /rerank`. SeoulDoc maps the returned array indexes back to its local
`evidence_id` and `place_id`; remote response metadata cannot change evidence
ownership.

Configure the main application with:

```text
RERANKER_URL=https://valerianfourel-seouldoctor-reranker.hf.space
RERANKER_API_TOKEN=<read token for the private Space>
RERANKER_TIMEOUT_SECONDS=20
RERANKER_MAX_CANDIDATES=64
```

If `RERANKER_API_TOKEN` is absent, the client falls back to `HF_TOKEN`.
Use a read-only runtime token after deployment. Keep a write token only for
Space updates.

The current Space requests T4 Small hardware and sleeps after 15 idle minutes.
It was verified with synthetic English and Korean reviews, then paused. Sending
stored review text to the service is a separate data-processing action. Run a
real-data smoke test or evaluation only when that transfer has been approved.


## Protect provider credits

Start with a private Space, or a protected Space if your Hugging Face plan supports it. The app limits each client to 12 chat requests per 60 seconds by default, but this is not a substitute for access control. Set hard spending limits in OpenRouter and OpenAI before making the Space public. Configure OpenRouter to deny providers that retain or train on prompts when your data policy requires it.

## Persist the indexes

The app downloads the facility data and a 224 MiB raw-review snapshot on its first start. Building the 8,484-document semantic index can take more than five minutes and consumes OpenAI embedding quota.

Default Space disk is ephemeral. Attach read-write persistent storage at
`/data` so `/data/seouldoc` and the Hugging Face cache survive restarts. See the
[Hugging Face Spaces storage guide](https://huggingface.co/docs/hub/main/spaces-storage).

After the facility snapshot, review snapshot, and existing Chroma collection
are present, build and activate Phase 3 from the backend working directory:

```bash
python -m search.indexes.cli publish \
  --root /data/seouldoc/search_indexes \
  --version 2026-09-02-v1 \
  --facilities /data/seouldoc/facilities.parquet \
  --reviews /data/seouldoc/reviews.parquet \
  --chroma /data/seouldoc/chroma_db

python -m search.indexes.cli activate \
  --root /data/seouldoc/search_indexes \
  --version 2026-09-02-v1
```

The measured release uses 1.7 GB. Publication does not change the active
release. Activation validates the complete package and atomically replaces
`active.json`. Set `SEARCH_INDEX_REQUIRED=true` only after a valid active
release exists. With that setting, a missing, stale, writable, or corrupt
release prevents startup. Phase 3 reports this release in `/health` but keeps
the legacy retriever live until Phase 4.

## Verify the deployment

Build and smoke-test the image locally without loading real credentials:

```bash
docker build --tag seouldoc-hf-eval:latest .
scripts/smoke_hf_image.sh
```

The smoke command uses dummy provider values, disables container networking, mounts the local source data read-only, and removes its temporary container on success or failure. It copies Chroma into the container's temporary writable storage because Chroma performs SQLite housekeeping during startup; the source index remains read-only.
When `backend/search_indexes/active.json` exists, it also mounts that release
read-only, requires successful Phase 3 validation, and checks that `/health`
reports the active version.

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

The runner checkpoints complete English and Korean conversations under `backend/tests/evaluation_runs/`. Those files are ignored by Git and created with owner-only permissions. The independent Likert grader sends selected facility fields and review excerpts to Qwen3.8 27B through OpenRouter. Run it only with authorization for that external processing and local transcript storage.
