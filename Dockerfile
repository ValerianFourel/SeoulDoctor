FROM node:20-bookworm-slim@sha256:2cf067cfed83d5ea958367df9f966191a942351a2df77d6f0193e162b5febfc0 AS frontend-builder

WORKDIR /build/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build


FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    LLM_PROVIDER=openrouter \
    OPENROUTER_CHAT_MODEL=openai/gpt-oss-120b \
    OPENROUTER_AGENT_MODEL=openai/gpt-oss-120b \
    ENABLE_RAW_REVIEWS=true \
    CHAT_RATE_LIMIT_REQUESTS=12 \
    CHAT_RATE_LIMIT_WINDOW_SECONDS=60 \
    HF_HOME=/data/.huggingface \
    FACILITIES_CACHE_PATH=/data/seouldoc/facilities.parquet \
    RAW_REVIEWS_PATH=/data/seouldoc/reviews.parquet \
    CHROMA_PATH=/data/seouldoc/chroma_db \
    SEARCH_INDEX_ROOT=/data/seouldoc/search_indexes \
    SEARCH_INDEX_REQUIRED=false \
    FRONTEND_STATIC_DIR=/home/user/app/frontend

RUN useradd --create-home --uid 1000 user \
    && mkdir -p /home/user/app/backend /home/user/app/frontend /data/seouldoc \
    && chown -R user:user /home/user /data

WORKDIR /home/user/app/backend

COPY backend/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY --chown=user:user backend/*.py ./
COPY --chown=user:user backend/search ./search
COPY --from=frontend-builder --chown=user:user /build/frontend/out /home/user/app/frontend

USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=10m --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=8)"]

CMD ["sh", "-c", "set -eu; release=/mnt/seouldoc-release; marker=/data/seouldoc/.release-20260905-ready; if [ ! -f \"$marker\" ]; then mkdir -p /data/seouldoc/chroma_db /data/seouldoc/search_indexes; chmod -R u+w /data/seouldoc/search_indexes; cp \"$release/sources/facilities.parquet\" /data/seouldoc/facilities.parquet; cp \"$release/sources/reviews.parquet\" /data/seouldoc/reviews.parquet; cp -R \"$release/chroma_db/.\" /data/seouldoc/chroma_db/; chmod -R u+w /data/seouldoc/chroma_db; cp -R \"$release/search_indexes/.\" /data/seouldoc/search_indexes/; chmod -R a-w /data/seouldoc/search_indexes; touch \"$marker\"; fi; exec uvicorn space_app:app --host 0.0.0.0 --port 7860 --proxy-headers --forwarded-allow-ips \"*\""]
