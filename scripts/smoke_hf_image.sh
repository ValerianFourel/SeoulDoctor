#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
image_name=${1:-seouldoc-hf-eval:latest}
container_name="seouldoc-hf-smoke-$$"

facilities="$repo_root/backend/local_facilities_cache.parquet"
reviews="$repo_root/backend/local_reviews_cache.parquet"
vectors="$repo_root/backend/chroma_db"
for path in "$facilities" "$reviews" "$vectors"; do
    if [ ! -e "$path" ]; then
        echo "Missing local smoke-test data: $path" >&2
        exit 2
    fi
done

cleanup() {
    docker rm --force "$container_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT HUP INT TERM

if [ "$(docker image inspect "$image_name" --format '{{.Config.User}}')" != "user" ]; then
    echo "Image must run as the non-root user named user" >&2
    exit 1
fi

docker run --detach \
    --name "$container_name" \
    --label seouldoc.purpose=offline-smoke \
    --network none \
    --env LLM_PROVIDER=openrouter \
    --env OPENROUTER_API_KEY=offline-smoke-not-a-secret \
    --env OPENAI_API_KEY=offline-smoke-not-a-secret \
    --env LLM_CHAT_MODEL=openai/gpt-oss-120b \
    --env LLM_AGENT_MODEL=openai/gpt-oss-120b \
    --env ENABLE_RETRIEVAL_DEBUG=false \
    --volume "$reviews:/data/seouldoc/reviews.parquet:ro" \
    --volume "$facilities:/data/seouldoc/facilities.parquet:ro" \
    --volume "$vectors:/smoke-input/chroma_db:ro" \
    --env CHROMA_PATH=/tmp/seouldoc-chroma \
    "$image_name" \
    sh -c 'cp -R /smoke-input/chroma_db /tmp/seouldoc-chroma && exec uvicorn space_app:app --host 0.0.0.0 --port 7860 --proxy-headers --forwarded-allow-ips "*"' \
    >/dev/null

attempt=1
while [ "$attempt" -le 60 ]; do
    if health=$(docker exec "$container_name" python -c \
        "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=5).read().decode())" \
        2>/dev/null); then
        docker exec "$container_name" python -c \
            "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=5).read(1)"
        printf '%s\n' "$health"
        echo "Hugging Face image smoke check passed; temporary container removed."
        exit 0
    fi
    sleep 2
    attempt=$((attempt + 1))
done

docker logs --tail 100 "$container_name" >&2
echo "Hugging Face image did not become healthy within 120 seconds" >&2
exit 1
