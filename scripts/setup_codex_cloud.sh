#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
backend_venv="$repo_root/backend/venv"
handoff_root="${SEOULDOC_HANDOFF_ROOT:-$repo_root/.codex-handoff}"
release_root="$handoff_root/release"
checkpoint_root="$handoff_root/checkpoints"
release_dataset="${SEOULDOC_RELEASE_DATASET:-ValerianFourel/seouldoc-app-release-20260905}"
release_revision="${SEOULDOC_RELEASE_REVISION:-3911d79dc31e6a6ccfa3f64a7e401b88893bf66a}"
results_dataset="${SEOULDOC_RESULTS_DATASET:-ValerianFourel/seouldoc-eval-handoff}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN must be configured as a Codex Cloud environment variable." >&2
  exit 2
fi

python -m venv "$backend_venv"
"$backend_venv/bin/python" -m pip install --upgrade pip
"$backend_venv/bin/python" -m pip install -r "$repo_root/backend/requirements.txt"
npm --prefix "$repo_root/frontend" ci

mkdir -p "$release_root" "$checkpoint_root"
RELEASE_ROOT="$release_root" RELEASE_DATASET="$release_dataset" RELEASE_REVISION="$release_revision" CHECKPOINT_ROOT="$checkpoint_root" RESULTS_DATASET="$results_dataset" HF_TOKEN="$HF_TOKEN" "$backend_venv/bin/python" - <<'PY'
import os

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["RELEASE_DATASET"],
    repo_type="dataset",
    revision=os.environ["RELEASE_REVISION"],
    local_dir=os.environ["RELEASE_ROOT"],
    token=os.environ["HF_TOKEN"],
)
snapshot_download(
    repo_id=os.environ["RESULTS_DATASET"],
    repo_type="dataset",
    local_dir=os.environ["CHECKPOINT_ROOT"],
    token=os.environ["HF_TOKEN"],
)
PY

required_paths=(
  "$release_root/sources/facilities.parquet"
  "$release_root/sources/reviews.parquet"
  "$release_root/chroma_db"
  "$release_root/search_indexes"
)
for required_path in "${required_paths[@]}"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Pinned release artifact is missing: $required_path" >&2
    exit 3
  fi
done

ln -sfn "$release_root/sources/facilities.parquet" "$repo_root/backend/local_facilities_cache.parquet"
ln -sfn "$release_root/sources/reviews.parquet" "$repo_root/backend/local_reviews_cache.parquet"

# Chroma opens its SQLite files in writable mode. Phase 3 rejects symlinked or
# writable index releases. Materialize each with the contract it expects.
if [[ ! -e "$repo_root/backend/chroma_db" ]]; then
  cp -a --reflink=auto "$release_root/chroma_db" "$repo_root/backend/chroma_db"
  chmod -R u+w "$repo_root/backend/chroma_db"
fi
if [[ ! -e "$repo_root/backend/search_indexes" ]]; then
  cp -al "$release_root/search_indexes" "$repo_root/backend/search_indexes"
  chmod -R a-w "$repo_root/backend/search_indexes"
fi

"$backend_venv/bin/python" -m unittest backend.tests.test_grounded_bilingual_suite backend.tests.test_grounded_journey_grader backend.tests.test_patient_journey backend.tests.test_qwen_likert_judge

echo "Codex Cloud setup complete."
echo "Pinned release: $release_dataset@$release_revision"
echo "Restored checkpoints: $results_dataset"
