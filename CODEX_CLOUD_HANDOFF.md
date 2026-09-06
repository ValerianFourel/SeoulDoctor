# Codex Cloud handoff: SeoulDoc live evaluation

## Goal

Resume the checkpointed bilingual SeoulDoc release evaluation from a Codex
Cloud checkout. A Luna agent operates the public scenario cards, the app runs
on the private Hugging Face Space, and Qwen3.8 27B grades the complete saved
journeys. Run the targeted gate first and the full suite only after it passes.

## Pinned resources

- GitHub repository: `ValerianFourel/SeoulDoctor`
- Working branch: `codex/modify-code-and-launch-evaluation`
- Application Space: `ValerianFourel/SeoulDoctor` (private, CPU Basic)
- Chat endpoint: `https://valerianfourel-seouldoctor.hf.space/chat`
- Release Dataset: `ValerianFourel/seouldoc-app-release-20260905`
- Release Dataset revision:
  `3911d79dc31e6a6ccfa3f64a7e401b88893bf66a`
- Verified Space source revision:
  `dfab9c80082a2b274916edc8d5ea40ed95e3b8a2`
- Private checkpoint Dataset:
  `ValerianFourel/seouldoc-eval-handoff-20260905`
- Seed checkpoint revision:
  `6602bd2f4e84d902ee52fb4651bc845f4378d96a`
- Local run lineage: `20260905-hf-release-eval-resume1`

The release Dataset contains:

```text
sources/facilities.parquet
sources/reviews.parquet
chroma_db/
search_indexes/
```

The last verified app health reported 8,484 facilities, 8,484 vector
documents, 1,791,749 raw reviews, Phase 3 version `2026-09-02-v1`, and
2,060,433 evidence rows.

## Codex Cloud environment

Use the `universal` container image and manual setup:

```bash
bash scripts/setup_codex_cloud.sh
```

Leave the maintenance script blank. Enable agent internet access for:

- `huggingface.co`
- `hf.co`
- `*.hf.space`
- `openrouter.ai`

Add the following in Codex Cloud as **Environment variables**, because the
agent needs them after setup:

```text
HF_TOKEN
OPENROUTER_API_KEY
SEOULDOC_EVAL_AUTH_TOKEN
SEOULDOC_APP_ENDPOINT
SEOULDOC_RELEASE_DATASET
SEOULDOC_RELEASE_REVISION
SEOULDOC_RESULTS_DATASET
```

Recommended non-secret values:

```text
SEOULDOC_APP_ENDPOINT=https://valerianfourel-seouldoctor.hf.space/chat
SEOULDOC_RELEASE_DATASET=ValerianFourel/seouldoc-app-release-20260905
SEOULDOC_RELEASE_REVISION=3911d79dc31e6a6ccfa3f64a7e401b88893bf66a
SEOULDOC_RESULTS_DATASET=ValerianFourel/seouldoc-eval-handoff-20260905
```

`SEOULDOC_EVAL_AUTH_TOKEN` must be a token that can call the private Space.
It may have the same value as `HF_TOKEN`. Do not put any value from the local
`.env` into Git, this document, a prompt, a log, or an evaluation artifact.

## Verified resume point

The original local audit checkpoint recorded:

- 178 backend tests passed before deployment.
- Local Docker build and release-mounted image smoke passed.
- The private Space was created without persistent storage.
- The 2.275 GB release was uploaded to the pinned private Dataset.
- The source-only Space mounted that Dataset and reached healthy status.
- Both BGE-M3 retriever and cross-encoder component probes passed.
- Two one-hour T4 Small jobs were launched for the retriever and reranker.

Those GPU jobs are temporary. Inspect their status before doing anything else.
Do not launch duplicates while either job is still available. Endpoint URLs and
tokens are operational state and must not be committed.

The setup script restores the latest private checkpoint Dataset into
`.codex-handoff/checkpoints/`. The seeded run is available at
`.codex-handoff/checkpoints/runs/20260905-hf-release-eval-resume1/`.

## Run protocol

From the repository root, verify the checkout and app health first:

```bash
git branch --show-current
backend/venv/bin/python -m unittest backend.tests.test_grounded_bilingual_suite backend.tests.test_grounded_journey_grader backend.tests.test_patient_journey backend.tests.test_qwen_likert_judge
```

Use a fresh run directory for the targeted Yongsan pair:

```bash
target_run="cloud-targeted-$(date -u +%Y%m%dT%H%M%SZ)"
target_dir=".codex-handoff/results/$target_run"

backend/venv/bin/python backend/tests/run_grounded_bilingual_suite.py --run-id "$target_run" --run-dir "$target_dir" --endpoint "$SEOULDOC_APP_ENDPOINT" --scenario reverse-yongsan-peds-03-en --scenario reverse-yongsan-peds-03-ko

backend/venv/bin/python scripts/sync_eval_checkpoint.py "$target_dir"
```

Proceed only when that command exits zero and `suite_report.json` says
`"passed": true`. Then run all eight scenarios with Qwen judging:

```bash
full_run="cloud-full-$(date -u +%Y%m%dT%H%M%SZ)"
full_dir=".codex-handoff/results/$full_run"

backend/venv/bin/python backend/tests/run_grounded_bilingual_suite.py --run-id "$full_run" --run-dir "$full_dir" --endpoint "$SEOULDOC_APP_ENDPOINT"

backend/venv/bin/python scripts/sync_eval_checkpoint.py "$full_dir"
```

The suite command is deliberately rerunnable against the same directory: it
reuses complete journeys and Qwen reviews. It refuses to silently replay an
incomplete journey. If interrupted, upload the current directory, diagnose the
checkpoint, and resume only when its existing files are internally consistent.

## Release decision

Report the deterministic hard gates, Qwen scores, pair gates, suite metrics,
failed scenario IDs, exact Git commit, Space revision, Dataset revision, and
checkpoint Dataset commit. Never summarize a partial or errored run as a pass.
Code changes belong on `codex/modify-code-and-launch-evaluation` and should return through a reviewed
GitHub diff or pull request.
