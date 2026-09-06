# Launch the complete SeoulDoc evaluation

This runbook prepares and launches every committed scenario family without changing a scenario after results are visible. Run each gate in order. Stop when a gate fails.

## Evaluation inventory

The launch includes four scenario families:

1. The sealed grounded suite contains eight English, Korean, and code-switch journeys in four language pairs.
2. The bilingual regression suite contains 12 English and Korean journeys. It covers specialty, location, review-language bridging, amenities, negative preferences, insufficient evidence, prompt injection, and subjective polarity.
3. The agentic stress ladder contains 10 progressive English, Korean, and mixed-language requests.
4. The randomized holdout contains six facilities with paired English and Korean cards, for 12 journeys. Regenerate it from the pinned release with seed `6688037892107667854` before patient runs.

The committed scenario definitions stay frozen during the run. Do not replace a failed case or edit a target, threshold, prompt, or oracle after the first application response.

## Required environment

Configure these secret variables in Codex Cloud:

```text
HF_TOKEN
OPENROUTER_API_KEY
SEOULDOC_EVAL_AUTH_TOKEN
```

`SEOULDOC_EVAL_AUTH_TOKEN` can contain the same credential as `HF_TOKEN`, but both names must be configured. The evaluation runners read only `SEOULDOC_EVAL_AUTH_TOKEN` when they call the private Space.

These non-secret variables are optional because the scripts contain pinned defaults:

```text
SEOULDOC_APP_ENDPOINT=https://valerianfourel-seouldoctor.hf.space/chat
SEOULDOC_RELEASE_DATASET=ValerianFourel/seouldoc-app-release-20260905
SEOULDOC_RELEASE_REVISION=3911d79dc31e6a6ccfa3f64a7e401b88893bf66a
SEOULDOC_RESULTS_DATASET=ValerianFourel/seouldoc-eval-handoff
```

Never print credential values. Never write them to prompts, logs, manifests, or result files.

## Prepare the checkout

Run the Cloud setup from the repository root:

```bash
bash scripts/setup_codex_cloud.sh
```

Run the offline launch preflight:

```bash
python scripts/prepare_eval_launch.py \
	--output .codex-handoff/results/launch-preflight.json
```

The report must set `ready` to `true`. This command checks the branch, credentials by presence only, and all 30 committed scenarios. It also records the 12-case randomized holdout plan. It does not contact the application or make paid model calls.

## Run the local gate

```bash
backend/venv/bin/python -m unittest discover -s backend/tests -p 'test_*.py'
backend/venv/bin/python -m pytest -q services/retriever
npm --prefix frontend run build
```

Do not start the live evaluation unless all three commands pass.

## Run the targeted gate

Create a new append-only directory and run the sealed Yongsan pair:

```bash
target_run="cloud-targeted-$(date -u +%Y%m%dT%H%M%SZ)"
target_dir=".codex-handoff/results/$target_run"

backend/venv/bin/python backend/tests/run_grounded_bilingual_suite.py \
	--run-id "$target_run" \
	--run-dir "$target_dir" \
	--endpoint "${SEOULDOC_APP_ENDPOINT:-https://valerianfourel-seouldoctor.hf.space/chat}" \
	--scenario reverse-yongsan-peds-03-en \
	--scenario reverse-yongsan-peds-03-ko

backend/venv/bin/python scripts/sync_eval_checkpoint.py "$target_dir"
```

Proceed only when the runner exits zero and `suite_report.json` contains `"passed": true`.

## Run all sealed grounded journeys

```bash
sealed_run="cloud-sealed-$(date -u +%Y%m%dT%H%M%SZ)"
sealed_dir=".codex-handoff/results/$sealed_run"

backend/venv/bin/python backend/tests/run_grounded_bilingual_suite.py \
	--run-id "$sealed_run" \
	--run-dir "$sealed_dir" \
	--endpoint "${SEOULDOC_APP_ENDPOINT:-https://valerianfourel-seouldoctor.hf.space/chat}"

backend/venv/bin/python scripts/sync_eval_checkpoint.py "$sealed_dir"
```

## Run the bilingual regression suite

```bash
regression_output=".codex-handoff/results/bilingual-$(date -u +%Y%m%dT%H%M%SZ).json"

backend/venv/bin/python backend/tests/run_openrouter_bilingual_eval.py \
	--endpoint "${SEOULDOC_APP_ENDPOINT:-https://valerianfourel-seouldoctor.hf.space/chat}" \
	--scenarios-file backend/tests/openrouter_bilingual_scenarios.json \
	--output "$regression_output"
```

Keep the output private and sync it as part of a uniquely named result directory.

## Run the agentic stress ladder

```bash
backend/venv/bin/python backend/tests/run_remote_agentic_stress.py \
	--endpoint "${SEOULDOC_APP_ENDPOINT:-https://valerianfourel-seouldoctor.hf.space/chat}" \
	--minimum-level 1 \
	--maximum-level 10
```

Capture standard output in the private result directory. A nonzero exit fails this gate.

## Regenerate the frozen random holdout

Use the committed seed and pinned release files. A new output directory is required.

```bash
holdout_run="random-holdout-6688037892107667854-$(date -u +%Y%m%dT%H%M%SZ)"
holdout_dir=".codex-handoff/results/$holdout_run"

python scripts/random_holdout_sampler.py \
	--facilities-parquet .codex-handoff/release/sources/facilities.parquet \
	--reviews-parquet .codex-handoff/release/sources/reviews.parquet \
	--source-revision 3911d79dc31e6a6ccfa3f64a7e401b88893bf66a \
	--output-dir "$holdout_dir" \
	--sample-size 6 \
	--seed 6688037892107667854 \
	--coordinator-model gpt-5.6-sol \
	--application-model openai/gpt-oss-120b
```

Give each patient agent only one entry from `public_casebook.json`. Keep `private_oracle.json` unavailable to patient agents. Each journey uses a unique result path. Grade all 12 journeys without replacement, then sync the redacted result directory.

## Report the release decision

Report each scenario family separately. Include the exact Git commit, application revision, Dataset revision, checkpoint commit, failures, facility ranks, presented ranks, exact-comment recall, evidence attachment, evidence ownership, grounding, and language consistency.

Do not collapse facility retrieval and exact-comment retrieval into one result. Do not report an incomplete run as a pass.
