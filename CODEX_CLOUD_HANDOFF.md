# Codex Cloud handoff: SeoulDoc live evaluation

## Fix-inference authorization, 2026-09-09

The user explicitly authorizes reading needed backend/.env credentials into
local task processes and removing the former prohibition from AGENTS.md. Never
print credentials or include them in artifacts. This supersedes historical
process-environment-only restrictions below. Work on fix-inference from the
NCS repair checkpoint. Run real inference scenarios before publication, using
existing resources and indexes. No new paid infrastructure is authorized.

The fresh synthetic diagnostic run uses scripts/search_repair_grading_v2.json,
frozen before its scored execution. It corrects applicability for predefined
clarification, empty hard-scope, injected translation-fault and independently
verified same-language cases. Null grades remain null and never count as passed
scores. All applicable dimensions still require 4/5, every original assertion
and publication blocker stays in force, and all nine predefined English live
cases must demonstrate real translation coverage. The old v1 gate stays
incomplete. Scenario messages/oracles and sealed holdout rules are unchanged.
The runner pins the new grading protocol hash separately from the manifest and
rejects an adaptive gate with a different protocol. Root grades fixed cases
before admitting the twelve adaptive conversations. Bounds: three hours per
phase, at most 80 app and 70 actor calls per phase, actor charges at most USD2,
sequential turns, app concurrency at most2, no new compute. Application and
translation charges are recorded separately from actor costs.


## Current NCS implementation handoff, 2026-09-08

For this task, the latest user instruction overrides the historical branch,
original-Space target and credential exceptions below. Integration is `ncs`;
the only deployment target is `ValerianFourel/SeoulDoctor-ncs-retriever`.
Never modify application `main` or deploy `ValerianFourel/SeoulDoctor`.
The separate NCS Space repository may use its own `main` deployment branch.

Read the latest `docs/PROJECT_STATE.md` and `planning/hf-deployment.md` updates
before resuming. The implementation checkpoint is `b7ed376679c530243eda41b42f420964090f429f` in
`/tmp/seouldoc-answer-reliability`, branch
`local/ncs-answer-reliability-20260908`. Local verification passed; no fresh live
patient gate is claimed. The previous failed adaptive records remain intact.

The NCS push reached `67f17dc213e4acedde80940b56df39bf52a45bf0`. Deployment
workflow run `34237945900` failed at `Sync application`; public annotations show
exit code 1 and the job-log endpoint returned 403. The specific remote cause is
unconfirmed. At 14:38 UTC, NCS still ran
`b13d2bdb6eb12153a23453382244630e3746f240`, with the old source marker. Direct
deployment is blocked by absent local `HF_TOKEN`. Do not evaluate the old
runtime as though it contains this implementation.

Use `HF_TOKEN` and `OPENROUTER_API_KEY` only from approved process environment
variables. Both were absent in this task; no `.env` was read. The existing NCS
Actions workflow can supply its repository `HF_TOKEN` secret to its process if
configured, but publication does not prove that secret or deployment succeeded.

The expired adaptive budget does not renew. A proposed fresh limit of 31 actor
calls, USD 2 actor charges, 60 application calls and 60 minutes remains pending
user authorization; app model costs are separate and unmeasured. No new paid
hardware or storage is authorized. Reuse existing resources only after checking
their current state. Stop on a failed targeted gate unless a fresh exception
explicitly permits diagnostic continuation.

Prepared offline tools and frozen settings are documented in
`/tmp/seouldoc-reliability-implementation/EVAL_PREFLIGHT.md`; the v3 runner is in
`/tmp/seouldoc-adaptive-tools-v3`. It is not enabled for live dispatch. Pin the
actual application/runtime revision and fresh absolute deadline before use.
Keep original-review visibility, actor completion, patient success and exact
target-evidence success as separate gates. The frozen mean >=4, every dimension
>=3 and hard gates remain unchanged. Upload authorized redacted checkpoints to
`ValerianFourel/seouldoc-eval-handoff` only when its process credential is
available; keep uploads off the app response path.

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
  `ValerianFourel/seouldoc-eval-handoff`
- Latest checkpoint revision:
  `ea0ec8f6fb5b462df0808193a95c297691d92880`
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
SEOULDOC_RESULTS_DATASET=ValerianFourel/seouldoc-eval-handoff
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

### Current diagnostic authorization, 2026-09-06

The user explicitly permits continuing through a failed targeted check to
collect diagnostic coverage of all 20 core, 10 stress, and 12 frozen holdout
cases. This overrides the stop-on-targeted-failure launch rule below for
diagnostic runs only. Record the failed gate, preserve sealed messages and
thresholds, and never label an incomplete or failed run as passed. Adaptive
patient conversations remain a separate mode. Use new run IDs, separate
checkpoint ownership, and identify the deployment revision for every result.

The user's latest authorization lifts the earlier USD 0.80 combined cap for
necessary temporary computation. Inspect existing resources first, bound every
job's duration and concurrency, and track costs. Prefer an initial 55-minute
window and reassess actual progress before further spending. This is not
authorization for persistent hardware, storage, or unrelated computation.

Configure `BGE_M3_RETRIEVER_EXPIRES_AT` and `RERANKER_EXPIRES_AT` on the backend
with timezone-aware ISO-8601 deadlines no later than the respective job
deadlines. The clients reject requests when their timeout plus 30 seconds
would reach expiry. Empty values are reserved for endpoints without temporary
leases; omitting them on temporary endpoints does not provide lifecycle
protection. These checks do not cancel jobs or stop billing: job-level
timeouts remain mandatory. `BGE_M3_MODEL_REVISION` defaults to the audited
`5617a9f61b028005a4858fdac845db406aefb181` and must match remote results.

Credentials must be present in the agent's approved process environment.
Do not print their values or include them in checkpoints.

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
`"passed": true`. The following historical command runs the full grounded suite;
it is not required for the current incremental-improvement discussion:

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

## Current NCS deployment and gate, 2026-09-09

Remote branch `ncs` contains the conversation harness and application repair.
The deployed/tested application commit is
`b346cabe48652d6c0532ed20bfe77fc8d973e77f`; Space commit
`279f82039b4de7f624bb5ccb22598625791d44fc` is running on the existing T4
Medium resource. Exact source-marker, CUDA model probe, and live GPU retrieval
plus reranking checks pass.

Release admission did not pass. The exact fixed run at
`.audit/ncs-conversation-20260909T124250Z/fixed-b346cabe/run.json` completed
4/11 real cases and stopped after a `ReadTimeout` on the Korean `명동` spelling
case. The exact fixture run at
`.audit/ncs-conversation-20260909T124250Z/fixtures-b346cabe/run.json` completed
13/13. Isolated Codex review is in `fixed-b346cabe/reviewed-gate.json` and says
`passed:false`. The adaptive OpenRouter suite was not admitted.

The Hugging Face API is current, but `www.seouldoc.io` is separately hosted on
Vercel and still serves older review-card assets without the expandable-original
control. Vercel CLI found no cached login, and the authorized environment has no
`VERCEL_TOKEN`. A production frontend deployment remains blocked on that
credential or a manual Vercel promotion. Do not infer a site deployment from the
Hugging Face Space update.
