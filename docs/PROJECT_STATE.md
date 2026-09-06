# SeoulDoc project state


## Verified simplification pass, 2026-09-06

Owner: side-conversation refactor session. Delivery branch:
`local/rag-visible-evidence-20260906`. Implementation commit:
`a7c26ce6159453e134611aab9400ba34a1d1ea95`.
Work was isolated on `local/debloat-shadowed-defaults` in
`/tmp/seouldoc-debloat-shadowed-defaults`, then fast-forwarded onto the delivery
branch after verification. Owned files: `backend/utils.py`,
`backend/tests/test_citywide_defaults.py`, and this coordination entry only.
Other sessions' source changes and dirty documentation were preserved.

Read-only inventory counted tracked Python, JavaScript, and TypeScript files
under backend, frontend, services, and scripts, excluding `next-env.d.ts`.
Counts include blank lines and tests, but exclude data and generated artifacts.
The initial baseline at `8c17793d0f00b6ee040185ea50d854f2c6b801d2` was
114 files and 37,474 lines. The main thread independently added two diagnostic
files with 274 lines in `903a2996b55698bfeddb61603a0a3165a1cf4a9a`.
After rebasing, the comparable baseline was 116 files and 37,748 lines.

| Metric | Rebased baseline | Refactor result |
| --- | ---: | ---: |
| Source files including tests | 116 | 117 |
| Source lines including tests | 37,748 | 37,748 |
| Lines in utils.py | 978 | 933 |
| Definitions of ensure_city_wide_defaults | 2 | 1 |
| Backend dependency declarations | 16 | 16 |
| Retriever declarations across five manifests | 25 | 25 |
| Frontend runtime dependencies | 6 | 6 |
| Frontend development dependencies | 12 | 12 |

The ranked deletion plan was:
1. Shadowed city-wide defaults definition: 45 lines, clear maintenance benefit,
   low behavior risk, cheap characterization and syntax-tree verification.
2. Repeated evaluation state defaults: possible dozens of lines, medium risk
   from runner-specific state contracts. Deferred.
3. Oversized main and retrieval modules: potentially large benefit, high risk
   and verification cost, active main-thread ownership. Deferred.
4. Dependency and compatibility removal: deletion benefit not proven, high
   deployment or privacy risk without usage evidence. Retained.

Only item 1 shipped. Python already replaced the first definition with the
second during module execution. The deleted body, docstring, six standalone
comments, and one inline comment could not affect subsequent callers.
The live function, public signature, imports, and all callers remain unchanged.
No runtime wrapper or new abstraction was introduced. The existing State shape,
Seoul-spelling behavior, coordinate truthiness, and privacy logging were retained.
The new 45-line test characterizes ten input combinations before and after
deletion. A whole-module AST comparison proved that the shadowed definition
was the only executable-code subtraction.

Focused characterization passed before and after deletion. Full backend
discovery passed 211 tests before rebase and 213 tests in 3.873 seconds after
rebase. The isolated frontend production build passed after rebase. Its initial
sandboxed attempt failed with a generic webpack error; the approved host retry
passed. Dependencies were reused, not installed or modified. No environment
files or real datasets were copied into the worktree.

Deslop and no-comments reviews were performed locally before commit. The
side-conversation prohibition on sub-agents prevented independent agent review.
No added comments, suppressions, defensive wrappers, casts, or compatibility
layers were found. No comments were restored, no extra fixes were accepted,
and no constraint encodings remain open. Existing branch changes outside this
pass were inspected only for overlap, not rewritten.

The Laziness Protocol limited this pass to one proven deletion. Subtract Before
You Add ruled out a replacement helper. Minimize Reader Load removed the
misleading second source of behavior. Model the Domain retained the existing
State object rather than adding a new representation. Prove It Works required
both characterization and AST equivalence, followed by full verification.

No sealed cases, thresholds, retrieval algorithms, public APIs, deployment
contracts, dependency manifests, credentials, or private artifacts changed.
Broader repository simplification is not complete. Remaining risks are the
deferred candidates and the absence of an independent review agent.
Next action: select another unowned, behavior-pinned simplification after the
main evaluation work, using this same narrow verification standard.

Updated: 2026-09-05

This file records the work shared between the local Codex session and Codex
Cloud. Update it before every handoff. Record observed results as verified
facts. Put untested expectations under assumptions or unresolved issues.

The consolidated [retrieval and evaluation work log](RETRIEVAL_EVAL_HANDOFF_2026-09-06.md)
records the current implementation, exact revisions, historical failures,
credential restart blocker, and remaining work. Its latest authorization summary
supersedes historical cost and diagnostic-stop restrictions below.

## Objective

Improve SeoulDoc's retrieval-augmented generation system through repeatable
English, Korean, and code-switch evaluations against the deployed private
Hugging Face application.

The immediate goal has two levels:

1. The system must return the correct facility under the requested specialty,
   location, radius, and negative constraints.
2. For review-dependent requests, the system must retrieve the selected
   facility's exact target comment, keep its evidence identity, and attach it
   to the response.

The evaluation must expose failures. Do not change targets, thresholds, or
scenario wording to make a run pass.

## Verified facts

- The coordination branch starts from code commit
  `d77ffac10639ef6c4c6f7d796b0a84ae1f84811a`.
- `backend/tests/run_grounded_bilingual_suite.py` writes a checkpoint after
  each stage. It refuses to treat partial output as a pass.
- The application model is `openai/gpt-oss-120b`. The saved full journeys are
  judged by `qwen/qwen3.8-27b`.
- The suite's default actor label is `gpt-5.6-luna-medium`. The sealed suite
  sends the scenario card messages exactly as committed.
- The private application Space is
  `ValerianFourel/SeoulDoctor`.
- The pinned release Dataset is
  `ValerianFourel/seouldoc-app-release-20260905` at revision
  `3911d79dc31e6a6ccfa3f64a7e401b88893bf66a`.
- The private result Dataset is
  `ValerianFourel/seouldoc-eval-handoff`. Its latest checkpoint
  revision is `ea0ec8f6fb5b462df0808193a95c297691d92880`.
- The release has 8,484 facilities, 8,484 facility vectors, 1,791,749 raw
  reviews, and 2,060,433 evidence rows.
- At `2026-09-05T10:22:29Z`, the local deployment checkpoint recorded a
  healthy Space, a successful BGE-M3 Yongsan component probe, and a successful
  cross-encoder component probe.
- At `2026-09-05T10:44:37Z`, the same checkpoint recorded that targeted Luna
  run `20260905-bge-m3-codeswitch-reciprocal-hf-retry1` timed out after
  600.103 seconds on turn one. The full suite was held.
- The checkpoint attributes that timeout to a non-progress cycle in
  `select_evidence_groups` after an evidence retry. It records a new regression
  passing after a local selector fix and a full backend result of 182 passing
  tests.
- The selector fix was uploaded to private Space commit
  `8de2068ab9487bc72c91b3a897ab765b6ee20d76`; the checkpoint says its rebuild
  was still awaiting a health gate.
- During this coordination inspection, the local main worktree remained on
  `codex-cloud` at `d77ffac10639ef6c4c6f7d796b0a84ae1f84811a` with uncommitted
  changes under `backend/search/` and its retrieval tests. The observed files
  include `evidence_retrieval.py`, `live_retrieval.py`,
  `test_evidence_retrieval.py`, and `test_live_retrieval.py`. Those changes are
  not part of this coordination branch.
- The two temporary T4 Small jobs were still available at the timeout. Their
  status after `2026-09-05T10:44:37Z` is not known from Git.
- No completed targeted or full end-to-end result appears after the timeout
  in local checkpoint lineage `20260905-hf-release-eval-resume1`.

## Important decisions and rationale

| Decision | Rationale |
|---|---|
| Keep hard rules server-owned. | A model may suggest evidence queries, but it must not change specialty, location, radius, prohibited rules, facility IDs, weights, or limits. |
| Search within one authoritative eligible scope. | Facility retrieval and evidence retrieval must obey the same geographic and hard-rule boundary. |
| Keep evidence identity through the response. | A review-dependent claim passes only when the evidence ID and facility ID still match the selected source comment. |
| Put new random cases in a separate holdout batch. | A recorded random sample detects overfitting without changing the regression casebook. |
| Use one branch and result path per simultaneous task. | Separate write targets prevent local and Cloud sessions from overwriting code or checkpoints. |
| Keep artifacts in private Hugging Face Datasets. | Git stays small, while Cloud can restore the pinned release and intermediate results. |

## Completed work

The code baseline includes:

- a typed search rule, scope, candidate, evidence, and turn-delta layer under
  `backend/search/`;
- versioned Phase 3 index loading and validation;
- bilingual lexical, dense, raw-comment, and BGE-M3 evidence retrieval;
- a fail-open remote cross-encoder adapter;
- facility-aware evidence attachment and source checks;
- checkpointed patient journeys, deterministic grading, and Qwen grading;
- authenticated requests to the private application endpoint without storing
  the authorization value;
- a private release Dataset, a private result Dataset, and a Cloud setup
  script.

The current deployment preparation also passed the release-mounted Docker
smoke test. The Space copied Chroma into writable storage and kept the Phase 3
release read-only.

## Prior evaluation result

The completed `20260904-gpu-crossencoder-v2` run did not pass:

- hard-gate rate: 0.125;
- scenario pass rate: 0.125;
- weighted mean: 3.556;
- language gap: 0.3375;
- reverse-target hit at 5: 0.833;
- mean reciprocal rank: 0.708.

The Mapo target comment surfaced in both Mapo journeys. The Yongsan target
facility ranked first, but its decisive comment did not appear. The current
BGE-M3 component probe now finds the Yongsan evidence in English and Korean.
That component result does not prove that the end-to-end application attaches
the comment.

## Tests actually run

These results were observed locally. Rows identify results that used a working
tree newer than the code baseline.

| Command or check | Result |
|---|---|
| `backend/venv/bin/python -m unittest discover -s backend/tests -p 'test_*.py'` | 181 tests passed on the committed baseline. The latest main-thread checkpoint reports 182 passing tests with its uncommitted selector fix. |
| `backend/venv/bin/python -m pytest -q services/retriever` | 5 tests passed. |
| `npm --prefix frontend run build` | Production build passed. |
| Focused journey and grader tests | A local checkout with its ignored transit fixture passed 36 tests. A fresh isolated coordination worktree ran the same 36 tests: 35 passed and one errored because `backend/tests/evaluation_runs/20260902-mapoderm-transit.json` is ignored and absent. |
| Private-endpoint authentication subset | The deployment checkpoint records 11 passing tests. |
| `bash -n scripts/setup_codex_cloud.sh` | Passed. |
| Coordination documentation checks | `git diff --check` and required-file and heading checks passed. |
| Python compilation of `scripts/sync_eval_checkpoint.py` | Passed. |
| Release-mounted Docker smoke | Passed with the counts listed under verified facts. |
| Live BGE-M3 and cross-encoder component probes | Passed. |

No current targeted or full end-to-end run is marked as passed. Do not infer
one from the component tests.

## Random comment-level holdout protocol

This protocol is approved as the next evaluation expansion. It is not yet
implemented as a generator.

1. Record the pinned Dataset revision, a random seed, and the sample size
   before viewing outcomes.
2. Sample facilities and review rows from the complete eligible source. Keep
   every valid sample, including difficult or unhelpful comments.
3. Record the selected `place_id` and `evidence_id` in a private oracle.
   Do not expose either ID or the facility name to the patient agent.
4. Read the sampled comment only after selection. Derive a realistic request
   from the comment, the facility specialty, and a location or travel
   constraint that can lead to that facility.
5. Create paired English and Korean cards when the comment supports both.
   Add code-switch cards as a separate stratum.
6. Freeze the generated casebook commit. Do not resample or rewrite a failed
   case during the comparison.
7. Assign one Luna patient agent to each card. Give each agent a unique result
   path. Patient agents may inspect the app response, but they may not inspect
   the private oracle.
8. Measure facility retrieval rank, presented rank, target evidence recall,
   target evidence attachment, evidence ownership, and response grounding.
   Report facility success and exact-comment success separately.
9. Save all successes and failures to the private result Dataset. A failed
   random case remains part of the holdout set.

The current source is facility-level. Until the repository proves that it has
stable doctor IDs and doctor-linked evidence, write scenarios about clinics,
hospitals, or facilities. Do not infer a specific doctor's qualifications from
a facility review.

## Parallel task ownership

| Task | Owner | Branch and write scope | Next action |
|---|---|---|---|
| Current deployed-app evaluation and selector fix | Local Codex main thread | `codex-cloud`; `backend/search/`, related retrieval tests, existing Hugging Face jobs, Space state, and local `.audit/20260905-hf-release-eval-resume1/` | Health-check Space commit `8de2068ab9487bc72c91b3a897ab765b6ee20d76`, rerun the unchanged targeted pair, then commit or checkpoint the selector fix. |
| Coordination record | This local coordination session | `coordination/codex-cloud-eval-20260905`; `AGENTS.md`, `docs/PROJECT_STATE.md`, and `docs/ENVIRONMENTS.md` only | Verify the remote tip, report its exact commit, then release this file ownership. |
| Random holdout generation | Codex Cloud after checkout | Create `cloud/random-holdout-<seed>`; own the new generator, its tests, and a new casebook file | Implement deterministic sampling and freeze the first sampled casebook before API calls. |
| Parallel patient runs | Luna Cloud agents after the casebook freezes | No tracked code edits. Each agent owns one `.codex-handoff/results/<batch>/<scenario-id>/` path | Run one card per agent and upload redacted checkpoints under distinct Dataset paths. |
| RAG fixes from measured failures | One Cloud or local implementer per failure | Create `cloud/rag-<issue>` or `local/rag-<issue>`; declare source-file ownership before editing | Reproduce one failure, change the owning module, run tests, then rerun the unchanged case. |
| Integration | User-designated local or Cloud coordinator | One integration branch at a time | Merge non-overlapping branches sequentially and reconcile this state file. |

## Unresolved issues

- The current state of the temporary GPU jobs is unknown. Query Hugging Face
  before starting replacement hardware.
- Private Space commit `8de2068ab9487bc72c91b3a897ab765b6ee20d76` has not passed its
  post-rebuild health gate in the recorded checkpoint.
- The new BGE-M3 component result has not passed the targeted end-to-end pair.
- The full sealed, regression, stress, and random suites have not run against
  the current deployed topology.
- The frozen random sample can be reproduced from the pinned release and seed,
  but automatic isolated patient execution and grading remain separate steps.
- The repository has facility-level evidence. Doctor-level identity and
  evidence ownership remain unverified.

## Assumptions

- The configured Hugging Face token can read the private Space and both private
  Datasets. Verify access during setup.
- The application endpoint remains at the URL recorded in
  `CODEX_CLOUD_HANDOFF.md`. Run the health preflight before any paid model call.

## Concrete next steps

1. Run `bash scripts/setup_codex_cloud.sh` and require a zero exit.
2. Run `python scripts/prepare_eval_launch.py` and require `"ready": true`.
3. Query the Space and GPU job status. Do not duplicate active paid jobs.
4. Run the unchanged targeted Yongsan pair.
5. Select the next diagnostic cases for the feature being improved.
6. Run the 12-case bilingual regression suite and the 10-level stress ladder.
7. Regenerate the 12-journey random holdout from the pinned seed and release.
8. Run each random card in an isolated patient context and grade every result.
9. Sync redacted checkpoints after each completed scenario or failure.
10. Create one RAG fix branch per measured failure and rerun unchanged cases.

## Handoff record

- Branch: `codex/modify-code-and-launch-evaluation`
- Integration commit: the commit containing this handoff update
- Current owner: evaluation launch coordinator
- Next owner: the Codex Cloud session that runs the gated evaluation
- Next action: run the local gate and the offline preflight, then run only the
  targeted Yongsan pair

## Launch preparation update — 2026-09-06

The integration owner merged the coordination branch into
`codex/modify-code-and-launch-evaluation`. This branch now contains the
application code, the sealed grounded runner, the 12-case bilingual regression
suite, the 10-level agentic stress ladder, and the deterministic randomized
holdout sampler.

The clean-checkout transit blocker is resolved. The recorded Kakao observation
now lives at `backend/tests/fixtures/20260902-mapoderm-transit.json`, and the
sealed casebook references that tracked fixture without weakening its hash or
route assertions.

`scripts/prepare_eval_launch.py` provides an offline, no-call preflight. It
validates the expected branch, the presence of all required credential names,
the three committed scenario inventories, and the sealed Yongsan targeted
pair. `docs/EVAL_LAUNCH_PLAN_2026-09-06.md` defines the gate order for 30
committed scenarios and the 12-journey random holdout.

No live evaluation ran during preparation. The next owner must run the local
gate and inspect current Hugging Face Space and GPU job state. Only then may the
owner run the unchanged targeted Yongsan pair. A targeted failure stops the
launch before the remaining suites.

## Live resource inspection — 2026-09-06

The private Hugging Face Space reported `RUNNING` on `cpu-basic`. Its
authenticated `/health` endpoint returned HTTP 200 with status `ok`, model
provider `openrouter`, application and agent model `openai/gpt-oss-120b`, 8,484
facilities, 8,484 vector documents, and 1,791,749 raw reviews.

The Hugging Face jobs API returned no running job in the 20 most recent jobs.
The two T4 Small jobs created on 2026-09-05 at 10:08 UTC are canceled. Do not
start replacement hardware for the evaluation suites. The targeted gate can
use the running CPU Basic Space and the configured remote model provider.

## Targeted live gate — 2026-09-06

Task owner: root evaluation coordinator. Result ownership was limited to
`.codex-handoff/results/targeted-20260906T125751Z/`. Patient agents made calls
only to the private `ValerianFourel/SeoulDoctor` Space.

The targeted gate failed and the remaining suites were not launched. The
English patient journey saved two of three required turns and remained active.
The suite runner rejected that incomplete checkpoint and did not replay it.
The Korean journey completed all three turns. Its deterministic grade failed
`retrieval_orchestration` and `retrieval_orchestration_each_turn`. The target
facility was retrieval rank 1 and presented rank 1, with its required decisive
evidence attached. Three contextual evidence IDs were missing. Qwen grading did
not run because journey integrity failed first.

The run made five application calls. No call was made after the targeted gate
failed. The redacted journeys, deterministic Korean grade, suite report, and
run summary were uploaded to private Dataset
`ValerianFourel/seouldoc-eval-handoff` at commit
`ea0ec8f6fb5b462df0808193a95c297691d92880`.

Unresolved failures:

- The English journey stopped after two successful recorded turns and cannot
  be resumed or replayed under the append-only run contract.
- The Korean journey did not satisfy the required retrieval action ordering on
  every search turn.
- The complete sealed, bilingual regression, stress, and random holdout suites
  remain blocked by the targeted gate.

The next action is to create a separate RAG fix branch, reproduce the Korean
retrieval orchestration failure without changing the sealed cards, and inspect
why the English third turn did not produce a complete checkpoint. After a fix,
run the same targeted pair with a new run ID.

## Retrieval-to-reply implementation, 2026-09-06

Owner: root local implementation session. Branch:
`local/rag-visible-evidence-20260906`, starting commit
`f8209d7d7ea6fe638dd6aa1aba316201d366f6b3`.
Owns sequential changes to the response/evidence path, service readiness,
related regression tests, verification scripts, and this coordination record.
The four pre-existing retrieval source/test modifications remain preserved;
untracked presentation and retriever draft files are outside this task.
Sealed scenario cards, targets, and thresholds remain unchanged.

The local BGE coverage audit verified 1,791,749 eligible review IDs with no
missing, duplicate, orphaned, or wrong-facility index entries. Index revision:
`a6a3ab6f70c67c15090d175efd4ecdea553b329e`; model revision:
`5617a9f61b028005a4858fdac845db406aefb181`; source revision:
`3911d79dc31e6a6ccfa3f64a7e401b88893bf66a`. Reuse these artifacts without
regeneration. The 128-token encoding truncates 57,134 source reviews; 989 rows
have empty learned sparse representations, with dense vectors present.
Saved row coverage is not evidence of live GPU readiness.

The user approved private process-only credential loading from backend/.env.
Never print, copy, commit, or upload those values. No replacement paid GPU
resources will be started under the current runbook restriction.

### Verified implementation checkpoint

Implementation commit: `af1bf165948625b2fe24983862ca416f4bb2355f`.
This integrates the pre-existing selector and ranking corrections, retaining
their related tests. Unrelated untracked workspace files remain untouched.

Local changes add full-eligible-scope lexical comment discovery before the
20-facility semantic shortlist, warning-first evidence selection, and mandatory
remote-service completion checks in the production search path. Failed retries
cannot erase an earlier reranker failure. Evidence carries its source digest.
The patient-visible renderer retains whole original comments and facility/ID
references, strips wrong-facility attachments, and replaces confident narrative
with a candidate-only response when retrieval or suitability is unresolved.
The frontend now shows that status and originals instead of labeling generated
translations as verbatim reviews. Source/test ownership includes
`frontend/components/ChatInterface.tsx`, `backend/evidence_response.py`,
`backend/search/readiness.py`, related new tests, and
`scripts/audit_visible_evidence.py` and `scripts/smoke_phase4_retrieval.py`.

The real search response path reproduced an array-valued business-hours crash;
its fix and regression pass. This is not yet traced to the historical deployed
500s. The readiness endpoint executes dense/sparse retrieval, local resolution,
and reranking. It explicitly does not verify CUDA execution. Per-stage evidence
search, reranking, and generation timings are recorded; no concurrency benchmark
has run. The old offline smoke expected the obsolete `indexed` status; it now
requires `complete` and no fallback. Sealed cases were not changed.

Verified tests: 205 backend tests, 36 focused evaluator tests, five retriever
tests, and frontend production build passed. The sandboxed retriever pytest
attempt stalled and was interrupted; the approved host run passed in 0.74 s.
Offline rerendering of the saved Yongsan EN/KO final responses made the complete
original decisive comment visible with correct ownership in both languages.
This is a rendering comparison, not a live after-evaluation or evidence that
recommendation suitability passes. No live evaluation ran during this change.

Authenticated inspection found private Space revision
`c291916971fe30c70db9f655a9fa727790b27e2e` running on CPU Basic with HTTP 200 health.
Both original GPU jobs are canceled; all 20 returned jobs are terminal. No
matching prior evaluation runner was found on the host. The credential review
accepted the user's explicit exception after clarification. No new hardware
was launched, and these code changes have not been deployed.

Remaining work: full-scope BGE discovery beyond the shortlist, explicit
supported/contradicted/unestablished semantic judgments for every requirement,
remaining constraint/refinement gaps, runtime model revision and CUDA proof,
endpoint lease handling, exact historical 500 reproduction, independent judge
repairs, and end-to-end/concurrency verification. The pipeline is not complete.
Next action: obtain a fresh bounded temporary-GPU authorization, inspect
resources again, then finish service integration and deploy for the unchanged
targeted gate. Do not launch a broader evaluation until that gate passes.

### Renewed resource authorization

The user has approved two replacement temporary T4 jobs, at most one hour
each, capped at USD 0.80 combined plus OpenRouter charges. This supersedes
the earlier no-replacement instruction for this pair only. No persistent
hardware or storage is authorized. Published T4 Small plus exposed-port cost
is USD 0.41/hour per job; use 55-minute timeouts to retain billing headroom.
Inspect active resources before launch and cancel or let both expire afterward.
Root continues ownership of the sequential service integration and evaluation.

Renewed preflight: authenticated resource inspection again found the CPU Basic
Space running at `c291916971fe30c70db9f655a9fa727790b27e2e` and all 20 returned
jobs terminal. No replacement jobs were launched. The tool approval reviewer
again rejected uploading the inspected metadata-only checkpoint, despite the
user's renewed approval, citing `.env` loading and `.codex-handoff` upload as
prohibited. The upload restriction conflicts with this runbook's explicit
redacted-checkpoint upload requirement. Resolve that execution-policy conflict
before spending the newly approved GPU budget. The prior checkpoint remains
local and no new live evaluation has started.

### Endpoint lease checks, 2026-09-06

Root retains sequential ownership on `local/rag-visible-evidence-20260906`.
Parent commit: `0b9ebffd07c23bcb39895a248e5eb2c3876c0fdb`.
This unit owns `backend/search/service_lease.py`, the two remote clients,
their configuration and main wiring, `test_service_lease.py`, and the handoff.
Both clients now refuse network calls when a configured temporary lease has
insufficient remaining time for the request plus a 30-second margin. Production
BGE responses must match the audited model revision, not only the model name.
No endpoint configuration or paid resource was changed. Expiry fields must be
set during deployment; these checks are not automatic job cancellation.

Verified: 210 backend tests passed in 4.035 seconds, 36 focused evaluator tests
passed, and the frontend production build passed. The focused client command
initially failed because `search` was absent from its Python import path;
rerunning with `PYTHONPATH=backend` passed all 16 tests. The standard full
discovery command passed without that override. Five new lease/revision tests
exercise expiry, margin, unchanged candidates, zero network calls after expiry,
and mismatched revisions. Sealed cases remain unchanged.

The latest user authorization permits broad diagnostic coverage after a failed
targeted gate, without weakening release requirements. The handoff now records
that override and the existing USD 0.80 compute cap.

Current blocker: credential-presence checks in both sandbox and approved host
processes found `HF_TOKEN`, `OPENROUTER_API_KEY`, and `SEOULDOC_EVAL_AUTH_TOKEN`
absent. No .env file was read in this unit. The current process-only credential
rule requires these variables to be injected before authenticated inspection,
deployment, model selection, uploads, or evaluation can proceed. No live calls,
new charges, deployment, or remote checkpoint upload occurred. The pipeline and
the 42-case evaluation remain unfinished. Next action: inject the three
credentials into the agent process environment, then inspect existing resources
and complete service integration before using the bounded GPU window.

### Credential and resource recheck, 2026-09-06

The user explicitly authorized loading `backend/.env` into the process for
this task and lifted the USD 0.80 cap for necessary temporary computation.
No permanent infrastructure is authorized. Root will still bound job durations
and concurrency and report costs. This supersedes the older cap, not the
requirements to inspect resources and clean up afterward.

Credential loading succeeded with all three required names present, without
printing values. Authenticated inspection found the private app still at
`c291916971fe30c70db9f655a9fa727790b27e2e`, running on CPU Basic. All 20 listed
jobs are terminal. The private results Dataset revision is
`33b85912daf0ff08f55c9990f99c2aad37048d3c`.

The execution approval reviewer rejected the metadata-only checkpoint upload
twice. The second review accepted the private upload authorization but rejected
the explicit dotenv load under its interpretation of AGENTS.md, despite the
user's specific exception. No workaround was attempted. This is an execution
approval conflict, not absent credentials or absent user authorization. No paid
jobs, app deployment, or new evaluations were launched. The user-edited
AGENTS.md was read and preserved without modification or staging.

Root also corrected the obsolete retriever guide that instructed agents to
build the already-existing corpus. Next action: resolve the execution-layer
credential restriction through approved environment injection or reviewer
policy, verify checkpoint upload, then continue service integration and live
evaluation. The broad retrieval and 42-case evaluation objectives remain open.

### Diagnostic queue ownership

Root now owns `scripts/run_parallel_diagnostics.py` and its regression tests.
The restarted session has process-environment credentials, and the prior local
checkpoint uploaded at `1eabb4c5867302670529d4204a1fbe3d6f35663b`.
A fresh targeted diagnostic run uses `diagnostic-targeted-20260906-resume2`.
The planned shared queue evaluates the current deployed revision, not the local
implementation. It explicitly labels missing GPU readiness and provisional
independent judgments. No after-deployment success can be inferred from it.

### Incremental improvement scope, 2026-09-06

The user has deferred aggregate release-gate discussion and requested a ranked
list of small product improvements. Historical results and scenario definitions
remain preserved. Current reporting focuses on observed patient-facing failures,
not a release score. Root owns this coordination entry. The explicitly requested
`incremental_report` subagent owns only `docs/EVAL_RESULTS_2026-09-06.md` for a
documentation-only cleanup. No application change, deployment, or evaluation is
authorized by this narrower reporting task alone. Existing broader permissions
are not exercised in this unit.

### Main publication preparation, 2026-09-06

The user authorized removing the remaining aggregate-check documentation
references and publishing the task code to `main`. Root owns this sequential
documentation and integration step. Historical raw results and test definitions
remain unchanged. The current [diagnostic report](EVAL_RESULTS_2026-09-06.md)
records the observed feature failures without the deferred aggregate score.
The publication includes the existing retrieval-to-reply implementation and
evaluation tools. It does not deploy the Hugging Face application.
Backend discovery passed 216 tests in 3.988 seconds and the frontend production
build passed. User changes to `AGENTS.md`, presentation files, package files,
and retriever drafts remain outside the publication.

## AI-Native Builder assessment setup, 2026-09-06

Owner: Codex assessment setup session. Branch `ncs`, isolated worktree
`/tmp/seouldoc-ncs`. Starting commit
`64e1fc2ff1d5d8d51be5cadba6e9f308b9e8ce8f`, also the observed local
`origin/main` integration reference. Owns `planning/`, the assessment section
in `README.md`, and this appended coordination entry only on `ncs`.
Original-worktree changes to this file and AGENTS.md remain outside this branch.
The user's current assessment instruction prohibits product changes, paid
compute, deployment, visibility changes, and private artifact publication.
Historical permissions above are not exercised for this task.

Setup delivers the baseline, plan, reviewer workflow, decisions, factual agent
worklog, demo outline, submission checklist, and session-export limitations.
Local isolated checks passed: 36 focused evaluator tests, frontend production
build, and setup shell syntax. No live evaluation or resource inspection ran.
Historical model/index/Dataset revisions remain unchanged and are not freshly
verified; no assessment resources or remote checkpoints were created.
Repository-only live recommendations remain blocked by private data, model API
access, and retrieval services. Public repository access is unverified.
See `planning/handoff.md` for commit resolution and remaining tasks.
Next action: select one reproducible preference-refinement journey and identify
the smallest measurable change, including a bounded synthetic fixture if needed.

## Assessment Space deployment preparation, 2026-09-06

Owner: Codex ncs deployment session. Branch `ncs`, parent
`31eca96c3919d489d979b88efbe25cb949a64df3`. Owns the new ncs sync workflow,
`scripts/sync_ncs_spaces.py`, its deployment tests, and planning updates.
The user now requests an isolated Hugging Face deployment following ncs and GPU
BGE-M3 retrieval. This supersedes setup-only scope for that deployment.
Existing application Space, main branch, and sealed evaluations stay unchanged.
Use the existing corpus embeddings; live GPU query encoding is the missing step.
Initial process checks found HF_TOKEN and OPENROUTER_API_KEY absent. No remote
inspection, creation, hardware allocation, or deployment has run in this unit.

Local verification completed: 220 backend tests passed in 4.115 seconds,
including four new deployment-boundary tests. Frontend production build passed.
Both source-manifest dry runs passed. Live provisioning and sync remain blocked
by absent credentials; no remote success is claimed.

Branch publication checkpoint: ncs commit
e0f99935ac68a57e4dea3bc36bde70834302ef1d was pushed to origin/ncs.
The initial sandbox attempt failed DNS; the approved retry succeeded.
The pre-push hook reported three nonblocking medium PII/internal findings
without details; public-submission review remains pending. No Space sync ran
and no GitHub secret or enabling variable was configured by this session.

### Authorized credential loading and live provisioning

The user explicitly authorized loading HF_TOKEN, OPENROUTER_API_KEY, and
SEOULDOC_EVAL_AUTH_TOKEN from backend/.env into this deployment process.
No values are printed or committed. The read-only inspection succeeded: original
Space 8fb1fc893eaac192bb514047d8e0577055a425df remains RUNNING on CPU Basic;
all 20 returned jobs are terminal. Pinned semantic manifest matches 1,791,749
reviews and model revision 5617a9f61b028005a4858fdac845db406aefb181.
Root additionally owns backend/restore_release.py, Dockerfile startup wiring,
retriever GPU verification, their tests, and deployment records on ncs only.

## Private ncs Spaces created, 2026-09-06

Created ValerianFourel/SeoulDoctor-ncs and
ValerianFourel/SeoulDoctor-ncs-retriever as private Docker Spaces on CPU Basic.
The user will select GPU hardware manually in the retriever Space settings.
No GPU allocation or persistent storage purchase was performed. Required
process-authorized credentials were set as Space secrets without printing them.
The original Space remains unchanged at 8fb1fc893eaac192bb514047d8e0577055a425df.

The ncs application now restores its pinned private release at startup instead
of depending on an external mount. The retriever explicitly selects CUDA and
performs a bilingual embedding probe exposed at /ready/gpu. It will not become
ready on CPU. Local checks: 222 backend tests and five retriever tests passed;
frontend production build passed. Live GPU inference and comment display are
pending the manual GPU upgrade. Automatic GitHub sync still needs NCS_HF_TOKEN
and NCS_HF_SYNC_ENABLED configured in GitHub; source upload is manual for now.

## Combined ncs GPU application, 2026-09-07

Owner: root ncs deployment session; parent dc2e2ee5fc86f5d33609f1612208206ca606eb6b.
User requests the full main application at SeoulDoctor-ncs-retriever, retaining
the GPU they enabled. Remote main was fetched and remains
64e1fc2ff1d5d8d51be5cadba6e9f308b9e8ce8f; frontend and core search match it.
Owns Dockerfile, backend/space_runtime.py and space_app.py, deployment sync and
workflow, focused tests, and planning records. No ranking/UI redesign is planned.
Live inspection proved the requested Space uses NVIDIA L4 and BGE-M3 CUDA query
inference succeeds; its root 404 is an API-only deployment. The original Space
also now uses L4, a user-managed change which this task will not modify.
The earlier ncs Dockerfile accidentally replaced HEALTHCHECK CMD and omitted
startup CMD. Correct this deployment bug with a regression check.

Combined deployment prepared and locally verified: 225 backend tests passed,
five retriever tests passed, and frontend build passed. The root routing test
proves HTML, existing API routes, and GPU readiness can coexist. User authorized
additional app keys; they were privately configured on the existing L4 Space.
The ncs sync target is now that combined Space only. Live verification pending.

## NCS review presentation ownership

Owner: side-conversation presentation session, on `ncs` at
`75a2c83bec70982f48cecdb4029823832794d1bf`.
Owns `backend/evidence_response.py`, new `backend/review_presentation.py`,
response-language/presentation wiring in `backend/main.py` and `backend/models.py`,
`frontend/components/ChatInterface.tsx`, new review display component, related
presentation tests, and `planning/02-review-presentation.md`. Current checkout
was clean at inspection. Retrieval/ranking and deployment files are outside scope.
No subagents, live model calls, deployment, or hardware changes are planned.

The user additionally assigns the translation-policy subsection in `AGENTS.md`
to this session and requires selected-review translation through Qwen-27B
(`qwen/qwen3.8-27b`), independently of the application response model.

Presentation implementation is locally verified: 233 backend tests passed,
frontend production build passed, and four synthetic browser cases passed.
Qwen translation uses `qwen/qwen3.8-27b` through a separate OpenRouter client.
Original evidence remains in the API; patient cards hide IDs and low-information
comments and show labeled translations with original toggles. No live Qwen or
BGE-M3 call ran. Details and limitations are in `planning/02-review-presentation.md`.
Next action: a bounded live translation-faithfulness check before deployment.

## Review-presentation simplification ownership

The same side-conversation owner continues on ncs at committed HEAD
`75a2c83bec70982f48cecdb4029823832794d1bf`, preserving the uncommitted formatting
implementation. This unit owns a narrow cleanup in `backend/review_presentation.py`,
the unused language import in `backend/main.py`, focused characterization tests,
and `planning/03-review-presentation-refactor.md`. No signatures, retrieval,
frontend behavior, model choice, or deployment changes are planned.

The simplification passed 20 focused tests before and after each unit, exact
before/after comparisons for 360 synthetic presentation cases and 11,172 Hangul
syllables, full discovery of 235 tests, and the frontend build. Local deslop and
comment reviews completed without subagents. No behavior change or live call
was introduced. `planning/03-review-presentation-refactor.md` records the path,
retained checks, deletions, and limits. The combined work remains uncommitted.
Next action remains a bounded live Qwen translation-faithfulness check.

## NCS session reentry handoff

Owner: side-conversation handoff session, branch `ncs`, committed HEAD
`75a2c83bec70982f48cecdb4029823832794d1bf`. This unit owns only
`planning/resume-ncs.py`, `planning/04-session-reentry.md`, and this appended
entry. Existing product edits remain uncommitted and untouched. The permanent
worktree is `/home/valerian/Seoul/SeoulDoc/.worktrees/ncs`.

The user now requests a simple translation API and local checker, superseding
the Qwen next action above. The local draft uses optional Google NMT translation;
its latest behavior still needs full verification. The missing-review report
requires distinguishing empty API evidence from hidden frontend evidence.
This handoff does not claim a fix or deployment. The latest credential check
confirmed HF_TOKEN presence but remote metadata calls failed with ConnectionError.
The user authorizes the terminal launcher to load the three named credentials
privately into a resumed Codex process. It changes no network or hardware policy.
Next action: reproduce the specialty/location-only request and inspect original
review attachment in its API response. See `planning/04-session-reentry.md`.

## Missing original comments investigation, 2026-09-07

Owner: root continuation of the presentation handoff, branch `ncs`, HEAD
`75a2c83bec70982f48cecdb4029823832794d1bf`, permanent `.worktrees/ncs` checkout.
Read the state, environments, Cloud handoff, and planning reentry script/notes.
Preserving all existing uncommitted presentation changes. This sequential fix
owns `backend/search/live_retrieval.py`, `backend/search/indexes/repository.py`,
their tests, and `planning/05-original-comment-visibility.md`. No parallel owner
is running in this session. Scope is original-comment attachment without changing
facility ranking, embedding artifacts, or preference evidence requirements.
Live source inspection: ncs Space is RUNNING at
`d765a828f621d5fb1b9273b960748e63100a8e26`. Its constraint collector matches local
code, but both new presentation modules are absent remotely. Credentials used
only from the process environment; Google translation key is absent.

## NCS Donate button, 2026-09-07

Owner: Donate-button side conversation. Branch `ncs`, code HEAD
`75a2c83bec70982f48cecdb4029823832794d1bf`. Owns only
`frontend/components/HeaderMenu.tsx` and this appended record. User requested
a minimal local edit on ncs; preserve concurrent comment work and Git state.

Removed the Donate link markup; menu and brand remain. TypeScript checking
(`tsc --noEmit --incremental false`) passed in an isolated frontend copy under
`/tmp`, and the source diff check passed. The isolated production build exited
1 with a generic webpack error and no detailed diagnostic. No backend tests or
live checks ran for this header-only edit. No commit, push, deployment, or branch
switch occurred. Next action: include the header edit in the main thread's next
verified ncs build and deployment. Main-thread build artifacts were untouched.

Verified local fix: general originals are now attached for searches without
review requirements, outside ranking and preference-coverage calculations.
Scope-checked source sampling scans at most 100 records per shortlisted facility
and displays at most three useful originals, retaining source identity. The
presentation regex was also corrected for the existing TypeScript target;
`frontend/components/ReviewEvidence.tsx` is included in this continuation's scope.
Full backend discovery passed 240 tests; frontend build and six synthetic browser
cases passed, including Korean originals without translation or presentation
metadata. The initial attachment regression failed before the fix. No live chat
or evaluation ran, and no deployment was performed. Prior uncommitted work is
preserved, with HEAD still `75a2c83bec70982f48cecdb4029823832794d1bf`.
Remaining limits: bounded first-100 source sampling, preference-dependent recall,
live translation faithfulness, and deployment of the combined uncommitted draft.
Next action: review/commit that draft and sync its exact revision to the ncs Space,
then verify a specialty/location request. See planning/05-original-comment-visibility.md.

## Seven-case mini retrieval evaluation, 2026-09-07

Owner: root, branch ncs, source commit ea515261e6e415797ca18c1426e97e2b3e1ab48a.
User authorizes a seven-case retrieval-only evaluation with a 60-second measured
deadline and at most four simultaneous requests. Owns scripts/mini_retrieval_eval.py,
backend/mini_retrieval.py, the protected Space entrypoint wiring, related tests,
and planning/06-mini-retrieval-eval.md. Expected IDs and source quotations stay
in ignored private fixtures, never in endpoint inputs. No generation, translation,
judge, browser automation, corpus rebuild, or new paid hardware in this task.

## Startup optimization brief, 2026-09-07

Root recorded the user's single-Space startup request in
planning/07-startup-optimization.md. Current ncs source is
b76d29fa238a6b62fd90588158e317d7e76940b8; uploaded Space source is
67a3cf8ab75c6cd45a49dfd4fcf5a342cc49e3c8. The earlier deployment did not become
active within the bounded wait; this is not a measured application-startup
bottleneck. Preserve committed comment/translation/header work and private
seven-case fixtures. Startup changes remain a plan, with no measured improvement.
Next action: verify the pending evaluator deployment and run its frozen cases,
then measure separate cold-start stages before implementing the startup brief.

## Ncs deployment and mini-evaluation verification

Root verified app source b76d29fa238a6b62fd90588158e317d7e76940b8 at live Space
67a3cf8ab75c6cd45a49dfd4fcf5a342cc49e3c8, RUNNING on existing L4. Website/health
returned 200, Donate markup is absent, and BGE CUDA readiness passed. No hardware,
visibility, main or original-Space changes occurred. Startup note is committed
at 22175d07; optimization itself is not implemented.

Warm seven-case retrieval run: 3/7 clinic hits, 4/7 review hits, 2/7 selected hits,
zero observed ownership errors, zero timeouts, 14.61 seconds measured execution
and 1.203 seconds warmup. All cases are incomplete: reranker unconfigured, and
English semantic requests fail. The earlier 16.93-second diagnostic lacked a GPU
warmup gate; it is retained separately and the runner is corrected. Both runs and
seven private labels are checkpointed at seouldoc-eval-handoff revision
 ef41a2509efd9e58a620b364285ea887e88f6add, runs/mini-retrieval-20260906T230140Z/.
Backend checks: 243 passed; frontend production build passed. No generation,
translation, browser automation or LLM judging was used in the evaluation.
Root owns the runner, protected API and planning records. Next action for the
latest startup brief: measure separate cold-start stages before optimizing;
retrieval follow-up remains English semantic failures and missing reranking.

## Current-changes planning note, 2026-09-07

Root added planning/08-current-changes.md as the consolidated handoff for branch
ncs at code/runner checkpoint ce7102fee1f722625a758945313760a5285b1680. This
update owns that note and this entry only. It distinguishes deployed app source
b76d29fa from later runner/documentation commits, records the verified incomplete
seven-case results and private checkpoint, and preserves startup optimization as
pending work. No new application change, evaluation or deployment ran for this
note. Next action: measure cold-start stages before implementing note 07.

## Card summaries, comment controls and Google translation, 2026-09-07

Root owns frontend/components/ChatInterface.tsx and ReviewEvidence.tsx,
backend/review_presentation.py, its tests, and planning/09-card-review-controls.md
on ncs, parent b0bbea14ab7094fe718f4d13ec4129b2bbb8c816. User requests normal
facility summaries in place of per-card candidate text, five comments at a time,
a small collapse arrow, and use of the newly provided GOOGLE_TRANSLATE_API_KEY.
The user specifically authorizes use of that key from backend/.env; load it only
inside a private process, never display or commit it. Existing global incomplete
search messaging and source evidence remain preserved. No ranking or location
logic change is part of this UI/translation task.

## Fifty-draw retrieval evaluator, 2026-09-07

Root owns scripts/mini_retrieval_eval_50.py, its focused tests, planning/10-fifty-case-retrieval.md, and this entry on ncs, parent
14a9e7c3ad92694a8a2d571c6342ab5e4138aa96. The original seven-case command,
fixtures and results remain unchanged. Fifty replacement draws are frozen from
an explicitly restricted 19-pair population: 18 unique pairs, seven clinics,
20 English/20 Korean/10 mixed. No ranking or application change belongs here.

Local verification: 252 backend tests and frontend build passed. The previously
authorized card/translation image built at Space source
2e9a6e65b1037f347791fb265cbe061ebcb070c7 (app Git source 14a9e7c).
Hugging Face reports RUNTIME_ERROR: Scheduling failure: not enough hardware
capacity. Requested hardware remains l4x1; no hardware/storage change or restart
was made. Website returned 503. Live retrieval and translated-card verification
are blocked by infrastructure scheduling, not demonstrated application startup
latency. Next action: once the existing L4 Space is RUNNING, verify its exact
runtime revision and CUDA readiness, then execute the frozen suite in a fresh
private run directory.

Evaluator implementation is committed at
2b186c6116997b6036c1c551fc51b9b7b1d35508. The bounded deployment gate produced
50 BLOCKED rows, no retrieval requests, and no measured quality/latency results.
Private fixtures/provenance/results are synced at Dataset revision
ef941a0ce929df871f635c303c82191570e85a69 under
runs/mini-retrieval-50-20260906T232853Z/. See planning/10-fifty-case-retrieval.md
for source pins, limitations, tests, reproduction command and the next action.
The earlier card/translation deployment blocker is recorded in note 09.

Final local checks: 252 backend tests, 36 focused evaluation/grader tests, five
retriever tests (bounded host retry), frontend build, shell syntax and diff
checks passed. The sandboxed retriever attempt stalled and was interrupted.
No live retrieval retry followed the blocked checkpoint.

Final evaluator code commit: 8bd1368e32b16f53c74b1460c4c79e4d5c569cd3 on ncs.
A final review added an explicit private-label check for a target selected under
the wrong clinic even when its returned place_id agrees with that wrong card.
Eight expansion tests and 253 full backend tests passed after this correction.
No further live call ran; the frozen blocked checkpoint remains unchanged.
Owner and next action remain as recorded above.
