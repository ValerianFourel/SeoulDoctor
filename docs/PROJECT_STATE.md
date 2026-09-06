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
