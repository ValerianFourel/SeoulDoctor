# SeoulDoc project state

Updated: 2026-09-05

This file records the work shared between the local Codex session and Codex
Cloud. Update it before every handoff. Record observed results as verified
facts. Put untested expectations under assumptions or unresolved issues.

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
- The repository contains eight sealed scenarios in
  `backend/tests/grounded_bilingual_scenarios.json`. They form four English,
  Korean, or code-switch pairs.
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
| Gate the full run behind a targeted run. | This limits paid calls and gives a clear failure point before all eight journeys and Qwen grading. |
| Keep the existing eight scenarios sealed. | Stable cases make before-and-after RAG comparisons meaningful. |
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
5. If the targeted pair passes, run the eight sealed journeys.
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
