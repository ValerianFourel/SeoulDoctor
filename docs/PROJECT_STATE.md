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
  `ValerianFourel/seouldoc-eval-handoff-20260905`. Its seeded checkpoint
  revision is `6602bd2f4e84d902ee52fb4651bc845f4378d96a`.
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
  changes to `backend/search/evidence_retrieval.py` and
  `backend/tests/test_evidence_retrieval.py`. Those changes are not part of
  this coordination branch.
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
| Current deployed-app evaluation and selector fix | Local Codex main thread | `codex-cloud`; `backend/search/evidence_retrieval.py`, `backend/tests/test_evidence_retrieval.py`, existing Hugging Face jobs, Space state, and local `.audit/20260905-hf-release-eval-resume1/` | Health-check Space commit `8de2068ab9487bc72c91b3a897ab765b6ee20d76`, rerun the unchanged targeted pair, then commit or checkpoint the selector fix. |
| Coordination record | This local coordination session | `coordination/codex-cloud-eval-20260905`; `AGENTS.md`, `docs/PROJECT_STATE.md`, and `docs/ENVIRONMENTS.md` only | Push the branch and report its exact commit. |
| Random holdout generation | Codex Cloud after checkout | Create `cloud/random-holdout-<seed>`; own the new generator, its tests, and a new casebook file | Implement deterministic sampling and freeze the first sampled casebook before API calls. |
| Parallel patient runs | Luna Cloud agents after the casebook freezes | No tracked code edits. Each agent owns one `.codex-handoff/results/<batch>/<scenario-id>/` path | Run one card per agent and upload redacted checkpoints under distinct Dataset paths. |
| RAG fixes from measured failures | One Cloud or local implementer per failure | Create `cloud/rag-<issue>` or `local/rag-<issue>`; declare source-file ownership before editing | Reproduce one failure, change the owning module, run tests, then rerun the unchanged case. |
| Integration | User-designated local or Cloud coordinator | One integration branch at a time | Merge non-overlapping branches sequentially and reconcile this state file. |

## Unresolved issues

- The target Codex Cloud environment name and link were not supplied. The
  prompt contained the placeholder `[PASTE ENVIRONMENT NAME AND LINK]`.
- The current state of the temporary GPU jobs is unknown. Query Hugging Face
  before starting replacement hardware.
- Private Space commit `8de2068ab9487bc72c91b3a897ab765b6ee20d76` has not passed its
  post-rebuild health gate in the recorded checkpoint.
- The new BGE-M3 component result has not passed the targeted end-to-end pair.
  The latest attempt timed out on turn one.
- The full eight-scenario suite has not run against the new deployed topology.
- The selector fix is not committed to the GitHub code branch. Cloud must not
  edit its two owned files while the local main thread is working on them.
- A clean checkout cannot finish `scripts/setup_codex_cloud.sh`: its final
  focused test expects the ignored file
  `backend/tests/evaluation_runs/20260902-mapoderm-transit.json`, and the setup
  script does not restore that file.
- The random holdout generator and a frozen random casebook do not exist yet.
- The repository has facility-level evidence. Doctor-level identity and
  evidence ownership remain unverified.
- The local main thread and the Cloud session are not synchronized until the
  Cloud session performs the checkout and read acknowledgement described in
  `docs/ENVIRONMENTS.md`.

## Assumptions

- The selected Cloud environment will provide a Luna model and multi-agent
  execution. This was requested by the user but was not verified from the
  placeholder environment.
- The configured Hugging Face token can read the private Space and both private
  Datasets. Verify access during setup.
- The application endpoint remains at the URL recorded in
  `CODEX_CLOUD_HANDOFF.md`. Run the health preflight before any paid model
  call.

## Concrete next steps

1. Fill in the Cloud environment name and link in
   `docs/ENVIRONMENTS.md`.
2. In Codex Cloud, select
   `coordination/codex-cloud-eval-20260905` at the reported coordination
   commit.
3. Read this file and `docs/ENVIRONMENTS.md`, then report the checked-out
   commit. This completes the synchronization handshake.
4. Keep the selector files assigned to the local main thread. That owner must
   health-check Space commit `8de2068ab9487bc72c91b3a897ab765b6ee20d76`,
   rerun the unchanged targeted pair, and publish or checkpoint its code.
5. On a separate `cloud/setup-transit-fixture` branch, repair the clean-checkout
   fixture contract without skipping or weakening the transit assertion. Pin
   any private artifact and verify its hash.
6. Run `bash scripts/setup_codex_cloud.sh` and require a zero exit after that
   repair.
7. Query the Space and GPU job status. Do not duplicate active paid jobs.
8. If the targeted pair passes, run all eight sealed scenarios and Qwen
   grading.
9. On a separate branch, implement the random holdout generator and freeze a
   first batch with its seed and source revision.
10. Run the frozen random scenarios with separate Luna agents.
11. Create one RAG fix branch per measured failure. Compare each change against
    the same sealed and random cases.

## Handoff record

- Branch: `coordination/codex-cloud-eval-20260905`
- Code baseline:
  `d77ffac10639ef6c4c6f7d796b0a84ae1f84811a`
- Coordination commit: the exact branch tip reported after push
- Current owner: local coordination session
- Next owner: Codex Cloud after it checks out the reported commit and reads the
  two coordination files
- Next action: run the synchronization handshake, then inspect live resource
  status before resuming the targeted gate
