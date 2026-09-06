# Retrieval and evaluation work log, 2026-09-06

## Current outcome

This log describes the earlier implementation and credential-recovery session.
The subsequent diagnostic run attempted all 42 cases, completed 37 conversations,
and recorded five HTTP 500 failures. See the newer
[diagnostic report](EVAL_RESULTS_2026-09-06.md) for those results and limitations.
The intended retrieval pipeline remains unfinished.
The local implementation has passed regression tests, but it has not been
deployed. Existing review embeddings were audited and must be reused.
No new GPU jobs or live evaluation calls occurred during that earlier local
implementation and credential-recovery work.

This log consolidates the session history. The latest dated entries in
[Project state](PROJECT_STATE.md) and the current
[Cloud handoff](../CODEX_CLOUD_HANDOFF.md) remain the coordination records.
Historical results below are not results for the new local implementation.

## Code ownership and workspace

Owner: root local implementation session.
Branch: `local/rag-visible-evidence-20260906`.
Current code commit: `8c17793d0f00b6ee040185ea50d854f2c6b801d2`.

The main implementation commits are:

- `af1bf165948625b2fe24983862ca416f4bb2355f`: retrieval-to-reply changes,
  incomplete-search handling, readiness probes, tests, and verification scripts.
- `8c17793d0f00b6ee040185ea50d854f2c6b801d2`: temporary endpoint lease checks
  and strict BGE model-revision validation.

The first commit preserves and integrates pre-existing selector and ranking
corrections. Those corrections were not all authored during this session.
The workspace also contains user changes to `AGENTS.md`, untracked presentation
files, root package files, and retriever drafts. None was discarded.
Current documentation updates are uncommitted at the time of this log.

## What was implemented locally

The application now has these changes:

- Full-eligible-scope lexical comment discovery before the semantic facility
  shortlist. BGE discovery itself still operates on a shortlist.
- Warning-first evidence selection and preservation of original comment text,
  evidence identity, facility ownership, and the review-source digest.
- Production completion checks that require remote retrieval and reranking.
  A failed retry cannot erase an earlier service failure.
- A patient-visible evidence renderer that preserves whole original comments
  and source IDs, removes wrong-facility attachments, and replaces confident
  prose with a candidate-only notice when retrieval or suitability is unresolved.
- Frontend status notices and original comments instead of generated
  translations presented as verbatim reviews.
- A fix for an array-valued business-hours serialization crash. The historical
  deployed amenities failures have not yet been traced to this exact cause.
- A retrieval readiness endpoint that executes semantic retrieval, local ID
  resolution, and reranking. It explicitly does not prove CUDA execution.
- Separate evidence-search, reranking, and answer-generation timing fields.
- Client lease checks that reject requests when a configured service deadline
  cannot accommodate the request timeout plus a 30-second margin.
- Strict production BGE model-revision checks against the audited revision.

Lease checks depend on deployment setting `BGE_M3_RETRIEVER_EXPIRES_AT` and
`RERANKER_EXPIRES_AT`. They do not cancel jobs or stop billing. Job-level
timeouts and cleanup remain required.

The implementation includes `scripts/audit_visible_evidence.py` for objective
visibility checks. That script does not claim to judge translation fidelity or
recommendation suitability. The offline facility smoke script was corrected
to require the current `complete` status rather than an obsolete status name.

## Verified local checks

The latest application-code verification recorded:

| Check | Observed result |
| --- | --- |
| Full backend unittest discovery | 210 passed in 4.035 seconds |
| Focused journey and grader tests | 36 passed |
| Focused lease, semantic client, reranker, and completion tests | 16 passed with `PYTHONPATH=backend` |
| Retriever service tests from the prior implementation checkpoint | 5 passed in the approved host run |
| Frontend production build | Passed |
| Sealed scenario messages and thresholds | Unchanged by this implementation |

The focused client command first failed on its import path, then passed with
`PYTHONPATH=backend`. Full discovery passed without that override. An earlier
sandboxed retriever test process stalled and was interrupted. The host rerun
passed. No browser-based frontend verification has been completed.

Offline rerendering of the saved Yongsan final responses made the complete
decisive original comment visible with correct ownership in English and Korean.
That was not a fresh retrieval or live application run. Correct recommendation
interpretation still requires independent verification.

An offline facility search measured about 30.05 ms in English and 27.50 ms in
Korean within a 979-facility scope. It used a precomputed facility probe vector.
Those numbers do not measure live BGE query encoding, chat latency, or concurrency.

## Existing embeddings and coverage

The recorded audit verified these revisions:

| Artifact | Exact revision |
| --- | --- |
| `ValerianFourel/seouldoc-bge-m3-review-index` | `a6a3ab6f70c67c15090d175efd4ecdea553b329e` |
| `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` |
| `ValerianFourel/seouldoc-app-release-20260905` | `3911d79dc31e6a6ccfa3f64a7e401b88893bf66a` |

Source review SHA-256:
`0e8c6f4ff09b75becf7821ba08bf8b19b6533b824a26ca4e7c6451277dfbfafa`.

The source contains 1,805,211 rows, of which 13,462 are blank exclusions.
All 1,791,749 eligible review IDs have dense vectors. The audit found no
missing, duplicate, orphaned, wrong-facility, or builder-to-app ID mismatches.
Dense vectors have 1,024 dimensions and finite values. Artifact hashes matched.

There are 989 empty learned-sparse rows, with dense vectors present. The
128-token encoding limit truncates 57,134 original reviews. These limitations
make full-text lexical retrieval important but do not justify rebuilding the
existing corpus to restore a stopped service.

The audited raw-review ID is derived from facility ID, review index, and
stripped original text. The holdout sampler used a different ID format.
An explicit evaluator mapping remains necessary. Facility-vector counts alone
are not evidence of comment-level embedding coverage.

## Historical evaluation findings

Earlier evaluations used a deployed application with failed remote retrieval
and reranking calls. Their results describe that degraded deployment.

- The adaptive Yongsan pilot completed two three-turn conversations. English
  recommended the clinic without adequately disclosing the nurse conflict.
  Korean qualified or rejected it more appropriately. A confident judge score
  did not reliably reflect this distinction.
- A Korean-to-English case missed the decisive target comment.
- The 12-case regression run recorded five application errors. Patient agents
  stopping as satisfied did not establish passes. Some judge citations failed
  validation.
- The stress runner reported six passes and four failures. Its native checks
  and timings need review before treating them as recommendation-quality or
  concurrency measurements.
- The frozen 12-case holdout was only partially executed across the original
  run and a continuation. It must not be reported as 12 completed cases.

An older completed run, `20260904-gpu-crossencoder-v2`, had a 0.125 hard-gate
rate and weighted mean 3.556. Different run revisions and modes must remain
separate in any before-and-after comparison.

Important evaluator limitations include early-stopping patients, unsupported
judge explanations, invalid citations, evidence-ID mismatches, and rewarding
target presentation without distinguishing endorsement from rejection.
Internal attachment alone does not prove patient-visible evidence use.

## Checkpoints retained locally

These ignored directories contain the relevant artifacts, not source code:

- `.codex-handoff/results/bge-coverage-audit-20260906T191212Z-1e44c8/`
- `.codex-handoff/results/retrieval-evidence-checkpoint-20260906T194207Z/`
- `.codex-handoff/results/adaptive-bilingual-pilot-20260906T181638Z-fcaef1/`
- `.codex-handoff/results/all-eval-20260906T182845Z-b50fd4/`
- `.codex-handoff/results/holdout-parallel-continuation-20260906T185708Z-3802e0/`

The frozen holdout seed is `6688037892107667854`. Its existing public cards
and private oracle must remain frozen and separate. No checkpoint was discarded
or silently replayed during the recent implementation work.

The metadata-only local implementation checkpoint has not been uploaded.
Earlier evaluation uploads do not imply that this newer checkpoint is remote.

## Last observed resources and costs

Authenticated inspection found private Space `ValerianFourel/SeoulDoctor`
running on CPU Basic at revision
`c291916971fe30c70db9f655a9fa727790b27e2e`. This is not the local code revision.
All 20 jobs returned by the inspection were terminal. The original retrieval
and reranker jobs were canceled.

The private results Dataset last reported revision
`33b85912daf0ff08f55c9990f99c2aad37048d3c`.
These are observations, not guarantees of current remote state after restart.

Recent local implementation and recovery attempts incurred no new paid jobs
or evaluation calls. The earlier adaptive pilot recorded about USD 0.014215
for patient and judge calls, excluding application-model charges. A complete
cost accounting for historical runs has not been established.

The user lifted the old USD 0.80 cap for necessary temporary computation.
Authorization still excludes permanent hardware, persistent storage, and
unrelated spending. Bounded durations, concurrency, cost tracking, resource
inspection, and cleanup remain required.

The user also permits diagnostic continuation through failed targeted checks
for all 20 core, 10 stress, and 12 frozen holdout cases. That permission does
not waive release gates or allow failed and incomplete cases to count as passes.

## Credential blocker and restart state

The initial shell failure involved `bwrap` and loopback network permissions.
After the user's host-side repair, shell execution and patching worked.

Initially the agent process lacked `HF_TOKEN`, `OPENROUTER_API_KEY`, and
`SEOULDOC_EVAL_AUTH_TOKEN`. The user then explicitly authorized reading
`backend/.env` into a process for this task. A presence-only check confirmed all
three keys, and authenticated read-only Hugging Face inspection succeeded.
No credential values were printed or added to artifacts.

The tool approval layer nevertheless rejected the checkpoint-upload command
because it loaded `.env`. On reconsideration, it accepted the upload scope but
still rejected dotenv loading. This is an execution-policy conflict, not absent
keys, missing user authorization, or a failed Hugging Face authentication check.
No indirect execution workaround was attempted.

The user received a terminal command that loads only the three required keys
with `python-dotenv`, starts `codex resume`, and configures environment inheritance
with an explicit allowlist. That restart has not been verified. Starting a child
Codex process inside this session cannot change this session's parent environment.
A shared app-server process may also require a restart if it retains the old
environment. Sandbox and approval protections must remain enabled.

## Remaining work in dependency order

1. Restore approved process-environment credential availability after restart.
   Verify presence only, authenticated resource access, and redacted checkpoint
   upload before consuming paid GPU time.
2. Inspect active processes and remote resources. Preserve partial conversations
   and assign new run IDs rather than silently replaying old ones.
3. Complete full-eligible-scope dense and sparse BGE discovery, then merge with
   lexical results and rerank without losing requirement coverage or ownership.
4. Add explicit supported, contradicted, and unestablished requirement
   assessments. Similarity scores and risk-query matches are not semantic proof.
   Verify constraint replacement and English, Korean, and code-switch equivalence.
5. Finish authenticated service deployment, exact runtime model pinning,
   actual CUDA inference checks, index readiness, and job lifecycle handling.
   The current web-health checks do not prove GPU inference.
6. Reproduce historical amenities and insufficient-evidence failures from saved
   requests and tracebacks. Verify negative-review warnings and incomplete-search
   behavior in the deployed application, not just the local renderer.
7. Repair evaluator integrity checks and holdout ID mapping. Verify current
   OpenRouter model availability, conduct bounded model pilots, and freeze model
   choices and grading prompts before scored runs. Independent second review is
   required for disputed or consequential decisions.
8. Finish the shared queue for all scenario families. Keep conversation turns
   sequential and run independent scenarios and completed-conversation judges
   concurrently. Separate app, model, and GPU limits. Keep uploads asynchronous.
9. Rerun required tests, deploy, verify component operations, and run the targeted
   gate. Continue diagnostic coverage after failures only with those failures
   preserved. Adaptive conversations remain a separately identified mode.
10. Benchmark concurrency under measured latency, error rates, and capacity.
    Complete all 42 cases and objective evidence checks, upload redacted results,
    report exact revisions and costs, and verify resource cleanup.

The final report must distinguish retrieval, selection, ownership, visible
quotation or translation, faithfulness, and the evidence's effect on the
recommendation. Fluent answers, attached trace evidence, or unsupported numeric
judge scores do not establish success.
