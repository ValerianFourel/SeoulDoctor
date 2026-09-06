# Seven-case retrieval evaluation

Run from the repository root:

```bash
python scripts/mini_retrieval_eval.py
```

The script uses the existing backend environment when invoked with system Python.
Required process credentials: HF_TOKEN and SEOULDOC_EVAL_AUTH_TOKEN. It never
loads an environment file. Default target is the private ncs Space; override
MINI_RETRIEVAL_ENDPOINT only to test another authorized compatible deployment.

Seven evaluator-only labels live at `.audit/mini-retrieval/fixtures.json`, excluded
from Git. Override MINI_RETRIEVAL_FIXTURES for a restored private fixture file.
Each stores query, location, specialty, expected facility/evidence IDs, original
source text and locator. Queries are three English, three Korean and one mixed,
without clinic names or source quotations. Preparation read at most 20 source
records from each of at most three facilities per specialty, through the existing
facility index. Seven distinct clinics, districts and specialties were selected;
no full review-corpus scan or new embeddings were used.

The token-protected `/internal/retrieval` API rejects extra fields, including
labels. It receives only query/location/specialty. It calls the same
CandidateRetrievalAdapter, scope compiler, immutable index, facility query
encoder, BGE client and configured reranker used by the app. The harness starts
at structured retrieval: it supplies the whole patient query as a comment term
rather than running the conversational model's intent extraction. It does not
measure conversation understanding, generation, translation or LLM judgment.
The API reuses app resources and an equivalent embedding client with retries
turned off. No target evidence is inserted into candidates.

Normal limits are printed from policy objects: facility channel 200, scope
comment discovery 100, shortlist 20, lexical quota 5 per constraint/facility,
CPU pool 512, initial rerank budget 224, three presented comments. The configured
reranker may impose its existing smaller cap. The evaluation disables the normal
32-candidate coverage retry, as requested; it does not increase any candidate
limit. Missing reranker or semantic retrieval remains an incomplete result.

Preparation and warmup are timed separately. The warmup encodes a non-case query
using the existing embedding model. GPU services are already warm app processes.
Four worker processes reuse HTTP sessions, each case has one request, read
requests time out after at most 30 seconds, and all workers are terminated at
the 60-second overall execution deadline. Unfinished cases remain explicit
timeouts, with ownership unknown. No automatic replay occurs.

Outputs distinguish top-five facility rank, target review in retrieval admissions,
target review among that clinic's selected card comments, and facility ownership.
Reported rank is within the normal top five; '-' means outside that window.
Returned IDs stay in the evaluator, not patient-facing markup. Totals and private
JSON checkpoints retain failures. A pass also requires complete retrieval; an
incomplete pipeline can show a hit without being called a passing case.

Status: implementation locally verified with 243 backend tests. Deployment and
seven measured cases are pending; no result is asserted yet.

## Verified execution and deployment

App source b76d29fa238a6b62fd90588158e317d7e76940b8 is RUNNING on L4 at Space
67a3cf8ab75c6cd45a49dfd4fcf5a342cc49e3c8. Website HTML and health returned 200;
Donate markup is absent. BGE readiness passed a real CUDA probe on NVIDIA L4,
1024 dense dimensions, nonempty sparse outputs, pinned model revision.

The first 16.93-second run started while GPU readiness was still 503. Its
warmup checked only the facility encoder, a harness bug. It is retained as a
diagnostic, not the warm-service measurement. The script now requires `/ready/gpu`
to return ready before any measured case; readiness failure produces blocked rows.

The corrected warm run spent 1.203 seconds in warmup and 14.61 seconds in measured
execution. Fixture loading was below 0.001 seconds; manual fixture selection was
not timed and must not be reported as that loading time. All seven source labels
were independently checked through seven indexed ID lookups for clinic ownership,
complete original text and source locator.

| Case | Language | Clinic rank in top 5 | Review hit | Selected hit | Ownership errors | Seconds | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | English | — | No | No | 0 | 1.70 | Incomplete |
| 2 | English | 2 | No | No | 0 | 1.74 | Incomplete |
| 3 | English | — | No | No | 0 | 1.77 | Incomplete |
| 4 | Korean | 2 | Yes | Yes | 0 | 12.19 | Incomplete |
| 5 | Korean | 5 | Yes | Yes | 0 | 9.00 | Incomplete |
| 6 | Korean | — | Yes | No | 0 | 12.77 | Incomplete |
| 7 | Mixed | — | Yes | No | 0 | 8.05 | Incomplete |

Totals: clinic hits 3/7, review hits 4/7, selected-comment hits 2/7, ownership
errors 0, timeouts 0. Exit 1 is a failed run. Reranking is not configured on the
Space. English cases still report semantic `request_failed` after GPU readiness;
Korean/mixed cases report semantic `ok`. This evaluation did not fix those
application/service failures, substitute a reranker, or reinterpret missing IDs
as hits. Retrieval hit means present in constraint-retrieval admissions, not
merely found during earlier full-scope facility discovery.

Both results, source-label fixtures and provenance were preserved privately at
Dataset `ValerianFourel/seouldoc-eval-handoff`, revision
`ef41a2509efd9e58a620b364285ea887e88f6add`, under
`runs/mini-retrieval-20260906T230140Z/`. The command restores only the pinned seven
fixtures if the local file is missing; it never downloads the corpus. Results
remain outside Git. No automatic case retry or target-dependent retrieval ran.

Backend discovery passed 243 tests and frontend build passed. The later runner
warmup correction passed backend discovery again. Startup-stage measurements and
optimizations remain pending in note 07. Next retrieval action: diagnose English
semantic request failures and restore an authorized reranker, then run a new
explicit comparison against these unchanged fixtures.
