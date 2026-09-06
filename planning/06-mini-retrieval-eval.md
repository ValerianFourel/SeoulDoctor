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
