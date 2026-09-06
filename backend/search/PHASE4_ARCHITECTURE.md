# Phase 4 retrieval architecture

## Problem

The live request path already compiles hard rules and builds a complete eligible
scope, but it previously sent that scope to global Chroma retrieval, in-memory
BM25, and truncated evidence search. Phase 4 must query the immutable indexes
inside the scope without moving distance, specialty, or exclusion rules into
the retriever.

## Usage

`execute_search()` builds `ScopeSelection` with the active release version and
calls `CandidateRetrievalAdapter.rank()` once. The adapter returns the same
dataframe shape used by answer generation. If the active path fails, it calls
the legacy ranker with that same complete eligible dataframe.

## Shape

[`live_retrieval.py`](live_retrieval.py) owns the temporary migration boundary:

- bind `ReadonlyIndex` to the authoritative `ScopeSelection`;
- embed the primary query once;
- search facility BM25, exact scoped dense vectors, and full indexed evidence;
- collapse evidence documents to one rank per facility;
- combine channel ranks with versioned weighted reciprocal-rank fusion;
- run at most one rule-owned lexical and evidence retry when channels disagree;
- blend fused relevance with distance after the hard radius filter;
- attach aggregate private telemetry and preserve the response dataframe shape;
- fall back atomically to the legacy ranker without changing scope.

The public operation is one `rank()` call. Channel limits and weights are
server-owned `RRFPolicy`, not request fields. Both indexed and fallback results
pass the scope assertion before serialization.

## Synthesis decision

Two designs were compared. The selected base used one deep adapter because it
keeps fallback and migration policy out of `main.py`. The second design supplied
the pure rank-fusion boundary, deterministic tie rules, explicit rejection of
out-of-scope hits, and a single-embedding requirement. The retry reuses the
primary dense result and changes only lexical and evidence wording.

We rejected separate public services for each channel because callers would
need to coordinate ordering, limits, failure behavior, and retry policy. We
also rejected putting fusion on `ScopedIndex`; the repository owns storage and
scope safety, while fusion is serving policy.

## Tradeoffs

- The adapter temporarily knows the legacy dataframe method signature so
  `main.py` does not duplicate fallback policy.
- A deterministic retry is smaller than the GPT OSS refinement controller.
  Phase 5 will own model-planned evidence requests, caching, and stop budgets.
- The initial RRF weights remain evaluation inputs. Recorded bilingual recall
  gates must approve them before the legacy path can be removed.

## Verification

Run:

```bash
PYTHONPATH=backend backend/venv/bin/python -m unittest backend.tests.test_live_retrieval
backend/venv/bin/python scripts/smoke_phase4_retrieval.py
```

The smoke command uses the active production release and its vectors without a
network call. It exercises English and Korean queries inside an actual 5 km
scope and fails if indexed retrieval falls back or makes multiple embedding
calls.
