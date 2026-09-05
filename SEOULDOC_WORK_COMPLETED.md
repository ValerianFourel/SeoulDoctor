# SeoulDoc implementation and evaluation record

Date: 2026-09-04

This document records the work requested and completed during the SeoulDoc
build, RAG redesign, Hugging Face packaging, and bilingual evaluation work. It
distinguishes implemented behavior from planned or still-failing behavior.

## 1. Product goal

SeoulDoc helps a user find a medical facility in Seoul from English, Korean,
or code-switched conversation. The LLM acts as a language bridge and query
planner. The server remains responsible for location, distance, eligibility,
retrieval scope, evidence identity, and response safety.

The intended search contract is:

1. Understand the user’s language and medical intent.
2. Preserve constraints across turns.
3. Use a 5 km radius when the user does not specify a distance.
4. Treat an explicit radius as a hard search boundary.
5. Search Korean and English evidence for the same request.
6. Return facilities with attached, source-linked review evidence.
7. Refine retrieval iteratively until the request is covered or the bounded
   retry budget is exhausted.

## 2. Requests and decisions captured

The conversation established these project decisions:

- OpenRouter is the LLM provider.
- The serving model is `openai/gpt-oss-120b`.
- The grading model is `qwen/qwen3.8-27b`.
- Luna is used as the patient/scenario actor for the recorded evaluation.
- Sol was used for evaluation scenario and success-condition design.
- English, Korean, and mixed-language comments must be reachable through one
  bilingual retrieval policy.
- BM25 and dense retrieval are both used at broad candidate-recall stage.
- A cross-encoder is reserved for final fine-grained reranking in the Hugging
  Face deployment.
- User distance is explicit when supplied and otherwise defaults to 5 km.
- Evaluation scenarios cover radius changes, Seoul map coordinates, transit
  distance, refinement, code switching, evidence wording, and random/seeded
  doctor review targeting.
- The deployment target changed from Render to a Hugging Face Docker Space.
- Evaluation conversations and grader outputs are saved as audit artifacts.

No API key or `.env` value is recorded in this file. Hugging Face secrets are
configured through Space Settings, not committed to Git.

## 3. Chronological implementation record

### Foundation and provider setup

The app was moved to an OpenRouter-compatible configuration. Environment
variables and deployment documentation describe the provider, chat model,
agent model, embedding key, geocoding keys, raw-review access, and index root.
The production default is `openai/gpt-oss-120b`.

The project was packaged as a single Docker-based Hugging Face Space with the
Next.js frontend and FastAPI backend exposed on port 7860. The deployment guide
covers secrets, persistent `/data` storage, index publication, startup
validation, health checks, and the local Docker smoke test.

### Evaluation design

The evaluation work added bilingual scenario definitions, a patient journey
runner, deterministic journey grading, an independent Qwen Likert judge, and
privacy-safe evidence citation catalogs. Scenarios test:

- implicit 5 km defaults and explicit radius expansion;
- English and Korean equivalents;
- Mapo dermatology and Yongsan pediatrics reverse-target retrieval;
- required review evidence and contextual review evidence;
- preserved specialty/location/radius across refinement turns;
- removal of one criterion without deleting unrelated criteria;
- English-to-Korean and Korean-to-English code switching;
- facility coordinates and real Seoul map/transit checks.

The evaluator separates deterministic gates from model judgment. The
deterministic layer checks state, scope, distance, retrieval methods, trace
ordering, evidence identity, language, and transport receipts. Qwen3.8 27B
grades intent state, refinement, orchestration, evidence ranking,
geography/transport, and language/medical safety on a Likert scale.

### Typed bilingual search state

The new `backend/search/` package introduces server-owned contracts for:

- search rules and evidence requirements;
- geographic scope selection;
- turn deltas and a deterministic reducer;
- facility and evidence retrieval hits;
- immutable index manifests and read-only repositories;
- weighted reciprocal-rank fusion;
- retrieval telemetry and bounded retries.

The LLM now proposes a turn delta rather than rewriting the entire state.
Deterministic safeguards compile English, Korean, and code-switched phrases for
distance, specialties, diseases, comment qualities, hours, and removals.
Station/map-pin coordinates remain point-radius searches even when reverse
geocoding also supplies a district. A local search defaults to 5 km unless the
user gives another radius.

### Immutable indexed retrieval

The Phase 3 index builder publishes source-hash-bound, immutable artifacts for
facility vectors, facility lexical search, structured evidence, and verbatim
reviews. The active release is opened read-only and validated against the
mounted facility and review snapshots at startup.

The measured release contains:

- 8,484 facilities;
- 1,791,749 nonempty verbatim reviews;
- 268,684 facility fact and summary records;
- 2,060,433 total evidence records;
- approximately 1.7 GB on disk.

The serving adapter binds retrieval to the complete eligible facility scope,
then runs facility BM25, facility dense retrieval, structured evidence BM25,
and multilingual verbatim-comment search. Weighted RRF fuses the channels.
One deterministic retry searches unmet evidence terms and reuses the primary
embedding. Detailed retrieval receipts stay behind
`ENABLE_RETRIEVAL_DEBUG`.

### Hugging Face packaging

The repository includes `Dockerfile`, `backend/space_app.py`, Space health
metadata, a Docker smoke script, and `HUGGINGFACE_DEPLOYMENT.md`. The Space
expects secrets such as `OPENROUTER_API_KEY` and `OPENAI_API_KEY` in Hugging
Face Settings. The index can be persisted under `/data/seouldoc` and activated
atomically. `SEARCH_INDEX_REQUIRED=true` makes startup fail closed when the
validated index is absent or stale.

The cross-encoder remains gated. It is intended to run only when the deployment
target is the Hugging Face Space and must have a deterministic RRF fallback.

## 4. Files and responsibilities

- `backend/main.py`: FastAPI lifecycle, chat routing, extraction integration,
  authoritative scope construction, search execution, and response boundary.
- `backend/models.py`: typed state plus private/public serialization rules.
- `backend/query_facets.py`: bilingual aliases and retrieval-term expansion.
- `backend/search/turn_delta.py`: validated turn delta and state reducer.
- `backend/search/rules.py`: search-rule compilation and evidence requirements.
- `backend/search/scope.py`: specialty/geography/radius eligibility.
- `backend/search/live_retrieval.py`: scoped BM25/dense/evidence retrieval,
  fusion, retry, coverage, and telemetry.
- `backend/search/indexes/`: immutable index build, manifests, activation, and
  read-only repository access.
- `backend/tests/grounded_bilingual_scenarios.json`: sealed scenario casebook.
- `backend/tests/run_grounded_bilingual_suite.py`: recorded journey runner.
- `backend/tests/grounded_journey_grader.py`: deterministic grading gates.
- `backend/tests/qwen_likert_judge.py`: OpenRouter Qwen Likert grading.
- `backend/tests/probe_reverse_target.py`: offline seeded target probe.
- `backend/tests/test_*.py`: contract, state, scope, index, retrieval, privacy,
  and evaluator regressions.
- `backend/space_app.py`, `Dockerfile`, `HUGGINGFACE_DEPLOYMENT.md`: Space
  entrypoint and deployment procedure.
- `RAG_REDESIGN_PLAN.md`: architecture and migration plan.
- `RAG_TIGHTENING_PLAN_2026-09-03.md`: implementation record plus the dated
  2026-09-04 remediation plan.

## 5. Verification completed

The backend regression suite was run on 2026-09-04:

```text
Ran 149 tests in 4.185s
OK
```

The local service health check reported OpenRouter with
`openai/gpt-oss-120b`, 8,484 facilities, 1,791,749 reviews, and the expected
index/evidence counts when the validated release was mounted.

The relevant prior commits are:

- `851503c3` — ship SeoulDoc as a Hugging Face Docker Space;
- `37b659da` — build grounded bilingual retrieval evaluation;
- `354a163d` — use Qwen3.8 27B for Likert grading;
- `4c42e72` — bound Qwen Likert reasoning;
- `57935211` — record Qwen live grading.

## 6. Latest full evaluation

Audit directory:

`.audit/20260904-rag-tightening-full-qwen27b-retry1/`

Configuration:

- app: `openai/gpt-oss-120b` through OpenRouter;
- judge: `qwen/qwen3.8-27b` through OpenRouter;
- patient actor: Luna medium effort;
- endpoint: local FastAPI `/chat`;
- eight bilingual scenarios, four language pairs.

The run completed without runner errors but did not pass release gates. The
important result was diagnostic: the target facility usually reached the
displayed top three, while decisive evidence and refinement state were less
reliable. Several runs ended with `iteration_limit` and no terminal
`finish_search`. One Korean response failed the language gate.

There is also an evidence-schema false negative. In the Mapo journeys, the
decisive review was present in the target facility’s returned evidence, but the
evidence object lacked `place_id`; the grader requires that association before
counting the review. Yongsan has a separate genuine miss where the required
review was not consistently surfaced.

## 7. Known unfinished work

The following items are documented, not claimed as complete:

1. Add `place_id` to emitted evidence records and test its public serialization.
2. Make negative constraints deterministic for aggressive upselling,
   unfriendly nurses, overprescribing, and equivalent Korean wording.
3. Preserve prior requirements during refinements, removing only the named
   criterion when the user explicitly removes it.
4. Make evidence coverage per facility and per requirement rather than global.
5. Retrieve and attach decisive evidence before ranking a facility highly.
6. Add bounded targeted passes for missing evidence and a truthful terminal
   search receipt.
7. Add a constrained English/Korean output fallback.
8. Re-run the complete eight-scenario suite and verify the Hugging Face-only
   cross-encoder against latency and memory budgets.

The dated remediation plan in
`RAG_TIGHTENING_PLAN_2026-09-03.md` contains the ordered implementation and
release gates for these items.

## 8. Reproduction commands

Run backend tests:

```bash
backend/venv/bin/python -m unittest discover -s backend/tests -p 'test_*.py'
```

Probe the seeded reverse target without an LLM:

```bash
PYTHONPATH=backend backend/venv/bin/python backend/tests/probe_reverse_target.py
```

Build or activate the indexed release:
