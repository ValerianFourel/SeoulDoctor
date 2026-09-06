# Separate 50-draw retrieval suite

Owner: root, branch `ncs`. Implementation commit:
`2b186c6116997b6036c1c551fc51b9b7b1d35508`. The original
`scripts/mini_retrieval_eval.py`, its seven fixtures, historical checkpoints,
and planning note 06 are unchanged. This expansion changes no application,
retrieval ranking, embeddings, hardware, or deployment configuration.

## Frozen sample

The eligible population is **19 existing facility-comment pairs from seven
intentionally selected clinics**, covering seven districts and specialties.
This is a restricted, previously inspected shortlist, **not a full-corpus random
sample or an independent holdout**. The prior preparation kept up to three
35–450-character comments from the first 20 indexed verbatim records per clinic.
That length and clinic selection bias remains part of this suite.

Sampling uses Python `random.Random(20260907).choices(population, k=50)`, in the
stored population order, uniformly **with replacement**. Each pair remains
eligible on every draw. No failed, negative, generic, or duplicate case was
replaced. Draw IDs are `draw-001` through `draw-050`; the first 20 receive English
queries, the next 20 Korean, and the final 10 mixed-language queries. The result
is **50 draws, 18 unique pairs, seven unique facilities**. Duplicate draws carry
their full weight in draw metrics but are not independent quality evidence.

All 19 eligible pairs were verified with indexed evidence-ID lookups: facility
ownership, complete original text and source locator matched. No review-corpus
scan was needed. Sources are the existing release
`3911d79dc31e6a6ccfa3f64a7e401b88893bf66a`, local index `2026-09-02-v1`,
and BGE index revision `a6a3ab6f70c67c15090d175efd4ecdea553b329e`.

The coding agent authored source-supported paraphrases before any retrieval
attempt. No separate query-generation API, translation service, or judging model
was used. Clinic names, expected IDs and source quotations are excluded from
queries. Only query/location/specialty cross the retrieval boundary. Original
text, ownership labels, alternative language paraphrases, and query notes remain
in private fixtures. Duplicate pair/language draws reuse the same query.

No sampled comment was unqueryable. Generic kindness is weakly distinctive and
is labeled accordingly. Cost observations and negative staff/parking experiences
are retained as requests to inspect those experiences, not promises of low
prices or positive recommendations. The labels measure evidence relevance, not
proof that a facility satisfies a preference or that a particular doctor has a
trait. Review language does not establish staff language ability.

Fixture SHA-256:
`cacc38529f2e6ef471f56cec5efeae0db6efcfa7245a0c59a47175373d4e9fe0`.
Private provenance records the population hash and freeze timestamp. The timed
preparation-script/verification interval was 0.082 seconds (see the private
provenance for the precise value); the separately recorded interval from saved
query-preparation script through final review was 116.014 seconds. That interval
includes evaluator implementation. Initial query drafting was not instrumented:
these figures are a lower bound, **not total query-authoring time**.

## Run and reproduce

Restore the private fixture directory from Dataset
`ValerianFourel/seouldoc-eval-handoff`, revision
`ef941a0ce929df871f635c303c82191570e85a69`, path
`runs/mini-retrieval-50-20260906T232853Z/`.
Access requires approved process credentials; no environment file is read.
Keep the restored files under ignored `.audit/`. To retry after service recovery,
copy only frozen fixtures/provenance/population into a **fresh run directory**;
never overwrite the existing `results.json`.

```bash
python scripts/mini_retrieval_eval_50.py \
  --fixtures .audit/NEW-RUN-ID/fixtures.json \
  --size 50 --deadline 300 \
  --expected-space-revision 2e9a6e65b1037f347791fb265cbe061ebcb070c7
```

To reproduce the draws without retrieval, load the private `population.json`
and call `sample_cases(population, 20260907, ['en']*20 + ['ko']*20 + ['mixed']*10)`
from `scripts/mini_retrieval_eval_50.py`. Compare the serialized fixtures/hash.
The population contains the frozen paraphrases, so no model call is needed.

The runner verifies the exact active ncs revision, warms the facility encoder,
and requires the real CUDA `/ready/gpu` probe before measured execution. It
uses the existing protected endpoint and worker/scorer from the seven-case
runner. Four worker processes reuse HTTP sessions with fresh searches, no
retries, at most 30 seconds per request and a configurable overall deadline
(default 300 seconds, maximum 600). Warmup is separate from measured execution.

The unchanged endpoint supplies facility limit 200, discovery limit 100,
shortlist 20, lexical quota 5, pool 512, initial rerank budget 224, and three
selected comments. Coverage retry is disabled (0), not the normal policy's 32.
The server's actual limits are saved/printed after successful warmup. No limits
were reduced or increased for this suite. Generation, translation, intent-model
calls, browser automation, and LLM judging are outside retrieval execution.

Every completed response is checkpointed atomically, retaining all draws.
Final unfinished requests are timeouts; queued cases are explicitly not started;
unknown ownership is not zero verified errors. Client termination does **not**
cancel server computation. Do not immediately replay timed-out work: inspect
server readiness/activity first. The runner refuses an existing result file.

Per-draw results include rank within the top five, admission hit, selection under
the target clinic, ownership errors, completeness, request latency, and service
failure details. A hit from incomplete retrieval does not pass. Draw totals,
language counts and per-unique-pair counts/hits are separate; unique-pair groups
retain their sample sizes instead of pretending repeated draws are independent.
Median/p95 use returned valid-response latencies, including incomplete responses;
timeouts are reported separately, not folded into that latency distribution.
Successful-case throughput counts only complete passing cases per measured
second. Selected-ID success does not prove browser visibility or translation.

## Actual attempt and checks

Run ID: `mini-retrieval-50-20260906T232853Z`. The one bounded attempt exited 1
at the deployment gate. HF reported `RUNTIME_ERROR` with
**“Scheduling failure: not enough hardware capacity”**; current hardware was
null, requested hardware remained `l4x1`. All **50/50 draws are BLOCKED**, with
ownership unknown for all 50. No retrieval request was issued, no encoder/GPU
warmup ran, and measured retrieval execution was 0 seconds. Deployment-gate
inspection took 0.139 seconds. Median/p95 and throughput are unavailable, not
measured zero latency or a successful fast run. Zero observed hits/errors in
this checkpoint must not be interpreted as retrieval-quality scores.

| Language | Draws | Blocked | Measured responses |
| --- | ---: | ---: | ---: |
| English | 20 | 20 | 0 |
| Korean | 20 | 20 | 0 |
| Mixed | 10 | 10 | 0 |

All private fixtures, population, provenance and checkpoint rows were uploaded
to the pinned private Dataset revision above. The gate attempt used evaluator
SHA-256 `f3467cca7ef7737d9369400cd746c0c82d5c08be5eed3403c64262873f449220`;
the final code additionally labels in-progress checkpoints honestly, prints
runtime diagnostics, records its own hash, and reports unavailable throughput
as null. The blocked artifact was retained unchanged after those refinements.
No second live attempt was made.

Local checks: 252 backend tests passed, including seven focused expansion tests;
36 required evaluation/grader tests passed; frontend production build passed.
Five retriever tests passed in 0.73 seconds on a bounded host retry; the
sandboxed attempt stalled and was interrupted. Shell syntax and Git diff checks
passed.
Tests cover replacement and seed reproducibility, duplicate draw IDs, label
isolation, variable size/language allocations, incomplete-hit failure, and a
bounded process test that retains finished, timed-out and queued draws.

English semantic `request_failed` and missing reranking remain unresolved
findings from the seven-case run. The blocked 50-case attempt neither confirms
nor clears them. Application fixes belong in a separate task.

Next action: when the existing ncs L4 Space is RUNNING, verify its exact runtime
revision and CUDA readiness, then execute these unchanged fixtures once in a
fresh private run directory and sync that checkpoint. Do not allocate replacement
hardware or regenerate embeddings to unblock this evaluator.
