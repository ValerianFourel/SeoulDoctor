# Diagnostic evaluation results, 2026-09-06

All 42 scenarios were attempted. Thirty-seven conversations completed, and five
stopped with HTTP 500 errors. This report focuses on selected diagnostic
observations that guide incremental product improvements. It is not a
comprehensive release judgment or verification of the finished retrieval
pipeline. Judge-only recovery is recorded separately. Original run artifacts
and scenario definitions remain unchanged.

## Run identity

- Branch: `local/rag-visible-evidence-20260906`.
- Runner commit: `903a2996b55698bfeddb61603a0a3165a1cf4a9a`.
- Deployed Space commit: `c291916971fe30c70db9f655a9fa727790b27e2e`.
- Targeted run: `diagnostic-targeted-20260906-resume2`.
- Full run: `parallel-diagnostics-20260906-resume3`.
- Private results Dataset: `ValerianFourel/seouldoc-eval-handoff`.
- Targeted upload: `7cfca82b25739aae360c2869f918acf8fd7e8cc5`.
- Original full-run upload: `fda549fe6a391bcefb0db9dea803a15f638d0c1d`.

Local application changes were not deployed in this run. No GPU jobs or paid
hardware were started. The inspected Space used CPU Basic, and all 20 listed
jobs were terminal. Credentials came from process environment variables.

## Coverage and measured failures

The queue combined fixed-message journeys, twelve adaptive regression
conversations, ten stress cases, and twelve frozen random holdouts. Adaptive
conversations are a separate mode. The legacy regression projection omits
`hidden_constraints`, so those conversations do not prove that every intended
staged requirement was exercised.

The targeted Yongsan run completed both languages. Diagnostic-continuation
authorization allowed the remaining cases to run despite observed failures.

| Measurement | Result |
| --- | --- |
| Completed conversations | 37 of 42 |
| Successful application turns | 89 |
| Full-run application requests | 94 |
| Stress native passes | 6 of 10 |
| Frozen holdout conversations completed | 12 of 12, three turns each |
| Median application request latency | 31.89 seconds |
| p95 application request latency | 63.15 seconds |
| Original queue duration, including judges | 1,670.70 seconds |

Latency includes failed requests and excludes the client semaphore queue.
Four scenario workers shared two concurrent application slots and four model
slots. Turns stayed sequential within each conversation. This is one measured
concurrency setting, not a concurrency sweep or complete per-stage benchmark.

The HTTP 500 cases were `en-amenities-negative-03`,
`ko-amenities-negative-03`, `en-insufficient-evidence-04`,
`ko-insufficient-evidence-04`, and `stress-09`. They remain failures, not passes.
Semantic service calls failed on 64 successful turns. Reranker calls failed on
59. Native stress checks also include brittle exact-phrase and method-name
checks, so their failures are not interchangeable with recommendation errors.

## Decisive evidence

Among six comment-driven final responses, all six target IDs appeared
in internal admission records. Four target comments remained attached to the
correct facility. Neither code-switching target remained on the final cards.
No final reply contained its complete original target text.

The complete-original-text check is stricter than faithful translation or a
faithful excerpt. It does not establish that every paraphrase was wrong.
Faithfulness and recommendation effect are not validated by this string check.

In the targeted Yongsan run, both languages attached the mixed review. The
English final reply praised the clinic without the nurse warning. The Korean
reply explicitly disclosed the criticism of nurses. Internal retrieval success
therefore did not establish consistent patient-visible interpretation.

For the twelve holdouts, one final response presented the sampled facility.
No final response attached the exact sampled comment or showed its complete
original text. All twelve source-ID mappings were resolved against the original
Parquet rows. Sampled facilities are not assumed to be the only suitable answer
to broad requests.

An offline replay of the local evidence renderer showed the complete original
text for the four target comments that survived selection. It could not restore
the two missing code-switching targets. This was not a deployed after-test and
did not verify recommendation suitability.

## Models and evaluator limitations

The catalog snapshot records these canonical model names:

- Patient and judge: `openai/gpt-5.6-luna-20260709`.
- Independent judge: `qwen/qwen3.8-27b-20260814`.
- Deployed application: `openai/gpt-oss-120b`.

Catalog names do not prove immutable provider weights. Selection reused prior
pilots rather than a fresh comparative benchmark. Patients received no private
target comments, IDs, or grading oracles. Judges received separate contexts.

The original 3,000-token judge budget caused malformed or missing output,
especially when reasoning consumed the allowance. Of 74 original reviews,
45 failed output or citation validation. Those records are preserved.
Judge-only recovery uses an 8,192-token budget with reasoning disabled and at
most two attempts. It never replays application conversations.

Literal citation checks reject unsupported quotations. They do not validate
the judge's interpretation or rationale. Scores remain provisional, and no
overall validated Likert grade is reported. Incomplete conversations are not
graded as passes. Recovery counts and reported costs are stored in the final
`objective_summary.json`; application-model costs are excluded from that ledger.

## Artifact provenance and remaining work

The existing embedding audit recorded 1,791,749 eligible reviews, complete
dense coverage with 1,024 finite dimensions, and no missing IDs, duplicates, or
ownership mismatches. It also recorded 989 empty sparse representations and
57,134 reviews truncated above 128 tokens. No embeddings were regenerated.

- BGE-M3 model revision: `5617a9f61b028005a4858fdac845db406aefb181`.
- Index Dataset revision: `a6a3ab6f70c67c15090d175efd4ecdea553b329e`.
- Source Dataset revision: `3911d79dc31e6a6ccfa3f64a7e401b88893bf66a`.
- Source SHA-256: `0e8c6f4ff09b75becf7821ba08bf8b19b6533b824a26ca4e7c6451277dfbfafa`.

The latest local backend test run passed 216 tests in 4.330 seconds. Those
tests do not override the live failures. Full GPU readiness, deployment of the
local fixes, full-scope BGE retrieval, and a trustworthy live before/after
comparison remain unfinished. The smallest deployment candidates are the local
incomplete-search warning and original-comment renderer. Their offline checks
do not prove live recommendation quality. The renderer does not implement a
requirement-by-requirement semantic suitability assessment. GPU restoration,
that assessment, and broader retrieval changes remain separate work.
