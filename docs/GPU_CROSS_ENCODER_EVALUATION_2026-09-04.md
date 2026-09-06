# GPU cross-encoder evaluation — 2026-09-04

## Status

The Hugging Face GPU integration works, but the bilingual evaluation suite does not pass.

The cross-encoder improved decisive-review attachment from 0 of 6 to 3 of 6 reverse-target scenarios and raised the Qwen mean from 3.45 to 3.556. It did not solve Yongsan review retrieval, iterative-search completion, negative-constraint handling, or Korean response consistency. One code-switch target also fell outside the displayed top five.

Authoritative run: `20260904-gpu-crossencoder-v2`

Final status: `finished`, no runner errors, suite `passed=false`.

## What was implemented

### Private Hugging Face reranker

- Space: [ValerianFourel/SeoulDoctor-Reranker](https://huggingface.co/spaces/ValerianFourel/SeoulDoctor-Reranker)
- Runtime: T4 Small during the evaluation.
- Service: Hugging Face Text Embeddings Inference.
- Model: `BAAI/bge-reranker-v2-m3`.
- API: authenticated Docker Space endpoints at `/health` and `/rerank`.
- Billing control: 15-minute sleep setting; the Space was explicitly paused after the journeys were saved.
- Privacy: the user explicitly approved transmitting evaluation conversations and stored review excerpts to the private Space and to OpenRouter for this run.

Hugging Face documents authenticated Space API access in [Spaces API endpoints](https://huggingface.co/docs/hub/en/spaces-api-endpoints). This project uses a direct Docker HTTP endpoint rather than a Gradio-generated route.

### SeoulDoc integration

The backend now has a fail-open remote evidence-reranking client. It:

- sends only a query and the selected evidence text to the private reranker;
- keeps facility IDs and all local metadata authoritative;
- validates every returned index and score;
- falls back to the original evidence order on request or schema failure;
- reranks evidence before the three-evidence-per-facility presentation cap;
- records whether reranking was used and why in private retrieval diagnostics;
- leaves facility weighted-RRF scoring independent from cross-encoder scores.

The reranking query includes English and Korean evidence terms plus positive and negative preference language.

### Fixes made during the evaluation

The first real batches exposed duplicate evidence IDs across the primary and retry searches. TEI returned one score per input position, while the adapter expected one score per unique evidence ID. The adapter rejected the response and safely fell back.

A focused regression test reproduced the failure before the fix:

- Test: `test_duplicate_evidence_ids_are_sent_once_and_removed_from_output`
- Before: `outcome.used == false`, reason `invalid_response`
- Fix: deduplicate by evidence ID before the GPU request and before successful output assembly.
- After: reranker and adjacent retrieval tests passed 12 of 12.
- Real-search verification: the GPU reranked 213 unique evidence records with `used=true` and `reason=ok`.

The Qwen judge also cited the valid field `results[0].distance_km`, but the evaluator catalog originally allowed only the whole result object. This caused two otherwise-complete suite attempts to stop during grading.

A second regression test reproduced and fixed that evaluator defect:

- The projected citation catalog now includes individual result fields.
- Qwen judge tests passed 8 of 8.
- Existing journeys were regraded without replaying app searches.

## Evaluation design

The authoritative run used:

- 8 scenarios grouped into 4 English/Korean or code-switch pairs;
- exact public scenario messages;
- GPT-OSS-120B through OpenRouter as the app model;
- Qwen3.8-27B through OpenRouter as the independent Likert judge;
- the Phase-3 index with 8,484 facilities and 2,060,433 evidence records;
- 1,791,749 raw verbatim reviews;
- saved Kakao transit observations plus independent distance checks;
- retrieval debug records with fresh run IDs, actions, coverage, termination, candidates, and reranker outcomes.

The suite requires all hard gates, a scenario mean of at least 4.0, no dimension below 3, a suite mean of at least 4.2, at least 90% scenario passes, a language gap no greater than 0.3, all reverse targets in the top five, and MRR of at least 0.5.

## Results

| Metric | Prior baseline | GPU run | Change | Required | Pass |
|---|---:|---:|---:|---:|---|
| Hard-gate rate | 0.250 | 0.125 | -0.125 | 1.000 | No |
| Scenario pass rate | 0.250 | 0.125 | -0.125 | 0.900 | No |
| Weighted mean | 3.450 | 3.556 | +0.106 | 4.200 | No |
| English mean | 3.300 | 3.725 | +0.425 | — | — |
| Korean mean | 3.600 | 3.388 | -0.213 | — | — |
| Language gap | 0.300 | 0.3375 | +0.0375 | <= 0.300 | No |
| Reverse target hit@5 | 1.000 | 0.833 | -0.167 | 1.000 | No |
| Mean reciprocal rank | 0.806 | 0.708 | -0.097 | 0.500 | Yes |

This was not a perfectly isolated A/B test: GPT-OSS query interpretation and generated responses were rerun, so model variability can affect facility ranks and language behavior. The strongest attributable GPU result is the change in attached evidence, supported by successful reranker traces on every search turn.

### Scenario outcomes

| Scenario | Hard gate | Qwen mean | Scenario pass | Main outcome |
|---|---:|---:|---:|---|
| Radius expansion, English | Yes | 4.80 | Yes | Complete search and correct language |
| Radius expansion, Korean | No | 3.90 | No | Final response failed Korean-language checks |
| Mapo dermatology, English | No | 3.65 | No | Target rank 1 with decisive review; search ended at iteration limit |
| Mapo dermatology, Korean | No | 3.15 | No | Target rank 1 with decisive review; negative state and completion failed |
| Yongsan pediatrics, English | No | 3.30 | No | Target rank 1 but decisive review absent; iteration limit |
| Yongsan pediatrics, Korean | No | 2.30 | No | Target rank 1 but decisive review absent; polarity/state and completion failed |
| Code switch, English to Korean | No | 3.15 | No | Target retrieval rank 20, not displayed; Korean response failed |
| Code switch, Korean to English | No | 4.20 | No | Target displayed rank 4 with decisive review; completion failed |

No bilingual pair passed every pair gate.

## Did the GPU retrieve the right comments?

Partially.

| Reverse-target scenario | Baseline decisive review | GPU decisive review | GPU target rank |
|---|---:|---:|---:|
| Mapo, English | Missing | Found | Presented 1 |
| Mapo, Korean | Missing | Found | Presented 1 |
| Yongsan, English | Missing | Missing | Presented 1 |
| Yongsan, Korean | Missing | Missing | Presented 1 |
| Code switch, English to Korean | Missing | Missing | Not displayed; retrieval 20 |
| Code switch, Korean to English | Missing | Found | Presented 4 |

The Mapo decisive review is `review:325180cb5153b903bfa2`. It was attached in both direct Mapo searches and the Korean-to-English code-switch search.

The Yongsan decisive review is `review:0fca6e1982b0a016e01c`. It did not appear anywhere in either saved Yongsan journey. The target facility itself was rank 1, so facility retrieval was correct while evidence selection was not.

All three Mapo contextual reviews and all three Yongsan contextual reviews also remained missing. The cross-encoder improved one decisive evidence path, but it did not provide complete evidence coverage.

## GPU execution evidence

Every authoritative journey recorded successful GPU calls:

| Journey | Rerank calls | Successful | Pre-dedup candidate counts |
|---|---:|---:|---|
| Radius English | 3 | 3 | 200, 7, 600 |
| Radius Korean | 3 | 3 | 200, 7, 400 |
| Mapo English | 2 | 2 | 146, 741 |
| Mapo Korean | 2 | 2 | 328, 713 |
| Yongsan English | 3 | 3 | 400, 20, 38 |
| Yongsan Korean | 3 | 3 | 270, 13, 43 |
| Code switch English to Korean | 2 | 2 | 341, 414 |
| Code switch Korean to English | 2 | 2 | 207, 512 |

All recorded reasons were `ok`. The configured maximum sent to the GPU remains 256 unique candidates; the trace counts describe the input before client-side deduplication and clipping.

## What still does not pass

### 1. Cross-encoding cannot recover evidence that first-stage retrieval does not admit

The cross-encoder only reorders retrieved evidence. It cannot find a review that BM25 and indexed evidence retrieval did not place in its candidate window. The absent Yongsan review shows that first-stage review recall remains the primary bottleneck.

The current diagnostics do not retain the full pre-rerank evidence-ID list, so they cannot prove whether the Yongsan review was absent from all candidates or merely outside the 256-item GPU window. This observability gap should be closed using IDs or hashes, without logging full review text.

### 2. Coverage repeatedly ends at `iteration_limit`

Most review-dependent final turns recorded:

- `retrieval_status=incomplete`;
- `coverage_sufficient=false`;
- `termination_reason=iteration_limit`;
- no `finish_search` action.

The GPU ranks evidence but does not change coverage reasoning or execute another targeted retrieval. It therefore cannot repair this hard gate by itself.

### 3. Negative constraints are inconsistently represented

Examples include:

- Korean Mapo did not retain the explicit aggressive-upselling avoidance term.
- Korean Yongsan reduced “unfriendly nurses” to a generic “unfriendly” term.
- Hard positive and hard negative fields remained empty where the scenario expected strong polarity.
- A negative Yongsan nurse review was presented as normal evidence rather than clearly labeled as a warning.

A relevance cross-encoder treats a negative review mentioning the requested concept as highly relevant. It does not know whether that text supports or violates the preference unless polarity is represented in scoring and presentation.

### 4. Language enforcement remains unreliable

The Korean radius response and English-to-Korean code-switch response failed deterministic language checks. Retrieval quality does not fix response-language generation.

### 5. Facility ranking is still unstable in one code-switch direction

The English-to-Korean code-switch target fell from baseline presented rank 2 to GPU-run retrieval rank 20 and disappeared from the displayed results. Evidence cross-encoding is intentionally not used as a facility relevance score, so this is primarily a query/state/facility-ranking issue and may also reflect LLM run-to-run variability.

## Recommended next changes

1. Add facility-scoped second-stage review recall. After selecting the facility shortlist, search all reviews belonging to each shortlisted facility using bilingual positive, negative, and disease terms. Union this with global BM25/dense evidence before cross-encoding.
2. Track constraint coverage explicitly. Maintain one status per positive, negative, and excluded constraint. Retry only missing constraints with targeted queries, and terminate only after each required constraint is supported or explicitly unresolved.
3. Separate support and risk evidence. Cross-encode positive support and negative-warning queries independently. Do not use one relevance score for both. Present negative matches as warnings, not endorsements.
4. Preserve polarity in state. Map phrases such as “avoid aggressive upselling” and “unfriendly nurses” into exact negative fields; preserve removed constraints separately so they cannot reappear.
5. Enforce response language after generation. Run a deterministic script-ratio check and regenerate once in the requested language when it fails.
6. Improve reranker diagnostics. Record the deduplicated candidate count, clipped count, selected evidence IDs or hashes, and final cross-encoder rank for selected evidence. This makes candidate recall distinguishable from reranker failure.
7. Re-run an isolated retrieval A/B. Freeze the saved search rules and candidate lists, then compare local ordering against GPU ordering without another LLM interpretation pass. Use the full bilingual suite only after that component test passes.

## Verification and artifacts

Authoritative artifacts:

- `.audit/20260904-gpu-crossencoder-v2/suite_report.json`
- `.audit/20260904-gpu-crossencoder-v2/journey-*.json`
- `.audit/20260904-gpu-crossencoder-v2/deterministic-*.json`
- `.audit/20260904-gpu-crossencoder-v2/qwen-*.json`

Prior baseline:

- `.audit/20260904-rag-tightening-full-qwen27b-retry1/suite_report.json`

Non-authoritative diagnostic attempt:

- `.audit/20260904-gpu-crossencoder-v1/`
- It preserved useful failure evidence but omitted retrieval debug data and ended with one judge-citation validation error.

Focused verification performed:

- Duplicate-evidence regression failed before the adapter fix.
- Reranker and adjacent retrieval tests: 12 passed.
- Qwen result-field citation regression failed before the evaluator fix.
- Qwen judge tests: 8 passed.
- Complete backend unit suite: 155 passed.
- Real 213-item reranker preflight: used successfully.
- Authoritative suite: 8 scenarios finished, 0 runner errors, suite failed.
- Hugging Face Space runtime after the run: paused.

## Operational state

- The paid T4 Space is paused.
- The local evaluation server remains available with retrieval debug enabled for this local process only.
- Production should not expose private retrieval diagnostics.
- Do not commit `backend/.env` or any token.
- A runtime deployment should use a restricted read token where possible; keep the write token only for Space administration.
