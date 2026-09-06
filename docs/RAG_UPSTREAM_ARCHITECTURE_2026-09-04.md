# SeoulDoc upstream evidence retrieval architecture

Date: 2026-09-04
Status: implemented for facility-first retrieval, facility-scoped lexical evidence, bounded local distinctiveness, reranking, coverage retry, and diagnostics

## Problem

The failing evaluations were not primarily a GPU-capacity problem. The correct facility often reached the displayed top three, but its decisive verbatim review disappeared before reranking. A cross-encoder cannot recover evidence that never enters its candidate window.

The redesign therefore treats facility discovery and review evidence discovery as separate stages. Review evidence is retrieved only after a bounded facility shortlist is frozen.

## Implemented pipeline

```text
User request
    |
Structured bilingual rules
    |-- hard eligibility and distance
    |-- disease evidence
    |-- desired/support evidence
    |-- avoided/risk evidence
    '-- removed constraints excluded
    |
Facility retrieval: geo scope + facility BM25 + facility dense vectors
    |
Frozen top-20 facility shortlist
    |
Facility x constraint x evidence-role review cells
    |
Per-cell BM25 retrieval from each facility's review collection
    |
Fair admission + deduplication + bounded local distinctiveness
    |-- CPU pool: at most 512 unique reviews
    |-- initial reranker window: 224
    '-- reserved targeted-retry window: 32
    |
Multilingual cross-encoder reranking by role
    |
Coverage table for displayed facilities
    |
At most one targeted facility-scoped retry for missing required cells
    |
finish_search: complete | partial_evidence
    |
Supporting reviews | warnings | unverified preferences
```

## Core invariants

1. Hard eligibility and distance are applied before ranking. Evidence cannot bring an ineligible facility back into scope.
2. The displayed facilities must be members of the frozen shortlist.
3. Review retrieval is facility-scoped. A review can only be attached to its owning facility.
4. Disease, support, and risk are independent retrieval and reranking roles.
5. Removed constraints do not produce retrieval queries.
6. Warning evidence never becomes positive justification or a facility-ranking boost.
7. The reranker receives no duplicate evidence IDs.
8. The initial and retry reranker budgets together cannot exceed 256.
9. Search terminates explicitly as `complete` or `partial_evidence`; missing evidence does not end as an unexplained iteration limit.

## Bilingual query decomposition

Each evidence requirement has independent English and Korean variants. For example, a pediatric request can produce:

- Disease: `pediatrics`, `pediatric clinic`, `소아과`, `소아청소년과`
- Support: `kind to children`, `gentle with children`, `아이에게 친절`, `아이 진료를 잘함`
- Risk: `unfriendly nurses`, `rude nursing staff`, `간호사가 불친절`, `간호사 불친절`

The channels are searched independently and fused only after each facility and constraint has received a fair opportunity to contribute candidates. Deterministic extraction also preserves the known bilingual risks “aggressive upselling” and “unfriendly nurses.”

## Common versus distinctive reviews

Repeated generic comments should not consume the evidence window, while one unusual comment must not be treated as automatically true.

The current admission policy:

1. Normalizes review text with Unicode NFKC.
2. Clusters exact and near-duplicate reviews using character n-gram Jaccard similarity.
3. Computes local rarity as `1 / sqrt(cluster_size)`.
4. Computes a bounded concreteness signal from the review text.
5. Computes local distinctiveness as:

```text
distinctiveness = clamp(0.70 * local_rarity + 0.30 * concreteness, 0, 1)
```

6. Uses distinctiveness as a small tie-break contribution with weight `0.10`, while round-robin cell admission and cluster diversity prevent repetitive templates from monopolizing the window.

This is intentionally conservative. Distinctiveness affects attention, not truth. The output keeps local cluster size separate from corroboration count, and a singular review is presented as a report rather than established fact.

The present score is local to the candidate pool. Corpus-wide document frequency and semantic-cluster rarity require a new immutable review-vector index and are described under “Remaining work.”

## Evidence roles and presentation

Evidence is scored and displayed in separate groups:

- Supporting reviews: evidence for desired qualities
- Warnings: evidence reporting avoided qualities or risks
- Unverified preferences: required preferences without adequate attached evidence

A relevance match for “unfriendly nurses” therefore becomes a warning. It cannot be rendered as an ordinary reason to recommend the clinic.

## Deterministic coverage and retry

Coverage is evaluated for every displayed facility and required constraint. Missing cells receive one targeted retry against that facility with a larger lexical quota. The 256-item GPU allowance is split into 224 initial candidates and 32 retry candidates so the retry cannot silently exceed the service limit.

After the retry:

- `finish_search: complete` means every required displayed-facility cell has usable evidence.
- `finish_search: partial_evidence` means one or more required cells remain missing and must be disclosed as unverified.

## Admission diagnostics

Every candidate can record:

- facility and constraint ownership
- evidence role
- lexical rank
- dense rank, when a review-dense index supplies one
- fusion rank
- admission to the GPU window
- cross-encoder rank and score
- attachment to a displayed facility
- rejection reason
- local cluster size and distinctiveness

These fields expose exactly where decisive evidence was lost.

## Acceptance gate result

The component probe ran against active index `2026-09-02-v1`, containing approximately 1.79 million reviews. Each target competed inside a 20-facility shortlist in both English and Korean.

| Gate | Mapo | Yongsan |
|---|---:|---:|
| Decisive review recall@256 | 100% | 100% |
| Decisive review attachment@3 | 100% | 100% |
| Correct facility ownership | 100% | 100% |
| Risk polarity | 100% | 100% |
| English/Korean candidate overlap | 100% | 100% |
| Duplicate GPU evidence IDs | 0 | 0 |
| Finish status | complete | complete |

Mapo admitted 125 unique candidates. Yongsan admitted the full 224-candidate initial window. Neither case required the reserved retry window.

The machine-readable result is stored at `.audit/20260904-upstream-recall/component_report.json`.

## Verification

The complete backend regression suite passes: 162 tests.

Focused tests cover:

- facility scope and per-facility quotas
- fair per-cell admission
- cross-cell deduplication without starving another queue
- bilingual variants
- support/risk separation
- bounded novelty preference over repeated generic comments
- deterministic negative-preference extraction
- reranker scores
- explicit completion telemetry

## Committed next stage: BM25 plus BGE-M3 sparse and dense

The next retrieval stage will combine three independent review-recall channels:

1. Local BM25 for exact lexical evidence.
2. BGE-M3 learned sparse retrieval for multilingual token weighting.
3. BGE-M3 dense retrieval for paraphrases and cross-language semantic matches.

This is a committed design direction, not a claim that all three channels are live. The status boundary is:

| Capability | Status on 2026-09-04 |
|---|---|
| Facility shortlist, local review BM25, cell quotas, distinctiveness, coverage retry | Implemented |
| BGE-M3 HTTP client and configuration | Implemented and unit-tested |
| Facility-scoped BM25, BGE-M3 RRF fusion, and fail-open fallback | Implemented and unit-tested |
| Scope-bound evidence-ID resolver and ownership checks | Implemented and unit-tested |
| Private Space runtime source in `services/retriever/production.py` and `Dockerfile` | Implemented; dependency and runtime testing plus deployment pending |
| Corpus encoder, release builder, and full 1.79-million-review release | Pending |
| Deployed-service ablation and full scenario evaluation | Pending |

BGE-M3 sparse retrieval is described as learned sparse rather than SPLADE in this design. BGE-M3 produces its own lexical weights. The remote service must use the model's native sparse and dense outputs rather than treating a generic SPLADE endpoint as equivalent.

### Why BM25 stays

BM25 remains a required channel after BGE-M3 is added. It provides behavior that the semantic channels should not be asked to replace:

- Exact Korean phrases, clinic names, rare treatment terms, numbers, and unusual review wording remain directly searchable.
- Corpus document frequency naturally reduces the rank of repeated generic terms and rewards rarer lexical evidence.
- Its rank and matched terms are easy to audit during decisive-review evaluation.
- It runs inside the application against the already validated 1.7 GB index and remains available when the private GPU Space is sleeping or unavailable.

BGE-M3 sparse adds learned multilingual token weights. BGE-M3 dense adds semantic and cross-language recall. BM25 protects literal recall and the fail-open path. The combination is stronger than treating any one channel as the replacement for the others.

### Selected service boundary

The BGE-M3 review index will live in a separate private Hugging Face GPU Space. The application Space keeps the facility index, review BM25, original review text, evidence metadata, and final ownership checks.

The production service source now exists in `services/retriever/production.py` with its Docker image. It pins BGE-M3, validates the v2 dense and sparse CSR release artifacts, validates the caller's complete expected-release identity, and performs exact top-k scoring only inside the requested facility ranges. It can also download a pinned private Dataset revision. This source has not yet passed dependency installation or runtime tests and has not been deployed.

The full review corpus contains 1,791,749 nonempty reviews. A 1,024-dimensional float16 dense matrix alone is approximately 3.42 GiB. Learned sparse weights, row maps, facility ranges, manifests, and model cache bring the semantic addition to roughly 5 GB before operational headroom. Adding those files to the application Space would increase cold-start time, disk pressure, and failure coupling for every frontend and chat deployment.

The private service instead owns only versioned semantic artifacts and returns ranked evidence IDs. It does not return review text or control facility ownership. The application resolves every returned ID against its local immutable evidence store and rejects unknown, duplicated, out-of-scope, or wrongly attributed references.

This split has a network dependency and a second deployment to operate. The local BM25 fallback makes that trade acceptable. A semantic outage lowers recall but does not take the doctor search offline.

### Target retrieval flow

```text
Structured bilingual constraints
    |
Geo scope + facility BM25 + facility dense retrieval
    |
Frozen top-20 facility shortlist
    |
Facility x constraint x role x language query cells
    |------------------------------------|
    |                                    |
Local review BM25                 Private BGE-M3 service
                                         |-- learned sparse
                                         '-- dense
    |------------------------------------|
    |
Validate returned IDs and facility ownership locally
    |
Per-cell rank fusion with reciprocal rank fusion
    |
Fair admission, deduplication, and distinctiveness
    |
Cross-encoder reranking of the best 64-128 reviews in batches
    |
Coverage check and one targeted retry
    |
Supporting reviews | warnings | unverified preferences
```

The application sends one batched semantic request for the complete shortlist and all query cells. It does not send one request per facility or per constraint. Runtime requests contain query text and facility IDs, not stored review text.

The semantic service stores review rows in facility order. It can score only the row ranges belonging to the top-20 shortlist, so it does not need to search all 1.79 million reviews for every request. Large facility ranges can be scanned in fixed chunks while retaining an exact top-k heap.

### Reciprocal rank fusion

Raw BM25, learned-sparse, and dense scores are not comparable. Fusion therefore uses ranks:

```text
cell_score(review) =
    w_bm25   / (k + bm25_rank)
  + w_sparse / (k + sparse_rank)
  + w_dense  / (k + dense_rank)
```

The first experiment will use `k = 60` and equal channel weights. We will tune weights only from the retrieval ablation results.

English and Korean query variants are searched independently. The implemented RRF groups candidates by facility, constraint, role, and query language, then combines available BM25, learned-sparse, and dense ranks within that group. The existing admission step deduplicates evidence IDs across language groups while preserving their diagnostic records.

### Semantic release lifecycle

The semantic release is built outside the application request process:

1. Read verbatim reviews in stable facility and evidence-ID order.
2. Pin the BGE-M3 model and tokenizer to immutable revisions.
3. Encode dense and learned-sparse outputs in batches on temporary GPU hardware.
4. Store normalized float16 dense vectors, sparse token IDs and weights, facility row ranges, and stable evidence IDs.
5. Write and validate a manifest containing the raw-review source digest, evidence-ID schema, model revision, counts, shapes, and artifact hashes.
6. Publish the release to private persistent storage.
7. Configure the private retrieval Space to load one explicit release ID.
8. Activate or roll back by changing the pinned release, never by mutating a live index in place.

The local and remote releases must report the same raw-review SHA-256 digest. A release mismatch disables both semantic channels for that request.

### Failure fallback

Semantic retrieval is fail-open inside the evidence layer:

- A timeout, sleeping Space, HTTP error, or open circuit returns the local BM25 candidates.
- A source-digest mismatch, unknown evidence ID, duplicate reference, invalid score, or wrong facility rejects the complete remote response.
- A cross-encoder failure preserves the three-channel RRF order.
- One larger targeted semantic retry is allowed only for missing displayed-facility cells. A timed-out request is not repeated during the same search.
- Remaining missing required evidence terminates as `partial_evidence` and is disclosed as unverified.

A semantic failure must not escape to the broad legacy fallback in `CandidateRetrievalAdapter`. Losing one optional channel must not replace the validated facility shortlist or discard successful BM25 evidence.

### Required telemetry

Each evidence admission record must include:

- BM25, BGE-M3 sparse, and BGE-M3 dense ranks
- query cell, language, constraint, role, and facility
- rank after RRF
- semantic release ID and model revision
- remote status and network, query-encoding, sparse-search, and dense-search latency
- local ownership-validation result
- cross-encoder admission, rank, and score
- attachment or rejection reason

Application health must report whether the semantic service is configured, available, bound to the expected release, and using the same review-source digest. Evaluation preflight must fail before paid scenarios if semantic retrieval is required but these checks do not pass.

### Rollout phases

1. **App integration, complete.** Configuration, the remote client, scope-bound evidence-ID resolution, three-channel RRF, fail-open BM25 behavior, and admission diagnostics are implemented and unit-tested.
2. **Private service source, complete.** The production FastAPI runtime, pinned BGE-M3 loader, v2 artifact validator, expected-release checks, facility-scoped exact top-k search, and optional pinned Dataset download are implemented. Dependency installation and runtime behavior still need testing.
3. **Build benchmark.** Encode a 100,000-review sample, measure throughput and artifact density, then set GPU time, storage, and cold-start budgets for the full corpus.
4. **Semantic release.** Implement the corpus encoder and builder, then create and validate the full sparse and dense artifacts from the same raw-review snapshot as the local index.
5. **Runtime test and deployment.** Test the production image with a validated release, deploy the private Space, and verify source-digest binding, BM25 fallback, health, batching, exact facility-scoped results, and warm latency.
6. **Component gates.** Run the live retrieval ablations before enabling the cross-encoder or spending OpenRouter evaluation budget.
7. **Scenario evaluation.** Rerun only failed Mapo, Yongsan, and code-switch scenarios first. Run the complete eight-scenario suite after those pass.

### Ablation and acceptance gates

Every target set must be measured with these configurations:

1. BM25 only
2. BM25 plus BGE-M3 sparse
3. BM25 plus BGE-M3 sparse plus BGE-M3 dense
4. Three-channel retrieval plus the cross-encoder

The three-channel candidate generator must meet all of these gates before the full LLM evaluation:

- Mapo and Yongsan decisive-review recall@256 is 100% in English and Korean.
- Decisive-review attachment@3 is 100%.
- Both English-to-Korean and Korean-to-English code-switch targets are recalled.
- Correct facility ownership and risk polarity are 100%.
- English and Korean candidate overlap is at least 80%.
- No duplicate evidence IDs enter the cross-encoder window.
- No BM25-only exact-match target regresses when sparse and dense channels are enabled.
- Remote timeout and release-mismatch tests return usable BM25 results.
- Warm p50 and p95 semantic latency, response size, and scoped review count are recorded before rollout.

Do not run another expensive eight-scenario LLM evaluation until the full semantic release passes these component gates. This keeps retrieval recall, cross-encoder behavior, and LLM behavior as separate failures that can be measured and fixed independently.
