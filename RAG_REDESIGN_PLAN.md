# RAG redesign plan

Status: in progress

Progress:

- Phase 1 completed on 2026-09-02. Immutable contracts and the deterministic
  bilingual compiler live under `backend/search/`. The existing extraction path
  runs the compiler in shadow mode. At the Phase 1 checkpoint, all 105 backend
  tests passed.
- Phase 2 completed on 2026-09-02. `ScopeBuilder` is the authoritative owner of
  specialty, geography, radius, required-attribute, and prohibited-rule
  eligibility. The legacy RAG adapter receives the complete eligible scope, and
  the result boundary checks scope membership before serialization. Silent
  radius expansion was removed. All 115 backend tests pass. Exhaustive checks
  over the 8,484-facility corpus match independent dataframe and Haversine
  filtering for the 5 km point scenario and all 25 Seoul districts, with
  identical English and Korean rule hashes and facility IDs.
- Phase 3 completed on 2026-09-02. The offline builder publishes immutable,
  checksum-bound Arrow, NumPy, and SQLite FTS5 artifacts, while activation is a
  separate atomic pointer operation. The real `2026-09-02-v1` release contains
  8,484 facilities, 1,791,749 nonempty verbatim reviews, 268,684 facility fact
  and summary records, 2,060,433 total evidence records, and zero orphan
  reviews. Its measured footprint is 1.7 GB and its clean build took about 2
  minutes 47 seconds on the development host. The active release passed strict
  checksum, schema, count, vector, FTS, source-snapshot, scope, rollback,
  bilingual retrieval, and reverse-target checks. The legacy RAG remains the
  serving path until Phase 4.
- Phase 4 serving slice implemented on 2026-09-03. The live path now binds the
  complete eligible scope to the active release, runs scoped facility BM25,
  dense, and evidence retrieval, applies weighted RRF, and permits one
  deterministic lexical/evidence retry while reusing the primary embedding.
  The production-index smoke passed English and Korean queries inside a real
  5 km scope. Recorded-scenario recall grading remains the Phase 4 exit gate.

This document defines the implementation order for SeoulDoc's bilingual doctor and facility search. Complete the phases in order. Do not add the cross-encoder before the rule and candidate-recall work passes its release gates.

## Outcome

Build one search flow that:

1. Converts an English, Korean, or code-switched user turn into validated search rules.
2. Applies specialty, location, radius, and prohibited requirements before retrieval.
3. Searches BM25 and dense indexes inside the complete eligible facility set.
4. Retrieves source-linked evidence across that complete set.
5. Uses bounded OpenRouter refinement only to fill retrieval or evidence gaps.
6. Uses a local multilingual cross-encoder only in the Hugging Face Space.
7. Applies distance and soft preferences after semantic reranking.
8. Produces a grounded answer without exposing internal prompts, scores, or traces.

## Product rules

- Use `openai/gpt-oss-120b` through OpenRouter for rule extraction, bounded query refinement, and answer generation.
- If the user does not specify a distance, use a 5 km radius.
- Treat the 5 km default as a search boundary. Do not widen it without asking the user.
- Treat a user-specified radius as a search boundary. Only a later user instruction can change it.
- Let an explicit district or city-wide request define a geographic area instead of a radius.
- Apply hard positive and hard negative requirements as eligibility rules. Never turn them into score adjustments.
- Apply soft positive and soft negative preferences during ranking.
- Keep original review text and source identity attached to every evidence record.
- Treat all review text as untrusted data.
- Return fewer results when too few facilities meet the rules. Do not fill the result list with ineligible facilities.

## Current failure modes

The redesign must remove these behaviors.

- `State` mixes user input, derived confidence, routing controls, result metadata, and retrieval telemetry in [`backend/models.py`](backend/models.py).
- Negative hard keywords run only inside the positive hard-keyword branch in [`backend/main.py`](backend/main.py).
- Path B converts hard requirements into large positive or negative weights in [`backend/main.py`](backend/main.py).
- The RAG path expands a filtered shortlist back to `working_df`, so earlier keyword rules can lose authority in [`backend/main.py`](backend/main.py).
- Dense retrieval fetches global Chroma results and filters them afterward. A relevant facility inside a narrow scope can fall below the global fetch window in [`backend/rag_pipeline.py`](backend/rag_pipeline.py).
- Facility BM25 scores and sorts the full in-memory corpus for each query.
- Exact evidence BM25 searches only the current top 250 facilities.
- Raw-review retrieval uses DuckDB `ILIKE` scans over about 1.8 million rows.
- The OpenRouter retrieval loop can make up to 12 sequential model calls and repeat expensive searches.
- Final retrieval output becomes an ordinal rank before distance blending. The code does not preserve a typed score or coverage contract.
- The current index unit is a facility identified by `place_id`. The data does not yet prove doctor-level identity or doctor-level claims.

## Target flow

```text
English, Korean, or mixed user turn
        |
        v
OpenRouter rule proposal
        |
        v
Deterministic SearchRules compiler
        |
        v
Complete EligibleScope
        |
        v
Scoped facility BM25 + scoped facility dense retrieval
        |
        v
Full-scope exact and review evidence retrieval
        |
        v
Bounded coverage refinement
        |
        v
HF-only multilingual cross-encoder
        |
        v
Soft-preference, evidence, and distance ranking
        |
        v
Grounded OpenRouter answer and public facility cards
```

## Core contracts

Add immutable types under `backend/search/`.

```python
@dataclass(frozen=True)
class RuleProvenance:
    source: Literal[
        "user_explicit",
        "default_5km",
        "verified_context",
        "taxonomy",
    ]
    source_span: tuple[int, int] | None
    turn_id: str


@dataclass(frozen=True)
class DistanceRule:
    anchor: GeoPoint
    max_km: float
    provenance: RuleProvenance


@dataclass(frozen=True)
class HardEligibility:
    specialty_ids: frozenset[str]
    geography: GeographyRule
    prohibited_facility_ids: frozenset[str]
    prohibited_taxonomy_ids: frozenset[str]
    required_attributes: tuple[RequiredAttribute, ...]


@dataclass(frozen=True)
class SoftPreference:
    concept_id: str
    polarity: Literal["positive", "negative"]
    weight_class: Literal["weak", "normal", "strong"]
    provenance: RuleProvenance


@dataclass(frozen=True)
class EvidenceRequirement:
    requirement_id: str
    terms_en: tuple[str, ...]
    terms_ko: tuple[str, ...]
    match_mode: Literal["exact_phrase", "all_terms", "semantic"]
    source_types: frozenset[str]
    support_required: bool


@dataclass(frozen=True)
class SearchRules:
    schema_version: str
    original_query: str
    language: Literal["en", "ko", "mixed"]
    hard: HardEligibility
    soft: tuple[SoftPreference, ...]
    evidence: tuple[EvidenceRequirement, ...]
    provenance: Mapping[str, RuleProvenance]
    rules_hash: str


@dataclass(frozen=True)
class EligibleScope:
    index_version: str
    rules_hash: str
    facility_bitmap_ref: str
    facility_count: int
    scope_digest: str


@dataclass(frozen=True)
class Candidate:
    facility_id: str
    lexical_rank: int | None
    dense_rank: int | None
    evidence_rank: int | None
    matched_rule_ids: frozenset[str]


class SearchEngine:
    async def search(
        self,
        turn: UserTurn,
        previous_rules: SearchRules | None,
    ) -> SearchOutcome:
        raise NotImplementedError
```

`SearchEngine.search()` is the public backend operation. HTTP handlers must not coordinate its internal stages.

## Module ownership

| Module | Owns | Must not own |
|---|---|---|
| `backend/search/contracts.py` | Immutable rules, scope, candidate, evidence, and result types | Provider calls |
| `backend/search/rules.py` | Rule proposal validation, bilingual canonicalization, provenance, and the 5 km default | Candidate ranking |
| `backend/search/scope.py` | Specialty, geographic, radius, and prohibited-rule filtering | Soft preferences |
| `backend/search/indexes/manifest.py` | Index compatibility, checksums, and active version | Request policy |
| `backend/search/retrieval/facilities.py` | Scoped facility BM25, scoped dense retrieval, and per-channel ranks | Hard-rule interpretation |
| `backend/search/retrieval/evidence.py` | Exact, factual, and review evidence lookup with source identity | Answer prose |
| `backend/search/fusion.py` | Bilingual variant fusion, channel quotas, and shortlist formation | Eligibility |
| `backend/search/refinement.py` | Coverage checks, deduplication, cache keys, and stopping | Hard-rule mutation |
| `backend/search/rerank/hf.py` | Local model lifecycle, pair rendering, batching, timeout, and fallback | Eligibility and distance |
| `backend/search/ranking.py` | Soft preferences, evidence support, distance, and deterministic ties | Retrieval orchestration |
| `backend/search/answering.py` | Grounded answer input and public serialization | Private telemetry |
| `backend/search/engine.py` | One-turn orchestration | Search-policy implementations |

Reduce `backend/main.py` to HTTP and conversation wiring. Keep `backend/rag_pipeline.py` as a temporary compatibility adapter, then delete it after cutover.

## Rule compilation

OpenRouter produces an untrusted `SearchRulesDraft`. The compiler then:

1. Validates specialty values against the canonical specialty taxonomy.
2. Resolves geographic names and coordinates through the existing location services.
3. Parses explicit numeric distance in English and Korean.
4. Adds a 5 km `DistanceRule` when the user did not specify a distance.
5. Classifies each constraint as hard, soft, or evidence-only.
6. Applies deterministic bilingual aliases from `backend/query_facets.py`.
7. Attaches provenance to each accepted rule.
8. Rejects unsupported hard concepts or asks for clarification.
9. Produces a stable `rules_hash` for caching and evaluation.

The compiler must ignore client-supplied confidence, routing mode, candidate IDs, and ranker controls. Keep a signed rule receipt if the application remains stateless between turns.

## Authoritative scope

`ScopeBuilder` is the only owner of hard-rule semantics. It must:

- Intersect specialty and structured-attribute indexes.
- Apply district, city-wide, or point-radius geography.
- Use geographic cells for coarse filtering and exact Haversine distance for the final radius check.
- Remove prohibited facility and taxonomy matches.
- Return a complete bitmap and count.
- Return zero results when no facility qualifies.
- Never return a truncated eligible-ID list.

Every retrieval method receives `EligibleScope`. The answer stage runs one final membership assertion through the same scope policy before exposing a facility.

## Persistent retrieval indexes

### Facility index

Build one document per facility with separate fields for:

- Canonical specialty IDs and labels.
- Korean words and Korean character n-grams.
- English words.
- Exact normalized phrases.
- District, neighborhood, and geographic cell.
- Trusted facility facts and generated summaries.
- Stable facility ordinal and `place_id`.

Use a persistent fielded BM25 engine. Tantivy is the first local candidate, but benchmark the Python binding and Korean indexing before choosing it.

Store normalized multilingual facility embeddings in a memory-mapped matrix or a filtered dense index. At 8,484 facilities, exact similarity over the eligible bitmap may beat filtered ANN in both correctness and latency. Switch to filtered FAISS or a search service only after a benchmark proves that exact scoring misses the target latency.

### Evidence index

Build a persistent lexical index over every review and factual evidence record. Store:

- `facility_id`.
- Evidence ID.
- Original text.
- Source type and source locator.
- Language.
- Normalized searchable fields.
- Exact phrase positions or original-text offsets.
- Dataset version.

Intersect evidence postings with `EligibleScope`. Search the full eligible set, not the current top 250 facilities. Evidence hits may add a facility to the recall set.

Do not add dense embeddings for all raw reviews in the first release. First measure full-scope evidence BM25. Add review-level dense retrieval only if the bilingual and reverse-target evaluations show a recall gap.

### Index manifest

Build indexes outside the request path. Each published version must contain:

- Facility and review dataset hashes.
- Schema version.
- Analyzer and normalization versions.
- Embedding model ID and immutable revision.
- Embedding dimensions.
- Facility and evidence counts.
- Stable ID-map checksum.
- Build commit.
- File hashes.

Write a new version to a temporary directory. Validate every file and count. Activate the version through an atomic pointer change. Keep the previous validated version for rollback. Do not mutate the active index at application startup.

## Hybrid candidate generation

Build deterministic English, Korean, and mixed query variants from `SearchRules`.

1. Run facility BM25 inside `EligibleScope`.
2. Run dense similarity inside `EligibleScope`.
3. Run exact and review evidence retrieval inside `EligibleScope`.
4. Fuse variants inside each channel.
5. Fuse lexical, dense, and evidence channels with versioned weighted RRF.
6. Preserve each source rank and matched rule ID on the candidate.
7. Apply channel quotas before forming the reranker shortlist.

Start evaluation with 200 results per facility channel and at most 64 facilities for cross-encoder reranking. Treat both values as measured configuration, not permanent constants.

## Bounded refinement

Run facility retrieval once. Give GPT OSS 120B a compact coverage report after the first retrieval pass.

Allow only these planner actions:

- `REQUEST_EVIDENCE` for an unmet requirement.
- `SOFT_REQUERY` with a new bilingual wording for a soft preference.
- `STOP`.

Do not give the planner fields for specialty, location, radius, prohibited rules, facility IDs, weights, or limits.

Stop when any condition is true:

- All required evidence requirements have support.
- One soft requery has run.
- A round adds no facility or evidence.
- A normalized query hash repeats.
- The planner targets no unmet requirement.
- The wall-clock, model-call, query, or evidence budget expires.

Cache results by index version, scope digest, query hash, and retrieval channel. Run the cross-encoder once after refinement finishes.

## Hugging Face cross-encoder

Enable the cross-encoder only through explicit Hugging Face deployment settings.

```text
DEPLOYMENT_TARGET=hf_space
RERANKER_ENABLED=true
RERANKER_REQUIRED=true
RERANKER_MODEL_ID=<multilingual-model>
RERANKER_MODEL_REVISION=<immutable-commit>
RERANKER_MAX_CANDIDATES=64
RERANKER_BATCH_SIZE=8
RERANKER_MAX_CONCURRENCY=1
RERANKER_TIMEOUT_MS=<measured-on-target-hardware>
```

Use `NoOpReranker` for other deployments. Keep OpenRouter as the generative LLM provider.

The Hugging Face implementation must:

- Load one model and tokenizer during application startup.
- Pin the model and tokenizer revisions.
- Warm the model before readiness succeeds.
- Allow one active inference stream.
- Bound the queue, candidate count, token count, batch size, and request deadline.
- Discard all partial scores after timeout or inference failure.
- Fall back to the deterministic pre-reranker order for request-time failures.
- Fail readiness when `RERANKER_REQUIRED=true` and the model cannot load.
- Report model revision, readiness, queue depth, latency, timeout count, and fallback count.
- Keep scores and operational details out of public responses.

Render one deterministic query and document pair per facility. Include the normalized bilingual intent, canonical specialty, soft preferences, and evidence requirements in the query. Include trusted facility fields and two or three source-labeled evidence snippets in the document. Do not include distance. Do not include entire facility profiles or unlimited review histories.

Choose the reranker model through an English, Korean, and code-switch benchmark on the target Space hardware. Do not choose a large model based only on public benchmark scores. CPU latency and memory can make a theoretically better model unusable.

Install reranker dependencies through a Hugging Face-specific requirements file. Update `Dockerfile` to install those dependencies only in the Space image.

## Final ranking

The final ranker receives only eligible candidates. It combines:

- Cross-encoder rank when available.
- Pre-reranker RRF rank.
- Required evidence coverage.
- Soft positive preference matches.
- Soft negative preference matches.
- Distance from the verified origin.

Do not let GPT OSS 120B or the cross-encoder choose component weights. Version the weights in configuration and calibrate them against the bilingual evaluation set.

Recompute eligibility and exact distance before serialization. Return fewer than five facilities when the scope contains fewer valid facilities.

## Answer and evidence safety

Give the answer model only final eligible facility packets and bounded evidence records. Each evidence record must retain its facility ID, source ID, original text, source type, and source locator.

Validate the generated answer before returning it:

- Reject a facility ID that is not in the final ranked set.
- Reject an evidence ID that was not provided to the model.
- Reject modified text presented as a direct quotation.
- Keep original text separate from a translated summary.
- Use a deterministic English or Korean fallback when answer validation fails.

Keep retrieval scores, candidate lists, model prompts, and traces behind the existing private debug boundary.

## Health and telemetry

Report these component states through the health response:

- Active index version and manifest validity.
- Facility and evidence counts.
- Dense model compatibility.
- Reranker enabled, required, loaded, or degraded state.
- Reranker model revision.
- Queue saturation, timeout, and fallback counters.
- Last successful index load.

Record private structured telemetry for stage latency, scope count, channel hit counts, candidate overlap, refinement stop reason, evidence coverage, reranker duration, and fallback reason. Do not log raw medical queries, review text, exact addresses, or facility candidate IDs without the existing consent controls.

## Implementation phases

### Phase 1: lock the rule contract

Create `backend/search/contracts.py` and `backend/search/rules.py`.

Add tests for:

- The implicit 5 km default.
- Explicit English and Korean distances.
- Radius preservation across refinement turns.
- Hard positive and hard negative classification.
- Soft positive and soft negative classification.
- English and Korean canonical-rule parity.
- Rejection of client-supplied derived controls.
- Unknown specialty and location clarification.

Run the compiler in shadow mode. Keep current search results unchanged.

Exit when all existing tests and all new rule tests pass with zero hard-rule or distance violations.

### Phase 2: make scope authoritative

Create `backend/search/scope.py`.

Move specialty, geographic, radius, and prohibited filtering behind `ScopeBuilder`. Pass the resulting complete scope into the old RAG path through an adapter. Remove silent radius expansion from the authoritative path.

Exit when exhaustive dataframe filtering and `ScopeBuilder` return the same IDs for the full bilingual scenario set.

### Phase 3: build immutable indexes

Add the offline builder and loader under `backend/search/indexes/`.

Build facility BM25, facility dense, and full review/evidence BM25 artifacts. Publish them under a versioned directory. Load the active version read-only at startup.

Exit when manifest validation, counts, referential integrity, and rollback pointer tests pass.

Completed with a real release and an idempotent repeat publication. Repeating
the same version validated and reused the release in about 23 seconds without
rebuilding the evidence database. A deterministic raw-review target was
retrieved at rank 1 inside its facility scope, and exact dense scoring returned
the target facility at rank 1 with cosine score 1.0.

### Phase 4: replace candidate retrieval

Create the facility and evidence retrievers. Replace global Chroma over-fetch, per-request full BM25 scoring, top-250 evidence search, and raw-review `ILIKE` scans.

Run the old and new retrievers in shadow mode against recorded scenarios.

Exit when the new path has zero scope violations and meets the approved recall gates.

The serving implementation and production-index smoke are complete. The
recorded English, Korean, code-switched, and reverse-target recall comparison is
still required before this phase clears its exit gate.

### Phase 5: bound iteration

Create the coverage checker and refinement controller. Replace the current tool loop with the restricted planner actions and cache.

Exit when repeated queries cause no repeated retrieval work and every run records a deterministic stop reason.

### Phase 6: add the HF-only reranker

Add the Hugging Face implementation, lifecycle wiring, configuration, dependencies, health fields, and tests. Ship the code with reranking disabled.

Enable it first in a staging Space. Compare enabled and disabled runs on the same index version and scenario set.

Exit when reranking improves the holdout ranking metrics and meets the measured Space latency and memory limits.

### Phase 7: cut over and remove legacy policy

Make `SearchEngine` authoritative. Delete Path A and Path B policy, the old twelve-step tool loop, and duplicate final filters after a rollback drill succeeds.

Exit when the compatibility adapter has no production callers and all release evaluations pass through `SearchEngine`.

## Evaluation gates

### Hard correctness gates

- Zero specialty, location, radius, and prohibited-rule violations.
- Missing distance always compiles to 5 km.
- No radius change occurs without a new user instruction.
- Every returned facility belongs to the complete eligible scope.
- Exact evidence outside the old top 250 is retrievable.
- Every quoted claim keeps the correct facility and source identity.
- Raw review instructions cannot alter rules, tools, facility IDs, or citations.
- Public response snapshots contain no private ranking or trace fields.

### Retrieval gates

- Measure eligible-facility Recall@200 against exhaustive labeled retrieval.
- Measure evidence Recall@100 against sealed reverse-target cases.
- Keep English and Korean recall within five percentage points.
- Keep paired English and Korean top-five Jaccard at or above 0.6. Target 0.8 before full rollout.
- Recover a facility that ranks below the old global dense window but is relevant inside a narrow scope.
- Recover decisive evidence from a facility outside the old top 250.

### Reranker gates

- Measure nDCG@5 and MRR with reranking enabled and disabled.
- Keep reranking disabled if it does not improve the approved holdout set.
- Verify deterministic fallback for timeout, queue saturation, invalid output, and model failure.
- Measure warm latency and peak memory on the real Space hardware.
- Set the production timeout only after the target-hardware measurement.

### End-to-end gates

- Run the grounded English and Korean journeys through the deployed Space.
- Run the Qwen 3.8 27B Likert judge after deterministic gates pass.
- Run random-facility reverse targeting from review evidence.
- Compare user-side distance, application distance, and map-route distance.
- Run the real reranker in the Hugging Face container smoke test.
- Preserve all current API privacy and release-boundary tests.

## Rollout and rollback

Use one authoritative path per phase. Shadow paths may record private comparisons but must not change user results.

Roll out in this order:

1. Shadow rule compiler.
2. Shadow scope builder.
3. Shadow indexed retrieval.
4. New retrieval with reranker disabled.
5. Reranker canary in the Hugging Face Space.
6. Full Hugging Face rollout.

Rollback by disabling the new stage and selecting the previous validated index manifest. Keep the old retrieval path for one release after cutover, then delete it.

## Doctor-level data limit

The current system indexes facilities, not verified doctors. Audit the source dataset before making doctor-level claims.

If verified doctor records exist, add a stable `doctor_id`, associate evidence with that doctor, and rank doctor entities inside an eligible facility scope. If those records do not exist, use the words `clinic`, `hospital`, or `facility` in the UI and generated answer. A reranker cannot infer a specific doctor's qualifications from facility-level reviews.

## Definition of done

The redesign is complete when:

- One typed rule contract controls all search stages.
- One scope builder owns every hard rule.
- BM25, dense retrieval, and evidence retrieval search inside the same complete scope.
- The OpenRouter planner cannot edit hard rules.
- The Hugging Face cross-encoder runs only in the Space and has a tested deterministic fallback.
- Distance remains correct and visible in every returned result.
- English and Korean evaluation gates pass.
- Index publication and rollback are reproducible.
- Legacy Path A, Path B, global over-fetch, top-250 evidence indexing, and review `ILIKE` retrieval are gone.
