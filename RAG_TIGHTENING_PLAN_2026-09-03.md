# SeoulDoc RAG tightening plan

Date: 2026-09-03
Status: core implementation complete; Hugging Face reranker remains gated

## Goal

Make English, Korean, and code-switched searches compile to the same search
policy, preserve exact user constraints across turns, run every required
retrieval channel inside the eligible facility scope, and return enough safe
receipt data for the evaluation to verify what happened.

The target is the failed run `20260903-phase4-qwen27b`. A change counts only
when it improves the deterministic gates and the repeated bilingual scenario
run. Prompt fluency alone does not count.

## Findings from the failed run

1. Station locations carry valid coordinates, but `detect_search_mode()`
   changes them to `zone` when reverse geocoding also supplies a district.
   `compile_legacy_state_rules()` then creates a district `AreaRule` and drops
   the point radius.
2. The first local Jamsil request inherited the old city-wide 25 km value. An
   explicit English 1 km refinement later inherited 0.5 km.
3. The LLM returns full replacement lists. The controller guesses whether a
   turn is additive from words such as `and`, so removal and replacement have
   no first-class meaning. This turned removed parking into a negative rule and
   lost Tuesday-evening hours.
4. The server creates `last_retrieval_run_id`, metadata, and trace data, but
   production serialization removes them. The evaluator therefore cannot
   prove which stages ran.
5. The new indexed adapter uses scoped lexical, dense, and evidence retrieval,
   but its method names and trace shape do not match the actual operations or
   the evaluation contract. Evidence search does not report structured facts
   and verbatim comments as separate coverage channels.
6. The Mapo target can rank first in both languages. The corpus is not the
   primary blocker. State custody, query equivalence, evidence coverage, and
   final ranking are.

## Architecture decision

Use a typed turn delta and a deterministic reducer. GPT OSS proposes the delta;
the server validates it and owns all state transitions. The HTTP handler calls
one operation and does not coordinate individual retrieval stages.

### Caller view

```python
proposal = propose_turn_delta(message, previous_state)
delta = compile_turn_delta(message, proposal)
next_state = reduce_search_state(previous_state, delta)
outcome = search_engine.search(message, next_state)
return public_chat_response(outcome, evaluation_receipt=eval_mode)
```

### Core types

```python
@dataclass(frozen=True)
class SearchDelta:
    specialty: str | None
    location: LocationUpdate | None
    distance_km: float | None
    citywide: bool | None
    add_hard: tuple[str, ...]
    add_soft: tuple[str, ...]
    add_negative: tuple[str, ...]
    remove_concepts: tuple[str, ...]
    required_hours: tuple[str, ...]
    comment_requirements: tuple[EvidenceRequirementDraft, ...]
    response_language: Literal["English", "Korean"]

@dataclass(frozen=True)
class SearchReceipt:
    run_id: str
    status: Literal["complete", "incomplete", "error"]
    termination_reason: str
    actions: tuple[RetrievalActionReceipt, ...]
    methods: frozenset[str]
    coverage: tuple[RequirementCoverage, ...]
```

`SearchDelta` contains user intent only. It cannot contain candidate IDs,
scores, alpha values, or tool routing controls. `SearchReceipt` contains stage
names and counts but no raw medical query, prompt, score, review text, or
candidate ID.

### Alternatives rejected

- A larger prompt that rewrites the whole state lost because one malformed
  completion can still overwrite distance, polarity, and location.
- More controller heuristics lost because `main.py` already has overlapping
  early routes and merge branches. Adding another branch would make state
  behavior harder to predict.

## Implementation phases

### Phase 1: lock down turn semantics

- Add failing unit tests for station-as-point behavior, implicit 5 km, explicit
  1 km, constraint deletion, Tuesday-evening preservation, and code-switch
  replacement.
- Add the `SearchDelta` compiler and reducer as pure functions.
- Parse numeric English and Korean distances deterministically. Explicit values
  override inherited labels and radii.
- Treat a station, landmark, address, or map pin with coordinates as a distance
  search. Only an explicit district request becomes a zone search.
- Remove a concept from every positive and negative collection when the user
  says it is no longer required.
- Keep opening hours separate from review preferences.

Exit gate: the six state regressions pass without an LLM or network call.

### Phase 2: tighten the meta-prompt

- Give GPT OSS the previous public state and ask for a `SearchDelta`, not a full
  replacement state.
- Include explicit operations: `set`, `add`, `remove`, and `replace_context`.
- Require bilingual canonical terms for each comment requirement.
- Reject unknown fields and retry malformed JSON once. Use deterministic
  extraction for distance, city-wide scope, and known removal phrases even
  when the model fails.

Exit gate: recorded English, Korean, and code-switch turns compile to the same
canonical rules where their meaning is equivalent.

### Phase 3: make retrieval policy server-owned

- Always run scoped facility BM25 and scoped dense retrieval for a search turn.
- Search structured or summary evidence separately from verbatim reviews.
- Search comment requirements with both Korean and English variants.
- Fuse channel ranks with versioned reciprocal-rank-fusion weights.
- Keep one bounded retry. Retry only unmet requirements and stop on repeated
  query hashes or no new evidence.
- Record each real stage in `SearchReceipt`; never manufacture a stage that did
  not execute.

Exit gate: every search run has a fresh ID, an ordered receipt, a coverage
decision, and a deterministic stop reason.

### Phase 4: validate evidence and answer generation

- Attach every verbatim review to one facility ID.
- Require decisive evidence before claiming a review-dependent preference.
- Generate prose only from the final eligible facility packets.
- Validate facility IDs, evidence IDs, quotations, language, radius, and final
  membership before returning the response.
- Use a deterministic English or Korean fallback when validation fails.

Exit gate: no cross-facility evidence, unsupported hours, parking, transport,
or review claims appear in response snapshots.

### Phase 5: add the Hugging Face-only cross-encoder

- Keep reranking disabled unless `DEPLOYMENT_TARGET=hf_space` and
  `RERANKER_ENABLED=true`.
- Rerank at most 64 eligible candidates after retrieval and coverage refinement.
- Pin the model revision and bound token count, batch size, queue length,
  concurrency, and timeout.
- Drop all partial scores on error and fall back to the deterministic RRF order.
- Load and warm one model during startup. Fail readiness only when
  `RERANKER_REQUIRED=true`.

Exit gate: the enabled reranker improves holdout nDCG@5 or MRR without breaking
the Space memory and latency budgets. Otherwise it stays disabled.

### Phase 6: rerun the decision suite

- Run focused tests, the full backend suite, six offline target probes, and the
  same eight bilingual journeys under a new run ID.
- Compare target ranks, top-five overlap, evidence coverage, hard gates, and
  Qwen scores with `20260903-phase4-qwen27b`.

Release requires 100% hard-gate rate, at least 90% scenario pass rate, weighted
mean at least 4.2, language gap at most 0.3, 100% target hit@5, and MRR at least
0.5.

## Implementation order for this change

1. Phase 1 state regressions and reducer.
2. Evaluation-safe receipt plus honest indexed-stage trace.
3. Meta-prompt migration to delta output.
4. Bilingual evidence coverage and bounded refinement.
5. HF-only reranker wiring, disabled by default.
6. Focused, full-suite, and live scenario verification.

## Not in scope

- Training or fine-tuning GPT OSS or Qwen.
- Exposing prompts, scores, medical queries, exact addresses, review text, or
  candidate IDs in telemetry.
- Claiming doctor-level identity from facility-level records.
- Changing sealed target IDs or success thresholds to make the evaluation pass.

## Implementation record — 2026-09-03

Completed:

- Added a typed `SearchDelta` compiler and deterministic state reducer. Exact
  English and Korean distances, the 5 km local default, context replacement,
  constraint removal, comment-term canonicalization, and Tuesday-evening
  requirements no longer depend on the model rewriting the whole state.
- Changed the GPT OSS extraction contract to propose a delta with explicit
  refine/replace operations, removed concepts, hours, and exact distance.
- Fixed station coordinates so a reverse-geocoded district cannot silently
  replace point-radius search.
- Added hard structured-hours filtering and bilingual aliases for English,
  Korean, and code-switched review requirements.
- Split scoped retrieval into facility BM25, facility semantic, structured
  evidence BM25, and multilingual verbatim-comment search. Added versioned RRF,
  per-requirement evidence queries, a bounded coverage retry, and truthful
  complete/incomplete termination receipts.
- Kept detailed retrieval receipts behind `ENABLE_RETRIEVAL_DEBUG`; normal
  responses do not expose queries, scores, review text, or candidate IDs.

Verification evidence:

- TDD regressions were observed red before implementation for the missing turn
  delta, station mode, Tuesday-evening scope, and comment canonicalization.
- The full backend suite passes: 147 tests, 0 failures.
- The live indexed app loaded 8,484 facilities and 2,060,433 evidence records.
- Live run `20260903-rag-tightening-final` passed all 12 deterministic gates.
  Its final state used a 5 km Jamsil Station radius, removed parking, required
  Tuesday evening, preferred `thorough` and `clear explanations`, and avoided
  `long wait`.
- The final retrieval receipt was `complete` with `finish_search`; all four
  retrieval channels returned results, coverage was sufficient, and every
  returned facility satisfied the radius and structured-hours scope.
- The complete three-turn conversation and deterministic report are stored in
  `.audit/20260903-rag-tightening-final/`.

Intentionally gated:

- Phase 5's Hugging Face cross-encoder is not enabled yet. It should only ship
  after it improves holdout nDCG@5 or MRR within the Space memory and latency
  budget; the current server-owned RRF path is the verified fallback.
- The single live scenario proves the repaired regression, not the Phase 6
  release thresholds for the entire eight-journey bilingual/Qwen suite. Run
  that broader suite before treating the model evaluation as release-complete.

## Remediation plan after the 2026-09-04 full evaluation

Date: 2026-09-04
Status: diagnosis complete; implementation pending

### Observed failure

The correct facility usually reaches the displayed top three, but the
decisive verbatim review is not reliably attached. The full retry run also
showed lost negative constraints, searches ending at `iteration_limit` without
`finish_search`, and one Korean language failure.

The evaluation exposed an additional contract defect: Mapo's decisive review
was present in the returned facility evidence, but evidence records omitted
`place_id`, so the deterministic grader rejected them. Yongsan also has a real
retrieval/state problem: the required review is not consistently surfaced and
later refinements can discard earlier requirements.

### Phase 1 — repair the evidence contract

- Add `place_id` to every record emitted by
  `backend/search/live_retrieval.py`.
- Verify that `place_id` survives `backend/models.py` public serialization.
- Add a regression test requiring `place_id`, `evidence_id`, `source_type`,
  `is_verbatim`, and non-empty `text` on every returned evidence item.
- Re-run the reverse-target grader to separate false negatives from genuine
  retrieval misses.

### Phase 2 — make bilingual constraints deterministic

- Represent required positives, soft preferences, hard exclusions, and
  explicitly removed criteria as separate typed fields.
- Add English/Korean mappings for aggressive upselling, unfriendly nurses,
  overprescribing, clear explanations, fast treatment, and long waits.
- Let the model propose facets, but never let it erase an existing constraint
  without an explicit user removal instruction.
- Add exact English, Korean, and code-switch polarity tests.

### Phase 3 — preserve refinement state

- Preserve specialty, location, radius, and prior evidence requirements unless
  the user explicitly replaces them.
- Interpret “no longer important” as removal of only the named criterion.
- Interpret “instead require” as adding the new requirement while retaining
  unrelated requirements.
- Add the exact three-turn Yongsan conversation as a regression fixture.

### Phase 4 — make evidence coverage facility-aware

- Run BM25 and dense retrieval over the complete eligible facility scope.
- Group evidence by facility and requirement.
- Score coverage per facility, not globally across the corpus.
- Rerank candidates using requirement coverage before distance blending.
- Attach the strongest three to five evidence records per facility, including
  decisive verbatim reviews.
- Keep the cross-encoder as a final per-facility reranker for the Hugging Face
  deployment only.

### Phase 5 — make the iterative loop converge

- Pass 1: broad bilingual BM25 plus dense retrieval.
- Pass 2: targeted queries for unmet requirements.
- Pass 3: targeted decisive-review retrieval when required evidence is still
  absent.
- Bound retries, deduplicate query hashes, and stop when no new evidence is
  found.
- Emit `finish_search` only after the final pass and record honest coverage and
  termination status.

### Phase 6 — enforce response language

- Validate Hangul output for Korean requests and Latin output for English.
- Run one constrained translation/rewrite fallback when the check fails.
- Preserve medical names and quoted review text during translation.

### Phase 7 — verification gates

Run unit tests, evidence-contract tests, Mapo and Yongsan reverse-target tests,
code-switch tests, the full eight-scenario suite, and random-doctor review
probes.

Release requires decisive evidence on every reverse-target scenario, no lost
negative constraints, no unintended refinement resets, terminal search
receipts, passing language gates, 100% hard-gate rate, at least 90% scenario
pass rate, weighted mean at least 4.2, target hit@5 of 100%, and MRR of at
least 0.5.

### Implementation order

1. Evidence `place_id` contract and tests.
