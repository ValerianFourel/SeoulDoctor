# NCS search correction plan

Owner: root. Scope: ncs. Starting application source:
5a9167e6e2c47452035915234c8946265cffb9fd. Last verified Space revision:
a358796dba5fadcc0256a039d1604a0ac9fdd64e.

This is an implementation and verification plan. No search fixes or API scenario
runs have been completed in this planning task. No push or deployment is planned
until the checks below pass. Existing sealed evaluation cases remain unchanged.

## Confirmed mechanisms and remaining questions

- The extraction prompt asks for specialty only when explicitly requested.
  A symptom-only request can leave specialty unset. The authoritative scope
  correctly enforces a specialty once supplied. Fix interpretation, not the
  overwritten legacy medium-confidence filtering path.
- Geocoding accepts the first successful provider result without validating
  whether it matches the requested place. The reported eastern-Seoul results
  indicate an incorrect Myeongdong anchor. Capture actual state and provider
  results before attributing it to spelling, stale state, or provider ranking.
- EvidenceRetriever.collect returns empty evidence when no evidence constraints
  were compiled. This does not prove the source has no patient reviews.
- Search scope is built once. There is no bounded specialty-preserving radius loop.
- Review translation is skipped if answer synthesis and verification leave less
  than seven seconds of their 45-second budget. This is a confirmed failure
  path, not yet proof of the cause of this user's untranslated response.
- The generic incomplete-search sentence is hard-coded in the fallback answer.

## Implementation order

1. Reproduce the exact two-turn conversation through /chat with sequential state.
   Save sanitized request/response evidence and timings. Record specialty intent,
   requested location, resolved coordinates, radius, retrieval reason codes,
   evidence counts, answer stages and translation outcome separately.
2. Resolve specialty and location before ranking facilities. Distinguish explicit,
   inferred and unresolved specialty intent. Recognize musculoskeletal symptom
   requests without requiring the word orthopedics. For vague foot complaints,
   ask one short clarification rather than silently mixing specialties. Normalize
   common Myeongdong spellings, validate the returned place, and never label
   unrelated coordinates as the user's requested location. Preserve corrected
   specialty and location on subsequent turns; clear superseded scope coverage.
3. Keep the resolved specialty mandatory. Start with the requested/default area.
   If no eligible specialist exists and radius is soft, widen through the next
   bounded radii, such as 1, 2, 5, 10 and 25 km. Start from the actual current
   radius, stop as soon as eligible specialists are found, and disclose widening.
   Do not substitute unrelated specialties or pad to five cards. An explicit hard
   limit such as "only within 1 km" requires a suggestion rather than silent widening.
4. Always run facility-owned original-review retrieval for shortlisted candidates.
   Use the query, visit reason and specialty even without extra review constraints.
   Keep server relevance order and relevant negative/mixed evidence. Target the
   existing three-first/seven-next presentation and at least two usable relevant
   reviews where the source supports them. Never duplicate, invent, or borrow
   reviews to fill a quota. Distinguish no source reviews from failed retrieval.
5. Reserve a bounded translation budget independently of answer generation.
   Prioritize initially visible reviews; preserve original evidence unchanged.
   Retain translation-first display and Show original. Record missing credentials,
   provider failure, skipped budget and unsupported translation distinctly.
6. Generate useful next-step guidance from the actual search outcome. Supply the
   LLM with resolved specialty/location, attempted radii, available evidence and
   actions the app can perform. Ask for the missing detail that would improve
   results, or offer a relevant retry or wider area. Remove the generic sentence.
   Do not disguise incomplete retrieval as a complete search. Keep a concise,
   actionable fallback when the answer model itself fails.

## Before-push API scenario matrix

Freeze expected outcomes before implementation. Use local API regression fixtures
for faults and controlled data, then exercise the real configured retrieval stack
from an isolated local candidate server. Do not inject faults into the public Space.

| Case | Required result |
| --- | --- |
| Exact foot/myeondong then orthopedi/Jonggak transcript | Correct scope or focused clarification; corrected second turn replaces first scope |
| myeondong, Myeongdong, Myeong-dong, 명동 | Correct central-Seoul anchor; no unrelated Myeonmok coordinate acceptance |
| Explicit Myeonmok request | Preserve Myeonmok; do not over-normalize legitimate locations |
| Foot/ankle pain in English, Korean and mixed language | Appropriate specialty interpretation and useful review query |
| Ambiguous foot issue | One focused clarification; no unexplained mixed-specialty list |
| Explicit orthopedic doctor request | Every returned card matches orthopedics |
| No orthopedist in initial soft radius | Widen area while preserving specialty; report actual radius |
| Only within 1 km | Never silently exceed the hard distance limit |
| Specialty-only query without review attributes | Retrieve real relevant originals despite empty extra constraints |
| First-page review availability | At least two when frozen source evidence supports it; three initially where available |
| Slow or invalid answer model | Reviews and their translations survive answer failure |
| Translation service timeout/error | Original retained; internal cause recorded; no fabricated English text |
| Retrieval service unavailable | Honest, specific retry/refinement guidance; failed retrieval never reported as no source reviews |
| Contradictory or negative relevant reviews | Preserve exact text, ownership and effect on recommendation |

Supplement fixed cases with 12 seeded adaptive patient conversations of 2–6 turns
using the existing inhabited-evaluation tooling and pinned OpenRouter patient
models. Keep patient context separate from grading oracles. Bound app concurrency
to two and keep turns sequential. Retain partial runs and all errors.

Use agents for independent location/specialty and evidence/translation review.
Root grades conversations on 1–5 Likert scales for intent fidelity, specialty and
location accuracy, review relevance, translation faithfulness and useful guidance.
Require every deterministic assertion to pass and every completed conversation
to score at least 4 on each dimension. Any wrong specialty, wrong location,
invalid ownership or fabricated text blocks publication regardless of averages.
Report application errors and evaluator errors separately. Publish observed results,
not a promised pass. Rerun the exact unchanged failing cases after fixes.

## Delivery

Run focused evaluator tests before live evaluation, full backend tests, the
frontend build, and desktop/mobile browser checks. Review the diff independently.
Push only the verified ncs changes. Deploy the exact tested source when authorized,
then verify actual runtime revision, real API behavior and the rendered browser.
Do not regenerate embeddings or purchase infrastructure for this correction.
