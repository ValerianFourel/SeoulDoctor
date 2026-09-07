# Note 2: Patient-facing review presentation

## Scope and work plan

Presentation only. Preserve BGE-M3, ranking, indexes, original text, evidence IDs,
facility ownership, and source metadata. No deployment or hardware changes.

- [x] Read poteto-mode principles and project context.
- [x] Reproduce the reported duplication, internal labels, and short comments.
- [x] Trace the cause and sketch the presentation contract.
- [x] Implement filtering, translation fallback, and a single card display.
- [x] Verify the same behavior with tests and a rendered browser inspection.
- [x] Record actual checks, limitations, and next action.

Poteto-mode's investigation/design/review steps run sequentially here because
this side conversation prohibits subagents. No PR or deployment is part of
this request. Test fixtures are invented, not copied patient reviews.

## Design sketch

Keep the original evidence object. Add `presentation` with a status of `hidden`,
`original`, `translated`, or `unavailable`, plus the response language and text
only for a successful translation. Hidden records retain their original text
and IDs in the API. Original and translated text never share a storage field.

A backend presentation module owns language choice, eligibility, and one bounded
translation batch using a separate OpenRouter client pinned to `qwen/qwen3.8-27b`. The renderer preserves
ownership and recommendation caveats and stops duplicating quotes. A small
frontend component renders the presentation state and an original-text toggle.

A frontend-only approach would leave API consumers without translation status
and duplicate the filter. A corpus-level filter would change retrieval. The
chosen boundary applies after selection and before translation/serialization.

Model the Domain motivates explicit presentation states. Laziness Protocol
keeps retrieval unchanged and removes duplicated rendering. Boundary Discipline
keeps model-output validation in the translation adapter. Prove It Works requires
synthetic failure cases and browser evidence, not a claim of live retrieval.

## What changed

The reply no longer repeats reviews, IDs, or internal requirement labels. Cards
show substantive reviews once, with automatic translations labeled and originals
available through a toggle. Unavailable translations offer the original instead.
The unsupported unconditional “English OK” badge was removed. Retrieval roles
are not presented as proof that a requirement is met; incomplete-search and
recommendation cautions remain.

The API retains selected original reviews, including hidden short comments, IDs,
facility ownership, and source metadata. Filtering annotates presentation only.
An explicit response-language request persists across code-switching turns;
each result also stores its review language so later turns do not relabel it.

The user requested Qwen-27B for selected-comment translation. The call is pinned
to `qwen/qwen3.8-27b` through a separate OpenRouter client, independently of the
response model. `AGENTS.md` now records this policy. Translation receives only
eligible selected texts, without facility IDs, hidden comments, or private
conversation context. The existing process-provided OpenRouter credential is
required; missing access produces an unavailable status rather than another model.

## Verification actually performed

- Reproduced the old formatter with an invented review. The new regression
  failed because original text and internal source labels appeared in the reply.
- Eight new focused tests cover language choice, filtering, translation model
  selection, emoji preservation, both language directions, timeout, malformed or
  truncated output, and API serialization. Together with existing response and
  serialization checks, the focused run passed 18 tests.
- Updated ordinary renderer/integration tests to assert intact reviews in cards
  and no duplicate quotations in replies. Sealed evaluation cases and historical
  results were not changed.
- Required backend discovery passed 233 tests in 4.400 seconds after the Qwen
  change. The sandboxed attempt stalled in existing FastAPI routing tests and was
  interrupted; the approved host retry passed. Existing deprecation warnings remain.
- `npm --prefix frontend run build` passed with static generation and type checks.
- Browser verification against the built frontend passed four invented API
  fixtures: English translation, Korean translation at 390px width, translation
  unavailable, and no substantive reviews. Checked no visible internal IDs,
  no duplicated review, no English badge, working original toggles, no injected
  links, no horizontal overflow, and no JavaScript page errors. English and
  Korean screenshots were visually inspected. Translation output was mocked.
- Chromium and the local HTTP server required host execution because the sandbox
  blocks browser startup and listening sockets. No live application was contacted.
- `git diff --check` passed. Local style/comment review found no new suppressions,
  unsafe TypeScript casts, or retrieval changes. No independent agent review ran.

Re-run backend checks from the repository root:

```bash
PYTHONPATH=backend backend/venv/bin/python -m unittest backend.tests.test_review_presentation backend.tests.test_evidence_response backend.tests.test_chat_response_privacy
backend/venv/bin/python -m unittest discover -s backend/tests -p 'test_*.py'
npm --prefix frontend run build
```

## Limits and next action

Korean filtering is a heuristic, not morphological or semantic analysis. It
retains text with at least 12 Hangul syllables, or at least three letter-bearing
units containing six Hangul syllables. Other text needs at least three words.
This preserves longer Korean sentences without spaces while excluding brief
praise and isolated jamo. It can still admit unhelpful longer text and exclude
short useful warnings; source records and risk metadata remain intact.

Translations use one request per response, at most 20 distinct reviews and
16,000 source characters, with a 15-second timeout and no retries. Comments
outside this budget remain available as originals. Invalid output, truncation,
changed emoji sequences, or changed numbers produce an unavailable status.
These checks cannot prove that negation, staff roles, and every nuance were
translated faithfully. No live Qwen translation was evaluated in this task.
Language handling covers the product's English/Korean response languages, using
explicit request patterns and the established conversation language.

No BGE-M3 tuning, corpus changes, live retrieval verification, deployment,
hardware change, or main-branch merge occurred. Earlier evaluators that demand
literal quotations in reply text may fail under the new card-based presentation;
they remain unchanged and are not reported as passing live evaluations.

Next action: inspect a bounded live Qwen translation of selected comments for
faithfulness, then verify the actual API and cards on the assessment deployment
through a separately authorized deployment workflow.
