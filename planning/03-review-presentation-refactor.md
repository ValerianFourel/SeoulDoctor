# Note 3: Simplify the review-presentation path

## Workflow

- [x] Read poteto-mode principles, coordination, environments, and handoff.
- [x] Map the selected runtime path with how, using a sequential local pass.
- [x] Inventory dead paths, normalization, validation, wrappers, state, and branches.
- [x] Pin current behavior before editing.
- [x] Select the design and explain the deletions.
- [x] Implement in separately verified units.
- [x] Run deslop and no-comments locally and record verification.

The side conversation prohibits subagents. How, deslop, and no-comments run
locally; no independent review is claimed. This pass preserves the preceding
uncommitted presentation work on ncs. It does not revisit retrieval or deployment.

## Runtime map

`backend/main.py:chat_endpoint` selects the response language, then the search
path calls `finalize_evidence_response`. That function copies selected evidence,
checks facility ownership, retains recommendation cautions, and invokes
`prepare_review_presentations`. The latter marks display eligibility and sends
one bounded batch of selected texts to Qwen-27B when translation is needed.
`serialize_results_for_chat` keeps provenance and presentation fields in JSON.
`frontend/components/ReviewEvidence.tsx` displays eligible reviews once and offers
originals through a toggle. Translation does not change corpus records or ranking.

## Findings and selected design

- Delete the now-unused `detect_language` import from main. The old utility has
  other potential users and is not deleted from its owning module.
- Delete the redundant Hangul regex in the English translation decision. Every
  precomposed Hangul syllable already satisfies the existing non-ASCII alphabetic
  predicate, so the extra scan cannot change the result.
- Select the presentation status first and construct its object once. Then enter
  the translation queue only for unavailable translations. This removes repeated
  object construction from the three eligibility branches.
- Keep ownership validation and translation validation. They protect different
  trust boundaries and are not duplicate checks.
- Keep the `_symbols` helper. Its two uses express the same preservation rule;
  inlining it would duplicate the predicate.
- Keep the request character counter and deduplication map. They enforce bounded
  work and share a translation across duplicate source text.
- No repeated normalization or dead runtime wrapper was demonstrated in scope.

The selected design preserves module boundaries and signatures. Architect is
not triggered; arena is not needed because no competing target design remains.
Laziness Protocol favors these small deletions over introducing another layer.
Model the Domain keeps the existing presentation states as the organizing shape.

## Verification

Before editing, 20 focused tests passed, including new characterization of
shared translations for duplicate source text and preservation of originals when
the translation budget is exceeded. The same 20 tests passed after each unit.

A temporary before/after harness compared complete evidence objects, provider
request payloads, and client options for 360 deterministic synthetic cases.
It covered both response languages, successful output, timeouts, malformed JSON,
wrong output lengths, truncated output, missing clients, duplicate text, hidden
reviews, and oversized reviews. All results matched. An additional check compared
the English translation decision for all 11,172 precomposed Hangul syllables;
all matched. No private review or model call was used.

Required backend discovery passed 235 tests in 4.148 seconds on the approved
host runner. The existing sandbox routing-test limitation was already reproduced
in the preceding unit, so this pass used that verified execution route.
The frontend production build and `git diff --check` passed. No frontend source
changed during this refactor; the preceding browser checks were not rerun.

Deslop and no-comments were performed locally over the incremental cleanup.
No new wrappers, compatibility paths, suppressions, or comments were added.
Zero comments were deleted or restored. The two existing comments in the
presentation module explain Korean tokenization and provider-failure privacy;
both still explain non-obvious constraints. No independent comment-sicko agent
ran, as prohibited by this side conversation.

The cleanup removes one unused import and one redundant regex scan, centralizes
three initial presentation-object assignments into one, and reduces translation
queue nesting by one level. It adds two net lines in the presentation module;
the benefit is less duplicated structure, not a large line-count reduction.
No alternative design or behavior change was implemented and reverted.

The existing uncommitted feature changes remain uncommitted. No PR, merge,
push, deployment, or hardware change occurred in this pass. No live Qwen or
BGE-M3 result is implied. The remaining next action is the bounded live
translation-faithfulness check described in Note 2.
