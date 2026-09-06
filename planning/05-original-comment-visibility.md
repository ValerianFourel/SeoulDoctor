# 05. Original comments on facility cards

Owner: root continuation of the presentation handoff. Branch `ncs`, worktree
`/home/valerian/Seoul/SeoulDoc/.worktrees/ncs`, committed HEAD
`75a2c83bec70982f48cecdb4029823832794d1bf`. This fix and the preceding presentation
work remain uncommitted; no prior draft was discarded or published.

## Findings

1. `ConstraintEvidenceRetriever.collect` returns empty groups with
   `no_constraints` when the request has no review-dependent requirements.
   Specialty/location-only requests therefore attach no original reviews even
   when the facility's source index contains them. A synthetic regression through
   the real ranking adapter reproduced the empty array before the fix.
2. The local Google translation draft preserves originals with status
   `unavailable` when no key is configured. Missing translation is not the cause
   of the empty API array. HF and model credentials were present in the process;
   `GOOGLE_TRANSLATE_API_KEY` was absent. No environment file was read.
3. Authenticated source inspection found the ncs Space RUNNING at
   `d765a828f621d5fb1b9273b960748e63100a8e26`. Its collector matched local source,
   but `backend/review_presentation.py` and `frontend/components/ReviewEvidence.tsx`
   both returned 404. Local draft changes are not deployed there.
4. The current frontend draft failed its production build because the Unicode
   regex literal requires an ES6 TypeScript target. Constructing the same regex
   with `new RegExp` preserves the letter check with the existing target.

## Change

The ranking adapter now attaches general originals for the no-review-constraints
path, using a scope-checked read of the existing immutable evidence index.
It scans at most 100 original records per shortlisted facility in source order
and attaches at most three with letters. It reuses the local display filter,
retains short Korean comments and negative feedback, and resolves source IDs
through the existing resolver. The API retains original text, facility ownership,
source locator, and source digest. These samples do not enter facility rank
fusion, preference coverage, or supporting-evidence groups. They are general
patient reviews, not a claim of relevance or suitability. Empty sources remain
empty. No corpus, embeddings, ranking weights, or translation provider changed.

The first-100 scan is deliberately bounded: facilities whose first 100 records
are all noise may still have no displayed comment. Preference-dependent requests
continue through the existing retrieval pipeline and may have their own missing
evidence failures. This change does not hide incomplete-search warnings.

## Verification

- New attachment regression failed before the fix with an empty evidence array.
- Full backend discovery passed **240 tests** after the fix.
- Real synthetic index test verified source resolution, scope rejection,
  deterministic selection, empty input, and quota bounds.
- Adapter-to-presentation-to-public-serialization regression verified Korean
  originals without a translation key, source digest and ownership, empty
  facilities, and unchanged facility order when sample reviews are removed.
- Frontend production build passed after the regex fix. The initial sandbox
  attempt exited without a diagnostic; the host build exposed the actual error.
- Chromium checks passed six synthetic response cases: English translation,
  Korean translation at mobile width, translation unavailable, noise-only input,
  absent presentation metadata, and a short Korean original. No page errors or
  horizontal overflow; internal IDs stayed hidden and original toggles worked.
- `git diff --check` passed. No live chat, translation, or evaluation ran.

A test initially expected an oversized quota to raise, but the existing helper
clamps it; the test was corrected to verify that bound. The earlier browser
fixtures expected 'Very good' to be hidden; expectations were corrected for the
user's current short-comment policy. These are test corrections, not evidence
of additional product failures. Synthetic screenshots remain outside Git.

## Next action

Review and commit the combined presentation draft and this attachment fix, then
sync that exact commit to the existing ncs Space and verify a specialty/location
request's API and cards. Do not sync HEAD now: it excludes these local changes.
No deployment, hardware change, main merge, or private data publication occurred.

## Deployment authorization

The user requested committing and deploying the combined draft to the existing
ncs GPU Space, including removal of the Donate button from HeaderMenu. This
supersedes the prior local-only checkpoint. The remote main reference remains
64e1fc2ff1d5d8d51be5cadba6e9f308b9e8ce8f and is already in this branch's history.
Deployment completion and live checks will be recorded after verification.
