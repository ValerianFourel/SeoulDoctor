# 04. Session reentry and current scope

This is a handoff record, not a session transcript or proof of deployment.

## Restart from your terminal

Exit the current Codex interface first, then run:

```bash
cd /home/valerian/Seoul/SeoulDoc
backend/venv/bin/python .worktrees/ncs/planning/resume-ncs.py
```

Select the intended NCS conversation in the picker. The installed CLI supports
`resume --all --cd`; `--all` includes sessions started in the original directory.
You can pass an explicit session ID to the script instead of choosing one.
The launcher requires the `ncs` branch and explicitly selects its worktree.
It does not switch branches, restore files, start services, or deploy anything.

The user explicitly authorized loading the three credentials from the original
checkout's `backend/.env` into the resumed process. The launcher uses that file
first, then existing environment variables, and never prints credential values.
It requires `HF_TOKEN`, `OPENROUTER_API_KEY`, and `SEOULDOC_EVAL_AUTH_TOKEN`.
This is a specific exception to the default `.env` restriction, not permission
to display or publish secrets. Environment inheritance does not remove network
restrictions or prove that a token is valid. The optional Google translation
key is not loaded by this launcher; configure it separately when authorized.

Send this message after selecting the session:

> Read AGENTS.md, then docs/PROJECT_STATE.md, docs/ENVIRONMENTS.md, and
> CODEX_CLOUD_HANDOFF.md in that order, including their latest updates. Then read
> planning/04-session-reentry.md and the current planning records. Verify the ncs
> worktree, HEAD, and dirty files before proceeding. Continue the bounded review
> visibility investigation described here, preserving all ongoing work. Check
> credential presence without printing values. Do not change main, hardware,
> repository visibility, embeddings, or retrieval ranking.

## Place within the project

SeoulDoc already provides conversational facility search, retrieval services,
original-review storage, and evaluation tools. The NCS assessment builds on that
product. Its baseline is `64e1fc2ff1d5d8d51be5cadba6e9f308b9e8ce8f`.
The immediate improvement is a review experience that patients can read and
reviewers can trace through the API, before broader recommender improvements.

Observed local state at this handoff:

- Branch: `ncs`.
- Committed HEAD: `75a2c83bec70982f48cecdb4029823832794d1bf`.
- Worktree: `/home/valerian/Seoul/SeoulDoc/.worktrees/ncs`.
- The original checkout is on `local/rag-visible-evidence-20260906` and has
  unrelated changes. Do not move or overwrite them.
- The presentation changes are uncommitted. HEAD alone does not contain them.
- Investigation target: private Space `ValerianFourel/SeoulDoctor-ncs-retriever`.
  Inspect its actual source and runtime revisions before attributing a failure
  to the local draft. Do not alter the original SeoulDoctor Space.

## Current update

The user reports facility results followed by “No original review is available
to quote.” The goal is to find where originals disappear and restore their
display with the smallest reversible change.

The latest user direction supersedes the Qwen translation requirement recorded
in notes 02 and 03. The current local draft uses Google Cloud Translation Basic
with `model=nmt`, an optional `GOOGLE_TRANSLATE_API_KEY`, and a simple local
letter-presence filter. It keeps short meaningful comments and substantive text
with emojis, while excluding numeric, emoji-only, and isolated-jamo noise.
This deliberately relaxes the earlier one-or-two-word exclusion.

Required behavior remains:

- Show available original reviews when translation fails or is unconfigured,
  including responses without the new presentation metadata.
- Label translations, offer originals, and preserve complete source text,
  evidence IDs, facility ownership, and provenance in API serialization.
- Hide internal identifiers from patient-facing text and avoid duplicate full
  reviews in the reply and cards.
- Preserve incomplete-search warnings. Retrieved comments alone do not establish
  service availability. Review language does not establish staff language skills.
- Never invent a quotation when the API contains no original review.

The presentation owner has unfinished changes in `backend/evidence_response.py`,
`backend/review_presentation.py`, `backend/main.py`, `backend/models.py`, related
response/presentation tests, `frontend/components/ChatInterface.tsx`,
`frontend/components/ReviewEvidence.tsx`, `AGENTS.md`, and planning records.
Inspect the full diff before editing. Coordinate any expansion into retrieval
files with their owner. This side conversation does not use subagents.

## Evidence and remaining checks

Notes 02 and 03 record earlier presentation tests and a 235-test backend pass
plus a frontend build. Those results predate the current translation rollback
and do not validate the latest draft. No new application tests ran while writing
this restart handoff. The latest credential inspection confirmed a nonempty
`HF_TOKEN` in the local file, but both authenticated metadata requests ended in
`ConnectionError`. Current token validity and Space state remain unverified by
that check. No deployment was performed by this handoff task.

Investigate the distinction between empty evidence from the backend and evidence
hidden by the UI. One lead is the collector's no-review-constraints path for a
specialty/location-only request. Treat it as a hypothesis until reproduced.
Do not assume translation caused the deployed failure.

After a fix, test reply and cards separately, absent presentation metadata,
translation failure, short Korean/English comments, noise filtering, emojis,
and preservation of source identity through serialization. Run:

```bash
backend/venv/bin/python -m unittest discover -s backend/tests -p 'test_*.py'
npm --prefix frontend run build
```

Inspect the rendered interface. Reuse existing dependencies. Do not launch the
full remote evaluation suite, regenerate embeddings, tune BGE-M3, or change
ranking for this presentation task. Keep private reviews and raw transcripts
out of Git. Record actual checks and limitations in numbered planning notes.
Review and commit only owned changes when ready; deployment status must be
reported separately from local test success. These new handoff files are local
and have not been published to GitHub by this task.

Single next action: reproduce a specialty/location-only request and inspect
whether its API carries original review records before changing the display.
