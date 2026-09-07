# Assessment baseline

- Source branch: `local/rag-visible-evidence-20260906`.
- Starting commit: `64e1fc2ff1d5d8d51be5cadba6e9f308b9e8ce8f`.
- Initial inspection UTC: `2026-09-06T21:13:15Z`.
- Baseline recorded before tracked assessment edits: `2026-09-06T21:14:22Z`.
- Assessment branch/worktree: `ncs`, `/tmp/seouldoc-ncs`.
- Original worktree: `/home/valerian/Seoul/SeoulDoc`.
- Remote: `git@github.com:ValerianFourel/SeoulDoctor.git`.
- Observed `origin/main` equals the starting commit. Local `main` is older at
  `4ef17c64875cf2cf6730a36cfd4cf4a2015ff493`; it was not used.
- No fetch or remote visibility check ran. Remote-tracking refs are local evidence.

`ncs` did not exist. An isolated worktree preserves these excluded original edits:
modified `AGENTS.md` and `docs/PROJECT_STATE.md`; untracked
`SeoulDoc_Project_Overview.png`, `SeoulDoc_Project_Overview.pptx`,
`docs/seouldoc-project-deck.html`, `generate_project_overview.py`, root
`package.json` and `package-lock.json`, and retriever drafts
`Dockerfile.incomplete`, `production.pinned-wrapper-draft.py`,
`production_draft.py`, `requirements-job.txt` under `services/retriever/`.
No stash, reset, or source-worktree edit was used. The user-supplied current
AGENTS instructions still govern this session despite the older committed copy.

## Checks actually performed

All run from the isolated committed baseline with existing dependencies reused.

| Check | Result |
| --- | --- |
| Focused evaluator unittest command in [reviewer setup](reviewer-setup.md) | PASS, 36 tests, 0.148 seconds |
| `npm --prefix frontend run build` | PASS, Next.js 14.2.35, static pages generated and types checked |
| `bash -n scripts/setup_codex_cloud.sh` | PASS, syntax only; bootstrap not executed |
| Initial `git diff --check` | PASS |
| Transit fixture tracked at `backend/tests/fixtures/20260902-mapoderm-transit.json` | Present |
| Standalone frontend lint | Not run; ESLint executable/config absent in inspected environment |
| Full backend suite / retriever pytest | Not run for documentation-only setup |
| Live app, model, GPU, or remote evaluation | Not run |

Python 3.12.11, Node v24.15.0, npm 11.12.1, and Codex CLI 0.153.4 were
observed. `gh` and a standalone `pstack` executable were not on PATH.
Frontend dependencies reuse the tracked `frontend/package-lock.json`; backend
uses `backend/requirements.txt`, with no Python lockfile found.

## Limitations and conflicting records

Earlier environment text says an ignored transit fixture is missing, but the
later project update and committed fixture resolve that claim. Local `main` and
older handoff branch names do not identify the chosen baseline.
The excluded deployment update reports retrieval readiness HTTP 503 on
2026-09-06. It is historical context, not a live check in this assessment.
No end-to-end success is claimed. Private corpus/index artifacts, external model
access, and retriever/reranker availability prevent repository-only live review.
The existing README's launch-ready wording is not an assessment validation.
