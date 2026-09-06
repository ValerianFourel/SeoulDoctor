# Reviewer setup

The repository can support offline tests and a frontend build after dependency
installation. A fresh clone cannot produce live recommendations using Git alone.
No credential-free demo mode was added in setup.

## Existing local environment

From `/tmp/seouldoc-ncs`, Python reuses the original ignored virtual environment
through `backend/venv`. The temporary frontend dependency link was used for the
build and removed before commit. Recreate it when continuing locally:

```bash
ln -s /home/valerian/Seoul/SeoulDoc/frontend/node_modules frontend/node_modules
```

These are machine-local conveniences, not submission dependencies. Keep the
symlink out of staging. No .env or private artifact is linked into the worktree.

## Fresh checkout

From repository root, use the existing manifests; do not install pstack to run
SeoulDoctor. Dependency installation requires network access.

```bash
python3 -m venv backend/venv
backend/venv/bin/python -m pip install -r backend/requirements.txt
npm --prefix frontend ci
```

Python has a requirements file rather than a lockfile, so a future fresh install
may resolve differently from the verified local Python 3.12.11 environment.
The frontend lockfile was reused with Node v24.15.0 and npm 11.12.1.

Run inexpensive offline checks:

```bash
backend/venv/bin/python -m unittest backend.tests.test_grounded_bilingual_suite backend.tests.test_grounded_journey_grader backend.tests.test_patient_journey backend.tests.test_qwen_likert_judge
npm --prefix frontend run build
bash -n scripts/setup_codex_cloud.sh
```

After application edits, run the required full backend suite and frontend build:

```bash
backend/venv/bin/python -m unittest discover -s backend/tests -p 'test_*.py'
npm --prefix frontend run build
```

For retrieval-service edits also run
`backend/venv/bin/python -m pytest -q services/retriever`.
`npm --prefix frontend run lint` is declared, but standalone ESLint is missing
in the inspected environment; do not claim a standalone lint pass.

## Application startup, conditional on data and services

Use two terminals from the repository root:

```bash
backend/venv/bin/python -m uvicorn main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000 npm --prefix frontend run dev
```

Open http://localhost:3000 and API docs at http://localhost:8000/docs.
These commands were inspected, not live-tested during setup. Frontend static
build success does not prove chat works. The production config exports static
files, so use the dev command above for local review rather than next start.

Live dependencies include the private facility/review Parquet snapshots,
Chroma and immutable search indexes; model provider access such as
`OPENROUTER_API_KEY` with `LLM_PROVIDER=openrouter`; and configured BGE-M3
retriever/reranker endpoints and authentication. Geocoding paths may require
`GOOGLE_MAPS_API_KEY` or `KAKAO_REST_API_KEY`. Configure approved process
environment values privately, never in planning files or logs.

The private release Dataset is `ValerianFourel/seouldoc-app-release-20260905`,
revision `3911d79dc31e6a6ccfa3f64a7e401b88893bf66a`. Private evaluation
checkpoints are in `ValerianFourel/seouldoc-eval-handoff`. See
[environments](../docs/ENVIRONMENTS.md) and
[handoff](../CODEX_CLOUD_HANDOFF.md) for authorized restoration details.
The Cloud setup script downloads private artifacts and was not executed here.
Existing artifact revisions must be verified before reuse; never regenerate
embeddings merely to get a demo running. Service health is unverified here.

## Bounded proposal for implementation

For the selected journey, consider 2–3 fictional facilities with invented
comments and stable fixture evidence IDs. Cover one consequential preference,
one refinement, and conflicting/missing evidence. Run real state, selection,
and rendering logic where possible; substitute only the unavailable dependencies.
Show a persistent “Synthetic demo data; no live retrieval” label and document
which stages are simulated. Keep the fixture separate from sealed evaluations.
This is proposed, not built, and cannot establish live retrieval quality.

## Review the assessment delta

```bash
git diff --stat 64e1fc2ff1d5d8d51be5cadba6e9f308b9e8ce8f..ncs
git diff 64e1fc2ff1d5d8d51be5cadba6e9f308b9e8ce8f..ncs -- README.md planning docs/PROJECT_STATE.md
git log --oneline 64e1fc2ff1d5d8d51be5cadba6e9f308b9e8ce8f..ncs
git diff --check 64e1fc2ff1d5d8d51be5cadba6e9f308b9e8ce8f..ncs
```

For final review, inspect the complete diff without the setup-only path filter.
Local project skills are pinned in skills-lock.json and explained in starter.md.
The planning and proof skills were used directly; no slash command or standalone
pstack binary is required. Debugging/review skills can guide later scoped work.
