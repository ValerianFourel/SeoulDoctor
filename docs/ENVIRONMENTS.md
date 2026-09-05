# SeoulDoc environments

Updated: 2026-09-05

Use this reference to reproduce the local or Cloud environment without copying
credentials into Git.

## Repository

- GitHub: `https://github.com/ValerianFourel/SeoulDoctor`
- Git remote: `git@github.com:ValerianFourel/SeoulDoctor.git`
- Coordination branch:
  `coordination/codex-cloud-eval-20260905`
- Code baseline:
  `d77ffac10639ef6c4c6f7d796b0a84ae1f84811a`

## Target Codex Cloud environment

- Name: not supplied
- Link: not supplied
- Source received:
  `[PASTE ENVIRONMENT NAME AND LINK]`
- Status: unresolved. Replace the placeholder after the environment is known.

Do not claim that this Cloud environment is synchronized yet. The Cloud
session must check out the reported coordination commit and complete the
handshake below.

## Local Linux environment

The active local repository uses Linux. Run commands from the repository root.

Install backend dependencies:

```bash
python -m venv backend/venv
backend/venv/bin/python -m pip install --upgrade pip
backend/venv/bin/python -m pip install -r backend/requirements.txt
```

Install frontend dependencies:

```bash
npm --prefix frontend ci
```

Run focused evaluation tests:

```bash
backend/venv/bin/python -m unittest backend.tests.test_grounded_bilingual_suite backend.tests.test_grounded_journey_grader backend.tests.test_patient_journey backend.tests.test_qwen_likert_judge
```

Run the full local checks:

```bash
backend/venv/bin/python -m unittest discover -s backend/tests -p 'test_*.py'
backend/venv/bin/python -m pytest -q services/retriever
npm --prefix frontend run build
bash -n scripts/setup_codex_cloud.sh
```

The retriever test command needs `pytest`. Install it in the local virtual
environment if it is absent:

```bash
backend/venv/bin/python -m pip install pytest
```

## Codex Cloud environment

Select the `universal` container image.

For the first coordination chat on this commit, use this bootstrap setup
script:

```bash
python -m venv backend/venv
backend/venv/bin/python -m pip install --upgrade pip
backend/venv/bin/python -m pip install -r backend/requirements.txt
npm --prefix frontend ci
```

This bootstrap installs dependencies and lets the Cloud agent repair the
clean-checkout blocker below. It does not restore the private release, run the
test gate, or authorize a live evaluation. Do not report full setup as passed.

After the fixture contract is repaired, replace the environment setup script
with:

```bash
bash scripts/setup_codex_cloud.sh
```

Automatic setup is not sufficient because this project must also restore two
private Datasets. The `universal` image, setup lifecycle, environment variable
lifetime, and caching behavior are documented in the
[official Codex Cloud environment reference](https://developers.openai.com/codex/cloud/environments).

Leave the maintenance script blank for the first uncached run. Reset the
environment cache when switching from the bootstrap setup to the full setup,
or after changing environment settings. Allow agent network access to:

- `huggingface.co`;
- `hf.co`;
- `*.hf.space`;
- `openrouter.ai`.

The setup script installs Python and frontend dependencies. It downloads the
pinned private release to `.codex-handoff/release/` and restores the latest
private checkpoints to `.codex-handoff/checkpoints/`.

Configure these names as Codex Cloud environment variables:

```text
HF_TOKEN
OPENROUTER_API_KEY
SEOULDOC_EVAL_AUTH_TOKEN
SEOULDOC_APP_ENDPOINT
SEOULDOC_RELEASE_DATASET
SEOULDOC_RELEASE_REVISION
SEOULDOC_RESULTS_DATASET
```

Use these non-secret values:

```text
SEOULDOC_APP_ENDPOINT=https://valerianfourel-seouldoctor.hf.space/chat
SEOULDOC_RELEASE_DATASET=ValerianFourel/seouldoc-app-release-20260905
SEOULDOC_RELEASE_REVISION=3911d79dc31e6a6ccfa3f64a7e401b88893bf66a
SEOULDOC_RESULTS_DATASET=ValerianFourel/seouldoc-eval-handoff-20260905
```

Do not paste credential values into tracked files, prompts, logs, or result
artifacts. Codex Cloud removes values stored as Cloud Secrets before the agent
phase. The eval agent needs `HF_TOKEN`, `OPENROUTER_API_KEY`, and
`SEOULDOC_EVAL_AUTH_TOKEN` during that phase, so configure those names as
environment variables in the private Cloud environment.

### Known clean-checkout blocker

The setup script is not yet a green clean-checkout gate. Its final focused test
expects this ignored local file:

```text
backend/tests/evaluation_runs/20260902-mapoderm-transit.json
```

In the isolated coordination worktree, the command ran 36 tests: 35 passed and
one errored when that file was absent. The script currently restores the
release and result Datasets into other paths and does not restore this fixture.

Resolve this on a dedicated `cloud/setup-transit-fixture` branch. Preserve the
assertion and the observation's integrity. A suitable repair may pin the file
in a private Dataset and restore it with a recorded hash, or replace the
ignored-file dependency with a tracked immutable test fixture after review.
Do not skip the test or call setup complete while it exits nonzero.

## Files and services that are not in Git

The Cloud checkout cannot read these local items:

- `backend/.env` and every credential value in it;
- `backend/local_facilities_cache.parquet`;
- `backend/local_reviews_cache.parquet`;
- `backend/chroma_db/`;
- `backend/search_indexes/`;
- `backend/tests/evaluation_runs/20260902-mapoderm-transit.json`;
- local `.audit/` directories and checkpoints other than the tracked
  `.audit/seouldoc-live-eval.tsv` and artifacts uploaded to the private result
  Dataset;
- local Docker images and containers;
- the local Codex main-thread process;
- untracked project-deck files and untracked retriever draft files.

The Cloud setup restores the four large data artifacts from the private release
Dataset. The seeded redacted checkpoint is in the private result Dataset. The
current setup does not place the ignored transit observation at the path the
focused test expects. Neither Dataset gives Cloud access to local processes,
current GPU job status, or checkpoints created after the last upload.

Query Hugging Face for the current Space and job state before launching paid
hardware. Do not infer current state from a one-hour job ID in an old
checkpoint.

## Handoff and synchronization handshake

The producer must push a dedicated task branch and report its exact commit. The
consumer then runs or verifies the equivalent of:

```bash
git fetch origin coordination/codex-cloud-eval-20260905
git checkout coordination/codex-cloud-eval-20260905
git rev-parse HEAD
git merge-base --is-ancestor d77ffac10639ef6c4c6f7d796b0a84ae1f84811a HEAD
sed -n '1,260p' docs/PROJECT_STATE.md
sed -n '1,260p' docs/ENVIRONMENTS.md
```

The consumer reports:

- the branch and `git rev-parse HEAD` value;
- confirmation that both coordination files were read;
- the task it accepts;
- the tracked files and result path it owns;
- its first concrete action.

Only after that acknowledgement may either session say that the consumer is
synchronized.

## Branch and ownership rules

- Use `local/<task>` for a local implementation task.
- Use `cloud/<task>` for a Cloud implementation task.
- Use one worktree per simultaneous code-changing task.
- Declare tracked-file ownership in `docs/PROJECT_STATE.md` before editing.
- Reserve `cloud/setup-transit-fixture` for the clean-checkout fixture repair.
  Do not include RAG changes on that branch.
- Keep parallel patient runs code-free and give each one a unique result path.
- If two changes need the same tracked file, run them sequentially or assign
  one integration owner. Do not rely on two agents agreeing to take turns.
- Before handoff, update `docs/PROJECT_STATE.md`, commit the task branch, and
  record the next action.
