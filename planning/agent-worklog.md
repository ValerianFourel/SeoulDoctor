# Agent worklog

This is a factual worklog, not a verbatim session export.

The user delegated assessment setup to Codex, including baseline inspection,
branch isolation, tooling checks, reviewer/planning artifacts, and a setup commit.
The user explicitly deferred product implementation. One agent performed this
work; no subagents or separate model review runs were launched.

On 2026-09-06, Codex read the supplied and local AGENTS instructions, project
state, environments, and Cloud handoff, including the latest dated records.
It inspected Git state, README, startup configuration, dependency manifests,
installed skills, and CLI help without reading credential files or datasets.
It created an isolated branch at current HEAD and reused existing environments.

Verified results: 36 focused offline tests passed; frontend production build
passed; setup shell syntax passed. No live evaluations, paid resources, or
application changes occurred. See baseline.md for exact scope and omissions.

Corrections during setup:

- Git worktree creation initially failed because sandboxed Git metadata was
  read-only. The approved retry created ncs at the same requested commit.
- A filename assumption used next.config.js; the actual file is next.config.mjs.
  Inspection was corrected. Some targeted searches named nonexistent test paths;
  these were inspection errors, not product defects or failed tests.
- The dependency symlink for frontend/node_modules appeared as untracked because
  the existing ignore rule covers directories. It is excluded from staging and
  removed after checks; no dependency content is committed.
- Older missing-transit-fixture notes conflict with the committed fixture and
  later project update. The baseline records the resolved status explicitly.

Applied skill guidance: unslop for prose, sequence-verifiable-units for planning,
prove-it-works for artifact review, and TypeScript guidance during API URL
inspection. Read root-cause debugging guidance for later use; no product bug
was diagnosed. pstack's 52 project skills and skills-lock.json exist; no global
model configuration or skill installation was performed.

Codex wrote the planning files and README entry and reviewed their paths,
commands, claims, and diff scope. Future implementation, debugging, recording,
public-access checks, and session-log publication remain pending.

## Deployment preparation follow-up

The user authorized a separate assessment Hugging Face deployment and GPU
retrieval milestone. Codex inspected deployment sources, current official Hub
documentation, and process credential presence. HF_TOKEN and OPENROUTER_API_KEY
were absent. It prepared a branch-scoped source synchronizer and GitHub workflow,
with four focused deployment-boundary tests passing. No actual Space or GPU
was created. The existing corpus is to be reused. GPU inference and visible
comment proof remain pending; the current health endpoint alone cannot prove CUDA.

Local verification completed: 220 backend tests passed in 4.115 seconds,
including four new deployment-boundary tests. Frontend production build passed.
Both source-manifest dry runs passed. Live provisioning and sync remain blocked
by absent credentials; no remote success is claimed.

Branch publication checkpoint: ncs commit
e0f99935ac68a57e4dea3bc36bde70834302ef1d was pushed to origin/ncs.
The initial sandbox attempt failed DNS; the approved retry succeeded.
The pre-push hook reported three nonblocking medium PII/internal findings
without details; public-submission review remains pending. No Space sync ran
and no GitHub secret or enabling variable was configured by this session.

## Private ncs Spaces created, 2026-09-06

Created ValerianFourel/SeoulDoctor-ncs and
ValerianFourel/SeoulDoctor-ncs-retriever as private Docker Spaces on CPU Basic.
The user will select GPU hardware manually in the retriever Space settings.
No GPU allocation or persistent storage purchase was performed. Required
process-authorized credentials were set as Space secrets without printing them.
The original Space remains unchanged at 8fb1fc893eaac192bb514047d8e0577055a425df.

The ncs application now restores its pinned private release at startup instead
of depending on an external mount. The retriever explicitly selects CUDA and
performs a bilingual embedding probe exposed at /ready/gpu. It will not become
ready on CPU. Local checks: 222 backend tests and five retriever tests passed;
frontend production build passed. Live GPU inference and comment display are
pending the manual GPU upgrade. Automatic GitHub sync still needs NCS_HF_TOKEN
and NCS_HF_SYNC_ENABLED configured in GitHub; source upload is manual for now.
