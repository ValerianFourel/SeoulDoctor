# Assessment handoff

Branch `ncs`; worktree `/tmp/seouldoc-ncs`. Owner: Codex assessment setup session.
Owned files: `planning/`, README assessment section, appended project-state entry.
Baseline commit `64e1fc2ff1d5d8d51be5cadba6e9f308b9e8ce8f`. The setup commit is the first commit after this baseline
on ncs, titled `Set up AI-Native Builder assessment on ncs`.
Resolve its exact hash without a self-referential commit field:

```bash
git log --reverse --format='%H %s' 64e1fc2ff1d5d8d51be5cadba6e9f308b9e8ce8f..ncs
git rev-parse HEAD
```

Completed: baseline isolation, reviewer instructions, planning and collaboration
artifacts, CLI export investigation, 36 focused passing tests, frontend build,
and shell syntax check. The setup commit changes documentation only.
Original worktree remains on its original branch with its edits preserved.
Observed integration ref origin/main already equals the baseline; no older main
was merged. Remote changes since that local observation remain unverified.

Unresolved: public repository access, repository-only live reproducibility,
implementation journey choice, demo recording, and active-session export.
No private data was copied, no service revision changed, and no assessment
resources or remote checkpoints exist. Historical revisions are in the existing
handoff and are not freshly verified. No new product/evaluation result exists.

The next session must verify HEAD, read the project-state/environment/handoff
records, and confirm ownership before editing. No receiving session has yet
confirmed synchronization. Open this worktree explicitly; the original checkout
was not switched. Recreate the local frontend dependency link described in
reviewer-setup.md if continuing locally.

Single next action: select one reproducible preference-refinement journey and
identify its smallest measurable improvement, including a bounded synthetic
fixture if needed for reviewer reproduction.

## Deployment follow-up

The setup-only phase is over: the user requested a separate Hugging Face
assessment deployment with GPU retrieval. Source synchronization and its tests
are prepared, but no live deployment has occurred. See hf-deployment.md.
Current next action: inject HF_TOKEN and OPENROUTER_API_KEY through the process
environment, then inspect live resources and provision the isolated Spaces.

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

Combined deployment prepared and locally verified: 225 backend tests passed,
five retriever tests passed, and frontend build passed. The root routing test
proves HTML, existing API routes, and GPU readiness can coexist. User authorized
additional app keys; they were privately configured on the existing L4 Space.
The ncs sync target is now that combined Space only. Live verification pending.
