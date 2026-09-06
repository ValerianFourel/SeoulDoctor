# Assessment Hugging Face deployment

Requested on 2026-09-06. The next milestone is a separate SeoulDoctor instance
following ncs, with accessible GPU BGE-M3 query encoding and one original
facility-owned comment returned. A random sample is sufficient for the initial
plumbing check. It does not prove that the most relevant comment was selected.

## Current state

Deployment is blocked, not completed. The current process has no HF_TOKEN or
OPENROUTER_API_KEY. No credentials were loaded from files. No Space was created,
no live resources were inspected, and no GPU charges were incurred in this unit.
The original SeoulDoctor Space remains untouched.

Prepared names, subject to checking existing resources before creation:

- Private application: ValerianFourel/SeoulDoctor-ncs, CPU Basic.
- Private GPU retriever: ValerianFourel/SeoulDoctor-ncs-retriever.

Use the existing two-service architecture. The GPU embeds incoming queries;
the review corpus already has BGE-M3 embeddings. Reuse the private review index
at a6a3ab6f70c67c15090d175efd4ecdea553b329e, model revision
5617a9f61b028005a4858fdac845db406aefb181. Verify its manifest and matching source
digest before wiring it to the app release
3911d79dc31e6a6ccfa3f64a7e401b88893bf66a. No corpus rebuild is planned.

## Branch synchronization

Hugging Face builds its own Space repository. The prepared GitHub workflow
.github/workflows/ncs-spaces.yml syncs the exact triggering ncs commit into
those repositories. The source revision is recorded as ncs-source.json.
This is file synchronization, not a native pointer to a GitHub branch.

scripts/sync_ncs_spaces.py defaults to an offline manifest preview:

```bash
backend/venv/bin/python scripts/sync_ncs_spaces.py app
backend/venv/bin/python scripts/sync_ncs_spaces.py retriever
```

It reads selected committed blobs only and excludes tests, environment files,
private data, planning/session logs, and unrelated workspace changes. The
retriever uses its own production files at the Space root. Existing Space
README configuration is preserved. Synchronization requires an existing private
Space with the variable NCS_SOURCE_BRANCH=ncs. It never allocates hardware,
creates repositories, sets secrets, or changes the original application.

After provisioning and verification, configure the GitHub Actions secret
NCS_HF_TOKEN and repository variable NCS_HF_SYNC_ENABLED=true, then publish ncs.
No GitHub credential is available in this process, and this configuration has
not been performed. The workflow stays disabled until the variable is set.
Once enabled, every push may rebuild the GPU Space and incur runtime charges;
keep hardware paused outside authorized development windows.

## Resume sequence

1. Inject HF_TOKEN and OPENROUTER_API_KEY through the agent environment, then
   resume. Inspect existing Spaces, jobs, permissions, and current pricing.
2. Reuse a suitable idle resource only if isolated from the original app;
   otherwise create the two private assessment Spaces. Do not overwrite an
   existing namesake without verifying ownership and purpose.
3. Configure the app's read-only release mount at /mnt/seouldoc-release.
   The current Dockerfile requires this mount and copies into ephemeral disk.
   Verify the provider's current mount API against the installed SDK before use.
   Preserve pinned source revisions. Do not buy persistent storage.
4. Configure retriever Dataset/model revisions and read access. Inspect the
   manifest to get its exact release ID and raw-review digest. Configure
   application retrieval URL/release identity and secrets privately.
5. Build on free CPU hardware first where feasible. Allocate one temporary
   T4 Small development window only after preflight, bound to 55 minutes and
   explicitly pause/downgrade at its end. Idle sleep is useful but is not a hard
   spending cap; visitors can wake a sleeping paid Space. Track actual charges.
6. Prove CUDA query execution, model revision, dense/sparse output, evidence-ID
   resolution, and facility ownership. Existing /health alone does not prove
   CUDA use. Select a sample with a recorded seed in a separate private run;
   do not publish the review text in Git or alter sealed cases.
7. Check the application loads and the sampled original comment appears with
   its facility. Full recommendation readiness also needs the existing reranker
   contract checked; do not label a BGE component probe as an end-to-end pass.
8. Record Space URLs, exact source/Space commits, results, and cleanup status;
   enable branch sync and continue the relevance improvement.

The installed deployment guide contains stale corpus-build and persistent-disk
advice. The newer retriever guide and current instructions require artifact
reuse and no persistent storage purchase. The installed huggingface_hub 1.3.2
also predates parts of the current online volume API; live inspection is needed.

References checked during preparation:

- https://huggingface.co/docs/hub/spaces-github-actions
- https://huggingface.co/docs/hub/spaces-gpus
- https://huggingface.co/docs/huggingface_hub/guides/manage-spaces

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
