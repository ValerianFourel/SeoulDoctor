# Combined ncs GPU deployment

## Current deployment checkpoint, 2026-09-08

The NCS Space is public and running on `t4-medium`, verified at source and
runtime revision `4138eb056b80f4689fb0ef87244cf128c44280b2`. The original
application remains separate. PR #4 merged the response changes into `ncs` at
`0bb08702bb07f3dc5ddbed943acfa2460c1db170`, but Actions run `34212865685`
skipped the disabled automatic sync. The response changes are not live yet.

The followup push `67dce71fab57f0a0ba61fff15fd590f1640e6779` used the explicit
deployment marker described below. Actions run `34214729943` executed but
failed before any upload because `NCS_HF_TOKEN` supplied no process credential.
The user subsequently specified the existing `HF_TOKEN` secret for deployment;
the workflow reference has been corrected to use it. A new explicit deployment
push `d7587caeb0539ac0a2e282982f95f16e96fb137b` launched Actions run `34215299377`,
which also received no token and failed before upload. A clean `ncs` checkout
for direct deployment passed its manifest dry run. Loading the local token from
`backend/.env` requires a specific exception to the current AGENTS prohibition;
that question is pending. Until the new running revision is verified, the
response changes must not be described as live.

For the already public target, run the following from a clean, committed `ncs`
checkout with an approved `HF_TOKEN` in the process environment:

```bash
backend/venv/bin/python scripts/sync_ncs_spaces.py app
backend/venv/bin/python scripts/sync_ncs_spaces.py app --apply --allow-public
```

For one deployment through GitHub Actions, prefix the pushed `ncs` commit title
with `[deploy ncs] `. This explicit request runs the existing workflow using
`HF_TOKEN`, even when automatic sync is disabled. Normal pushes still require
`NCS_HF_SYNC_ENABLED=true`. Checkout remains pinned to the pushed revision and
the workflow only deploys `ncs` to the fixed NCS Space. It does not modify `main`.

`--allow-public` acknowledges the target's current visibility. It does not change
visibility, hardware, or the fixed target. The source branch marker and parent
revision checks still apply. The sections below record the earlier private
deployment setup and its historical verification results.

## Original combined deployment setup

The active target is the private Space
https://huggingface.co/spaces/ValerianFourel/SeoulDoctor-ncs-retriever.
Despite its retained name, it now receives the complete application bundle.
The user requested the website and GPU retrieval at this same URL and selected
NVIDIA L4 hardware manually. This session does not alter that hardware.

## Source and runtime

Fetched main is 64e1fc2ff1d5d8d51be5cadba6e9f308b9e8ce8f. The frontend and
core search remain identical to main. Assessment changes run the website/API
on port 7860 and BGE-M3 on loopback port 7861 in the same container.
The backend uses its existing semantic HTTP client against that loopback service.
Separate Python environments preserve application and model dependency versions.

The build variable NCS_ENABLE_GPU=true includes the GPU environment. The runtime
supervises both processes and terminates the container if either exits.
GET / serves the exported main frontend. GET /health describes the application;
GET /ready/gpu forwards the live GPU service's bilingual encoding proof.
The core /ready/retrieval contract still includes reranking, which is a separate
requirement from BGE-M3 readiness and must not be claimed passed without checking.

The app restores the existing private release at
3911d79dc31e6a6ccfa3f64a7e401b88893bf66a. BGE-M3 reuses index revision
a6a3ab6f70c67c15090d175efd4ecdea553b329e and model revision
5617a9f61b028005a4858fdac845db406aefb181. No corpus embeddings are rebuilt.
HF_TOKEN, OPENROUTER_API_KEY, OPENAI_API_KEY, GOOGLE_MAPS_API_KEY, and
KAKAO_REST_API_KEY are private Space secrets. The user explicitly authorized
loading the additional app keys from the local environment file; values were
never printed or committed.

## Deployment

From ncs with HF_TOKEN already in the process environment:

```bash
backend/venv/bin/python scripts/sync_ncs_spaces.py app
backend/venv/bin/python scripts/sync_ncs_spaces.py app --apply
```

The first command previews committed files; the second uploads them to the fixed
assessment Space. The obsolete API-only root files are removed by this sync.
The GitHub workflow now targets only this combined Space. Automatic deployments
still need NCS_HF_TOKEN and NCS_HF_SYNC_ENABLED=true configured in GitHub.
This session uses direct uploads from the committed ncs branch.

## Verification checkpoint

Before replacement, / returned 404 while /ready/gpu confirmed NVIDIA L4,
cuda:0, two encoded queries, 1024 dimensions, and nonempty sparse embeddings.
That demonstrated the problem was the API-only deployment, not unavailable GPU.
The earlier app Dockerfile edit also incorrectly replaced HEALTHCHECK CMD and
omitted startup CMD; this was corrected and has a dedicated regression check.

Local checks passed: 225 backend tests in 4.169 seconds, five retriever tests,
and frontend production build. FastAPI routing tests stalled in the sandbox;
the approved host retry passed. Live combined deployment verification is pending.

The original SeoulDoctor Space remains unchanged. No additional GPU or storage
was purchased. The user manages the existing L4 lifetime.
