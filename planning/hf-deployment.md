# Combined ncs GPU deployment

## NCS reliability pushed; deployment failed, 2026-09-08

The reliability implementation is committed at
`b7ed376679c530243eda41b42f420964090f429f` on isolated fix branch
`local/ncs-answer-reliability-20260908`, based on the starting NCS revision
`9036fbc4f00888fe4e9d746b69b419272408aebe`. Local checks passed: 360 backend
tests, 36 required evaluator tests, frontend production build and 58/58 desktop
and mobile browser assertions. Browser/API checks use synthetic fixtures;
unchanged live replay and adaptive evaluation are still pending.

The explicit `HEAD:refs/heads/ncs` push to the verified repository
`github.com/ValerianFourel/SeoulDoctor.git` succeeded at
`67f17dc213e4acedde80940b56df39bf52a45bf0`. Its `[deploy ncs]` title triggered
the existing workflow for `ValerianFourel/SeoulDoctor-ncs-retriever`.
[Actions run 34237945900](https://github.com/ValerianFourel/SeoulDoctor/actions/runs/34237945900)
failed at `Sync application` after checkout and dependency setup passed. Public
annotations show only exit code 1. The unauthenticated job-log endpoint returned
403, so the specific remote failure cause was not confirmed. No deployment
configuration or hardware change was included.

At the preflight, the NCS Space source/runtime remained
`b13d2bdb6eb12153a23453382244630e3746f240`, source marker
`d7587caeb0539ac0a2e282982f95f16e96fb137b`, running on its existing T4 Medium.
The original Space remained
`8909ff673d2e1b5e7df97010f9e309f6a9dc4c7b`. Post-push public checks at 14:38 UTC
confirmed these same source/runtime revisions and healthy old NCS counts. The
reliability implementation is not live. No post-deployment chat, GPU readiness
or browser proof for this implementation is claimed.

Local `HF_TOKEN` and `OPENROUTER_API_KEY` process variables are absent. The
historical dotenv exception below is superseded by the current user instruction:
approved process variables only. Direct deployment and private checkpoint upload
remain blocked until that credential is supplied. This does not establish that
the workflow's secret was absent; that remote detail is unverified.
No expired evaluation budget or historical temporary-resource allowance renews.

## Current deployment checkpoint, 2026-09-08

The user approved process-only loading of the local `HF_TOKEN`. Direct upload
of committed NCS source `d7587caeb0539ac0a2e282982f95f16e96fb137b` succeeded,
creating Space revision `b13d2bdb6eb12153a23453382244630e3746f240`. That exact
revision is now running and healthy, with GPU readiness verified on Tesla T4.
All 89 managed files match the intended source. Four live API requests passed
36 response-preservation checks, and one further browser conversation passed
desktop/mobile checks. The original Space and hardware remain unchanged.

The failed before-test and successful after-test are checkpointed privately at
Dataset revision `82139f4d432bbf9e32d590d55bd1c8a84d7df63a`, under
`runs/response-polish-live-20260908T105322Z`. Local evidence is in
`.audit/response-polish-20260908-hf-token/`. The deployment used a direct process;
GitHub Actions still needs its own `HF_TOKEN` secret configured.

## Earlier deployment attempts

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
for direct deployment passed its manifest dry run. The user then approved
loading only `HF_TOKEN` from `backend/.env`; the direct deployment and live
verification described above succeeded.

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
