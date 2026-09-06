# Combined ncs GPU deployment

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
