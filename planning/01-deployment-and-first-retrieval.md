# Note 1: Deploy the existing retrieval path and verify one comment

The immediate goal is to get the application running on Hugging Face, then
retrieve one real comment to verify the current deployment. Pipeline tuning
comes later. This first check should establish what the deployed code actually
does before changing ranking, prompts, query expansion, or evidence selection.

## What the code changes provide

The existing application already had a BGE-M3 HTTP client and comment-ID
resolution. The assessment changes make that retrieval path deployable alongside
the website in the private `ValerianFourel/SeoulDoctor-ncs-retriever` Space:

- The container serves the website and application API on port 7860, with the
  BGE-M3 service on internal port 7861. The existing backend client calls that
  internal service. Separate Python environments preserve their dependencies.
- Startup restores the pinned application release and reuses the existing
  BGE-M3 index and model revision. It does not regenerate corpus embeddings.
- BGE-M3 explicitly requires CUDA. Its startup probe encodes an English and a
  Korean query and checks dense dimensions and nonempty sparse representations.
  The application exposes this service's readiness report at `/ready/gpu`.
- The runtime supervises both processes. Deployment wiring also corrects the
  Docker startup command and serves the frontend at the root URL.

The core search and frontend behavior remain inherited from the application
baseline. These changes enable deployment and inspection; they do not establish
that recommendation quality has improved.

## What is currently being attempted

First, bring up the combined application on the selected NVIDIA L4 hardware.
The deployment record reports a successful GPU encoding probe on the earlier
API-only service, whose root URL returned 404. The combined deployment is meant
to serve the website as well as the API at the same Space.

The latest startup output supplied for this note still reports
`RUNNING_APP_STARTING`. Its log stream disconnected after the startup banner.
That output alone establishes neither successful startup nor an application
crash. Live verification of the combined deployment remains pending.

Second, submit a natural-language query through the existing BGE-M3 retrieval
path, using the required facility scope, without tuning the pipeline. Inspect
the returned comment ID and resolve it against the pinned original source.
Then verify that the application API transmits the full original comment,
matching `evidence_id`, facility `place_id`, source locator, and source digest.
Do not inject an expected comment ID into retrieval to manufacture success.

## Completion criteria for this first check

1. Confirm the intended deployed revision is running and the application and
   GPU readiness endpoints respond successfully.
2. Retrieve a real comment through BGE-M3 and record the query, facility scope,
   returned ID, retrieval channel, and deployed model/index revisions privately.
3. Compare the API's original text and facility ownership with the source record
   identified by that ID. Check that the text is complete and is not a summary
   or translation presented as an original quotation.
4. Record success, failure, or the precise blocker before making retrieval
   changes. Keep private comments and raw response artifacts out of Git.

A passing GPU probe proves query encoding, not comment retrieval. A passing
single-comment check proves this deployment path for that request, not general
retrieval quality. `/ready/retrieval` also includes reranking, so its result must
be reported separately rather than inferred from `/ready/gpu`.

No new live check was run while writing this note. See
[deployment details](hf-deployment.md) for the implementation and recorded checks.
