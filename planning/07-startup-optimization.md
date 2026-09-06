# 07. Optimize startup of the existing ncs GPU Space

## Current state and ownership

Target: `ValerianFourel/SeoulDoctor-ncs-retriever`, private, existing L4 hardware.
Keep the full website and BGE-M3 in this one Space. Do not change main, the original
SeoulDoctor Space, visibility, secrets, hardware, storage, ranking, translation,
comment selection, or corpus embeddings. Use approved process credentials only.

Root owns this planning note and the sequential deployment/evaluation work.
Branch `ncs`, worktree `/home/valerian/Seoul/SeoulDoc/.worktrees/ncs`.
Source HEAD: `b76d29fa238a6b62fd90588158e317d7e76940b8`.
Current upload: Space `67a3cf8ab75c6cd45a49dfd4fcf5a342cc49e3c8`.
Verify actual runtime again before modifying or restarting anything.

Committed work to preserve:

- `ea515261e6e415797ca18c1426e97e2b3e1ab48a`: original comment attachment,
  optional Google translation presentation, short Korean originals, and Donate
  button removal. Passed 240 backend tests, frontend build, six synthetic browser
  cases; the final header build also passed.
- `b76d29fa238a6b62fd90588158e317d7e76940b8`: protected retrieval-only API and
  seven-case evaluator. Passed 243 backend tests and frontend build.
- Seven private labels at `.audit/mini-retrieval/fixtures.json`; IDs, clinic
  ownership, source locators and complete texts were checked against the existing
  index. No measured seven-case result yet. See note 06.

Before work, read AGENTS.md, docs/PROJECT_STATE.md, docs/ENVIRONMENTS.md,
CODEX_CLOUD_HANDOFF.md and the latest numbered planning notes. Preserve any new
uncommitted work. The current note is a work brief, not an implemented optimization.

## Observations, not timing claims

The first upload, Space `82d5e1e96bd80668880af1a4d1af4991c2b2ad0e`, built and reached
RUNNING_APP_STARTING, but did not become the active revision within a ten-minute
polling window. The previous runtime remained
`d765a828f621d5fb1b9273b960748e63100a8e26` during those observations. A limited
startup-log inspection contained only two events and no restoration/completion
markers. This does not identify the delay as model download or application code.
Build progress showed completion and image push without an ERROR marker.

The evaluator upload superseded that revision for a specific new API requirement;
its most recent observed state was BUILDING with no active runtime SHA. No
repeated restart has been issued to test a hypothesis. No additional hardware
or persistent storage was purchased.

Source inspection shows potential improvements, not measured bottlenecks:

- `space_runtime.py` restores the application release before launching either
  service, preventing independent retriever initialization from overlapping it.
- GPU dependency installation comes after application/frontend COPY steps in
  Dockerfile, so source changes can invalidate expensive build layers.
- Runtime model/index initialization must be traced before assuming download or
  GPU initialization accounts for the observed deployment delay.

## Bounded implementation plan

1. Record source/runtime SHAs and timestamped build, scheduling and startup
   transitions. Add structured duration markers if the current logs cannot
   separate dataset download, dataset copy, model download, retrieval-index
   download/load, CUDA initialization and the encoding readiness probe. Keep
   private data, request URLs and credentials out of logs. An instrumentation
   deployment is justified only if existing logs cannot provide the baseline.
2. Check current official Hugging Face documentation for supported model preload
   or Docker caching. Preload only required public BGE-M3 artifacts at pinned
   revision `5617a9f61b028005a4858fdac845db406aefb181` into a reusable build layer.
   Configure the runtime to use that exact cache/revision without a second copy.
   Never place credentials or private application/index datasets in image layers.
3. Overlap application release restoration with retriever initialization where
   independent. Start the backend only after restoration succeeds. Retain child
   supervision, failure propagation, signal handling and bounded cleanup.
4. Keep website availability separate from GPU and retrieval readiness. Preserve
   the real bilingual CUDA encoding probe and the existing index. HTML success
   must not imply a working retriever or reranker.
5. Add focused regressions for launch ordering, failed restoration, service
   failure and shutdown. Run the full backend suite, retriever tests and frontend
   build. Review the exact deployment manifest and diff. Deploy only the startup
   changes and already-authorized work to this same Space.
6. Compare equivalent before/after cold-start conditions with revisions and
   separate timing columns. Report unmeasured stages as unknown, not zero.
   Bound the experiment to one baseline and one optimized cold-start attempt;
   another build/restart requires a concrete diagnosed reason. Do not disrupt an
   active mini-evaluation to collect startup timings.

## Remaining work and next action

No startup optimization or measured speedup is claimed in this note. No complete
stage timings are available yet. Splitting out the frontend is a future option,
not part of this task.

Next action: verify the current evaluator deployment and preserve its seven-case
run; then establish a timestamped cold-start baseline before choosing changes.
