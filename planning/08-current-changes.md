# 08. Current changes and handoff

Recorded 2026-09-07. This note summarizes verified work and supersedes earlier
statements that the presentation changes are uncommitted or deployment is pending.
It is a worklog summary, not a verbatim agent transcript.

## Checkout and deployment

- Branch: `ncs`.
- Worktree: `/home/valerian/Seoul/SeoulDoc/.worktrees/ncs`.
- Code/runner checkpoint: `ce7102fee1f722625a758945313760a5285b1680`, pushed to origin.
- Working tree was clean before this documentation update.
- Last verified Space: `ValerianFourel/SeoulDoctor-ncs-retriever`, private, RUNNING
  on its existing NVIDIA L4. Space source and runtime both matched
  `67a3cf8ab75c6cd45a49dfd4fcf5a342cc49e3c8`.
- Deployed application source: `b76d29fa238a6b62fd90588158e317d7e76940b8`.
  Later branch commits update the local evaluator and documentation; they do not
  represent another application deployment.

Owner: root Codex continuation. This documentation update owns only this note
and the appended coordination entry in `docs/PROJECT_STATE.md`. No other session
is claimed to be synchronized. A receiving session must verify HEAD and read the
required project records before taking ownership.

## Implemented and deployed

Original comments are attached for searches reaching the retrieval adapter without
review-specific constraints. The adapter reads at most 100 original records per
shortlisted facility from the existing index, filters text noise locally, and
attaches at most three comments without changing facility ranking or preference
coverage. It preserves source IDs, ownership, locators and source digests.
This is general comment attachment, not a claim that the most relevant comment
is always selected. Other search paths and preference-dependent recall still need
separate verification.

Cards retain original Korean comments when translation is unavailable or
presentation metadata is absent. Short meaningful reviews and substantive text
with emojis remain visible. Optional selected-comment translation uses Google
Cloud Translation Basic with `model=nmt`; translated text is labeled and originals
remain accessible. Live translation faithfulness has not been verified.
The Unicode letter check was corrected to build with the existing TypeScript
target. The Donate button was removed.

The protected retrieval-only API accepts only query, location and specialty.
It calls the application's existing retrieval adapter and configured services,
without response generation, translation or judging. Expected IDs and source
quotations remain exclusively in the private evaluator fixtures.

## Seven-case evaluation

Run:

```bash
python scripts/mini_retrieval_eval.py
```

The command reuses the backend environment and approved process credentials.
It restores only the pinned seven private fixtures if needed. Fixtures cover
seven clinics, districts and specialties, with three English, three Korean and
one mixed query. IDs, ownership and full source labels were checked through
indexed lookups. Manual fixture selection was not timed; fixture loading is
reported separately and must not be mistaken for preparation time.

The runner uses at most four workers, bounded requests and a 60-second measured
deadline. Candidate limits remain normal; the coverage retry and transport retries
are disabled. The corrected warmup requires actual GPU readiness before cases run.

| Measurement | Verified warm run |
| --- | --- |
| Clinic hits in top five | 3/7 |
| Target review hits in retrieval admissions | 4/7 |
| Selected-comment hits | 2/7 |
| Observed ownership errors | 0 |
| Timeouts | 0 |
| Measured execution | 14.61 seconds |
| Warmup | 1.203 seconds |

All seven cases were marked incomplete and the command exited 1. The Space has
no configured reranker. The three English cases reported semantic request failures
even after CUDA readiness passed; Korean and mixed cases reported semantic success.
The earlier 16.93-second diagnostic lacked the GPU warmup gate and is retained
separately. These results are not a successful end-to-end retrieval evaluation.

Both runs, seven labels and provenance are stored privately in
`ValerianFourel/seouldoc-eval-handoff` at
`ef41a2509efd9e58a620b364285ea887e88f6add`, under
`runs/mini-retrieval-20260906T230140Z/`. No private fixtures, original source
quotations or generated result bundles were added to Git. See [note 06](06-mini-retrieval-eval.md)
for seven individual rows, limits and interpretation.

## Checks actually completed

- Backend discovery: 243 tests passed. Focused evaluator checks also passed
  after the runner updates.
- Frontend production build passed, including the Donate-button removal.
- Six synthetic browser presentation cases passed before deployment, including
  Korean originals, missing metadata, translation failure and short comments.
- Live website and health returned HTTP 200; Donate markup was absent.
- Live BGE readiness passed on NVIDIA L4 with 1024 dense dimensions, nonempty
  sparse output and pinned model revision
  `5617a9f61b028005a4858fdac845db406aefb181`.
- No browser automation, translation or LLM judge was used in the seven-case run.

These are recorded observations from the preceding work, not fresh live checks
performed while writing this note.

## Pending startup work and limitations

[Note 07](07-startup-optimization.md) records the startup optimization brief.
No startup optimization or measured speedup has been implemented. The earlier
bounded deployment wait expired; later deployment succeeded. Those observations
do not identify dataset, model, index or infrastructure scheduling durations.

Keep the single-Space architecture. Proposed work is to measure startup stages,
preload the pinned public model into a reusable cache, and overlap independent
application restoration and retriever initialization while preserving supervision
and truthful readiness. Private datasets and credentials must stay out of image
layers. Do not rebuild corpus embeddings or change retrieval quality settings
as part of startup work.

Remaining retrieval work includes English semantic request failures, missing
reranking, stronger comment relevance and live translation verification. The
first-100 attachment scan can miss useful comments beyond that bound. No hardware,
storage, visibility, main-branch or original-Space changes were made.

Single next action: establish measured cold-start stage timings under the
conditions in note 07 before choosing or deploying startup optimizations.
