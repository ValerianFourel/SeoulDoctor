# Session handoff: Korean evidence for English-speaking patients

This session made SeoulDoc's existing application usable alongside GPU retrieval
in the ncs assessment Space, added Google translation and simpler review cards,
and measured where comment retrieval still fails. It did not establish reliable
recommendations or finish startup optimization.

## Resume point and ownership

- Branch: `ncs`.
- Worktree: `/home/valerian/Seoul/SeoulDoc/.worktrees/ncs`.
- HEAD when this handoff was written: `f66708c6978dbb3c239c28f4aac900f59b8c4dbc`.
- Latest application cleanup commit: `1807fa81f5709514562935fcef37cede6d17ae85`.
  This cleanup was local at the previous handoff and is included in the
  requested ncs branch publication. No new Space deployment is part of this update.
- Last pushed application change: `66022eabf25ee0dde0ebeab4276aef78d7e6ef0b`.
- Owner: root Codex session. This final documentation task owns this note,
  the pointer in `planning/handoff.md`, and an appended PROJECT_STATE entry.
- Preserve the other session's uncommitted
  `planning/retrieval-evaluation-explained.md` and its PROJECT_STATE entry.

Read AGENTS.md, PROJECT_STATE, ENVIRONMENTS and CODEX_CLOUD_HANDOFF in their
required order, then this note. Historical handoffs describe older deployments;
the revisions below identify what was actually measured. No session
synchronization, new deployment or infrastructure change is implied by this note.

## What was achieved

### One website and retriever in the existing GPU Space

The target remains the private
[SeoulDoctor-ncs-retriever Space](https://huggingface.co/spaces/ValerianFourel/SeoulDoctor-ncs-retriever).
The combined application serves the website and API while BGE-M3 runs in the
same GPU Space. The earlier API-only root returning `{"detail":"Not Found"}`
was addressed by serving the full application. The original SeoulDoctor Space
and main branch were not changed by this assessment work.

GPU hardware was left for the user to add manually on the Hugging Face website,
through the existing Space's hardware settings. A source-code upload does not
select a GPU or guarantee capacity. An L4 allocation previously failed with
“Scheduling failure: not enough hardware capacity.” The later successful
50-case run observed an already configured L40S (`l40sx1`). This session did
not purchase or switch that hardware. Future operators should inspect the
existing allocation and pending build before requesting any change; do not
create another Space to bypass a startup failure.

Website availability and retrieval readiness are separate checks. The live run
verified the facility encoder and a real CUDA encoding probe on NVIDIA L40S,
with 1,024 dense dimensions and nonempty sparse outputs. HTML alone is not proof
that GPU retrieval works. Existing corpus embeddings and indexes were reused.

The latest display release was uploaded as Space source
`4138eb056b80f4689fb0ef87244cf128c44280b2`, from Git `66022eab…`.
The last read-only observation still showed `RUNNING_APP_STARTING`, with runtime
SHA `2e9a6e65b1037f347791fb265cbe061ebcb070c7`. Activation and live checks of the
new three/seven-comment layout remain unverified. This note does not refresh
that remote observation. The later cleanup must not be deployed under the
current no-deployment instruction.

### Google translation of Korean review evidence

The application uses **Google Cloud Translation Basic v2**, `model=nmt`, with
`GOOGLE_TRANSLATE_API_KEY` held server-side through approved process credentials
or a private Space secret. Adding a key to a local environment file alone does
not configure the deployed Space. The key was separately configured privately
for the existing Space. Follow the active AGENTS.md credential policy; future
sessions must not read or copy environment files or expose key values.

Selected comments retain original text, evidence IDs, facility ownership and
source metadata. English translations are labeled automatic, with a Show
original control. Missing credentials, failed translation or missing
presentation metadata leave available originals visible. A translation failure
must not become an empty-evidence claim.

Translation batches are deduplicated and bounded to 100 distinct comments and
16,000 source characters, with a bounded request timeout. Three live synthetic
examples checked positive feedback, criticism of a receptionist, emojis and a
waiting-time detail. One real debugging finding was that Google correctly
translated a numeric duration as “two hours,” while a literal digit check
rejected it. The check now recognizes zero-to-twelve English time counts;
changed durations still fail. These probes do not establish translation
faithfulness for the full dataset. See [note 09](09-card-review-controls.md).

### A simpler layout and light comment filtering

The normal facility summary and its Read more control replace the repeated
per-card candidate warning. The response-level incomplete-search caution stays.
The summary is distinct from a verbatim patient comment and is not proof that
an individual doctor meets the patient's preferences. Donate was removed on ncs.

Review cards now show up to **three eligible comments initially**, then up to
**seven per page after Next**. Previous and a small collapse arrow keep long
review lists manageable. Translation labels and original toggles let an English
reader inspect the Korean evidence without showing internal IDs.

The local letter check excludes number-only, emoji-only and isolated-jamo noise.
Meaningful Korean comments and emojis within text remain eligible. The user's
later rule additionally excludes displayed English originals or translations
with three or fewer word tokens. Numbers count as tokens; contractions count
as one; emojis do not add words. This affects presentation, not the retrieval
corpus or evidence IDs. Unmarked Latin-only text uses a language heuristic.
A short but useful warning can be filtered out, so this remains a product
trade-off to review rather than a general measure of comment quality.

A light poteto-mode cleanup then made the usable-translation decision once,
shared by filtering and rendering. Public interfaces and behavior stayed the
same. See [layout details](11-comment-page-limits.md) and
[cleanup verification](12-light-cleanup.md).

## What the retrieval evaluation established

The original seven fixtures and command remain unchanged:

```bash
python scripts/mini_retrieval_eval.py
```

The separate 50-case suite samples with replacement from a **restricted
19-pair shortlist**, not the full corpus. Its frozen sample has 18 unique
facility-comment pairs, seven facilities, 20 English queries, 20 Korean queries
and 10 mixed queries. Repeated draws keep their weight and are not independent
quality evidence. Expected IDs, source text and labels remain private.

Run `mini-retrieval-50-20260906T233703Z` used application Space revision
`2e9a6e65b1037f347791fb265cbe061ebcb070c7`, before the latest layout change.
Only query, location and specialty entered the protected retrieval endpoint.
Four workers reused sessions; candidate limits stayed normal and coverage retry
was disabled. No answer generation, translation, browser or LLM judge ran in
the measured retrieval interval.

| Language | Draws | Clinic in top five | Target review admitted | Target selected | Complete passes |
| --- | ---: | ---: | ---: | ---: | ---: |
| English | 20 | 9 | 11 | 1 | 0 |
| Korean | 20 | 9 | 14 | 7 | 0 |
| Mixed | 10 | 8 | 6 | 4 | 0 |
| Total | 50 | 26 | 31 | 12 | 0 |

All requests returned, with zero ownership errors and zero timeouts. **All 50
were incomplete**, so the run exited 1. Reranking was disabled in every case;
two English semantic requests failed. The 31 admission hits versus 12 selection
hits identify a useful investigation area, but do not by themselves prove that
the selector is faulty: facility rank and other pipeline stages also matter.

Warmup took 1.577 seconds. Measured execution took 146.962 seconds, below the
300-second deadline. Median response latency was 9.559 seconds and p95 was
14.584 seconds. Passing-case throughput was zero. Query preparation was not
repeated; its original timing limitations are recorded in
[note 10](10-fifty-case-retrieval.md).

The private fixtures, provenance, per-draw results and per-pair breakdown are
at Dataset `ValerianFourel/seouldoc-eval-handoff`, revision
`06dccdcccbc3b27c26c72f87be088d8cb79604a9`, under
`runs/mini-retrieval-50-20260906T233703Z/`. Earlier blocked and seven-case results
remain preserved. Selected-ID success does not prove browser visibility,
translation quality, or conversational understanding.

## Toward a more precise BGE-M3 retrieval pipeline

The application already combines facility retrieval with evidence retrieval.
`ScopeBuilder` applies location and specialty constraints; the existing facility
encoder and lexical channels supply candidates. BGE-M3 provides **dense and
sparse comment retrieval** against the pinned existing index. ColBERT vectors
are disabled. BGE-M3 is not the configured cross-encoder reranker, and it does
not replace every facility encoder in the application.

The evidence path combines retrieved candidates, records admissions, optionally
reranks, and selects comments under their owning facilities. Presentation then
translates and filters selected text. The quick evaluator enters this structured
retrieval path with the entire query as a comment term; it skips the chat
model's conversational intent extraction. Keep that limitation visible when
judging location changes or follow-up requests.

The proposed improvement is to make each stage accountable, without a backend
rewrite or an immediate index rebuild:

1. **Make the existing services complete.** Diagnose the two frozen English
   failures through redacted status and timing diagnostics. Verify model/index
   revision agreement, request validation, transport and timeout behavior before
   changing language processing. Restore an authorized compatible reranker;
   measure its cost and latency without allocating hardware automatically.
2. **Keep constraints separate from preferences.** For a single reproducible
   journey, verify that a new location or specialty replaces stale conversation
   state. Keep geography and specialty as hard scope filters. Treat kindness,
   explanation quality and treatment anxiety as evidence requirements, not
   unsupported claims that a facility meets them.
3. **Trace hybrid recall and ranking.** Encode the patient's preference with the
   pinned multilingual BGE-M3 model and search the existing dense and sparse
   indexes within the eligible scope. Record channel ranks and evidence IDs.
   Preserve current fusion weights and limits for the first comparison. Do not
   translate or rewrite targets to make them easier to retrieve.
4. **Trace admission through selection.** For each missed target, identify
   whether it was outside scope, absent from channels, excluded before reranking,
   demoted with its clinic, or not selected for the card. Check facility ownership
   at each boundary. Prefer comments directly supporting the requested aspect,
   retain contradictions and negative experiences, and avoid redundant generic
   comments. Any selector change needs a separate tested commit.
5. **Translate after evidence selection.** Use Google only for presentation,
   retaining Korean originals and explicit failure fallback. Audit the final
   displayed comment as well as its selected ID, because short-English filtering
   can remove selected evidence. Consider translating only visible pages as a
   separate bounded optimization, not an unmeasured change to retrieval.
6. **Compare on frozen cases.** Reuse both suites unchanged, with a fresh run ID,
   verified deployment, GPU warmup, fixed limits and private checkpoints. Require
   complete retrieval before counting a pass. Expand to a broader documented
   sample only as a separately identified suite after service failures are fixed.

These are next steps, not completed improvements. English-language reviews do
not prove staff speak English. Facility evidence does not identify the behavior
or qualifications of every doctor working there.

## Example task for the next session: explain facility misses

Start by fixing the incomplete pipeline, then investigate where the correct
facility loses rank. The 50-case run admitted the target comment in 31 draws,
but placed its facility in the top five in only 26. **Those totals alone do not
identify which cases overlap.** Compare the per-draw labels before attributing
the gap to any stage. Report both-hit, comment-only, facility-only and neither-hit
counts, retaining duplicate weighting and a separate unique-pair breakdown.

Use this order for relevance work after verifying the deployment:

1. **Fix the two English semantic failures and restore reranking.** Every case
   was incomplete, so establish a functioning baseline before tuning relevance.
   Reranking may help, but measure its effect rather than assuming an improvement.
2. **Trace every missed facility through the pipeline.** Determine whether
   specialty/location filters excluded it, initial candidates omitted it, the
   shortlist dropped it, or final ranking put it below fifth. Identify the first
   failing stage; each failure class needs a different fix. If existing telemetry
   cannot establish the stage, report it as unknown instead of inferring it from
   the final response.
3. **Check how relevant comments affect facility scores.** When strong matching
   evidence is retrieved for a clinic, determine whether it meaningfully affects
   facility ranking. Avoid rewarding clinics simply for having more reviews.
   A relevant negative review should qualify the recommendation, not
   automatically improve the clinic's suitability.
4. **Improve candidate recall only where it fails.** Reranking cannot recover a
   clinic that never reaches its candidate set. If diagnostics justify it, test
   broader facility discovery from comment matches or a modest shortlist
   increase, preserving hard constraints. Treat this as an explicit experimental
   variant; keep the frozen baseline limits and results unchanged.
5. **Test one change at a time on the frozen cases.** Track facility recall at
   five, target-comment selection, ownership, completeness and latency. Report
   sample counts and results by unique pair as well as all draws, so repeated
   pairs do not dominate the conclusion. Preserve failed and incomplete results.

Suggested prompt for the main thread:

> For every facility miss in the frozen 50-case evaluation, identify the first
> stage where the target facility disappears or falls below fifth. Separate
> eligibility, candidate discovery, shortlist, and final-ranking failures.
> Report the counts before proposing the smallest fix. Keep target IDs
> exclusively in evaluation diagnostics.

This is a future diagnostic task, not a completed finding or authorization to
inject target IDs into retrieval. Collect ordinary stage candidate IDs in
protected diagnostics and compare them with private labels in the evaluator.
Do not pass expected IDs or target quotations to the application as search
inputs. Keep diagnostic records private and distinguish the incomplete historical
run from any new complete baseline.

These tests reward finding a particular source clinic. Other returned clinics
may also be suitable. Improving target-facility recall is useful, but does not
by itself establish better patient recommendations.

## Remaining work and the next action

**First action:** inspect the active revision and GPU readiness of the existing
ncs Space without changing it. Establish whether display release `4138eb…` has
activated, then verify the supplied Mullae journey and the three/seven-comment
controls against real API evidence. Keep that UI check separate from retrieval
timing. Review the local cleanup before seeking any new deployment authorization.

After that, prioritize reranker availability and the two English semantic
failures before tuning ranking. The user's Mullae-to-Itaewon conversation also
needs a separate check for stale location state; this session did not fix it.

Cold-start optimization is still a brief in
[note 07](07-startup-optimization.md). Measure build, hardware scheduling,
dataset restoration, model/index loading, CUDA initialization and readiness
separately before changing startup. Hardware capacity errors are not measured
application startup latency. Keep the single-Space architecture.

Local verification finished with 253 backend tests, a production frontend build,
English desktop/Korean mobile browser checks and 3,336 identical before/after
HTML renders for the cleanup. Those checks passed; the retrieval quality gate
did not. No new tests or live checks were run just to write this handoff.

For assessment submission, repository access, a short demo and final baseline
comparison still need completion. Planning notes are factual work records, not
verbatim agent transcripts. No supported export of this active session was
produced; preserve the limitation in
[session-logs/README.md](session-logs/README.md). Keep private source data and
raw evaluation artifacts out of Git.
