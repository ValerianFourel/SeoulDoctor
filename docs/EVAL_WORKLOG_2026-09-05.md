# SeoulDoc evaluation worklog — 2026-09-05

## Objective

Build a reproducible randomized comment-level evaluation that selects new
Seoul medical facilities and one real patient comment per facility, creates
paired English/Korean patient scenarios, and keeps the target facility and
evidence hidden from patient agents.

The release evidence is facility-level. This work does not infer or invent
individual doctor identities, qualifications, or doctor-level claims.

## Repository state

- Working branch: `cloud/random-holdout-6688037892107667854`
- Base available in this container: `dea9a63`
- Work commits, oldest first:
  - `329dd7d` — define random holdout sampling rules
  - `ed18403` — add the streaming random holdout sampler
  - `cb5ee68` — support the release Parquet data directly
  - `27271d7` — keep private evaluation artifacts out of Git
- Configured remote: `https://github.com/ValerianFourel/SeoulDoctor.git`

The running process did not expose `GITHUB_TOKEN` or `GH_TOKEN`, and `gh auth
status` reported no authenticated GitHub host. The commits therefore remained
local at the end of this session. A restarted container may be required after
adding the GitHub environment variable.

## Implementation

### Sampling specification

`docs/RANDOM_HOLDOUT_EVAL.md` now fixes the validity and sampling rules before
outcomes are observed. Important properties include:

- one unsigned 64-bit seed per frozen sample;
- distinct facilities selected without outcome-based resampling;
- a separately prioritized qualifying comment for each facility;
- deterministic selection for the same seed, sampler version, and source;
- append-only output directories;
- public patient cards separated from the private oracle;
- facility retrieval and exact-comment retrieval reported separately.

### Memory-bounded sampler

`scripts/random_holdout_sampler.py` supports:

- the pinned release `facilities.parquet` and `reviews.parquet` files;
- JSON Lines and CSV normalized inputs, including gzip variants;
- streamed Parquet review batches rather than loading all reviews;
- a small facility metadata lookup and `O(sample_size)` selected full records;
- deterministic SHA-256 priorities for facilities and evidence;
- English/Korean experience themes derived from comments;
- English translations of common Korean medical specialties;
- basic contact-data, emergency-language, and prompt-like-facet rejection;
- paired public English/Korean patient cards;
- a private oracle containing the target facility and exact source comment;
- a redacted run manifest with Git, model, Dataset, scenario, and artifact
  metadata.

Private `.codex-handoff/` artifacts are ignored by Git.

### Automated tests

`backend/tests/test_random_holdout_sampler.py` covers:

- record validity and unsafe-field rejection;
- deterministic selection independent of source ordering;
- distinct-facility enforcement;
- streamed iterator consumption;
- public-card redaction;
- bilingual specialty rendering;
- public casebook/private oracle separation;
- release Parquet facility/review joining;
- failure when fewer eligible facilities exist than requested.

The focused suite passed 8 tests. The complete repository suite passed 23 tests
with one existing Pydantic V2 deprecation warning in `backend/models.py`.

## Frozen randomized sample

- Release Dataset: `ValerianFourel/seouldoc-app-release-20260905`
- Release revision: `3911d79dc31e6a6ccfa3f64a7e401b88893bf66a`
- Sampling seed: `6688037892107667854`
- Source reviews scanned: `1,805,211`
- Eligible comments: `683,249`
- Ineligible comments: `1,121,962`
- Selected facilities: `6`
- English/Korean pairs: `6`
- Total frozen scenarios: `12`
- Latest local frozen run: `20260905T125329Z-random-holdout-sol-root`

The public scenarios cover these general needs without exposing target names or
comments:

1. Dermatology near Seobinggo-dong; kind communication and careful treatment.
2. Korean medicine near Eungbong-dong; kind communication.
3. Orthopedics near Hagye 1-dong; clear explanations.
4. Internal medicine near Geumho 1-ga-dong; kind communication.
5. Dermatology near Nonhyeon 1-dong; kind communication, careful treatment,
   and cleanliness.
6. Rehabilitation medicine near Ichon 1-dong; kind communication.

Each need has one English card and one Korean card. Exact target facilities,
comments, `place_id` values, and evidence identifiers remain in the untracked
private oracle.

## Application checks

Authenticated requests using the Hugging Face token reached the private Space:

- `/health`: HTTP 200
- `/openapi.json`: HTTP 200
- facilities reported healthy: `8,484`
- raw reviews reported healthy: `1,791,749`
- Phase 3 facility index: `8,484`
- Phase 3 evidence index: `2,060,433`
- application model: `openai/gpt-oss-120b`

Unauthenticated Space requests returned a generic Hugging Face 404 page. Future
evaluation requests must authenticate without logging or persisting the token.

Three isolated GPT-5.6 Sol patient agents each received a different public card
and made one authenticated `/chat` request. They did not receive the private
oracle. The execution bridge failed to surface their captured response bodies,
so these calls cannot be graded. They were not repeated because completed paid
calls must not be duplicated.

## Known limitations

1. The random sampler and frozen cards are implemented, but automatic patient
   agent orchestration and result grading are not yet implemented as a
   repository command.
2. A patient agent should write its response into a unique untracked journey
   artifact before yielding control so tool-output loss cannot discard results.
3. Target grading still needs to measure eligible facility rank, presented
   facility rank, exact-comment recall, evidence attachment, and
   evidence-to-facility ownership.
4. The result checkpoint was not uploaded because the expected checkpoint sync
   tooling was absent from this older checkout.
5. GitHub push and pull-request creation were blocked because GitHub credentials
   were not visible to the running container.

## Exact next actions

1. Restart the container so `GITHUB_TOKEN` is present, then push
   `cloud/random-holdout-6688037892107667854` to `origin` and open a pull
   request.
2. Review this branch against the current coordination branch before merging;
   this container began from an older checkout that did not contain the newer
   grounded bilingual evaluation runner.
3. Add append-only patient journey capture before making more application
   calls.
4. Execute the 12 frozen scenarios with one isolated patient context per
   scenario, in bounded parallel waves.
5. Grade all valid frozen cases without replacement or resampling and report
   facility-level and comment-level retrieval separately.
6. Upload redacted checkpoints to the private result Dataset after every
   completed scenario or failure.

