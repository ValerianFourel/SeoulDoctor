# Assessment brief

SeoulDoctor is an existing English, Korean, and code-switching facility finder.
Before this assessment it already had a Next.js chat UI, FastAPI conversation
state, specialty/location constraints, hybrid retrieval, original-review
presentation, remote retriever/reranker clients, and evaluation tooling.
These are inherited capabilities, not assessment deliverables or proof of live readiness.

The provisional improvement goal is a better conversational recommendation:
discover a consequential preference, retrieve supporting comments, and explain
how evidence and uncertainty affect the choice. Select one reproducible journey
before deciding what to change. A backend rewrite is not assumed.

Budget the implementation at 150 minutes: 20 to reproduce and choose scope,
60 to implement, 35 to verify and review, 35 for documentation and recording.
Cut secondary features if the core journey exceeds this budget.

Acceptance criteria for the later build:

- A reviewer can reproduce one baseline-to-final journey with documented inputs.
- A meaningful new behavior is identifiable in the diff and visible response.
- Reply text, facility cards, evidence ownership, and recommendation meaning
  each have explicit checks. Include preference refinement and missing evidence.
- English-written reviews never establish English-speaking staff. Facility
  evidence never becomes an unsupported individual-doctor claim.
- Original quotations stay distinct from translations and summaries. Conflicting
  feedback and unestablished requirements remain visible.
- Planning, actual agent contributions, checks, trade-offs, and scope cuts are
  recorded. Available authentic logs and a short demo accompany the submission.

Setup delivers documentation and a dedicated branch only. Excluded now: product
edits, deployment, paid infrastructure, corpus downloads, index regeneration,
full remote evaluations, publication, and repository visibility changes.
Existing scenarios/results stay unchanged; any conversational assessment suite
must have a separate name and fixtures. Final submission needs a public repo
link, demo recording, planning artifacts, and available reviewed agent logs.
