---
title: Seoul Doctor Matchmaker
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# SeoulDoc: ncs assessment

SeoulDoc helps patients find medical facilities in Seoul through conversation. The existing Next.js/FastAPI application already had location and specialty search, facility summaries, review data, and retrieval services. This assessment adds GPU-backed comment retrieval and an English presentation of Korean evidence.

[Review the ncs changes](https://github.com/ValerianFourel/SeoulDoctor/pull/3) · [Assessment Space](https://huggingface.co/spaces/ValerianFourel/SeoulDoctor-ncs-retriever) · [Planning](planning/)

## How the work developed

The builder set the direction and revised the scope through working examples. Codex implemented, debugged, tested, and documented the changes.

| Problem | Proposed solution | Resolution |
| --- | --- | --- |
| The GPU endpoint returned `Not Found` instead of the website. | Keep the full app and BGE-M3 in one existing Hugging Face Space. | Combined website/API and retriever runtime. The builder manually selected GPU hardware in Hugging Face settings. Real CUDA encoding was verified; existing embeddings and indexes were reused. |
| Korean evidence was missing or unreadable for English users. | Attach real comments, then translate selected evidence with Google Translation Basic v2 NMT. | Bounded translation batches, privately configured API key, labeled translations and an original-text toggle. Translation failure keeps Korean originals visible. |
| Long comment lists and repeated warnings overwhelmed the cards. | Restore facility summaries and reduce visible reviews. | Three comments initially, up to seven on subsequent pages, navigation and collapse controls. English comments of three words or fewer and non-text noise are filtered for display. Donate was removed on ncs. |
| Retrieval looked plausible without proving that the intended evidence survived. | Preserve a seven-case baseline and add a frozen 50-draw evaluation. | Real pipeline measurements exposed incomplete services and a gap between comment retrieval and facility ranking. Relevance tuning remains next. |

One actual debugging example: Google translated a duration as “two hours,” but a digit-only validation check rejected it. A narrow time-word check fixed that false rejection while retaining changed-duration checks. Short-comment filtering can also hide useful warnings; it is a presentation trade-off, not a quality score.

## Retrieval pipeline

Query, location and specialty → eligible facility scope → existing facility lexical/dense search plus BGE-M3 dense/sparse comment search → candidate fusion and admissions → configured reranking → facility and comment selection → Google translation or original fallback → review cards.

BGE-M3 is separate from the facility encoder and cross-encoder reranker. Reranking was disabled in the measured run. Reviews describe patient experiences; they do not establish an individual doctor's abilities or staff language fluency.

## Proof of work

Recorded verification includes 253 backend tests, a frontend production build, browser checks of card controls, and 3,336 identical HTML comparisons for the light cleanup. Latest layout activation on the Space remains unverified; uploaded source and active runtime revisions are distinguished in the [handoff](planning/13-session-handoff.md).

The [50-case run](planning/10-fifty-case-retrieval.md) used four workers and a 300-second deadline, excluding generation, translation and LLM judging.

| Target comment retrieved | Expected facility in top five |
| ---: | ---: |
| 31/50 · 62% | 26/50 · 52% |

Why is 31 higher than 26? Comment retrieval searches a broader evidence pool, while the facility score checks only the top five facilities. Finding the target comment does not guarantee that its clinic reaches that final five. The totals do not reveal the exact overlap or where each facility lost rank; that requires per-case tracing.

Reranking was missing and two English semantic requests failed, so these are diagnostic hit counts from an incomplete pipeline. Execution took 146.962 seconds, plus 1.577 seconds warmup.

Sampling used replacement from a restricted 19-pair shortlist: 50 draws, 18 unique pairs, seven facilities, and 20 English/20 Korean/10 mixed queries. Duplicate draws are not independent quality evidence. Labels and source comments remain private. The original [seven-case suite](planning/06-mini-retrieval-eval.md) is preserved:

```bash
python scripts/mini_retrieval_eval.py
```

## Run and review

Follow [reviewer setup](planning/reviewer-setup.md) for dependencies and private data/service prerequisites. A fresh clone alone cannot run live retrieval. With those configured, use two terminals:

```bash
backend/venv/bin/python -m uvicorn main:app --app-dir backend --reload --port 8000
NEXT_PUBLIC_API_URL=http://localhost:8000 npm --prefix frontend run dev
```

Open http://localhost:3000. The [historical technical README](https://github.com/ValerianFourel/SeoulDoctor/blob/15ce04908f33ead9620ae4201523ddab980ca0e2/README.md) preserves the earlier detailed documentation.

## Next steps and records

First fix the English semantic failures and restore reranking. Then trace every frozen facility miss through eligibility, discovery, shortlist and final ranking before choosing one small change. Compare complete results by draw and unique pair; source-clinic recall alone does not prove better recommendations. Startup optimization, live layout verification, reviewer access and the demo recording remain open.

- [Session handoff and next-task brief](planning/13-session-handoff.md), [baseline](planning/baseline.md), and [decisions](planning/decisions.md).
- [Agent worklog](planning/agent-worklog.md) and numbered planning notes record collaboration. They are summaries, not verbatim transcripts.
- [Session-log status](planning/session-logs/README.md): no supported export of this active session was found in the inspected interfaces; no transcript was fabricated.
- [Private evaluation checkpoints](https://huggingface.co/datasets/ValerianFourel/seouldoc-eval-handoff/tree/06dccdcccbc3b27c26c72f87be088d8cb79604a9/runs/mini-retrieval-50-20260906T233703Z) require access. [Submission checklist](planning/submission-checklist.md) tracks remaining deliverables.
