# NCS search repair evaluation

Run from the repository root against the isolated candidate API. The runner
uses `scripts/search_repair_scenarios.json`: 14 fixed cases (24 conversations
including variants) and 12 adaptive patient briefs of 2–6 turns. Never point
fault fixtures at the public Space.

```bash
backend/venv/bin/python scripts/search_repair_eval.py \
  --base-url http://127.0.0.1:8000 \
  --run-dir .audit/search-repair-fixed-FRESH_ID \
  --phase fixed --application-revision COMMIT
```

The run directory must be new. Requests retain sequential conversation state.
Each response or error is checkpointed. There are no automatic retries. Budgets
are 80 application calls, 70 actor calls, USD 2 for the actor, and three hours.
Application calls have a 240-second timeout; actor calls have a 60-second timeout.
Application-model costs are separate and are not measured by this runner.

Fixture-dependent cases are recorded as `not_run` with a nonzero exit. They need
separate local API regression evidence. Review the deterministic findings and
judge packets, including the source ownership and translation evidence. A
successful HTTP response does not prove source accuracy or recommendation quality.

Adaptive runs require `OPENROUTER_API_KEY` in the process environment and a
Codex-reviewed fixed gate. The runner checks the model catalogue and price ceilings
before calling the pinned Qwen actor through Phala. It never gives the actor the
private oracle. An unavailable pinned model fails the run; do not silently change
models in a scored run.

```bash
backend/venv/bin/python scripts/search_repair_eval.py \
  --base-url http://127.0.0.1:8000 \
  --run-dir .audit/search-repair-adaptive-FRESH_ID \
  --phase adaptive --application-revision COMMIT \
  --fixed-gate .audit/search-repair-fixed-FRESH_ID/reviewed-gate.json
```

The reviewed gate requires `passed: true`, `reviewer_backend: "codex_subagents"`, the exact
`application_revision`, `manifest_sha256` from the fixed run record, and
`evidence_paths` referencing reviewed artifacts. This is a reviewer attestation;
the runner cannot establish that an external review happened.

Root grades completed conversations from 1 to 5 on intent fidelity, specialty
and location accuracy, review relevance, translation faithfulness, and useful
guidance. Every dimension must be at least 4. Unsupported or incomplete dimensions
remain unscored. Wrong location, wrong specialty, hard-radius violations, invalid
ownership, and fabricated text block publication regardless of average scores.
Keep transcripts and generated reports out of Git. Follow the Cloud handoff for
authorized private checkpoint uploads.
