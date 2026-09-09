# NCS conversation evaluation harness

The deployment smoke asks whether a person can refine a natural request, get a
useful clinic comparison, and see correctly attached original patient comments.
It extends the existing search-repair runner instead of replacing the sealed
grounded, stress, or random-holdout suites.

## Capability and test map

| Product behavior | Owning implementation | Verification |
| --- | --- | --- |
| `POST /chat`, response state, and result cards | `backend/main.py`, `backend/models.py` | `scripts/search_repair_eval.py` sends the real payload and carries the returned state sequentially. |
| Specialty, location, symptoms, hard radius, preferences, and replacements | `backend/query_facets.py`, `backend/search/turn_delta.py`, `backend/search/scope.py` | The seven smoke cases cover clarification, language evidence, staff-role conflict, wait withdrawal, location replacement, code switching, and relevant dental reviews. |
| Original review retrieval, ownership, and response claims | `backend/search/evidence_retrieval.py`, `backend/evidence_response.py`, `backend/review_presentation.py` | Turn assertions check IDs, verbatim text, ownership, specialty, radius, and the retired generic incomplete-search sentence. `scripts/search_repair_report.py` resolves every surfaced evidence ID against the pinned local index. |
| Translation-first cards and original reveal | `frontend/components/ReviewEvidence.tsx`, `frontend/components/ChatInterface.tsx` | API packets retain translation status and original text. `frontend/tests/review-visibility.cjs` verifies the real 3/7 pagination and reveal controls. A live browser check is reported separately from saved-response rendering. |
| Independent quality judgment | `scripts/conversation_eval.py`, `scripts/conversation_eval_config.json` | Immutable packets expose user-visible content and a private oracle only to the judge. Code validates hashes, score ranges, citations, adherence, critical findings, and computes the verdict independently of the judge's proposed verdict. |

The smoke cases are open-ended diagnostics. Exact-clinic and exact-comment
targets remain in `backend/tests/grounded_bilingual_scenarios.json`; their
deterministic ranks, attachment, ownership, and quotation checks remain in
`backend/tests/grounded_journey_grader.py`. The existing 20 core, 10 stress,
and 12 holdout inventory remains governed by `docs/EVAL_LAUNCH_PLAN_2026-09-06.md`.

## Judge decision

`scripts/conversation_eval_config.json` records `codex_subagents`. OpenRouter
runs the simulated patients. After the run, an isolated Codex subagent receives
only the rubric and immutable packet, with no implementation history or another
judge's verdict. Python exports and validates packets but does not claim it can
invoke Codex tools. If no isolated Codex judge is available, the report remains
pending.

The config also supports `openrouter` for unattended API judging and `both` for
two independent judgments. In `both` mode, a disagreement is not averaged into
a pass. Change the backend only by updating the config decision ID and reason.

## Commands

Validate the config and frozen seven-case selection without network requests:

```bash
backend/venv/bin/python scripts/conversation_eval.py dry-run \
  --output .audit/conversation-dry-run-FRESH_ID.json
```

Run the offline harness checks and the existing evaluator checks:

```bash
backend/venv/bin/python -m unittest \
  backend.tests.test_conversation_eval \
  backend.tests.test_search_repair_eval \
  backend.tests.test_grounded_bilingual_suite \
  backend.tests.test_grounded_journey_grader \
  backend.tests.test_patient_journey \
  backend.tests.test_qwen_likert_judge
```

Run the seven adaptive cases against a verified deployment after producing the
matching fixed and fixture gate described in `docs/NCS_SEARCH_REPAIR_EVAL.md`:

```bash
backend/venv/bin/python scripts/search_repair_eval.py \
  --base-url https://valerianfourel-seouldoctor-ncs-retriever.hf.space \
  --run-dir .audit/ncs-deployed-smoke-FRESH_ID \
  --phase adaptive --application-revision COMMIT \
  --fixed-gate .audit/ncs-fixed-FRESH_ID/reviewed-gate.json \
  --scenario adaptive-12-relevant-dental-reviews \
  --scenario adaptive-11-consultation-language \
  --scenario adaptive-08-staff-role-conflict \
  --scenario adaptive-09-withdraw-wait-preference \
  --scenario adaptive-10-replace-location \
  --scenario adaptive-06-language-switch \
  --scenario adaptive-01-foot-clarification
```

The runner uses `OPENROUTER_API_KEY` for the simulated patient. It sends only
the patient brief, current stage, and visible conversation to OpenRouter. The
private oracle stays in the run and judge packet.

Export isolated Codex packets:

```bash
backend/venv/bin/python scripts/conversation_eval.py judge \
  --run .audit/ncs-deployed-smoke-FRESH_ID/run.json \
  --output .audit/ncs-deployed-smoke-FRESH_ID/judging
```

The pending command exits nonzero because no judgment has been imported. Give
each file in `judging/packets/` to an isolated Codex subagent. Import its single
JSON bundle:

```bash
backend/venv/bin/python scripts/conversation_eval.py import-codex \
  --output .audit/ncs-deployed-smoke-FRESH_ID/judging \
  --judgments /path/to/codex-judgments.json
```

For unattended OpenRouter judging of the same frozen transcript, rerun `judge`
into a new output directory with `--backend openrouter`. Rejudging never sends
another application turn. A new application replay uses a fresh run directory
and the fixed-message runner; uncertain timed-out turns are never replayed.

Run the broader suites with the unchanged commands in
`docs/EVAL_LAUNCH_PLAN_2026-09-06.md`. Live model calls are explicit and are not
part of unit-test or CI execution.

## Result interpretation

Every applicable score must be at least 4/5, every objective assertion must
pass, the simulator must follow its staged brief, and no critical failure may
be present. Null scores, incomplete conversations, invalid judge JSON,
unresolved citations, transport errors, simulator drift, and unavailable judges
remain failed or pending states. A good alternative can satisfy an open-ended
scenario while a separate exact-comment target probe fails; reports keep those
outcomes separate.

The API proves returned evidence, not browser visibility. A release report must
also identify whether the deployed `seouldoc.io` UI rendered each clinic's
translation, original reveal, pagination, and citation navigation. Unknown UI
stages stay unknown.
