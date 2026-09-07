# Coding-agent session logs

Status at setup: no verbatim export of this active session was produced.
`../agent-worklog.md` is a human-readable factual summary, not an exported log.

Inspected Codex CLI 0.153.4 help for the top-level command, agents, exec,
app-server, and debug, plus the available tool catalog. No supported export of
this active conversation was exposed there. No raw session store was read or
copied. This is a limitation of the inspected interfaces, not a claim that all
Codex clients lack exports. UI export availability was not verified.

The installed CLI does support event capture for future noninteractive runs:

```bash
# Run only for a deliberately chosen future task, with a unique local output path.
codex exec --json -C /tmp/seouldoc-ncs 'YOUR SCOPED TASK' > /tmp/ncs-agent-events.jsonl
```

This captures that future run's JSONL events. It cannot retroactively export
this session, and it is not guaranteed to include every internal event.
`--output-last-message` saves only the final reply and is not a session export.
No extra agent run was started solely to manufacture a log.

Keep raw exports outside Git. Before including any available log, inspect for
credentials, private review/facility content, evaluation transcripts, and
repository-prohibited artifacts. Remove restricted content, mark redactions,
and record source client/version, capture dates, completeness, and review status.
Do not reconstruct missing turns. Recheck the actual client's export option at
submission time and retain this limitation if unavailable or unsuccessful.
