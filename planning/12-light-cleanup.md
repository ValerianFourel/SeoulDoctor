# Light cleanup of recent ncs changes

Owner: root, sequential work on ncs, starting commit
66022eabf25ee0dde0ebeab4276aef78d7e6ef0b.
Owns ReviewEvidence.tsx, this note, and an appended coordination entry.
Other uncommitted documentation is preserved. No deployment or infrastructure
change is authorized for this cleanup.

## Playbook checklist

- [x] Read the poteto-mode Principles section in full.

1. Pin the behavior contract first. Run the **how** skill over the affected subsystem to learn the contract, then write a characterization test, snapshot, or equivalence harness that captures current behavior before any structure moves. The harness makes "refactor" a checkable claim (**principle-prove-it-works**). If the area has no coverage, write the pin before touching structure. Type check and lint are not a pin.
2. Name the structure the code is missing per **principle-model-the-domain**: a state machine over scattered booleans, a table or registry over spread-out branching, a typed model over repeated shape assumptions, a reducer over ad hoc mutations. Boring code stays when the shape is already clear and local; the reshape must delete branches or invalid states, not add indirection.
3. Name the target shape. State what the module layout, types, and call graph should be if built today (**principle-foundational-thinking**, **principle-redesign-from-first-principles**). If the target crosses a function boundary, run the **architect** skill for parallel design exploration of the shape before the move.
4. Subtract before you add. Delete dead weight, collapse one-caller wrappers, drop redundant validators, and remove orphan references before introducing the new shape (**principle-subtract-before-you-add**). The smallest change that reaches the target shape ships (**principle-laziness-protocol**). A speculative cleanup that "might help" gets reverted, not left to ride.
5. Move in small behavior-preserving steps, each keeping the pin green. For API reshapes, migrate every caller and delete the old API in the same wave (**principle-migrate-callers-then-delete-legacy-apis**). No compatibility shims, no parallel old-and-new paths. Spot-check every rename against the actual files; renames silently miss usages in strings, prose, and back-references. Delegate the mechanical edits to a subagent using your configured refactoring model (default in poteto-mode's Models section) with a specific scope (file paths, the names being moved, the behavior to hold); review the diff yourself.
6. Prove behavior is unchanged on the real artifact, not "it compiles" (**principle-prove-it-works**). For larger reshapes, run an equivalence check: a script that diffs old-vs-new outputs, a recorded baseline replayed against the new code, or a smoke run on the matching surface via the relevant control skill. Own the verification yourself; do not trust a delegate's "looks good" summary.
7. Confirm the change earns its place. The success measure is reduced reader load (**principle-minimize-reader-load**): fewer layers between question and answer, less hidden state, fewer indirections without a second consumer. If the diff does not lower reader load somewhere, revert it.
8. Rebase into small ordered commits that tell the story. A subtraction commit, then the reshape, then any follow-on cleanup, so a single revert undoes one slice. Shape them with the **sequence-verifiable-units** principle skill, so each behavior-preserving slice stays green before the next. Run **Opening a PR**.

The user's no-subagent instruction replaces all delegated steps with sequential
local inspection. No API boundary changes or new abstraction are planned, so
architectural fan-out and API migration are not applicable. A PR is outside this
small branch-local cleanup; retain local verification and narrow commits instead.

## Trace and contract

ChatInterface passes a facility's retrieval_evidence and review_language into
ReviewEvidence. Each record retains original text, verbatim status and optional
presentation metadata. The component excludes source noise, selects a usable
translation only when its language matches, and filters short displayed English
comments. It then paginates the eligible records and renders labels, source
original toggles and collapse controls. Backend translation and retrieval are
outside this cleanup.

The duplicated decision is whether a translation is usable: filtering and
rendering both repeat the status/language/nonempty checks. The target shape is
one local display record per eligible source review, carrying its chosen
translation. No new module, exported type, helper layer or dependency is needed.
The existing API record and component props stay unchanged.

Throughput checkpoint: one owner, one production file, one equivalence harness,
then browser verification and required checks. Existing code already has the
small shape needed for pagination; leave its arithmetic and state unchanged.

## Verification and outcomes

Completed one small structural change. The component now computes translatedText
once and carries it beside the unchanged source review in the local display list.
Filtering and rendering share that decision. No new function, module, exported
type or dependency was added. The diff is 12 added and 14 removed lines in one
production file. There were no unused imports, dead functions or redundant
wrappers in that file worth removing. Pagination and language detection stayed
as they were. No speculative change was attempted or reverted.

The pre-edit characterization harness captured 3,336 complete HTML renders.
The post-edit output matched byte-for-byte, covering original text, short
English text, numbers, emojis, Korean text, noise, missing metadata, each
presentation status, matching and mismatched translation languages, empty
translations, page counts, out-of-range pages and collapsed content.
Local harness: /tmp/ncs-review-equivalence.cjs. Run it with --capture before a
change, then without that flag to compare. Its baseline is a local artifact,
not an evaluation transcript or a substitute for live retrieval.

The real production build passed. Browser interaction checks passed in English
desktop and Korean mobile layouts, preserving 3/7/2 pagination, Previous,
collapse/expand, filtering and original toggles. Full backend discovery passed
253 tests in 5.248 seconds. No new live model, translation or retrieval call ran.

Deslop and comment review were performed locally under the no-subagent rule.
No added comments, casts, broad exception handlers, suppressions or boundary
validation deletions were found. No comments were deleted or restored; no
constraint encodings or review findings remain open. Type validation and source
visibility safeguards remain necessary and unchanged.

Laziness Protocol limited the work to one proven duplicated decision. Model the
Domain kept source review and optional display translation together locally.
Minimize Reader Load removed the need to compare two translation predicates.
Prove It Works required exact rendered-output equivalence and browser checks,
not just a successful compile.

Existing live failures remain separate. All 50 retrieval cases were incomplete
because reranking was disabled; two English semantic requests failed. This
cleanup did not fix or rerun them. The prior display deployment was already
submitted before the no-deployment instruction; no cleanup deployment, restart,
new hardware or other infrastructure change was made.

Next action: review this local cleanup commit. Diagnose retrieval service
failures separately before repeating the unchanged evaluation fixtures.
