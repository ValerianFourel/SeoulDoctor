# Useful follow-up after a facility search

The NCS app returned the same generic caution for every successful search with
cards. It generated an answer first, then discarded it in
`finalize_evidence_response`. Changing the generation prompt alone could not
change the patient-visible reply.

## Measured baseline

On September 8, 2026, the existing public Spaces reported these active sources:

| Application | Space revision | Existing hardware |
| --- | --- | --- |
| NCS | `4138eb056b80f4689fb0ef87244cf128c44280b2` | `t4-medium` |
| Original | `8909ff673d2e1b5e7df97010f9e309f6a9dc4c7b` | `cpu-upgrade` |

Both exposed the application model as `openai/gpt-oss-120b` through OpenRouter.
The endpoints did not expose an immutable model revision or provider charges.
No hardware, deployment, secrets, or indexes changed during these checks.

The exact patient query was `i need a orthopedic doctor next to  Jonggak`.
Two fresh conversations on each application returned five cards in the same
order. A clinic 3.916 km from Jonggak appeared before one 0.888 km away.
Both applications resolved Jonggak and orthopedics but retained the default
5 km radius. Neither offered a useful refinement prompt.

NCS repeated its generic introduction exactly in both calls, which took
15.179 and 34.499 seconds. A separate browser conversation reproduced the
same sentence and card ordering. The original app responded in 15.784 and
14.722 seconds. Its prose named three clinics while displaying five and
asserted English-speaking staff without evidence establishing that claim.
One response also misstated the range of distances.

An NCS follow-up, `Keep it within 1 km of Jonggak, please.`, returned in
15.102 seconds. It retained the specialty, location, and coordinates, changed
the radius to 1 km, and returned three clinics at 0.432, 0.673, and 0.888 km.
Its introduction was still generic. This verifies a useful refinement the
response can suggest.

These are five sequential ad hoc API calls plus one browser conversation,
not a broad quality evaluation. The focused evaluator suite passed 36 tests
before the API calls. Redacted recordings remain under the local ignored
`.audit/response-polish-20260908-baseline/` path. Private Dataset upload was
unavailable because the approved process credentials were absent.

## Chosen response design

The response uses validated card facts and the existing `State`. It can name
the closest displayed option and its straight-line distance, then invite a
useful refinement. It must qualify the comparison as among the displayed
options. It must not claim that the closest result is the nearest clinic in
Seoul or the best medical choice.

The server continues to control evidence ownership, incomplete-search
disclosures, unresolved requirements, and negative or mixed feedback.
Language support, availability, and suitability are not inferred from review
language or a search match. Card order and retrieval scores stay unchanged.

We considered exposing a bounded LLM answer with a validation step. Checking
names and citations would not prove that all suitability claims were supported.
The observed original-app claims make that tradeoff concrete. A composer from
validated facts solves this defect with fewer moving parts and avoids the
generation call that NCS discards today.

The frontend's decimal-formatting regex also needs correction. It inserts a
line break before `0.` in a distance such as `0.9 km`. The response must retain
its intended paragraphs and decimal distances.

Poteto Mode's Fix Root Causes principle directed the change to the final
response boundary. Model the Domain kept the existing conversation state as
the source of context. Experience First favored one useful next step.
Build the Lever and Prove It Works require a replayable comparison and a
browser check of the actual reply.

## Verification and delivery

Implementation is verified at
`55d477ed8faa493919a6b80e125b5a05c83b32b7` on
`local/ncs-response-polish-20260908` in `/tmp/seouldoc-response-polish`.
The original application remains separate from NCS as requested in the
previous conversation. No candidate has been deployed.

The focused and adjacent tests passed 35 cases. Full backend discovery passed
264 tests in 5.261 seconds, and the frontend production build passed. The
sandboxed backend run stalled and was interrupted; its bounded host retry
passed. The sandboxed build failed with a generic webpack error; its host
retry passed. Independent code and comment reviews found no issues.

Desktop and mobile browser checks passed at 1280x720 and 390x844. The actual
message text matched the candidate finalizer output, including the unbroken
`0.9 km` distance and both paragraphs. Card order was unchanged. There was no
horizontal overflow or JavaScript error. Existing font-preload warnings appeared.

The recorded Jonggak response now reads:

> I found 5 options for orthopedics around Jonggak. Of the options shown,
> 광화문정형외과의원 is closest, about 0.9 km from Jonggak by straight-line distance.
>
> What would you like the doctor to help with? You can also narrow the search
> to within 1 km, or add language or access needs.

The local replay uses the recorded search results and translation responses.
It exercises the revised finalizer and frontend without claiming new live
retrieval or new translation quality. The public NCS response omits private
retrieval metadata, so replay uses an empty metadata mapping. Separate focused
tests cover incomplete and unresolved searches.

The runnable replay, output, screenshots, and logs remain in the ignored local
`.audit/response-polish-20260908-candidate/` directory. This comparison verifies
the response boundary and frontend, not new provider latency or retrieval
quality. One generation call is removed from the tested retrieval path; no
live latency improvement is claimed before deployment.

Remaining retrieval rank, translation quality, precomputed summary wording,
and original-app generation problems need separate measured changes.
