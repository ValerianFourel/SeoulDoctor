# 09. Facility summaries, five-review pages and Google translation

Owner: root, branch `ncs`, parent b0bbea14ab7094fe718f4d13ec4129b2bbb8c816.
This implements the user's three changes to the supplied Mullae/Itaewon example.
The example also shows location/recommendation-quality concerns; those are not
silently treated as fixed by this presentation update.

## Changes

- Restore the existing facility summary and its Read more/Show less control,
  including cards marked as provisional. Remove the repeated per-card candidate
  paragraph. The response-level incomplete-search message remains intact.
- Show at most five review records per facility at once. Previous/Next buttons
  preserve access to the remaining reviews without changing source order. A small
  arrow beside Patient reviews collapses or expands the whole review section,
  with accessible names, expanded state and controlled-region linkage.
- Use the user-provided GOOGLE_TRANSLATE_API_KEY privately with Google Translation
  Basic v2, `model=nmt`, plain text and the X-Goog-Api-Key header. The key stays
  server-side. Original text, IDs, facility ownership and metadata remain in the
  API; originals remain available behind labeled translations or on failure.
- Increase the per-response translation count bound from 20 to 100 distinct
  selected comments, retaining the existing 16,000-character cap, deduplication,
  one request and bounded timeout. The user's example exceeded 20 comments,
  which would otherwise leave later cards untranslated even with a valid key.

Google's [authentication guide](https://docs.cloud.google.com/translate/docs/authentication)
confirms Basic v2 supports API keys. The [API-key guide](https://docs.cloud.google.com/docs/authentication/api-keys-use)
recommends the header instead of placing keys in URLs. The [translate reference](https://docs.cloud.google.com/translate/docs/reference/rest/v2/translate)
allows up to 128 strings per request; the app remains below that limit.

## Real translation check and correction

The user specifically authorized using the newly added key in backend/.env.
Only that key was selected inside the private helper process; no key or env file
content was displayed, copied to Git, or put in the image.

Basic v2 successfully translated a positive review and a mixed review while
retaining the receptionist criticism and emoji. A third synthetic example said
that the patient waited over two hours without a proper explanation. Google
returned the correct phrase "two hours" for the original numeric duration, but
our literal digit check rejected it. The check now recognizes English zero-to-
twelve time counts for minutes/hours/days/weeks/months/years. A changed duration
still fails. This is a narrow equivalence, not a claim to validate all translation
meaning. Larger or differently expressed numbers may still fall back to originals.
All three live examples then passed with originals retained.

## Verification

- Full backend discovery: 245 tests passed.
- Frontend production build passed.
- New tests cover equivalent versus changed time counts and the 100-comment
  translation limit with one request and retained originals beyond the bound.
- Browser pagination/collapse checks and deployment status are recorded below
  after completion. Existing source evidence and search ordering were preserved.
- No retrieval evaluation rerun, index rebuild, ranking change, hardware purchase,
  visibility change, main merge, or original-Space deployment is part of this task.

Next action: deploy the verified sources and private key to the existing ncs
Space, then verify the active revision and translated-card behavior.

Browser verification passed in English desktop and Korean mobile layouts:
12 synthetic records paginated as 5/5/2, collapse hid all reviews, expand restored
the page, Previous worked, original toggles worked, and missing/failed translation
metadata retained Korean originals. Provisional cards showed the normal summary;
the old candidate paragraph was absent. No horizontal overflow, visible internal
IDs, or page errors were observed.

Deployment result: application source 14a9e7c3ad92694a8a2d571c6342ab5e4138aa96
was pushed to origin/ncs and uploaded as Space source
2e9a6e65b1037f347791fb265cbe061ebcb070c7. The image build and upload completed.
The bounded runtime wait ended with RUNTIME_ERROR: “Scheduling failure: not
enough hardware capacity.” Requested hardware remains l4x1; current hardware
is null. Website returned 503. This is an infrastructure scheduling blocker,
not a measured model/dataset startup bottleneck. No replacement hardware,
restart loop, or rebuild was started. The Google key was configured privately;
three live Basic-v2 translation checks passed before deployment. Live translated
card verification remains blocked. Next action: verify the exact runtime
revision and translated cards when the existing Space can acquire its L4.
