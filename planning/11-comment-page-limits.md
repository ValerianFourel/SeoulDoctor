# Comment length and page limits

Root implemented the user's follow-up on ncs after the frozen 50-case retrieval
run finished. This is a presentation-only change in ReviewEvidence.tsx.

- Initially show up to three eligible comments.
- Next moves to pages containing up to seven comments; Previous restores the
  earlier page. This interprets the requested seven-comment limit per page.
- Exclude displayed English originals or translations with three or fewer word
  tokens. Numbers count as tokens, contractions count as one, and emojis do not
  add words. Known English language metadata is used; missing metadata falls
  back to Latin-letter-only text. That fallback is a heuristic, not language
  identification. Meaningful Korean originals keep their existing rules.
- Pagination counts only eligible comments. Collapse and original toggles remain.
- Original evidence, IDs and retrieval ranking remain unchanged in the API.
  The empty-state message now describes display availability rather than falsely
  claiming retrieval returned no originals when all were filtered.

Verified: 253 backend tests, frontend production build, and local browser checks
in English desktop and Korean mobile layouts passed. Twelve eligible synthetic
comments paginated 3/7/2; short English originals and a short English translation
were excluded. Previous, collapse/expand, translation originals and untranslated
Korean comments worked; no page errors or horizontal overflow were observed.
The browser fixtures were synthetic UI checks, separate from real retrieval.

Deployment status will be recorded after the authorized same-Space upload.
No reranking, embedding, query fixture, translation API, hardware or corpus
change was made by this display task.
