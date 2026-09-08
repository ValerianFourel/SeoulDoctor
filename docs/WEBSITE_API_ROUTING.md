# Website API routing

The static website on `seouldoc.io` and `www.seouldoc.io` calls these public
APIs in order:

1. `https://valerianfourel-seouldoctor-ncs-retriever.hf.space`
2. `https://valerianfourel-seouldoctor.hf.space`

`frontend/lib/api.ts` owns this policy. Production ignores the historical
Vercel `NEXT_PUBLIC_API_URL` value. Vercel previews use the same pair unless
an explicit non-Render API override is configured. Pages served inside an
HF Space use their own origin. Local development uses `NEXT_PUBLIC_API_URL`
when configured, otherwise its own origin.

Only `/chat` and `/set_travel_preference` can fall back. Requests are sequential,
with 90 seconds per attempt. Network errors, timeouts, 5xx responses, and
malformed successful responses trigger fallback. Ordinary 4xx responses and
caller cancellation do not. Both failures produce a retry message. Each new
request starts with NCS again; there is no persistent backend selection.

The full serialized request, including conversation state, and consent header
are preserved on fallback. Cross-origin cookies are omitted because HF's
preflight does not allow credentialed requests. The backend reads consent from
`X-Cookie-Consent`; it does not need a session cookie for these endpoints.
Neither Space currently needs a browser API token. Never embed an HF token in
`NEXT_PUBLIC_*` variables or client code.

A successful HTTP response does not prove full retrieval readiness. The website
uses whatever retrieval capability the chosen backend exposes. This routing
change does not provision a reranker or change backend deployments.

Run the routing regression tests with Node 24 or newer:

```bash
node scripts/test_frontend_api.cjs
npm --prefix frontend run build
```

Vercel production follows GitHub `main`. Verify its deployment commit and a
fresh website bundle after publishing; an HF deployment does not update Vercel.
