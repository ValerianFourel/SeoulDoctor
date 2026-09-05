---
title: SeoulDoc BGE-M3 Scoped Review Retriever
emoji: 🔎
colorFrom: indigo
colorTo: cyan
sdk: docker
app_port: 7860
startup_duration_timeout: 1h
---

# SeoulDoc scoped review retriever

Private Hugging Face Space serving immutable, facility-scoped BGE-M3 sparse and
dense review retrieval. The service never receives or decides geographic rules;
the caller supplies an already-authorized list of facility IDs.

## Artifact contract

`RETRIEVER_RELEASE_DIR` contains exactly these files:

* `manifest.json`: schema `seouldoc.bge-m3-review-manifest/v1`, with
  `release_id`, `model_id` (must be `BAAI/bge-m3`), non-empty `model_revision`,
  `review_source_sha256`, `review_count`, `dimension`, and SHA-256 records for
  the other files.
* `reviews.jsonl`: one canonical UTF-8 JSON object per row, in ordinal order:
  `{"evidence_id", "facility_id", "text"}`. IDs are unique and non-empty.
* `facility_ranges.json`: object mapping each facility ID to `[start, end]`
  half-open row offsets. Ranges must be sorted, disjoint, and cover every row.
* `dense.npy`: float32, L2-normalized shape `(review_count, dimension)`.
* `sparse.jsonl`: one JSON object per row mapping string token IDs to finite,
  non-negative weights. Rows align exactly with `reviews.jsonl`.

The manifest digests all four payload files and the SHA-256 of `reviews.jsonl`
is also the review source digest. Startup fails closed for any mismatch,
malformed row, duplicate ID, non-normalized vector, or invalid range.

Build a production release offline with `python app.py build --input reviews.jsonl
--output release --model-revision <pinned-revision>`. The command intentionally
requires a supplied encoder; it does not download models or attempt the 1.79M
row build. The fixture builder in `tests/` demonstrates the exact format.

`POST /v1/retrieve` accepts `facility_ids`, batched `queries`, and
`limit_per_facility_per_channel` (maximums: 50 facilities, 64 queries, 20
results). It returns evidence IDs plus sparse and dense ranks for every
facility/query/channel. `GET /health` reports readiness and release metadata.
