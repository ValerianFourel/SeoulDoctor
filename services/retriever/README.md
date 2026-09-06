---
title: SeoulDoc BGE-M3 Review Retriever
emoji: 🔎
colorFrom: indigo
colorTo: cyan
sdk: docker
app_port: 7860
suggested_hardware: t4-small
startup_duration_timeout: 1h
pinned: false
---

# SeoulDoc BGE-M3 review retriever

This private Space searches the BGE-M3 learned-sparse and dense review indexes.
The public application supplies a bounded facility shortlist and bilingual query
cells. The service returns evidence IDs and ranks. The application resolves each
ID through its local immutable review store before using it.

The production entry point is `production.py`. `app.py`, `build_fixture.py`,
`test_app.py`, `Dockerfile.fixture`, and `README.fixture.md` are isolated fixture
assets. They never run in the production image.

## Required configuration

```text
BGE_M3_MODEL_REVISION=<pinned Hugging Face commit SHA>
RETRIEVER_RELEASE_DIR=/data/release
```

If the release is not mounted at that path, configure a pinned private Dataset:

```text
RETRIEVER_DATASET_REPO=OWNER/seouldoc-bge-m3-reviews
RETRIEVER_DATASET_REVISION=<pinned Dataset commit SHA>
HF_TOKEN=<read-only token>
```

The service refuses to start if the BGE-M3 revision differs from the revision
recorded in the semantic release.

## Immutable release format

`manifest.json` uses schema `seouldoc.bge-m3-review-manifest/v2`. It records the
release ID, raw-review snapshot SHA-256, exact model revision, review count,
dimension 1024, and the size and SHA-256 of every artifact below.

```text
manifest.json
evidence_ids.npy
facility_ids.npy
facility_ranges.json
dense.npy
sparse_indptr.npy
sparse_indices.npy
sparse_values.npy
```

- ID arrays contain fixed-width Unicode values and never use Python objects.
- `dense.npy` contains L2-normalized float16 or float32 BGE-M3 vectors.
- The three sparse arrays form a CSR matrix with uint64 offsets, uint32 token
  IDs, and non-negative float16 or float32 learned weights.
- Facility ranges are contiguous, disjoint, and cover every review exactly once.
- `review_source_sha256` identifies the same raw Parquet snapshot used by the
  application BM25 release. It is not the digest of a derived vector artifact.

The runtime scans only reviews belonging to at most 50 supplied facilities. It
does not run a global nearest-neighbor query.

## API

- `GET /health`
- `POST /v1/retrieve`

`POST /v1/retrieve` requires the caller's expected release ID and raw-review
digest. A mismatch returns HTTP 409. Request limits are 50 facilities, 64 query
cells, 2,000 characters per query, and 20 results per facility, query, and
channel.

## Existing production release

Reuse the existing private Dataset `ValerianFourel/seouldoc-bge-m3-review-index`
at revision `a6a3ab6f70c67c15090d175efd4ecdea553b329e`. Its BGE-M3 model revision
is `5617a9f61b028005a4858fdac845db406aefb181`. The source Dataset is
`ValerianFourel/seouldoc-app-release-20260905` at revision
`3911d79dc31e6a6ccfa3f64a7e401b88893bf66a`.

The recorded full audit matched all 1,791,749 eligible review IDs and facility
owners, with no missing, duplicate, or orphaned entries. Dense vectors exist
for every eligible review. There are 989 empty learned-sparse rows. Encoding
used a 128-token limit, exceeded by 57,134 original reviews, so full-text
lexical retrieval remains important. Do not regenerate this corpus to restore
an expired service.

Artifact coverage does not establish live GPU readiness. Verify authenticated
query inference, exact model/source revisions, both retrieval channels,
evidence resolution, and actual CUDA execution before evaluation.
