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

## Remaining build step

The runtime and app integration are implemented. The production 1.79-million
review release has not been encoded yet. Build it offline on a temporary GPU,
validate it against the raw-review digest and the component retrieval gates,
then publish it to a private Dataset repository. Do not build the corpus during
Space startup or a chat request.
