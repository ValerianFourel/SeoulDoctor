# Immutable search index architecture

## Problem

Phase 3 must bind 8,484 facility records, their existing dense vectors, and
1,791,749 nonempty review texts into one verifiable release. A failed or
interrupted build cannot damage the active release. Startup may validate and
report the new release, but the legacy RAG path remains live until Phase 4.

The source data establishes four constraints. `place_id` is the stable facility
identity. The current `text-embedding-3-small` vectors must be exported without
an API call and must match the current facility profile byte for byte. Review
text remains untrusted evidence and reviewer names never enter an artifact.
Every lexical and dense query must bind an `EligibleScope` before it ranks.

## Usage

The offline builder publishes a release without activating it:

```python
repository = IndexRepository(Path("/data/seouldoc/search_indexes"))
published = repository.publish(BuildRequest(
    version="2026-09-02-v1",
    facilities_path=Path("/data/seouldoc/facilities.parquet"),
    reviews_path=Path("/data/seouldoc/reviews.parquet"),
    chroma_path=Path("/data/seouldoc/chroma_db"),
    chroma_collection="seoul_med_agentic_v2",
    embedding_model="text-embedding-3-small",
    embedding_revision="provider-managed",
))
```

Activation is an explicit administrative operation:

```python
repository.activate("2026-09-02-v1")
active = repository.open_active()
```

Phase 4 will bind the complete scope before using any channel:

```python
with active.within(scope_selection) as scoped:
    lexical = scoped.search_facilities(query, limit=200)
    dense = scoped.search_dense(query_embedding, limit=200)
    evidence = scoped.search_evidence(query, limit=100)
```

## Shape

One release contains four immutable payloads and a canonical manifest:

```text
search_indexes/
  active.json
  versions/
    2026-09-02-v1/
      manifest.json
      facility_ordinals.arrow
      facility_vectors.npy
      facility_lexical.sqlite3
      evidence_lexical.sqlite3
```

`facility_ordinals.arrow` is the only durable `place_id` to ordinal map.
Ordinals follow bytewise UTF-8 `place_id` order and match rows in the normalized
float32 vector matrix. Both SQLite stores copy ordinals only where joins or
foreign-key validation need them.

The facility database contains fielded Unicode word postings, Korean character
n-grams, and a trigram substring channel. The evidence database stores original
text and source metadata beside word, Korean n-gram, and substring postings.
It contains every nonempty raw review plus deterministic summary, highlight,
amenity, medical, hours, phone, and address records. FTS row IDs equal their
document ordinals.

`review_facilities` in the manifest means facilities represented by at least
one indexed, nonempty verbatim review. Referential validation still checks the
facility ID on every source review row before applying the nonempty-text rule.
The source snapshot currently has 5,657 distinct facility IDs and 5,645 with
indexed nonempty reviews.

The manifest contains schema and analyzer versions, input byte digests, stable
ID and logical record digests, embedding model metadata, exact vector-stream
digest, counts, and byte size plus SHA-256 for every payload. It contains no
timestamp, hostname, absolute path, credential, or request policy. The logical
content ID excludes the release name and SQLite page layout, so identical
inputs converge even when a physical database build differs.

The public implementation has one lifecycle owner:

```python
@dataclass(frozen=True)
class BuildRequest:
    version: str
    facilities_path: Path
    reviews_path: Path
    chroma_path: Path
    chroma_collection: str
    embedding_model: str
    embedding_revision: str


class IndexRepository:
    def publish(self, request: BuildRequest) -> PublishedIndex: ...
    def activate(self, version: str) -> PublishedIndex: ...
    def open_active(self) -> ReadonlyIndex: ...


class ReadonlyIndex:
    def within(self, scope: ScopeSelection) -> ScopedIndex: ...
```

Manifest parsing, FTS syntax, Chroma export, staging paths, and compatibility
constants stay private. The manifest module validates disk structure only. It
does not interpret search rules or choose request limits.

## Synthesis decision

Candidate B was the base because its separate Arrow ordinal map made one
identity authority explicit and its data fields matched current SeoulDoc more
closely. The cross-judge scored it 28/30 and Candidate A 27/30.

The final shape takes Candidate A's deterministic content identity, explicit
publish versus activate split, complete scope-descriptor checks, and vector
source digest. It rejects Candidate B's timestamped identity and broad public
compatibility knobs. It rejects Candidate A's trigram-only evidence index and
evidence identity tuple that omitted the source index and text digest.

## Publication and rollback

The builder writes a unique staging directory beside `versions`, streams and
closes every artifact, writes the manifest last, and validates the stage with
the production loader. It fsyncs payloads and directories before an atomic
rename into `versions/<version>`. It never overwrites a published version.

If the version already exists, an equal logical content ID is a successful
no-op. Different content raises a collision error. Activation validates the
target, writes and fsyncs a complete temporary pointer, replaces `active.json`,
and fsyncs the root. A crash can leave a disposable stage or a complete inactive
release, but it cannot produce a partially active release. Rollback calls the
same activation operation with an older validated version.

## Validation boundary

The loader rejects unsafe versions, path traversal, symlinks, hard links,
unlisted files, SQLite sidecars, writable payloads, noncanonical JSON, pointer
or artifact digest mismatch, unsupported schemas, noncontiguous ordinals,
missing or extra facility IDs, orphan evidence, FTS row-count mismatch, and
invalid vector dtype, shape, dimension, finiteness, or L2 norm.

SQLite opens with `mode=ro&immutable=1`; NumPy uses a read-only memory map.
`ReadonlyIndex.within()` also verifies index version, scope count, scope digest,
ID uniqueness, and complete ID-to-ordinal resolution. Scoped FTS joins happen
inside the query before BM25 ordering. Dense scoring slices the same ordinal
array before its dot product.

## Tradeoffs accepted

- We keep word and substring postings because one tokenizer does not cover both
  English words and short Korean substrings well.
- We store review text in the evidence release so citations do not depend on a
  mutable Parquet file remaining mounted.
- We scan checksums and vector norms before serving because a fast startup is
  less valuable than serving a mixed or corrupt release.
- We record the historical vector revision as `provider-managed`. Exact profile
  and vector digests are honest; a made-up immutable OpenAI revision is not.

## Open risks

- The first Hugging Face deployment must confirm that `/data` provides local
  POSIX rename and directory-fsync behavior.
- The full evidence database size and build time must be measured on the target
  Space before setting operational disk and startup budgets.
- An immutable embedding revision can replace `provider-managed` only when a
  trustworthy provider or build record supplies it.
