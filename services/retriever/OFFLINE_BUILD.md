# Offline BGE-M3 corpus build

The chat application never encodes the review corpus. This job creates the
immutable sparse and dense review release consumed by `production_core.py`.

## Safety properties

- The raw Parquet file and BGE-M3 model revision are SHA-pinned in the release.
- Reviews use the same evidence-ID formula as the application's raw review store.
- Rows are ordered by facility ID and evidence ID, so each facility has one
  contiguous range.
- Each 25,000-review shard has a record digest and file digest. A rerun reuses a
  shard only when its source, model, settings, records, and bytes still match.
- Assembly writes into a temporary directory and writes `manifest.json` last.
- The final validator reads the actual arrays and checks hashes, shapes, dtypes,
  finite values, ownership ranges, and evidence-ID uniqueness.

## Benchmark

Install the build-only dependencies from `requirements-build.txt`, then run:

```bash
python3 offline_builder.py \
  --reviews /input/seoul_medical_reviews_merged.parquet \
  --work-dir /output/bge-m3-benchmark-100k-v1 \
  --release-id bge-m3-benchmark-100k-v1 \
  --model-revision 5617a9f61b028005a4858fdac845db406aefb181 \
  --batch-size 64 \
  --shard-size 25000 \
  --max-length 128 \
  --sparse-top-k 128 \
  --max-reviews 100000 \
  --dataset-repo ValerianFourel/seouldoc-bge-m3-review-index-benchmark
```

The benchmark uses the first 100,000 nonempty source reviews only to measure
throughput, peak memory, output size, and upload time. Never configure the live
retriever with the benchmark Dataset.

Estimate the complete run from the measured encoding time:

```text
full encoding time = 100k encoding time * 17.91749
```

## Complete release

Use a new work directory and private Dataset. Omit `--max-reviews`:

```bash
python3 offline_builder.py \
  --reviews /input/seoul_medical_reviews_merged.parquet \
  --work-dir /output/bge-m3-full-v1 \
  --release-id bge-m3-full-v1 \
  --model-revision 5617a9f61b028005a4858fdac845db406aefb181 \
  --batch-size 64 \
  --shard-size 25000 \
  --max-length 128 \
  --sparse-top-k 128 \
  --dataset-repo ValerianFourel/seouldoc-bge-m3-review-index
```

After upload, pin the returned Dataset commit in
`RETRIEVER_DATASET_REVISION`. Set `BGE_M3_MODEL_REVISION` to the model commit
shown above. Do not use `main` for either revision.

## SSH and recovery

Launch the Hugging Face Job with `--ssh`. Once its status is `RUNNING`, connect:

```bash
hf jobs ssh <job-id>
```

The Job requires an SSH public key registered in the Hugging Face account. Do
not share an SSH private key or Hugging Face token in chat. Store checkpoints
under the mounted read-write `/output` volume. If a Job stops, launch the same
command against that volume and the builder will verify and reuse complete
shards.
