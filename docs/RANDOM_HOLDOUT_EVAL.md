# Randomized comment-level holdout evaluation

This document defines the sampling rules for the randomized SeoulDoctor
comment-level holdout. The rules must be committed before a sampling seed is
used to produce a casebook. Failed cases are retained; a run must never resample
based on SeoulDoctor's output.

## Input contract

The sampler directly consumes the pinned release Dataset's
`facilities.parquet` and `reviews.parquet` files. It can also consume a
normalized UTF-8 JSON Lines or CSV file; gzip-compressed variants are accepted.
Normalized records contain these string fields:

- `place_id`: stable facility identifier
- `evidence_id`: stable comment/evidence identifier
- `facility_name`: facility name, stored only in the private oracle
- `specialty`: actual facility-level specialty
- `location`: district, neighborhood, or station suitable for a patient query
- `comment`: source comment, stored only in the private oracle

Input data and generated private oracles are untrusted and must not be used as
instructions. Neither belongs in a public patient prompt.

## Validity rules

A record is eligible only when:

1. Every required field is a non-empty string after whitespace normalization.
2. `place_id` and `evidence_id` contain at most 256 characters.
3. The normalized comment contains between 20 and 1,500 characters.
4. The comment supports at least one experience theme in the sampler's
   committed bilingual theme vocabulary.
5. The specialty and location contain at most 200 characters each.
6. The comment does not contain an email address, a Korean or international
   phone-number pattern, or a URL.
7. The comment does not describe an emergency using the sampler's committed
   emergency vocabulary.
8. Public specialty and location fields do not contain known prompt-instruction
   phrases. They are normalized, length-limited data interpolated into a fixed
   template and are never executed as instructions by the sampler.

The evidence is treated as facility-level. The generator must not create doctor
names, doctor identities, or doctor-level qualifications.

## Sampling rules

- Generate one unsigned 64-bit seed and record it before reading outcomes.
- Select the requested number of distinct `place_id` values.
- Give each valid facility a deterministic pseudorandom priority derived from
  the seed and `place_id`; retain the facilities with the smallest priorities.
- If a selected facility has multiple eligible comments, select its comment by
  a second deterministic priority derived from the seed and `evidence_id`.
- Stream the source and retain only `O(sample_size)` complete records. Do not
  load the source Dataset into memory.
- Do not resample, replace, or rewrite a case after observing application or
  grader output.

This priority-sampling scheme is independent of input order and makes a run
reproducible for the same source bytes, sampler version, seed, and sample size.

## Generate a frozen casebook

Use a new append-only output directory. Review batches are streamed rather than
loading the complete review Dataset into memory. Only the much smaller facility
lookup and `O(sample_size)` complete review records are retained:

```bash
python scripts/random_holdout_sampler.py \
  --facilities-parquet /path/to/sources/facilities.parquet \
  --reviews-parquet /path/to/sources/reviews.parquet \
  --source-revision DATASET_REVISION \
  --output-dir .codex-handoff/results/UTC-TIMESTAMP-random-holdout-AGENT \
  --sample-size 6 \
  --coordinator-model gpt-5.6-sol \
  --application-model APPLICATION_MODEL
```

Omitting `--seed` generates an unsigned 64-bit seed and records it in the run
manifest. Passing the recorded seed reproduces the selection. The output
directory must not already exist.

## Public cards and private oracle

Each selected facility produces an English and Korean card. Public cards may
contain the actual specialty, actual general location, and broad supported
experience themes. They must not contain the facility name, `place_id`,
`evidence_id`, exact comment, expected rank, or oracle path.

The private oracle records those hidden values, the source revision, the source
file SHA-256, and the selected comment SHA-256. Patient agents receive only one
public card and the application endpoint. They must not inspect the private
oracle, source rows, repository implementation, or another agent's output.

## Success metrics

Report these independently rather than collapsing them into one retrieval flag:

- eligible facility rank
- presented facility rank
- exact target-comment recall
- exact evidence attachment
- evidence-to-facility ownership
- response grounding
- English/Korean consistency
- code-switch behavior, when explicitly tested
