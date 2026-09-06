"""Run a real BGE-M3 bilingual retrieval smoke test against a full release."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import production
from production_core import QueryCell, SemanticRelease

TARGET_FACILITY = "35475921"
DECISIVE_EVIDENCE = "review:0fca6e1982b0a016e01c"
CONTEXTUAL_EVIDENCE = {
    "review:4a6eb658e301a6dd56b4",
    "review:59da9c4c8dc5bfda4c78",
    "review:0e8aab707ed89e4b0169",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--limit", type=int, default=20)
    return parser


def main() -> int:
    args = _parser().parse_args()
    os.environ["BGE_M3_MODEL_REVISION"] = args.model_revision
    encoder = production.PinnedBgeM3Encoder(args.model_revision)
    release = SemanticRelease(args.release_dir, encoder)
    queries = [
        QueryCell(
            query_id="support-en",
            constraint_id="child-care",
            role="support",
            language="en",
            text="no overprescribing, clear explanations, kind pediatric care",
        ),
        QueryCell(
            query_id="support-ko",
            constraint_id="child-care",
            role="support",
            language="ko",
            text="과잉 처방을 하지 않고 설명을 잘하며 아이에게 친절한 진료",
        ),
        QueryCell(
            query_id="risk-en",
            constraint_id="nurse-manner",
            role="risk",
            language="en",
            text="unfriendly or blunt nurses",
        ),
        QueryCell(
            query_id="risk-ko",
            constraint_id="nurse-manner",
            role="risk",
            language="ko",
            text="간호사가 불친절하거나 무뚝뚝함",
        ),
    ]
    results = release.retrieve([TARGET_FACILITY], queries, args.limit)
    target_rows = np.flatnonzero(release.evidence_ids == DECISIVE_EVIDENCE)
    ownership_ok = (
        len(target_rows) == 1
        and str(release.facility_ids[int(target_rows[0])]) == TARGET_FACILITY
    )
    by_query: dict[str, dict[str, object]] = {}
    for query in queries:
        candidates = [item for item in results if item["query_id"] == query.query_id]
        decisive = [
            item for item in candidates if item["evidence_id"] == DECISIVE_EVIDENCE
        ]
        returned_ids = {str(item["evidence_id"]) for item in candidates}
        by_query[query.query_id] = {
            "decisive_found": bool(decisive),
            "decisive_hits": decisive,
            "contextual_found": sorted(CONTEXTUAL_EVIDENCE & returned_ids),
            "returned": len(candidates),
        }

    passed = ownership_ok and all(
        bool(result["decisive_found"]) for result in by_query.values()
    )
    report = {
        "passed": passed,
        "release_id": release.manifest["release_id"],
        "review_count": release.manifest["review_count"],
        "facility_count": len(release.ranges),
        "target_facility": TARGET_FACILITY,
        "decisive_evidence": DECISIVE_EVIDENCE,
        "ownership_ok": ownership_ok,
        "queries": by_query,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
