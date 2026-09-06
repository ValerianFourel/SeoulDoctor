"""Offline reverse-target probe for a sealed SeoulDoc scenario."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import pandas as pd


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from distance import haversine  # noqa: E402
from patient_journey import atomic_write  # noqa: E402
from query_facets import expand_multilingual_retrieval_terms  # noqa: E402
from raw_review_store import RawReviewStore  # noqa: E402


DEFAULT_CASEBOOK = Path(__file__).resolve().parent / "grounded_bilingual_scenarios.json"
DEFAULT_FACILITIES = BACKEND_DIR / "local_facilities_cache.parquet"
DEFAULT_REVIEWS = BACKEND_DIR / "local_reviews_cache.parquet"


def _scenario(casebook: Mapping[str, Any], scenario_id: str) -> Mapping[str, Any]:
    scenarios = casebook.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("casebook scenarios must be a list")
    for scenario in scenarios:
        if isinstance(scenario, Mapping) and scenario.get("id") == scenario_id:
            return scenario
    raise ValueError(f"unknown scenario ID: {scenario_id}")


def _decisive_ids(target: Mapping[str, Any]) -> set[str]:
    requirements = target.get("evidence_requirements")
    requirements = requirements if isinstance(requirements, Mapping) else {}
    decisive = requirements.get("decisive")
    decisive = decisive if isinstance(decisive, list) else []
    return {
        str(item["evidence_id"])
        for item in decisive
        if isinstance(item, Mapping) and item.get("evidence_id")
    }


def run_probe(
    casebook_path: Path,
    scenario_id: str,
    facilities_path: Path,
    reviews_path: Path,
) -> dict[str, Any]:
    """Check whether raw bilingual evidence can retrieve a sealed target."""
    casebook = json.loads(casebook_path.read_text(encoding="utf-8"))
    scenario = _scenario(casebook, scenario_id)
    oracle = scenario.get("oracle")
    if not isinstance(oracle, Mapping):
        raise ValueError("scenario lacks an oracle")
    origin = oracle.get("origin")
    target = oracle.get("reverse_target")
    if not isinstance(origin, Mapping) or not isinstance(target, Mapping):
        raise ValueError("scenario lacks origin or reverse target data")

    maximum_distance = float(oracle["maximum_distance_km"])
    specialty = str(oracle["expected_specialty"])
    target_id = str(target["place_id"])
    facilities = pd.read_parquet(facilities_path)
    required_columns = {"place_id", "category", "lat", "lon"}
    missing_columns = required_columns - set(facilities.columns)
    if missing_columns:
        raise ValueError(
            "facility snapshot lacks columns: " + ", ".join(sorted(missing_columns))
        )
    facilities = facilities.copy()
    facilities["distance_km"] = haversine(
        float(origin["latitude"]),
        float(origin["longitude"]),
        facilities["lat"],
        facilities["lon"],
    )
    scope = facilities[
        facilities["category"].astype(str).str.contains(
            specialty, case=False, na=False, regex=False
        )
        & (facilities["distance_km"] <= maximum_distance)
    ]
    scope_ids = scope["place_id"].astype(str).tolist()
    positive_terms = oracle.get("positive_terms")
    negative_terms = oracle.get("negative_terms")
    positive_terms = (
        [str(term) for term in positive_terms if str(term).strip()]
        if isinstance(positive_terms, list)
        else []
    )
    negative_terms = (
        [str(term) for term in negative_terms if str(term).strip()]
        if isinstance(negative_terms, list)
        else []
    )
    terms = expand_multilingual_retrieval_terms([
        *positive_terms,
        *negative_terms,
    ])

    store = RawReviewStore(str(reviews_path))
    try:
        records = store.search_comments(
            scope_ids,
            query_terms=terms,
            limit=60,
            per_facility_limit=5,
        )
    finally:
        store.close()

    ranked_facility_ids = list(dict.fromkeys(record["place_id"] for record in records))
    target_rank = (
        ranked_facility_ids.index(target_id) + 1
        if target_id in ranked_facility_ids
        else None
    )
    target_records = [record for record in records if record["place_id"] == target_id]
    found_evidence_ids = {record["evidence_id"] for record in target_records}
    required_evidence_ids = _decisive_ids(target)
    target_row = scope[scope["place_id"].astype(str) == target_id]
    target_distance = (
        float(target_row.iloc[0]["distance_km"])
        if not target_row.empty
        else None
    )
    rank_limit = int(target.get("retrieval_rank_lte", 5))
    passed = bool(
        target_distance is not None
        and target_rank is not None
        and target_rank <= rank_limit
        and required_evidence_ids
        and required_evidence_ids.issubset(found_evidence_ids)
    )
    return {
        "schema_version": 1,
        "scenario_id": scenario_id,
        "selection_seed": target.get("seed"),
        "scope": {
            "specialty": specialty,
            "maximum_distance_km": maximum_distance,
            "facility_count": len(scope_ids),
        },
        "query_terms": terms,
        "retrieved_comment_count": len(records),
        "retrieved_facility_count": len(ranked_facility_ids),
        "target": {
            "place_id": target_id,
            "name": target.get("name"),
            "distance_km": target_distance,
            "retrieval_rank_1based": target_rank,
            "retrieval_rank_limit": rank_limit,
            "required_decisive_evidence_ids": sorted(required_evidence_ids),
            "found_evidence": [
                {
                    "evidence_id": record["evidence_id"],
                    "score": record["score"],
                    "matched_terms": record["matched_terms"],
                }
                for record in target_records
            ],
        },
        "passed": passed,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe a sealed target using local facilities and raw comments."
    )
    parser.add_argument("--casebook", type=Path, default=DEFAULT_CASEBOOK)
    parser.add_argument("--scenario", default="reverse-mapoderm-02-en")
    parser.add_argument("--facilities", type=Path, default=DEFAULT_FACILITIES)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = run_probe(
        args.casebook,
        args.scenario,
        args.facilities,
        args.reviews,
    )
    if args.output:
        atomic_write(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
