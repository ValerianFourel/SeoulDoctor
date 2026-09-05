#!/usr/bin/env python3
"""Exercise Phase 4 against the active immutable index without network calls."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from search.contracts import (  # noqa: E402
    DistanceRule,
    GeoPoint,
    HardEligibility,
    RuleProvenance,
    SearchRules,
)
from search.indexes import IndexRepository  # noqa: E402
from search.live_retrieval import (  # noqa: E402
    CandidateRetrievalAdapter,
    RetrievalQuery,
)
from search.scope import ScopeBuilder  # noqa: E402


class LocalEmbeddingProbe:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.calls = 0

    def embedding_function(self, texts: list[str]) -> list[list[float]]:
        if len(texts) != 1:
            raise AssertionError("Phase 4 must embed one query at a time")
        self.calls += 1
        return [self.vector]

    def apply_combined_ranking(self, **_: object) -> pd.DataFrame:
        raise AssertionError("the real indexed smoke test must not use legacy fallback")


def main() -> None:
    index_root = Path(os.getenv(
        "SEARCH_INDEX_ROOT", BACKEND_ROOT / "search_indexes"
    ))
    facilities_path = Path(os.getenv(
        "FACILITIES_CACHE_PATH", BACKEND_ROOT / "local_facilities_cache.parquet"
    ))
    facilities = pd.read_parquet(facilities_path)
    release = IndexRepository(index_root).open_active()
    try:
        catalog = facilities[
            facilities["place_id"].astype(str).isin(release.facility_ids)
        ].copy()
        coordinates = catalog.dropna(subset=["lat", "lon"])
        if coordinates.empty:
            raise AssertionError("facility snapshot has no usable coordinates")
        anchor_row = coordinates.iloc[0]
        provenance = RuleProvenance("default_5km", None, "phase4-smoke")
        query_seed = "Phase 4 bilingual 5 km smoke"
        rules_hash = sha256(query_seed.encode("utf-8")).hexdigest()
        rules = SearchRules(
            schema_version="1",
            original_query=query_seed,
            language="mixed",
            hard=HardEligibility(
                specialty_ids=frozenset(),
                geography=DistanceRule(
                    GeoPoint(float(anchor_row["lat"]), float(anchor_row["lon"])),
                    5.0,
                    provenance,
                ),
                prohibited_facility_ids=frozenset(),
                prohibited_taxonomy_ids=frozenset(),
                required_attributes=(),
            ),
            soft=(),
            evidence=(),
            provenance={"geography": provenance},
            rules_hash=rules_hash,
        )
        scope = ScopeBuilder().build(
            catalog,
            rules,
            index_version=release.version,
        )
        eligible = scope.restrict_dataframe(catalog)
        if not scope.facility_ids:
            raise AssertionError("5 km smoke scope is unexpectedly empty")
        if max(scope.distance_km_by_facility.values(), default=0.0) > 5.000001:
            raise AssertionError("5 km scope contains an out-of-radius facility")

        target_id = scope.facility_ids[0]
        target_ordinal = release.facility_ids.index(target_id)
        vector = release._release.vectors[target_ordinal].astype(float).tolist()
        summaries: list[dict[str, object]] = []
        for text, language in (
            ("friendly medical clinic", "English"),
            ("친절한 병원", "Korean"),
        ):
            pipeline = LocalEmbeddingProbe(vector)
            outcome = CandidateRetrievalAdapter(
                active_index=release,
                legacy_pipeline=pipeline,
            ).rank(
                scope=scope,
                eligible=eligible,
                rules=rules,
                query=RetrievalQuery(
                    text=text,
                    max_distance_km=5.0,
                    search_mode="distance",
                    specialty_confidence=1.0,
                    target_language=language,
                ),
            )
            if outcome.telemetry.status != "indexed":
                raise AssertionError(outcome.telemetry.as_private_dict())
            if pipeline.calls != 1:
                raise AssertionError("Phase 4 made more than one embedding call")
            scope.assert_contains_only(outcome.dataframe)
            summaries.append({
                "language": language,
                "top_facility_id": str(outcome.dataframe.iloc[0]["place_id"]),
                "scope_count": len(scope.facility_ids),
                "retry_ran": outcome.telemetry.retry_ran,
                "channel_hit_counts": dict(outcome.telemetry.channel_hit_counts),
            })
        print(json.dumps({
            "index_version": release.version,
            "radius_km": 5.0,
            "queries": summaries,
        }, ensure_ascii=False, sort_keys=True))
    finally:
        release.close()


if __name__ == "__main__":
    main()
