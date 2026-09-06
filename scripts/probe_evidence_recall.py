#!/usr/bin/env python3
"""Gate paid evaluations on decisive-review admission and attachment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

from search.contracts import (  # noqa: E402
    AreaRule,
    EligibleScope,
    EvidenceRequirement,
    HardEligibility,
    RuleProvenance,
    SearchRules,
)
from search.evidence_retrieval import (  # noqa: E402
    ConstraintEvidenceRetriever,
    EvidenceRecallPolicy,
)
from search.indexes import IndexRepository  # noqa: E402
from search.scope import ScopeSelection  # noqa: E402


@dataclass(frozen=True)
class Target:
    name: str
    facility_id: str
    decisive_evidence_id: str
    requirements: tuple[EvidenceRequirement, ...]


VERBATIM = frozenset({"verbatim_review"})
TARGETS = (
    Target(
        name="mapo_dermatology",
        facility_id="19521641",
        decisive_evidence_id="review:325180cb5153b903bfa2",
        requirements=(
            EvidenceRequirement(
                "cheilitis",
                ("cheilitis care", "lip inflammation"),
                ("구순염", "입술염"),
                "semantic",
                VERBATIM,
                True,
                "disease",
            ),
            EvidenceRequirement(
                "detailed_explanation",
                ("detailed explanation", "explains well"),
                ("자세한 설명", "설명 잘"),
                "semantic",
                VERBATIM,
                True,
                "support",
            ),
            EvidenceRequirement(
                "fast_treatment",
                ("fast treatment", "quick treatment"),
                ("빠른 진료", "신속한 진료"),
                "semantic",
                VERBATIM,
                True,
                "support",
            ),
        ),
    ),
    Target(
        name="yongsan_pediatrics",
        facility_id="35475921",
        decisive_evidence_id="review:0fca6e1982b0a016e01c",
        requirements=(
            EvidenceRequirement(
                "no_overprescribing",
                ("no overprescribing", "does not overprescribe"),
                ("과잉 처방 안", "과잉처방 안"),
                "semantic",
                VERBATIM,
                True,
                "support",
            ),
            EvidenceRequirement(
                "clear_explanations",
                ("clear explanations", "explains well"),
                ("자세한 설명", "설명 잘"),
                "semantic",
                VERBATIM,
                True,
                "support",
            ),
            EvidenceRequirement(
                "kind_children",
                ("kind with children", "gentle with children"),
                ("아이에게 친절", "아이한테 친절"),
                "semantic",
                VERBATIM,
                True,
                "support",
            ),
            EvidenceRequirement(
                "unfriendly_nurses",
                ("unfriendly nurses", "rude nurses"),
                ("간호사 불친절", "간호사 무뚝뚝"),
                "semantic",
                VERBATIM,
                True,
                "risk",
            ),
        ),
    ),
)


def scope_for(
    index_version: str,
    rules_hash: str,
    facility_ids: tuple[str, ...],
) -> ScopeSelection:
    payload = json.dumps(
        {
            "index_version": index_version,
            "rules_hash": rules_hash,
            "facility_ids": sorted(facility_ids),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = sha256(payload.encode("utf-8")).hexdigest()
    return ScopeSelection(
        descriptor=EligibleScope(
            index_version=index_version,
            rules_hash=rules_hash,
            facility_bitmap_ref=f"memory://{digest}",
            facility_count=len(facility_ids),
            scope_digest=digest,
        ),
        facility_ids=facility_ids,
        distance_km_by_facility={},
    )


def rules_for(target: Target, language: str) -> SearchRules:
    provenance = RuleProvenance("user_explicit", None, "component-probe")
    fingerprint = sha256(
        f"{target.name}:{language}".encode("utf-8")
    ).hexdigest()
    return SearchRules(
        schema_version="1",
        original_query=f"{target.name} decisive review probe",
        language=language,
        hard=HardEligibility(
            specialty_ids=frozenset(),
            geography=AreaRule("seoul", "city", "Seoul", provenance),
            prohibited_facility_ids=frozenset(),
            prohibited_taxonomy_ids=frozenset(),
            required_attributes=(),
        ),
        soft=(),
        evidence=target.requirements,
        provenance={"hard.geography": provenance},
        rules_hash=fingerprint,
    )


def run_target(active: object, target: Target) -> dict[str, object]:
    retriever = ConstraintEvidenceRetriever(
        policy=EvidenceRecallPolicy(),
    )
    runs = {}
    requires_risk = any(
        requirement.support_required and requirement.evidence_role == "risk"
        for requirement in target.requirements
    )
    shortlist = (
        target.facility_id,
        *tuple(
            facility_id
            for facility_id in active.facility_ids
            if facility_id != target.facility_id
        )[:19],
    )
    for language in ("en", "ko"):
        rules = rules_for(target, language)
        scope = scope_for(active.version, rules.rules_hash, shortlist)
        with active.within(scope) as scoped:
            result = retriever.collect(
                scoped_index=scoped,
                rules=rules,
                shortlisted_facility_ids=shortlist,
                displayed_facility_ids=(target.facility_id,),
            )
        admitted_ids = {
            event.evidence_id
            for event in result.admissions
            if event.admitted_to_gpu
        }
        presented_ids = {
            item.hit.evidence_id
            for item in result.by_facility[target.facility_id].presented
        }
        groups = result.by_facility[target.facility_id]
        runs[language] = {
            "finish_status": result.finish_status,
            "retry_ran": result.retry_ran,
            "shortlist_count": len(shortlist),
            "warning_count": len(groups.warnings),
            "candidate_count": len(admitted_ids),
            "decisive_recall_at_256": target.decisive_evidence_id in admitted_ids,
            "decisive_attachment_at_3": target.decisive_evidence_id in presented_ids,
            "ownership_ok": all(
                item.hit.facility_id == target.facility_id
                for item in groups.presented
            ),
            "risk_polarity_ok": (
                not requires_risk
                or bool(groups.warnings)
                and all("risk" in item.roles for item in groups.warnings)
            ),
            "duplicate_gpu_ids": (
                len(admitted_ids)
                != len({
                    item.hit.evidence_id for item in result.evidence
                })
            ),
            "admitted_ids": sorted(admitted_ids),
            "presented_ids": sorted(presented_ids),
        }

    english = set(runs["en"]["admitted_ids"])
    korean = set(runs["ko"]["admitted_ids"])
    union = english | korean
    overlap = len(english & korean) / len(union) if union else 1.0
    passed = all(
        run["decisive_recall_at_256"]
        and run["decisive_attachment_at_3"]
        and run["ownership_ok"]
        and run["risk_polarity_ok"]
        and not run["duplicate_gpu_ids"]
        for run in runs.values()
    ) and overlap >= 0.80
    return {
        "target": target.name,
        "facility_id": target.facility_id,
        "decisive_evidence_id": target.decisive_evidence_id,
        "english_korean_candidate_overlap": round(overlap, 6),
        "runs": runs,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index-root",
        type=Path,
        default=BACKEND / "search_indexes",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with IndexRepository(args.index_root).open_active() as active:
        targets = [run_target(active, target) for target in TARGETS]
        report = {
            "schema_version": 1,
            "index_version": active.version,
            "thresholds": {
                "decisive_review_recall_at_256": 1.0,
                "decisive_review_attachment_at_3": 1.0,
                "ownership": 1.0,
                "risk_polarity": 1.0,
                "english_korean_candidate_overlap": 0.80,
                "duplicate_gpu_ids": 0,
            },
            "targets": targets,
            "passed": all(target["passed"] for target in targets),
        }

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
