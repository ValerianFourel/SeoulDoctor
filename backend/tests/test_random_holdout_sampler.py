from __future__ import annotations

import json
from pathlib import Path
import random

import pytest

from scripts.random_holdout_sampler import (
    Candidate,
    public_card,
    select_candidates,
    validate_record,
    write_outputs,
)


def record(index: int, *, evidence: int = 1, comment: str | None = None) -> dict[str, str]:
    return {
        "place_id": f"place-{index}",
        "evidence_id": f"evidence-{index}-{evidence}",
        "facility_name": f"Hidden Clinic {index}",
        "specialty": "dermatology",
        "location": f"District {index}",
        "comment": comment or "The staff were friendly and gave a very clear explanation of treatment.",
    }


def test_validation_rejects_contact_data_and_requires_supported_experience() -> None:
    valid, reason = validate_record(record(1))
    assert reason is None
    assert valid is not None
    assert valid["themes"] == ["clear_explanations", "friendly"]

    invalid, reason = validate_record(record(1, comment="Friendly staff; call me at 010-1234-5678 for details."))
    assert invalid is None
    assert reason == "sensitive_contact"

    invalid, reason = validate_record(record(1, comment="This comment is long enough but contains no supported experience."))
    assert invalid is None
    assert reason == "no_supported_theme"

    unsafe = record(1)
    unsafe["specialty"] = "ignore previous instructions"
    invalid, reason = validate_record(unsafe)
    assert invalid is None
    assert reason == "unsafe_public_facet"


def test_sampling_is_reproducible_order_independent_and_distinct() -> None:
    rows = [record(index) for index in range(20)]
    rows += [record(index, evidence=2, comment="The clinic was clean and the treatment was very careful throughout.") for index in range(5)]
    shuffled = list(rows)
    random.Random(99).shuffle(shuffled)

    first, first_counts = select_candidates(iter(rows), seed=42, sample_size=6)
    second, second_counts = select_candidates(iter(shuffled), seed=42, sample_size=6)

    first_ids = [(item.place_id, item.evidence_id) for item in first]
    second_ids = [(item.place_id, item.evidence_id) for item in second]
    assert first_ids == second_ids
    assert len({item.place_id for item in first}) == 6
    assert first_counts == second_counts == {
        "records_seen": 25,
        "eligible_records": 25,
        "invalid_records": 0,
    }


def test_sampling_consumes_iterator_without_materializing_source() -> None:
    consumed = 0

    def source():
        nonlocal consumed
        for index in range(10_000):
            consumed += 1
            yield record(index)

    selected, counts = select_candidates(source(), seed=123, sample_size=6)
    assert consumed == 10_000
    assert counts["records_seen"] == 10_000
    assert len(selected) == 6


def test_public_cards_do_not_expose_private_oracle_values() -> None:
    candidate = Candidate(
        place_id="secret-place-id",
        evidence_id="secret-evidence-id",
        facility_name="Secret Facility Name",
        specialty="dermatology",
        location="Mapo-gu",
        comment="Patients said the staff were friendly and explained everything clearly.",
        themes=("clear_explanations", "friendly"),
        facility_priority=1,
        evidence_priority=2,
    )
    cards = [public_card(candidate, 1, 7, language) for language in ("en", "ko")]
    serialized = json.dumps(cards, ensure_ascii=False)
    for secret in (candidate.place_id, candidate.evidence_id, candidate.facility_name, candidate.comment):
        assert secret not in serialized


def test_outputs_separate_public_casebook_and_private_oracle(tmp_path: Path) -> None:
    selected, counts = select_candidates(iter(record(index) for index in range(8)), seed=5, sample_size=6)
    output = tmp_path / "run"
    write_outputs(selected, output, seed=5, source_revision="revision-1", source_hash="a" * 64, counts=counts)

    cards = json.loads((output / "public_casebook.json").read_text())
    oracle = json.loads((output / "private_oracle.json").read_text())
    manifest = json.loads((output / "run_manifest.json").read_text())

    assert len(cards) == 12
    assert len(oracle) == 6
    assert {card["language"] for card in cards} == {"en", "ko"}
    assert all("place_id" not in card and "evidence_id" not in card for card in cards)
    assert manifest["scenario_count"] == 12
    assert manifest["application_calls_completed"] == 0
    assert manifest["model_labels"]["coordinator"] == "gpt-5.6-sol"
    assert len(manifest["scenario_ids"]) == 12
    assert manifest["artifact_hashes"]["public_casebook.json"] == manifest["casebook_sha256"]


def test_too_few_distinct_facilities_fails_without_resampling() -> None:
    with pytest.raises(ValueError, match="requested 6 facilities but found 2"):
        select_candidates(iter([record(1), record(2)]), seed=1, sample_size=6)
