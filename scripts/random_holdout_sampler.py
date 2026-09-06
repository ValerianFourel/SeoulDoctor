#!/usr/bin/env python3
"""Build a reproducible, memory-bounded bilingual comment holdout casebook."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gzip
import hashlib
import heapq
import json
from pathlib import Path
import re
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from typing import Iterable, Iterator, TextIO


SAMPLER_VERSION = "1.0.0"
REQUIRED_FIELDS = (
    "place_id",
    "evidence_id",
    "facility_name",
    "specialty",
    "location",
    "comment",
)
THEMES = (
    ("clear_explanations", ("explain", "explanation", "설명"), "clear explanations", "자세한 설명"),
    ("friendly", ("friendly", "kind", "친절"), "kind communication", "친절한 소통"),
    ("careful", ("careful", "thorough", "꼼꼼", "세심"), "careful treatment", "꼼꼼한 진료"),
    ("short_wait", ("short wait", "quickly", "빠르게", "대기 시간이 짧"), "a manageable wait", "부담스럽지 않은 대기 시간"),
    ("clean", ("clean", "hygien", "깨끗", "청결"), "a clean environment", "청결한 환경"),
    ("foreigner_friendly", ("english", "foreigner", "외국인", "영어"), "communication with international patients", "외국인 환자와의 소통"),
)
EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?82[- ]?)?0?1[016789][- ]?\d{3,4}[- ]?\d{4}(?!\d)")
EMERGENCY_TERMS = ("suicide", "unconscious", "chest pain", "응급", "자살", "의식불명", "가슴 통증")
INSTRUCTION_TERMS = ("ignore previous", "system prompt", "follow these instructions", "이전 지시", "시스템 프롬프트")
SPECIALTY_EN = {
    "가정의학과": "family medicine",
    "내과": "internal medicine",
    "마취통증의학과": "pain medicine",
    "비뇨의학과": "urology",
    "산부인과": "obstetrics and gynecology",
    "소아청소년과": "pediatrics",
    "안과": "ophthalmology",
    "이비인후과": "ear, nose, and throat care",
    "재활의학과": "rehabilitation medicine",
    "정신건강의학과": "mental health care",
    "정형외과": "orthopedics",
    "치과": "dentistry",
    "피부과": "dermatology",
    "한의원": "Korean medicine",
}


@dataclass(frozen=True)
class Candidate:
    place_id: str
    evidence_id: str
    facility_name: str
    specialty: str
    location: str
    comment: str
    themes: tuple[str, ...]
    facility_priority: int
    evidence_priority: int


def normalize(value: object) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def supported_themes(comment: str) -> tuple[str, ...]:
    lowered = comment.casefold()
    return tuple(name for name, terms, _, _ in THEMES if any(term in lowered for term in terms))


def validate_record(record: dict[str, object]) -> tuple[dict[str, str] | None, str | None]:
    normalized = {field: normalize(record.get(field)) for field in REQUIRED_FIELDS}
    missing = [field for field, value in normalized.items() if not value]
    if missing:
        return None, f"missing:{','.join(missing)}"
    if len(normalized["place_id"]) > 256 or len(normalized["evidence_id"]) > 256:
        return None, "identifier_too_long"
    if len(normalized["specialty"]) > 200 or len(normalized["location"]) > 200:
        return None, "facet_too_long"
    public_facets = f'{normalized["specialty"]} {normalized["location"]}'.casefold()
    if any(term in public_facets for term in INSTRUCTION_TERMS):
        return None, "unsafe_public_facet"
    comment = normalized["comment"]
    if not 20 <= len(comment) <= 1500:
        return None, "comment_length"
    if EMAIL_RE.search(comment) or URL_RE.search(comment) or PHONE_RE.search(comment):
        return None, "sensitive_contact"
    lowered = comment.casefold()
    if any(term in lowered for term in EMERGENCY_TERMS):
        return None, "emergency"
    themes = supported_themes(comment)
    if not themes:
        return None, "no_supported_theme"
    normalized["themes"] = list(themes)  # type: ignore[assignment]
    return normalized, None


def priority(seed: int, namespace: str, value: str) -> int:
    payload = f"{SAMPLER_VERSION}\0{seed}\0{namespace}\0{value}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def input_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    suffixes = path.suffixes
    data_suffix = suffixes[-2] if suffixes and suffixes[-1] == ".gz" and len(suffixes) > 1 else path.suffix
    if data_suffix in (".jsonl", ".ndjson"):
        return "jsonl"
    if data_suffix == ".csv":
        return "csv"
    raise ValueError("cannot infer input format; use --format jsonl or --format csv")


def iter_records(path: Path, fmt: str) -> Iterator[dict[str, object]]:
    with open_text(path) as source:
        if fmt == "csv":
            yield from csv.DictReader(source)
            return
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            yield value


def iter_release_parquet(facilities_path: Path, reviews_path: Path, batch_size: int = 8192) -> Iterator[dict[str, object]]:
    """Join the small facility table to streamed review batches.

    The pinned release currently has thousands of facilities but millions of
    reviews. Keeping only the six public facility facets per place is bounded by
    facility count; complete review rows are streamed one batch at a time.
    """
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover - depends on deployment extras
        raise ValueError("Parquet input requires pyarrow") from error

    facility_columns = ("place_id", "name", "category", "address", "file_district", "file_dong")
    facility_table = parquet.read_table(facilities_path, columns=list(facility_columns))
    facilities: dict[str, dict[str, str]] = {}
    for row in facility_table.to_pylist():
        place_id = normalize(row.get("place_id"))
        if not place_id:
            continue
        location_parts = [normalize(row.get(field)) for field in ("file_dong", "file_district", "address")]
        location = next((part for part in location_parts if part), "")
        facilities[place_id] = {
            "facility_name": normalize(row.get("name")),
            "specialty": normalize(row.get("category")),
            "location": location,
        }

    review_file = parquet.ParquetFile(reviews_path)
    review_columns = ("place_id", "review_index", "review_text")
    for batch in review_file.iter_batches(batch_size=batch_size, columns=list(review_columns)):
        for review in batch.to_pylist():
            place_id = normalize(review.get("place_id"))
            facility = facilities.get(place_id)
            if facility is None:
                continue
            review_index = review.get("review_index")
            evidence_id = f"{place_id}:review:{review_index}"
            yield {
                "place_id": place_id,
                "evidence_id": evidence_id,
                "facility_name": facility["facility_name"],
                "specialty": facility["specialty"],
                "location": facility["location"],
                "comment": review.get("review_text"),
            }


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def select_candidates(records: Iterable[dict[str, object]], seed: int, sample_size: int) -> tuple[list[Candidate], dict[str, int]]:
    if sample_size < 1:
        raise ValueError("sample size must be positive")
    selected: dict[str, Candidate] = {}
    heap: list[tuple[int, str]] = []  # negative facility priority, place_id
    counts = {"records_seen": 0, "eligible_records": 0, "invalid_records": 0}

    for raw in records:
        counts["records_seen"] += 1
        record, _ = validate_record(raw)
        if record is None:
            counts["invalid_records"] += 1
            continue
        counts["eligible_records"] += 1
        place_id = str(record["place_id"])
        facility_rank = priority(seed, "facility", place_id)
        evidence_rank = priority(seed, "evidence", str(record["evidence_id"]))
        candidate = Candidate(
            place_id=place_id,
            evidence_id=str(record["evidence_id"]),
            facility_name=str(record["facility_name"]),
            specialty=str(record["specialty"]),
            location=str(record["location"]),
            comment=str(record["comment"]),
            themes=tuple(record["themes"]),  # type: ignore[arg-type]
            facility_priority=facility_rank,
            evidence_priority=evidence_rank,
        )

        current = selected.get(place_id)
        if current is not None:
            if evidence_rank < current.evidence_priority:
                selected[place_id] = candidate
            continue
        if len(selected) < sample_size:
            selected[place_id] = candidate
            heapq.heappush(heap, (-facility_rank, place_id))
            continue
        worst_priority = -heap[0][0]
        if facility_rank >= worst_priority:
            continue
        _, evicted = heapq.heapreplace(heap, (-facility_rank, place_id))
        del selected[evicted]
        selected[place_id] = candidate

    if len(selected) < sample_size:
        raise ValueError(f"requested {sample_size} facilities but found {len(selected)} eligible distinct facilities")
    return sorted(selected.values(), key=lambda item: item.facility_priority), counts


def theme_labels(names: tuple[str, ...], language: str) -> list[str]:
    label_index = 2 if language == "en" else 3
    by_name = {theme[0]: theme[label_index] for theme in THEMES}
    return [by_name[name] for name in names]


def public_card(candidate: Candidate, pair_index: int, seed: int, language: str) -> dict[str, object]:
    labels = theme_labels(candidate.themes, language)
    preferences = ", ".join(labels)
    scenario_id = f"random-{seed:016x}-{pair_index:02d}-{language}"
    if language == "en":
        specialty = SPECIALTY_EN.get(candidate.specialty, candidate.specialty)
        prompt = (
            f"You need {specialty} care near {candidate.location}. You especially value patient "
            f"experiences involving {preferences}. Ask SeoulDoc naturally for a few suitable options "
            "and for review evidence explaining why they fit. Do not name or hint at a particular facility."
        )
    else:
        prompt = (
            f"{candidate.location} 근처에서 {candidate.specialty} 진료를 받고 싶습니다. "
            f"다음과 같은 환자 경험을 특히 중요하게 생각합니다: {preferences}. 특정 의료기관을 지목하거나 "
            "암시하지 말고, 조건에 맞는 곳과 그 이유를 보여 주는 근거를 자연스럽게 요청하세요."
        )
    return {"scenario_id": scenario_id, "pair_id": f"random-{seed:016x}-{pair_index:02d}", "language": language, "patient_card": prompt}


def git_value(*args: str) -> str:
    result = subprocess.run(
        ("git", *args), capture_output=True, check=False, text=True, timeout=5
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def write_outputs(
    candidates: list[Candidate],
    output_dir: Path,
    seed: int,
    source_revision: str,
    source_hashes: dict[str, str],
    counts: dict[str, int],
    coordinator_model: str = "gpt-5.6-sol",
    application_model: str = "unknown",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    cards: list[dict[str, object]] = []
    oracle: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates, 1):
        cards.extend((public_card(candidate, index, seed, "en"), public_card(candidate, index, seed, "ko")))
        oracle.append(
            {
                "pair_id": f"random-{seed:016x}-{index:02d}",
                "place_id": candidate.place_id,
                "evidence_id": candidate.evidence_id,
                "expected_facility": candidate.facility_name,
                "specialty": candidate.specialty,
                "location": candidate.location,
                "themes": list(candidate.themes),
                "comment": candidate.comment,
                "comment_sha256": hashlib.sha256(candidate.comment.encode()).hexdigest(),
            }
        )
    casebook_path = output_dir / "public_casebook.json"
    oracle_path = output_dir / "private_oracle.json"
    casebook_path.write_text(json.dumps(cards, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    oracle_path.write_text(json.dumps(oracle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sampler_version": SAMPLER_VERSION,
        "seed": seed,
        "git_branch": git_value("branch", "--show-current"),
        "git_commit": git_value("rev-parse", "HEAD"),
        "casebook_commit": git_value("rev-parse", "HEAD"),
        "casebook_sha256": file_sha256(casebook_path),
        "model_labels": {
            "coordinator": coordinator_model,
            "patient_message_source": "isolated_agent_from_public_card",
            "application": application_model,
            "grader": None,
        },
        "requested_facilities": len(candidates),
        "scenario_count": len(cards),
        "scenario_ids": [card["scenario_id"] for card in cards],
        "source_revision": source_revision,
        "result_dataset_revision": None,
        "source_sha256": source_hashes,
        "counts": counts,
        "completed_stage": "casebook_frozen",
        "application_calls_completed": 0,
        "result": None,
        "artifact_hashes": {
            "public_casebook.json": file_sha256(casebook_path),
            "private_oracle.json": file_sha256(oracle_path),
        },
        "unresolved_issue": None,
        "exact_next_action": "Give each isolated patient agent only its public card and the verified application endpoint.",
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_seed(value: str) -> int:
    seed = int(value, 0)
    if not 0 <= seed < 2**64:
        raise argparse.ArgumentTypeError("seed must be an unsigned 64-bit integer")
    return seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source", type=Path, help="normalized JSONL or CSV records")
    source_group.add_argument("--reviews-parquet", type=Path, help="release Dataset reviews.parquet")
    parser.add_argument("--facilities-parquet", type=Path, help="release Dataset facilities.parquet; required with --reviews-parquet")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=6)
    parser.add_argument("--seed", type=parse_seed, help="unsigned 64-bit integer; generated when omitted")
    parser.add_argument("--format", choices=("auto", "jsonl", "csv"), default="auto")
    parser.add_argument("--coordinator-model", default="gpt-5.6-sol")
    parser.add_argument("--application-model", default="unknown")
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else secrets.randbits(64)
    try:
        if args.reviews_parquet:
            if args.facilities_parquet is None:
                raise ValueError("--facilities-parquet is required with --reviews-parquet")
            records = iter_release_parquet(args.facilities_parquet, args.reviews_parquet)
            source_hashes = {
                "facilities.parquet": file_sha256(args.facilities_parquet),
                "reviews.parquet": file_sha256(args.reviews_parquet),
            }
        else:
            if args.facilities_parquet is not None:
                raise ValueError("--facilities-parquet may only be used with --reviews-parquet")
            fmt = input_format(args.source, args.format)
            records = iter_records(args.source, fmt)
            source_hashes = {args.source.name: file_sha256(args.source)}
        candidates, counts = select_candidates(records, seed, args.sample_size)
        write_outputs(
            candidates,
            args.output_dir,
            seed,
            args.source_revision,
            source_hashes,
            counts,
            coordinator_model=args.coordinator_model,
            application_model=args.application_model,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps({"seed": seed, "facilities": len(candidates), "scenarios": len(candidates) * 2, "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
