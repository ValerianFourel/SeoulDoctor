"""Summarize saved diagnostics without model calls or changing scenario targets."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]
from patient_journey import atomic_write
from scripts.audit_visible_evidence import normalized


def target_checks(body, target):
    place = str(target["place_id"])
    text = normalized(target["comment"])
    source_id = target["source_evidence_id"]
    match = re.search(r":review:(\d+)$", source_id)
    evidence_id = None
    if match:
        identity = f"{place}|{int(match[1])}|{target['comment'].strip()}"
        evidence_id = "review:" + sha256(identity.encode()).hexdigest()[:20]
    cards = body.get("results") or []
    found = [(card, evidence) for card in cards for evidence in card.get("retrieval_evidence", [])
             if normalized(evidence.get("text", "")) == text]
    admissions = body.get("state", {}).get("last_retrieval_metadata", {}).get("evidence_admissions", [])
    return {
        "facility_presented": any(str(card.get("place_id")) == place for card in cards),
        "exact_comment_attached": bool(found),
        "correct_ownership": bool(found) and all(str(c.get("place_id")) == place
            and str(e.get("place_id")) == place for c, e in found),
        "exact_original_visible": bool(text) and text in normalized(body.get("response", "")),
        "mapped_target_admitted": bool(evidence_id) and any(e.get("evidence_id") == evidence_id for e in admissions),
        "mapping_available": evidence_id is not None,
        "faithful_translation": "not_assessed",
        "recommendation_use": "not_objectively_established",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--oracle", type=Path, required=True)
    args = parser.parse_args()
    oracle = json.loads(args.oracle.read_text())
    statuses, failures, methods, judges = Counter(), Counter(), Counter(), Counter()
    latencies, exact, total_turns = [], {}, 0
    for path in sorted(args.run_dir.glob("*/result.json")):
        record = json.loads(path.read_text())
        statuses[record["status"]] += 1
        if record["status"] == "running":
            continue
        if record.get("http_error"):
            failures[str(record["http_error"])] += 1
        for timing in record.get("timings", []):
            if timing["method"].upper() == "POST":
                latencies.append(timing["request_seconds"])
        conversation = json.loads((path.parent / "conversation.json").read_text())
        for turn in conversation["turns"]:
            total_turns += 1
            metadata = turn["body"].get("state", {}).get("last_retrieval_metadata") or {}
            methods["semantic:" + str(metadata.get("semantic_status", "not_run"))] += 1
            methods["reranker:" + str(metadata.get("reranker_reason", "not_run"))] += 1
        sid = record["scenario_id"]
        if sid in oracle:
            exact[sid] = [target_checks(t["body"], oracle[sid]) for t in conversation["turns"]]
        judge_file = path.parent / "judges.json"
        if judge_file.exists():
            for review in json.loads(judge_file.read_text())["reviews"]:
                judges["valid_citations" if review["citation_valid"] else "invalid_or_error"] += 1
    latencies.sort()
    ledger_path = args.run_dir / "model_usage.json"
    ledger = json.loads(ledger_path.read_text())["calls"] if ledger_path.exists() else []
    costs = [(call.get("usage") or {}).get("cost") for call in ledger]
    report = {"release_passed": False, "statuses": dict(statuses), "successful_app_turns": total_turns,
              "http_errors": dict(failures), "service_statuses": dict(methods), "judges": dict(judges),
              "request_latency_seconds": {"count": len(latencies),
                "median": latencies[len(latencies)//2] if latencies else None,
                "p95": latencies[min(len(latencies)-1, int(len(latencies)*0.95))] if latencies else None},
              "model_calls": len(ledger), "reported_model_cost_usd": sum(c for c in costs if isinstance(c, (int,float))),
              "calls_missing_reported_cost": sum(c is None for c in costs), "app_model_cost_excluded": True,
              "holdout_exact_comment_checks": exact,
              "limitations": ["Citation validation is not semantic judgment validation",
                 "Recorded failed app requests are included in latency", "No GPU-backed release verification"]}
    atomic_write(args.run_dir / "objective_summary.json", report)
    print(json.dumps({k:v for k,v in report.items() if k != "holdout_exact_comment_checks"}))


if __name__ == "__main__":
    main()
