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
from evidence_response import finalize_evidence_response


def target_checks(body, target):
    place = str(target["place_id"])
    text = normalized(target["comment"])
    source_id = target["source_evidence_id"]
    match = re.search(r":review:(\d+)(?:\.0+)?$", source_id)
    evidence_id = target.get("evidence_id")
    if match and evidence_id is None:
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
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    oracle = json.loads(args.oracle.read_text())
    sealed_targets = {}
    if args.reviews:
        import pyarrow.dataset as ds
        cases = json.loads((ROOT / "backend/tests/grounded_bilingual_scenarios.json").read_text())["scenarios"]
        target_cases = [case for case in cases if case.get("oracle", {}).get("reverse_target")]
        places = list({str(case["oracle"]["reverse_target"]["place_id"]) for case in target_cases}
                      | {str(target["place_id"]) for target in oracle.values()})
        table = ds.dataset(args.reviews, format="parquet").to_table(
            columns=["place_id", "review_index", "review_text"], filter=ds.field("place_id").isin(places))
        source, source_rows = {}, {}
        for row in table.to_pylist():
            text = str(row["review_text"] or "").strip()
            identity = f"{row['place_id']}|{int(row['review_index'])}|{text}"
            eid = "review:" + sha256(identity.encode()).hexdigest()[:20]
            source[eid] = {"place_id": row["place_id"], "source_evidence_id": eid, "evidence_id": eid, "comment": text}
            source_rows[(str(row["place_id"]), int(row["review_index"]))] = source[eid]
        for target in oracle.values():
            match = re.search(r":review:(\d+)(?:\.0+)?$", target["source_evidence_id"])
            if match:
                original = source_rows[(str(target["place_id"]), int(match[1]))]
                if normalized(original["comment"]) != normalized(target["comment"]):
                    raise ValueError("Holdout source identity mismatch")
                target["evidence_id"] = original["evidence_id"]
        for case in target_cases:
            decisive = case["oracle"]["reverse_target"]["evidence_requirements"]["decisive"]
            sealed_targets[case["id"]] = [source[item["evidence_id"]] for item in decisive]
    statuses, failures, methods, judges, repairs = Counter(), Counter(), Counter(), Counter(), Counter()
    latencies, exact, sealed_exact, rerendered, total_turns = [], {}, {}, {}, 0
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
        if sid in sealed_targets:
            sealed_exact[sid] = [[target_checks(t["body"], target) for target in sealed_targets[sid]]
                                 for t in conversation["turns"]]
            if conversation["turns"]:
                body = dict(conversation["turns"][-1]["body"])
                state = body.get("state", {})
                body["response"], body["results"] = finalize_evidence_response(
                    body.get("response", ""), body.get("results", []),
                    state.get("last_retrieval_metadata") or {}, state.get("language_pref", "English"))
                rerendered[sid] = [target_checks(body, target) for target in sealed_targets[sid]]
        judge_file = path.parent / "judges.json"
        if judge_file.exists():
            for review in json.loads(judge_file.read_text())["reviews"]:
                judges["valid_citations" if review["citation_valid"] else "invalid_or_error"] += 1
    latencies.sort()
    ledger_path = args.run_dir / "model_usage.json"
    ledger = json.loads(ledger_path.read_text())["calls"] if ledger_path.exists() else []
    costs = [(call.get("usage") or {}).get("cost") for call in ledger]
    repair_calls = 0
    for path in args.run_dir.glob("*/*-review-repair-v1.json"):
        attempts = json.loads(path.read_text())["attempts"]
        repair_calls += len(attempts)
        costs.extend((attempt.get("usage") or {}).get("cost") for attempt in attempts)
        repairs["valid_citations" if attempts[-1]["citation_valid"] else "invalid_or_error"] += 1
    report = {"release_passed": False, "statuses": dict(statuses), "successful_app_turns": total_turns,
              "http_errors": dict(failures), "service_statuses": dict(methods), "judges": dict(judges),
              "request_latency_seconds": {"count": len(latencies),
                "median": latencies[len(latencies)//2] if latencies else None,
                "p95": latencies[min(len(latencies)-1, int(len(latencies)*0.95))] if latencies else None},
              "model_calls": len(ledger), "repair_calls": repair_calls, "repair_judgments": dict(repairs),
              "reported_model_cost_usd": sum(c for c in costs if isinstance(c, (int,float))),
              "calls_missing_reported_cost": sum(c is None for c in costs), "app_model_cost_excluded": True,
              "holdout_exact_comment_checks": exact,
              "sealed_exact_comment_checks": sealed_exact,
              "offline_renderer_only_exact_comment_checks": rerendered,
              "renderer_comparison_is_live": False,
              "limitations": ["Citation validation is not semantic judgment validation",
                 "Recorded failed app requests are included in latency", "No GPU-backed release verification"]}
    atomic_write(args.run_dir / "objective_summary.json", report)
    print(json.dumps({k:v for k,v in report.items() if not k.endswith("exact_comment_checks")}))


if __name__ == "__main__":
    main()
