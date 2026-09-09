"""Prepare private conversations and source checks for independent inference grading."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
from html import escape
import json
import math
import os
from pathlib import Path
import sys


def finite(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def error_fields(value):
    return {key: value[key] for key in ("status", "error", "error_type", "error_reason", "error_category", "reason", "http_status")
            if key in value}


def review_records(card):
    for review in card.get("retrieval_evidence", []):
        yield "reviews", review
    for group, reviews in card.get("retrieval_evidence_groups", {}).items():
        if group in {"supporting", "warnings"}:
            for review in reviews:
                yield group, review
    for citation in card.get("answer_citations", []):
        yield "citations", citation


def response_body(turn):
    return turn.get("response", {}).get("body") or turn.get("app", {}).get("body") or {}


def resolve_originals(scoped, identities):
    resolved, errors = {}, {}

    def resolve(batch):
        try:
            for hit in scoped.resolve_evidence_ids(batch):
                resolved[hit.evidence_id] = hit
        except ValueError as error:
            if len(batch) > 1:
                middle = len(batch) // 2
                resolve(batch[:middle])
                resolve(batch[middle:])
            else:
                errors[batch[0]] = {"code": "unresolved_original", "error_type": type(error).__name__, "error": str(error)}
        except Exception as error:
            for identity in batch:
                errors[identity] = {"code": "index_read_failed", "error_type": type(error).__name__, "error": str(error)}

    identities = sorted(set(identities))
    for start in range(0, len(identities), 400):
        resolve(identities[start:start + 400])
    for identity in identities:
        if identity not in resolved and identity not in errors:
            errors[identity] = {"code": "resolver_omitted_original"}
    return resolved, errors


def verify_record(record, owner, collection, originals, resolution_errors, source_sha):
    identity = record.get("evidence_id")
    errors = []
    original = originals.get(identity) if isinstance(identity, str) else None
    if not isinstance(identity, str) or not identity:
        errors.append({"code": "missing_evidence_id"})
    elif identity in resolution_errors:
        errors.append(resolution_errors[identity])
    elif original is None:
        errors.append({"code": "unresolved_original"})
    if not isinstance(owner, str) or not owner or record.get("place_id") != owner:
        errors.append({"code": "card_owner_mismatch"})
    if original:
        if original.facility_id != owner or record.get("place_id") != original.facility_id:
            errors.append({"code": "source_owner_mismatch", "actual_owner": original.facility_id})
        if not original.is_verbatim or original.source_type != "verbatim_review":
            errors.append({"code": "source_is_not_original_review"})
        if collection == "citations":
            excerpt = record.get("original_excerpt")
            if not isinstance(excerpt, str) or not excerpt or excerpt not in original.original_text:
                errors.append({"code": "citation_excerpt_mismatch"})
        elif record.get("text") != original.original_text:
            errors.append({"code": "original_text_mismatch"})
    if collection != "citations" and (record.get("is_verbatim") is not True or record.get("source_type") != "verbatim_review"):
        errors.append({"code": "record_is_not_original_review"})
    if not source_sha or record.get("review_source_sha256") != source_sha:
        errors.append({"code": "source_revision_mismatch"})
    return {"evidence_id": identity, "collection": collection, "valid": not errors, "errors": errors}


def measurements(turn, body):
    metadata = body.get("state", {}).get("last_retrieval_metadata", {})
    cached = turn.get("measurements", {})
    answer = metadata.get("answer", {})
    calls = answer.get("calls", cached.get("answer_model_calls", []))
    translation = answer.get("translation", cached.get("translation", {}))
    call_durations = [finite(call.get("duration_ms")) for call in calls]
    return {"application_seconds": finite(turn.get("app", {}).get("duration_seconds")),
            "retrieval_ms": finite(metadata.get("elapsed_ms", cached.get("retrieval_elapsed_ms"))),
            "retrieval_status": metadata.get("retrieval_execution_status", cached.get("retrieval_execution_status")),
            "retrieval_reasons": metadata.get("retrieval_reason_codes", []),
            "answer_status": answer.get("status", cached.get("answer_status")),
            "answer_reason": answer.get("reason", cached.get("answer_reason")),
            "answer_model_ms": sum(call_durations) if call_durations and all(value is not None for value in call_durations) else None,
            "answer_calls": [{"stage": call.get("stage"), "duration_ms": finite(call.get("duration_ms")),
                              "usage": call.get("usage")} for call in calls],
            "translation": translation}


def summarize_case(case, originals, resolution_errors, source_sha):
    result = {"id": case.get("id"), **error_fields(case), "turns": [], "assertion_failures": [], "source_errors": []}
    for turn_index, turn in enumerate(case.get("turns", [])):
        body = response_body(turn)
        if not isinstance(body, dict):
            body = {}
        state = body.get("state", {})
        failures = list(dict.fromkeys(turn.get("assessment", {}).get("failures", [])
                                     + turn.get("fixture_assessment", {}).get("failures", [])))
        summary = {"index": turn.get("index", turn_index), **error_fields(turn),
                   "patient": turn.get("message"), "assistant": body.get("response"),
                   "state": {key: state.get(key) for key in ("specialty", "location", "latitude", "longitude", "max_distance_km", "language_pref", "visit_reason")},
                   "measurements": measurements(turn, body), "assertion_failures": failures,
                   "unscored": turn.get("assessment", {}).get("unscored", []),
                   "app": error_fields(turn.get("app", {})), "actor": error_fields(turn.get("actor", {})),
                   "response_error": {key: body[key] for key in ("error", "detail") if key in body}, "cards": []}
        result["assertion_failures"].extend({"turn": turn_index, "assertion": item} for item in failures)
        for card_index, card in enumerate(body.get("results", [])):
            reviews = card.get("retrieval_evidence", [])
            displayable = [review for review in reviews
                           if review.get("presentation", {}).get("status") not in {"hidden", "unavailable"}]
            checks = [verify_record(record, card.get("place_id"), collection, originals, resolution_errors, source_sha)
                      for collection, record in review_records(card)]
            errors = [{"turn": turn_index, "card": card_index, **check} for check in checks if not check["valid"]]
            result["source_errors"].extend(errors)
            statuses = {}
            for review in reviews:
                status = review.get("presentation", {}).get("status", "missing")
                statuses[status] = statuses.get(status, 0) + 1
            summary["cards"].append({**{key: card.get(key) for key in ("place_id", "name", "category", "address", "distance_km", "answer_status")},
                "review_count": len(reviews), "translation_status_counts": statuses,
                "first_reviews": [{"evidence_id": review.get("evidence_id"), "original": review.get("text"),
                                   "presentation": review.get("presentation", {})} for review in displayable[:3]],
                "source_checks": checks})
        result["turns"].append(summary)
    return result


def provider_usage(records):
    calls, errors, unidentified = {}, [], []
    duplicates = 0
    for origin, body in records:
        body = body.get("body", body) if isinstance(body, dict) else {}
        if not isinstance(body, dict):
            body = {}
        identity = body.get("id")
        if not isinstance(identity, str) or not identity:
            unidentified.append({"origin": origin, "error": body.get("error"), "reason": "completion_id_missing"})
            continue
        usage = body.get("usage") or {}
        if not isinstance(usage, dict):
            errors.append({"code": "invalid_provider_usage", "id": identity, "origin": origin})
            usage = {}
        value = {"id": identity, "model": body.get("model"), "provider": body.get("provider"),
                 "cost_usd": finite(usage.get("cost")), "prompt_tokens": finite(usage.get("prompt_tokens")),
                 "completion_tokens": finite(usage.get("completion_tokens")), "origins": [origin], "conflicts": []}
        if value["cost_usd"] is not None and value["cost_usd"] < 0:
            value["cost_usd"] = None
        if identity not in calls:
            calls[identity] = value
            continue
        duplicates += 1
        prior = calls[identity]
        prior["origins"].append(origin)
        for key in ("model", "provider", "cost_usd", "prompt_tokens", "completion_tokens"):
            if prior[key] is None:
                prior[key] = value[key]
            elif value[key] is not None and prior[key] != value[key]:
                prior["conflicts"].append(key)
                errors.append({"code": "provider_capture_conflict", "id": identity, "field": key, "origin": origin})
    known = [call for call in calls.values() if call["cost_usd"] is not None and not call["conflicts"]]
    return {"scope": "run actor and translation responses and every file in the supplied capture directories; capture files may include calls outside these runs",
            "unique_completions": len(calls), "duplicate_records": duplicates,
            "known_cost_usd": sum(call["cost_usd"] for call in known),
            "unknown_or_conflicting_cost_count": len(calls) - len(known),
            "unidentified_records": unidentified, "errors": errors, "calls": list(calls.values())}


def render_markdown(report):
    lines = ["# Inference conversations for review", "", "Judgment is pending. No Likert score or quality pass is assigned.", "",
             f'Original source revision: `{report["index"].get("review_source_sha256", "unavailable")}`.', "",
             f'Source errors: {report["source_error_count"]}. Assertion failures: {report["assertion_failure_count"]}.', ""]
    for run in report["runs"]:
        lines.extend([f'## {run.get("phase")} · {run.get("application_revision")}', "", f'Run status: {run.get("status")}. Input: `{run["path"]}`.', ""])
        if run["errors"].get("error_type") or run["errors"].get("error_reason"):
            lines.extend(["Run error: " + json.dumps(run["errors"], ensure_ascii=False), ""])
        for case in run["cases"]:
            lines.extend([f'### {case["id"]} · {case.get("status")}', ""])
            if case.get("error_type") or case.get("error_reason"):
                lines.extend(["Case error: " + json.dumps(error_fields(case), ensure_ascii=False), ""])
            for turn in case["turns"]:
                lines.extend([f'Patient, turn {turn["index"] + 1}: {escape(str(turn["patient"]))}', "",
                              escape(str(turn["assistant"])), "",
                              "State: " + json.dumps(turn["state"], ensure_ascii=False), "",
                              "Timing and status: " + json.dumps(turn["measurements"], ensure_ascii=False), ""])
                if turn["assertion_failures"]:
                    lines.extend(["Failed assertions: " + ", ".join(turn["assertion_failures"]), ""])
                if turn.get("status") == "failed" or turn["response_error"]:
                    lines.extend(["Turn errors: " + json.dumps({"turn": error_fields(turn), "app": turn["app"], "actor": turn["actor"], "response": turn["response_error"]}, ensure_ascii=False), ""])
                for card in turn["cards"]:
                    lines.extend([f'**{escape(str(card["name"]))}** · {card["category"]} · {card["distance_km"]} km', "",
                                  f'{card["review_count"]} originals. Translation status: {json.dumps(card["translation_status_counts"])}.', ""])
                    for review in card["first_reviews"]:
                        lines.extend([f'Original `{review["evidence_id"]}`:', "", *["> " + escape(line) for line in str(review["original"]).splitlines()], "",
                                      "Presentation: " + escape(json.dumps(review["presentation"], ensure_ascii=False)), ""])
                    bad = [check for check in card["source_checks"] if not check["valid"]]
                    if bad:
                        lines.extend(["Source errors: " + json.dumps(bad, ensure_ascii=False), ""])
    usage = report["provider_usage"]
    lines.extend(["## Provider usage", "", usage["scope"], "",
                  f'{usage["unique_completions"]} unique completions; {usage["duplicate_records"]} duplicate records; known cost USD {usage["known_cost_usd"]:.6f}.',
                  f'{usage["unknown_or_conflicting_cost_count"]} completion costs unknown/conflicting; {len(usage["unidentified_records"])} records lack a completion ID.', ""])
    provider_errors = [record for record in usage["unidentified_records"] if record.get("error")]
    if provider_errors:
        lines.extend(["Provider errors: " + json.dumps(provider_errors, ensure_ascii=False), ""])
    if report["errors"]:
        lines.extend(["## Preparation errors", "", json.dumps(report["errors"], ensure_ascii=False, indent=2), ""])
    return "\n".join(lines)


def prepare(run_paths, index_root, capture_dirs=()):
    runs, identities, provider_records, errors = [], set(), [], []
    for path in run_paths:
        try:
            run = json.loads(Path(path).read_text())
            if not isinstance(run, dict) or not isinstance(run.get("cases"), list):
                raise ValueError("run must be an object containing a cases array")
            runs.append((str(path), run))
            for case_index, case in enumerate(run.get("cases", [])):
                try:
                    for turn in case.get("turns", []):
                        if turn.get("actor", {}).get("body"):
                            provider_records.append((f'{path}/{case.get("id")}/{turn.get("index")}/actor', turn["actor"]["body"]))
                        body = response_body(turn)
                        if isinstance(body, dict):
                            translation = body.get("state", {}).get("last_retrieval_metadata", {}).get("answer", {}).get("translation", {})
                            if translation.get("completion_id"):
                                provider_records.append((f'{path}/{case.get("id")}/{turn.get("index")}/translation', {
                                    "id": translation["completion_id"], "model": translation.get("actual_model"),
                                    "provider": translation.get("actual_provider"), "usage": translation.get("usage")}))
                        for card in body.get("results", []) if isinstance(body, dict) else []:
                            identities.update(record["evidence_id"] for _, record in review_records(card)
                                              if isinstance(record.get("evidence_id"), str) and record["evidence_id"])
                except Exception as error:
                    errors.append({"path": str(path), "case_index": case_index,
                                   "error_type": type(error).__name__, "error": str(error)})
        except Exception as error:
            errors.append({"path": str(path), "error_type": type(error).__name__, "error": str(error)})
    for directory in capture_dirs:
        if not Path(directory).is_dir():
            errors.append({"path": str(directory), "error_type": "FileNotFoundError", "error": "capture directory missing"})
            continue
        for path in sorted(Path(directory).glob("*.json")):
            try:
                provider_records.append((str(path), json.loads(path.read_text())))
            except Exception as error:
                errors.append({"path": str(path), "error_type": type(error).__name__, "error": str(error)})
    index_info, originals, resolution_errors = {}, {}, {}
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
        import pandas as pd
        from search.indexes.repository import IndexRepository
        from search.rules import RulesCompiler
        from search.scope import ScopeBuilder
        with ExitStack() as stack:
            index = stack.enter_context(IndexRepository(index_root).open_active())
            rules = RulesCompiler().compile(original_query="Read-only original review audit across Seoul", turn_id="source-audit",
                                             proposal={"location": "Seoul", "is_citywide_search": True}).rules
            scope = ScopeBuilder().build(pd.DataFrame({"place_id": index.facility_ids}), rules, index_version=index.version)
            scoped = stack.enter_context(index.within(scope))
            index_info = {"status": "opened", "version": index.version, "content_id": index.manifest.content_id,
                          "review_source_sha256": scoped.review_source_sha256, "root": str(index_root)}
            originals, resolution_errors = resolve_originals(scoped, identities)
    except Exception as error:
        index_info.update(status="failed", error_type=type(error).__name__, error=str(error))
        errors.append({"stage": "index", **index_info})
        resolution_errors = {identity: {"code": "index_unavailable"} for identity in identities}
    summaries = []
    for path, run in runs:
        cases = []
        for case_index, case in enumerate(run.get("cases", [])):
            try:
                cases.append(summarize_case(case, originals, resolution_errors, index_info.get("review_source_sha256")))
            except Exception as error:
                failure = {"path": path, "case_index": case_index, "error_type": type(error).__name__, "error": str(error)}
                errors.append(failure)
                cases.append({"id": case.get("id") if isinstance(case, dict) else case_index,
                              "status": "report_preparation_failed", "error": failure,
                              "turns": [], "assertion_failures": [], "source_errors": []})
        summaries.append({"path": path, **{key: run.get(key) for key in ("phase", "status", "application_revision", "manifest_sha256", "grading_protocol_id", "grading_protocol_sha256", "budget")},
                          "errors": error_fields(run), "cases": cases})
    usage = provider_usage(provider_records)
    errors.extend(usage["errors"])
    return {"quality_pass": False, "grading_status": "pending_root_judgment", "index": index_info,
            "source_coverage": "all attached reviews, supporting/warning groups, and answer citations, including later pages; retrieval telemetry IDs are excluded",
            "unique_evidence_ids": len(identities), "resolved_evidence_ids": len(originals), "resolution_errors": resolution_errors,
            "source_error_count": sum(len(case["source_errors"]) for run in summaries for case in run["cases"]),
            "assertion_failure_count": sum(len(case["assertion_failures"]) for run in summaries for case in run["cases"]),
            "runs": summaries, "provider_usage": usage, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    report = prepare(args.run, args.index_root, args.capture_dir)
    for name, content in (("report.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)),
                          ("REPORT.md", render_markdown(report))):
        path = args.output_dir / name
        with path.open("x", encoding="utf-8") as output:
            os.chmod(path, 0o600)
            output.write(content)
    print(json.dumps({"output_dir": str(args.output_dir), "source_errors": report["source_error_count"],
                      "assertion_failures": report["assertion_failure_count"], "quality_pass": False}))
    raise SystemExit(1 if report["errors"] or report["source_error_count"] or report["assertion_failure_count"] else 0)


if __name__ == "__main__":
    main()
