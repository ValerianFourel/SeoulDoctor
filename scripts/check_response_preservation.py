"""Check the deployed NCS reply through four sequential public chat requests."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError


SPACE = "ValerianFourel/SeoulDoctor-ncs-retriever"
API = "https://valerianfourel-seouldoctor-ncs-retriever.hf.space"
INITIAL = "i need a orthopedic doctor next to  Jonggak"
FOLLOWUP = "Keep it within 1 km of Jonggak, please."
GENERIC = "See the facility cards and reviews below."


def request_json(url, payload=None):
    request = Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=180 if payload is not None else 30) as response:
        return json.load(response)


def write_record(directory, name, value):
    text = json.dumps(value, ensure_ascii=False, indent=2)
    text = re.sub(r"(?i)Bearer\s+[^\s\"]+|hf_[A-Za-z0-9]{12,}|sk-or-v1-[A-Za-z0-9]+", "[REDACTED]", text)
    with (directory / name).open("x") as stream:
        stream.write(text + "\n")


def inspect_revision(expected):
    info = request_json(f"https://huggingface.co/api/spaces/{SPACE}")
    runtime = info.get("runtime", {})
    observed = {"source": info.get("sha"), "runtime": runtime.get("sha"), "stage": runtime.get("stage")}
    return observed, observed == {"source": expected, "runtime": expected, "stage": "RUNNING"}


def check_reply(body, *, followup):
    reply = body["response"]
    cards = body["results"]
    state = body["state"]
    valid_distances = bool(cards) and all(
        isinstance(card.get("distance_km"), (int, float))
        and not isinstance(card["distance_km"], bool)
        and math.isfinite(card["distance_km"])
        and card["distance_km"] >= 0
        for card in cards
    )
    closest_sentence = None
    if valid_distances:
        closest = min(cards, key=lambda card: card["distance_km"])
        closest_sentence = (
            f"Of the options shown, {closest['name']} is closest, about "
            f"{closest['distance_km']:.1f} km from Jonggak by straight-line distance."
        )
    checks = {
        "guided_reply_preserved": reply.startswith("I found ") and "orthopedics around Jonggak" in reply,
        "generic_replacement_absent": GENERIC not in reply,
        "grounded_distance": closest_sentence is not None and closest_sentence in reply,
        "specialty_preserved": state["specialty"] == "정형외과",
        "anchor_preserved": state["location"] == "Jonggak",
        "cards_present": bool(cards),
        "evidence_ownership": all(
            evidence["place_id"] == card["place_id"]
            for card in cards for evidence in card.get("retrieval_evidence", [])
        ),
    }
    if followup:
        checks["radius_refinement"] = state["max_distance_km"] == 1.0 and bool(cards) and all(
            0 <= card["distance_km"] <= 1.0 for card in cards
        )
        checks["followup_guidance"] = "You can compare the cards below or refine this search further." in reply
    else:
        checks["useful_question"] = "What would you like the doctor to help with?" in reply
        checks["refinement_options"] = "within 1 km" in reply and "language or access needs" in reply
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-space-sha", required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    directory = args.output_dir or Path(".audit") / datetime.now(timezone.utc).strftime("response-api-%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True, exist_ok=False)
    report = {"expected_space_sha": args.expected_space_sha, "api_calls": 0, "cases": [], "passed": False}
    try:
        before, ready = inspect_revision(args.expected_space_sha)
        report["before"] = before
        if not ready:
            raise RuntimeError("The expected Space revision is not running; no chat requests sent")
        for attempt in (1, 2):
            state = {}
            for label, message in (("initial", INITIAL), ("followup", FOLLOWUP)):
                payload = {"message": message, "current_state": state}
                name = f"{attempt}-{label}"
                record = {"case": name, "request": payload, "started_utc": datetime.now(timezone.utc).isoformat()}
                write_record(directory, name + "-request.json", record)
                started = time.monotonic()
                report["api_calls"] += 1
                try:
                    body = request_json(API + "/chat", payload)
                except Exception as error:
                    record["error"] = {"type": type(error).__name__, "message": str(error)}
                    if isinstance(error, HTTPError):
                        record["error"].update(status=error.code, body=error.read(8192).decode(errors="replace"))
                    record["latency_seconds"] = round(time.monotonic() - started, 3)
                    write_record(directory, name + ".json", record)
                    report["cases"].append({"name": name, "error": record["error"]})
                    raise
                elapsed = round(time.monotonic() - started, 3)
                record.update(response=body, latency_seconds=elapsed)
                write_record(directory, name + ".json", record)
                checks = check_reply(body, followup=label == "followup")
                report["cases"].append({"name": name, "checks": checks, "latency_seconds": elapsed})
                state = body["state"]
        after, stable = inspect_revision(args.expected_space_sha)
        report["after"] = after
        report["passed"] = stable and all(all(case["checks"].values()) for case in report["cases"])
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    write_record(directory, "summary.json", report)
    print(json.dumps({"passed": report["passed"], "api_calls": report["api_calls"], "report": str(directory / "summary.json")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
