"""Audit a saved journey against a private comment target; never calls an LLM."""

import argparse
from html import unescape
import json
from pathlib import Path
import re


def normalized(text):
    text = re.sub(r"(?m)^> ?", "", text)
    return " ".join(re.sub(r"\\([\\`*_{}\[\]()#+!|>])", r"\1", unescape(text)).split())


def audit(journey, target):
    output = []
    for turn in journey.get("turns", []):
        body = turn.get("response", {}).get("body", {})
        if not isinstance(body, dict):
            body = {}
        metadata = body.get("state", {}).get("last_retrieval_metadata", {})
        admissions = [event for event in metadata.get("evidence_admissions", [])
                      if event.get("evidence_id") == target["evidence_id"]]
        attached = [(card, item) for card in body.get("results", [])
                    for item in card.get("retrieval_evidence", [])
                    if item.get("evidence_id") == target["evidence_id"]]
        original = normalized(target["text"])
        visible = normalized(body.get("response", ""))
        ownership = bool(attached) and all(
            str(card.get("place_id")) == str(item.get("place_id")) == str(target["place_id"])
            for card, item in attached
        )
        output.append({
            "retrieved": bool(admissions) or bool(attached),
            "survived_evidence_selection": bool(attached),
            "correct_facility_ownership": ownership,
            "original_comment_visible": bool(original) and original in visible,
            "original_text_preserved": bool(original) and original in visible and ownership
            and all(item.get("is_verbatim") is True
                    and normalized(item.get("text", "")) == original for _, item in attached),
            "translation_fidelity": "not_assessed",
            "recommendation_use": "requires_independent_review",
            "retrieval_status": metadata.get("retrieval_status", "unknown"),
        })
    return {"journey_status": journey.get("status", "unknown"), "turns": output,
            "passed": False,
            "limitation": "Objective visibility checks do not grade recommendation suitability."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journey", type=Path)
    parser.add_argument("private_target", type=Path,
                        help="JSON containing evidence_id, place_id, and exact original text")
    args = parser.parse_args()
    report = audit(json.loads(args.journey.read_text()), json.loads(args.private_target.read_text()))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
