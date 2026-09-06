"""Re-review invalid judge records without replaying any app conversation."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]
import requests
from patient_journey import atomic_write
from scripts.run_parallel_diagnostics import DIMENSIONS, JUDGE_PROMPT, validate_judge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--workers", type=int, default=3, choices=range(1, 5))
    args = parser.parse_args()
    pending = []
    for path in args.run_dir.glob("*/judges.json"):
        reviews = json.loads(path.read_text())["reviews"]
        for item in reviews:
            if not item["citation_valid"]:
                pending.append((path.parent, item["model"]))

    def review(task):
        directory, model = task
        output = directory / (model.replace("/", "-") + "-review-repair-v1.json")
        if output.exists():
            return {"scenario": directory.name, "status": "existing_record_preserved"}
        conversation = json.loads((directory / "conversation.json").read_text())
        if not conversation["complete"]:
            return {"scenario": directory.name, "status": "incomplete_not_graded"}
        messages = [{"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": json.dumps({"dimensions": DIMENSIONS,
                     "turns": conversation["turns"]}, ensure_ascii=False)}]
        record = {"scenario": directory.name, "provisional": True, "original_review_preserved": True,
                  "model": model, "max_tokens": 8192,
                  "reasoning": {"effort": "none", "exclude": True}, "attempts": []}
        for attempt in range(2):
            begin = time.perf_counter()
            try:
                response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]},
                    json={"model": record["model"], "messages": messages, "temperature": 0,
                          "max_tokens": record["max_tokens"], "reasoning": record["reasoning"],
                          "response_format": {"type": "json_object"}}, timeout=90)
                response.raise_for_status()
                raw = response.json()
                item = {"usage": raw.get("usage"), "finish_reason": raw["choices"][0].get("finish_reason")}
                content = raw["choices"][0]["message"].get("content")
                try:
                    parsed = json.loads(content) if isinstance(content, str) else None
                except json.JSONDecodeError:
                    parsed = None
                item["citation_valid"] = validate_judge(parsed, conversation["turns"])
                item["review"] = parsed
            except Exception as error:
                item = {"citation_valid": False, "error_type": type(error).__name__}
            item["seconds"] = time.perf_counter() - begin
            record["attempts"].append(item)
            atomic_write(output, record)
            if item["citation_valid"]:
                break
        return {"scenario": directory.name, "citation_valid": record["attempts"][-1]["citation_valid"]}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(review, path) for path in pending]):
            print(json.dumps(future.result()), flush=True)


if __name__ == "__main__":
    main()
