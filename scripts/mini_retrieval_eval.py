#!/usr/bin/env python3
"""Run seven private fixtures against warm application retrieval in 60 seconds."""

import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if sys.prefix == sys.base_prefix and (ROOT / "backend/venv/bin/python").exists():
    os.execv(str(ROOT / "backend/venv/bin/python"), [str(ROOT / "backend/venv/bin/python"), __file__, *sys.argv[1:]])

import requests


def session():
    client = requests.Session()
    client.headers.update({"Authorization": "Bearer " + os.environ["HF_TOKEN"],
                           "X-SeoulDoc-Eval-Token": os.environ["SEOULDOC_EVAL_AUTH_TOKEN"]})
    return client


def worker(tasks, results, endpoint, deadline):
    client = session()
    while time.monotonic() < deadline:
        try:
            index, payload = tasks.get_nowait()
        except queue.Empty:
            return
        started = time.monotonic()
        results.put((index, "started", started))
        try:
            response = client.post(endpoint + "/internal/retrieval", json=payload,
                                   timeout=(3, min(30, max(.1, deadline - started))))
            response.raise_for_status()
            results.put((index, "result", (response.json(), time.monotonic() - started)))
        except requests.Timeout:
            results.put((index, "error", ("TIMEOUT", time.monotonic() - started)))
        except (requests.RequestException, ValueError) as error:
            results.put((index, "error", (type(error).__name__, time.monotonic() - started)))


def score(fixture, response):
    facilities = response["facilities"]
    rank = next((i for i, item in enumerate(facilities, 1)
                 if item["place_id"] == fixture["expected_facility_id"]), None)
    identity = fixture["expected_evidence_id"]
    retrieved = any(item["evidence_id"] == identity for item in response["retrieved"])
    selected = rank is not None and any(item["evidence_id"] == identity for item in facilities[rank-1]["selected"])
    errors = sum(item["place_id"] != card["place_id"] for card in facilities for item in card["selected"])
    errors += sum(item["place_id"] != fixture["expected_facility_id"]
                  for item in response["retrieved"] if item["evidence_id"] == identity)
    return rank, retrieved, selected, errors


def main():
    preparation = time.monotonic()
    fixture_path = Path(os.getenv("MINI_RETRIEVAL_FIXTURES", ROOT / ".audit/mini-retrieval/fixtures.json"))
    fixtures = json.loads(fixture_path.read_text())
    if len(fixtures) != 7 or sorted(item["language"] for item in fixtures) != ["en"]*3 + ["ko"]*3 + ["mixed"]:
        raise ValueError("Exactly seven fixtures, with 3 English, 3 Korean and 1 mixed query, are required")
    endpoint = os.getenv("MINI_RETRIEVAL_ENDPOINT", "https://valerianfourel-seouldoctor-ncs-retriever.hf.space").rstrip("/")
    print(f"Fixture preparation: {time.monotonic()-preparation:.3f}s; private fixtures: {fixture_path}", flush=True)
    warm_started = time.monotonic()
    warm_error = None
    try:
        with session() as client:
            response = client.get(endpoint + "/internal/retrieval/warmup", timeout=(3, 30))
            response.raise_for_status()
            warm = response.json()
            print("Application limits and services: " + json.dumps(warm), flush=True)
    except (requests.RequestException, ValueError, KeyError) as error:
        warm_error = type(error).__name__
    print(f"Warmup: {time.monotonic()-warm_started:.3f}s; status={warm_error or 'OK'}", flush=True)
    started = time.monotonic()
    output = {}
    if warm_error is None:
        context = mp.get_context("spawn")
        tasks, results = context.Queue(), context.Queue()
        for index, fixture in enumerate(fixtures):
            tasks.put((index, {name:fixture[name] for name in ("query", "location", "specialty")}))
        deadline = started + 60
        processes = [context.Process(target=worker, args=(tasks, results, endpoint, deadline)) for _ in range(4)]
        for process in processes:
            process.start()
        try:
            while len(output) < 7 and time.monotonic() < deadline:
                try:
                    index, kind, data = results.get(timeout=max(.001, deadline-time.monotonic()))
                except queue.Empty:
                    break
                if kind != "started":
                    output[index] = (kind, data)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=.1)
    seconds = time.monotonic()-started
    totals = [0, 0, 0, 0]
    rows=[]
    print("case lang clinic@5 rank review selected ownership seconds status")
    for index, fixture in enumerate(fixtures):
        kind, data = output.get(index, ("error", ("BLOCKED:"+warm_error if warm_error else "TIMEOUT:overall_deadline", seconds)))
        response, elapsed = data
        rank, review, selected, errors = score(fixture, response) if kind == "result" else (None, False, False, 0)
        status = ("PASS" if rank and review and selected and not errors and response["status"] == "complete" else "FAIL:"+response["status"]) if kind == "result" else response
        for i, value in enumerate((bool(rank), review, selected, errors)):
            totals[i] += value
        print(f"{index+1} {fixture['language']} {int(bool(rank))} {rank or '-'} {int(review)} {int(selected)} {errors if kind=='result' else '?'} {elapsed:.2f} {status}")
        rows.append({"case":index+1,"rank":rank,"review_hit":review,"selected_hit":selected,
                     "ownership_errors":errors if kind=='result' else None,"seconds":elapsed,"status":status,
                     "service_status": {key:response.get(key) for key in ('semantic_status','reranker_reason','channel_hit_counts')} if kind=='result' else {}})
    print(f"TOTAL facility hits {totals[0]}/7; review hits {totals[1]}/7; selected-comment hits {totals[2]}/7; ownership errors {totals[3]}; total {seconds:.2f}s")
    report=fixture_path.parent / (time.strftime('run-%Y%m%dT%H%M%SZ', time.gmtime())+'.json')
    report.write_text(json.dumps({"rows":rows,"totals":totals,"execution_seconds":seconds,"warmup_error":warm_error},indent=2)+'\n')
    return 0 if all(row['status']=='PASS' for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
