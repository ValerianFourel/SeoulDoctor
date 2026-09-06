#!/usr/bin/env python3
"""Replacement-sampled private retrieval suite; the seven-case runner stays unchanged."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if sys.prefix == sys.base_prefix and (ROOT / 'backend/venv/bin/python').exists():
    os.execv(str(ROOT / 'backend/venv/bin/python'), [str(ROOT / 'backend/venv/bin/python'), __file__, *sys.argv[1:]])

sys.path.insert(0, str(ROOT / 'scripts'))
import requests
from mini_retrieval_eval import score, session, worker

ENDPOINT = 'https://valerianfourel-seouldoctor-ncs-retriever.hf.space'


def sample_cases(population, seed, languages):
    if not population or not languages:
        raise ValueError('Population and language allocation must be nonempty')
    draws = random.Random(seed).choices(population, k=len(languages))
    return [dict(pair, case_id=f'draw-{i+1:03d}', language=language,
                 query=pair['queries'][language]) for i, (pair, language) in enumerate(zip(draws, languages))]


def payload(case):
    return {key: case[key] for key in ('query', 'location', 'specialty')}


def validate(fixtures, size):
    if len(fixtures) != size or size < 1:
        raise ValueError('Fixture count must equal suite size')
    if len({case['case_id'] for case in fixtures}) != size:
        raise ValueError('Every draw needs a unique case ID')
    if size == 50 and sorted(case['language'] for case in fixtures) != ['en']*20 + ['ko']*20 + ['mixed']*10:
        raise ValueError('50 draws require 20 English, 20 Korean, 10 mixed')
    for case in fixtures:
        payload(case)
        if case['expected_facility_id'] in case['query'] or case['expected_evidence_id'] in case['query']:
            raise ValueError('Target label leaked into query')
        if case.get('source_text') and case['source_text'] in case['query']:
            raise ValueError('Source quotation leaked into query')


def result_row(case, outcome=None, started=None, now=None, blocked=None, pending=False):
    row = dict(case_id=case['case_id'], language=case['language'],
               expected_facility_id=case['expected_facility_id'], expected_evidence_id=case['expected_evidence_id'],
               rank=None, review_hit=False, selected_hit=False, ownership_errors=None,
               complete=False, seconds=None, service_status={}, status='NOT_STARTED:deadline')
    if blocked:
        row['status'] = 'BLOCKED:' + blocked
    elif outcome:
        kind, (response, elapsed) = outcome
        row['seconds'] = elapsed
        if kind == 'result':
            try:
                rank, review, selected, errors = score(case, response)
                errors += sum(item['evidence_id'] == case['expected_evidence_id']
                              and item['place_id'] != case['expected_facility_id']
                              and item['place_id'] == card['place_id']
                              for card in response['facilities'] for item in card['selected'])
                complete = response['status'] == 'complete'
                row.update(rank=rank, review_hit=review, selected_hit=selected, ownership_errors=errors,
                           complete=complete, service_status={key: response.get(key) for key in
                           ('status', 'semantic_status', 'reranker_reason', 'channel_hit_counts')},
                           status='PASS' if rank and review and selected and not errors and complete else 'FAIL:'+response['status'])
            except (KeyError, TypeError, ValueError):
                row['status'] = 'ERROR:invalid_response'
        else:
            row['status'] = response
    elif started is not None:
        row.update(status='RUNNING' if pending else 'TIMEOUT:overall_deadline', seconds=max(0, now-started))
    elif pending:
        row['status'] = 'PENDING'
    return row


def execute(fixtures, endpoint, seconds, checkpoint, worker_target=worker):
    context = mp.get_context('spawn')
    tasks, results = context.Queue(), context.Queue()
    for index, case in enumerate(fixtures):
        tasks.put((index, payload(case)))
    started = time.monotonic()
    deadline = started + seconds
    output, active = {}, {}
    processes = [context.Process(target=worker_target, args=(tasks, results, endpoint, deadline)) for _ in range(4)]
    for process in processes:
        process.start()
    try:
        while len(output) < len(fixtures) and time.monotonic() < deadline:
            try:
                index, kind, data = results.get(timeout=max(.001, min(.2, deadline-time.monotonic())))
            except queue.Empty:
                if not any(process.is_alive() for process in processes):
                    break
                continue
            if kind == 'started':
                active[index] = data
            else:
                output[index] = (kind, data)
                checkpoint([result_row(case, output.get(i), active.get(i), time.monotonic(), pending=True)
                            for i, case in enumerate(fixtures)], time.monotonic()-started)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=.2)
        tasks.close()
        results.close()
    now = time.monotonic()
    rows = [result_row(case, output.get(i), active.get(i), now) for i, case in enumerate(fixtures)]
    checkpoint(rows, now-started)
    return rows, now-started


def metrics(rows, seconds):
    latencies = sorted(row['seconds'] for row in rows if row['service_status'] and row['seconds'] is not None)
    def percentile(values, proportion):
        if not values:
            return None
        index = (len(values)-1)*proportion
        lower = int(index)
        return values[lower] + (values[min(lower+1,len(values)-1)]-values[lower])*(index-lower)
    passed = sum(row['status']=='PASS' for row in rows)
    return dict(draws=len(rows), facility_hits=sum(bool(row['rank']) for row in rows),
                review_hits=sum(row['review_hit'] for row in rows), selected_hits=sum(row['selected_hit'] for row in rows),
                ownership_errors=sum(row['ownership_errors'] or 0 for row in rows),
                ownership_unknown=sum(row['ownership_errors'] is None for row in rows),
                complete=sum(row['complete'] for row in rows), passes=passed,
                successful_cases_per_second=passed/seconds if seconds else None,
                response_latency_count=len(latencies), median_seconds=statistics.median(latencies) if latencies else None,
                p95_seconds=percentile(latencies,.95),
                statuses={status:sum(row['status']==status for row in rows) for status in sorted({row['status'] for row in rows})})


def breakdown(rows, seconds):
    pairs = sorted({(row['expected_facility_id'],row['expected_evidence_id']) for row in rows})
    return dict(all_draws=metrics(rows,seconds), unique_pairs=len(pairs),
                unique_facilities=len({row['expected_facility_id'] for row in rows}),
                by_language={lang:metrics([row for row in rows if row['language']==lang],seconds) for lang in sorted({row['language'] for row in rows})},
                by_pair=[dict(facility_id=fac,evidence_id=eid,**metrics([row for row in rows if (row['expected_facility_id'],row['expected_evidence_id'])==(fac,eid)],seconds)) for fac,eid in pairs])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', required=True, type=Path)
    parser.add_argument('--size', type=int, default=50)
    parser.add_argument('--deadline', type=float, default=300)
    parser.add_argument('--expected-space-revision', required=True)
    args=parser.parse_args()
    if not 0 < args.deadline <= 600:
        parser.error('Deadline must be between 0 and 600 seconds')
    loading=time.monotonic()
    raw=args.fixtures.read_bytes()
    fixtures=json.loads(raw)
    validate(fixtures,args.size)
    provenance=json.loads(args.fixtures.with_name('provenance.json').read_text())
    if hashlib.sha256(raw).hexdigest()!=provenance['fixture_sha256']:
        raise ValueError('Frozen fixture hash mismatch')
    report_path=args.fixtures.with_name('results.json')
    if report_path.exists():
        raise ValueError('Existing run retained: choose a fresh run directory, never silently replay')
    metadata=dict(fixture_sha256=provenance['fixture_sha256'],provenance=provenance,
                  evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_revision=args.expected_space_revision, runtime=None,
                  fixture_loading_seconds=time.monotonic()-loading,deadline_seconds=args.deadline,workers=4,
                  cancellation='Terminating client processes does not cancel server computation; no automatic retries or subsequent run.',
                  warmup=None,warmup_error=None)
    warm_started=time.monotonic()
    try:
        with requests.Session() as client:
            runtime_response=client.get('https://huggingface.co/api/spaces/ValerianFourel/SeoulDoctor-ncs-retriever/runtime',
                                        headers={'Authorization':'Bearer '+os.environ['HF_TOKEN']},timeout=(3,15))
            runtime_response.raise_for_status()
            runtime=runtime_response.json()
            metadata['runtime']=runtime
            if runtime.get('stage')!='RUNNING' or runtime.get('sha')!=args.expected_space_revision:
                raise ValueError('Authorized deployment revision is not active and RUNNING')
        with session() as client:
            response=client.get(ENDPOINT+'/internal/retrieval/warmup',timeout=(3,30))
            response.raise_for_status()
            warm=response.json()
            gpu=client.get(ENDPOINT+'/ready/gpu',timeout=(3,30))
            gpu.raise_for_status()
            warm['gpu']=gpu.json()
            if warm['gpu'].get('ready') is not True:
                raise ValueError('GPU probe not ready')
            if warm.get('index_version')!=provenance['index_version']:
                raise ValueError('Index version mismatch')
            metadata['warmup']=warm
    except Exception as error:
        metadata['warmup_error']=type(error).__name__
    metadata['warmup_seconds']=time.monotonic()-warm_started
    print('DEPLOYMENT '+json.dumps(metadata['runtime']),flush=True)
    print('WARMUP '+json.dumps(metadata['warmup'])+'; seconds='+str(metadata['warmup_seconds']),flush=True)
    def checkpoint(rows, seconds):
        report=dict(metadata,rows=rows,execution_seconds=seconds,metrics=breakdown(rows,seconds))
        temporary=report_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        temporary.replace(report_path)
    checkpoint([result_row(case,blocked='pending') for case in fixtures],0)
    if metadata['warmup_error']:
        rows=[result_row(case,blocked=metadata['warmup_error']) for case in fixtures]
        seconds=0
        checkpoint(rows,seconds)
    else:
        rows,seconds=execute(fixtures,ENDPOINT,args.deadline,checkpoint)
    print('case lang rank review selected ownership complete seconds status semantic reranker')
    for row in rows:
        print(row['case_id'],row['language'],row['rank'] or '-',int(row['review_hit']),int(row['selected_hit']),
              row['ownership_errors'],row['complete'],round(row['seconds'],3) if row['seconds'] is not None else '-',
              row['status'],row['service_status'].get('semantic_status'),row['service_status'].get('reranker_reason'))
    summary=breakdown(rows,seconds)
    print('TOTAL '+json.dumps({key:value for key,value in summary.items() if key!='by_pair'})+'; total_seconds='+str(seconds))
    return 0 if all(row['status']=='PASS' for row in rows) else 1


if __name__=='__main__':
    raise SystemExit(main())
