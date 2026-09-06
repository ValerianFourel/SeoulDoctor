"""Evaluator regressions use synthetic labels and never call retrieval services."""
import json
from pathlib import Path
import sys
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import mini_retrieval_eval_50 as suite


def partial_worker(tasks, results, endpoint, deadline):
    index, request = tasks.get(timeout=1)
    results.put((index,'started',time.monotonic()))
    if index == 0:
        results.put((index,'result',({'facilities':[], 'retrieved':[], 'status':'incomplete'},.01)))
    else:
        time.sleep(10)


class ReplacementSuiteTests(unittest.TestCase):
    def pair(self):
        return dict(expected_facility_id='facility-secret', expected_evidence_id='review-secret',
                    source_text='private source', location='Mapo-gu', specialty='치과',
                    queries={'en':'Please explain treatment','ko':'치료 과정을 알려주세요','mixed':'치료 explanation 부탁해요'})

    def test_replacement_retains_duplicate_pairs_with_unique_draw_ids(self):
        cases=suite.sample_cases([self.pair()],42,['en']*50)
        self.assertEqual(len(cases),50)
        self.assertEqual(len({case['case_id'] for case in cases}),50)
        self.assertEqual(len({case['expected_evidence_id'] for case in cases}),1)

    def test_seed_is_reproducible(self):
        population=[dict(self.pair(),expected_evidence_id=str(i)) for i in range(10)]
        first=suite.sample_cases(population,14,['ko']*30)
        self.assertEqual(first,suite.sample_cases(population,14,['ko']*30))
        self.assertNotEqual(first,suite.sample_cases(population,15,['ko']*30))

    def test_only_three_fields_cross_request_boundary(self):
        case=suite.sample_cases([self.pair()],1,['en'])[0]
        sent=suite.payload(case)
        self.assertEqual(set(sent),{'query','location','specialty'})
        self.assertNotIn('secret',json.dumps(sent))
        self.assertNotIn('private source',json.dumps(sent))

    def test_variable_sizes_and_language_allocation(self):
        for size in (1,7,23):
            cases=suite.sample_cases([self.pair()],1,['en']*size)
            suite.validate(cases,size)
            with self.assertRaises(ValueError): suite.validate(cases,size+1)
        cases=suite.sample_cases([self.pair()],1,['en']*20+['ko']*20+['mixed']*10)
        suite.validate(cases,50)
        cases[0]['case_id']=cases[1]['case_id']
        with self.assertRaises(ValueError): suite.validate(cases,50)

    def test_incomplete_hits_do_not_pass(self):
        case=suite.sample_cases([self.pair()],1,['en'])[0]
        evidence={'place_id':'facility-secret','evidence_id':'review-secret'}
        response={'facilities':[{'place_id':'facility-secret','selected':[evidence]}], 'retrieved':[evidence], 'status':'incomplete'}
        row=suite.result_row(case,('result',(response,1)))
        self.assertTrue(row['selected_hit'])
        self.assertEqual(row['status'],'FAIL:incomplete')
        self.assertEqual(suite.metrics([row],1)['passes'],0)

    def test_deadline_retains_results_and_unfinished_draws(self):
        cases=suite.sample_cases([self.pair()],1,['en']*7)
        checkpoints=[]
        rows,elapsed=suite.execute(cases,'unused',1,lambda rows,seconds:checkpoints.append(rows),worker_target=partial_worker)
        self.assertEqual(len(rows),7)
        self.assertLess(elapsed,3)
        self.assertEqual(rows[0]['status'],'FAIL:incomplete')
        self.assertTrue(any(row['status']=='TIMEOUT:overall_deadline' for row in rows))
        self.assertTrue(any(row['status']=='NOT_STARTED:deadline' for row in rows))
        self.assertEqual(len(checkpoints[-1]),7)
        self.assertEqual(suite.breakdown(rows,elapsed)['unique_pairs'],1)
        self.assertEqual(suite.breakdown(rows,elapsed)['all_draws']['draws'],7)

    def test_unfinished_latency_is_not_suite_duration_and_ownership_is_unknown(self):
        case=suite.sample_cases([self.pair()],1,['en'])[0]
        row=suite.result_row(case,started=10,now=13)
        self.assertEqual(row['seconds'],3)
        self.assertIsNone(row['ownership_errors'])
        self.assertIsNone(suite.result_row(case)['seconds'])
