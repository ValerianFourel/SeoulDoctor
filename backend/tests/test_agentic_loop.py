import json
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))


if importlib.util.find_spec("chromadb") is None:
    chromadb = types.ModuleType("chromadb")
    chromadb.PersistentClient = object
    chromadb_utils = types.ModuleType("chromadb.utils")
    chromadb_utils.embedding_functions = types.SimpleNamespace()
    sys.modules["chromadb"] = chromadb
    sys.modules["chromadb.utils"] = chromadb_utils

if importlib.util.find_spec("rank_bm25") is None:
    rank_bm25 = types.ModuleType("rank_bm25")
    rank_bm25.BM25Okapi = object
    sys.modules["rank_bm25"] = rank_bm25


from rag_pipeline import RAGPipeline  # noqa: E402


class AgenticLoopTests(unittest.TestCase):
    def test_specific_bm25_requires_the_literal_term(self):
        class TokenScoreIndex:
            def __init__(self, corpus):
                self.corpus = corpus

            def get_scores(self, query_tokens):
                return np.array([
                    sum(document.count(token) for token in query_tokens)
                    for document in self.corpus
                ], dtype=float)

        pipeline = object.__new__(RAGPipeline)
        pipeline.specific_evidence_records = [
            {
                "place_id": "clinic-1",
                "text": "The consultation was professional.",
                "source_type": "review_summary",
                "is_verbatim": False,
            },
            {
                "place_id": "clinic-2",
                "text": "The consultation was unprofessional.",
                "source_type": "review_summary",
                "is_verbatim": False,
            },
        ]
        pipeline.specific_bm25_corpus = [
            ["the", "consultation", "was", "professional"],
            ["the", "consultation", "was", "unprofessional"],
        ]
        pipeline.specific_bm25 = TokenScoreIndex(pipeline.specific_bm25_corpus)

        result = pipeline.specific_bm25_search(
            "professional",
            exact_terms=["professional"],
        )

        self.assertEqual(result["facility_ids"], ["clinic-1"])
        self.assertEqual(result["matches"][0]["matched_terms"], ["professional"])

    def test_deterministic_agent_fuses_dense_and_specific_results(self):
        pipeline = object.__new__(RAGPipeline)
        pipeline.groq_client = None
        pipeline.vector_search = lambda query, n_results: {
            "ids": [["clinic-1", "clinic-2"]],
            "documents": [["General summary one", "General summary two"]],
            "distances": [[0.1, 0.2]],
        }
        pipeline.bm25_search = lambda query, n_results: {
            "ids": ["clinic-1"],
            "scores": [2.0],
        }
        pipeline.specific_bm25_search = lambda query, exact_terms, n_results, candidate_ids=None: {
            "matches": [{
                "place_id": "clinic-1",
                "text": "Parking is available next to the clinic.",
                "source_type": "review_summary",
                "matched_terms": ["parking"],
                "score": 3.0,
                "is_verbatim": False,
            }],
            "facility_ids": ["clinic-1"],
        }

        result = pipeline.agentic_search(
            "dentist with parking",
            exact_terms=["parking"],
            n_results=10,
        )

        self.assertEqual(result["method"], "agentic")
        self.assertEqual(result["ids"][0][0], "clinic-1")
        self.assertEqual(
            result["methods"]["clinic-1"],
            ["bm25_specific", "facility_bm25", "facility_semantic"],
        )
        self.assertEqual(
            result["evidence"]["clinic-1"][0]["matched_terms"],
            ["parking"],
        )
        self.assertEqual(result["matched_exact_terms"]["clinic-1"], ["parking"])
        self.assertEqual(result["trace"][0]["action"], "deterministic_hybrid")
        self.assertEqual(result["trace"][1]["action"], "finish_search")

    def test_function_calling_agent_iterates_and_translates_multilingual_comment(self):
        class FakeCompletions:
            def __init__(self):
                self.requests = []
                self.sequence = [
                    ("call-1", "search_facilities", {
                        "query": "친절하게 설명하는 산부인과",
                        "limit": 10,
                    }),
                    ("call-2", "search_multilingual_comments", {
                        "query_terms": [
                            "kind explanation", "친절", "설명", "clear explanation"
                        ],
                        "limit": 10,
                    }),
                    ("call-3", "select_comment_evidence", {
                        "selections": [{
                            "evidence_id": "review:one",
                            "translated_text": "The doctor explains things kindly and clearly.",
                            "relevance_reason": "Directly matches the requested explanation style.",
                        }],
                    }),
                    ("call-4", "assess_search_coverage", {
                        "satisfied_constraints": ["kind, clear explanation"],
                        "missing_constraints": [],
                        "evidence_sufficient": True,
                        "reason": "A translated verbatim comment supports the result.",
                    }),
                    ("call-5", "finish_search", {
                        "reason": "Every requested constraint has direct evidence.",
                    }),
                ]

            def create(self, **kwargs):
                self.requests.append(kwargs)
                call_id, name, arguments = self.sequence.pop(0)
                tool_call = types.SimpleNamespace(
                    id=call_id,
                    function=types.SimpleNamespace(
                        name=name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    ),
                )
                tool_calls = [tool_call]
                if call_id == "call-1":
                    tool_calls.append(types.SimpleNamespace(
                        id="premature-finish",
                        function=types.SimpleNamespace(
                            name="finish_search",
                            arguments=json.dumps({
                                "reason": "Issued before reading search results."
                            }),
                        ),
                    ))
                message = types.SimpleNamespace(content=None, tool_calls=tool_calls)
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=message)]
                )

        class FakeRawReviewStore:
            def search_comments(
                self, facility_ids, query_terms=None, limit=30
            ):
                self.last_request = {
                    "facility_ids": facility_ids,
                    "terms": query_terms,
                    "limit": limit,
                }
                return [{
                    "evidence_id": "review:one",
                    "place_id": "clinic-1",
                    "facility_name": "테스트의원",
                    "text": "의사 선생님이 친절하고 이해하기 쉽게 설명해 주세요.",
                    "language": "Korean",
                    "source_type": "verbatim_review",
                    "source_field": "review_text",
                    "source_index": 1,
                    "visit_date": "25.11.5.수",
                    "scraped_at": "2026-01-01",
                    "is_verbatim": True,
                    "matched_terms": ["친절", "설명"],
                    "score": 2.0,
                }]

        completions = FakeCompletions()
        pipeline = object.__new__(RAGPipeline)
        pipeline.groq_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=completions)
        )
        pipeline.raw_review_store = FakeRawReviewStore()
        pipeline.vector_search = lambda query, n_results: {
            "ids": [["clinic-1"]],
            "documents": [["Name: 테스트의원\nReview summaries: detailed explanations"]],
            "distances": [[0.1]],
        }
        pipeline.specific_bm25 = None
        pipeline.specific_evidence_records = []

        result = pipeline.agentic_search(
            "Find a gynecologist whose comments say she explains things kindly.",
            n_results=10,
            candidate_ids=["clinic-1"],
            target_language="English",
        )

        evidence = result["evidence"]["clinic-1"][0]
        self.assertTrue(evidence["is_verbatim"])
        self.assertEqual(evidence["language"], "Korean")
        self.assertEqual(
            evidence["translated_text"],
            "The doctor explains things kindly and clearly.",
        )
        self.assertIn("multilingual_comment_search", result["methods"]["clinic-1"])
        self.assertTrue(all("tools" in request for request in completions.requests))
        exposed_names = {
            tool["function"]["name"]
            for tool in completions.requests[0]["tools"]
        }
        self.assertIn("search_multilingual_comments", exposed_names)
        self.assertIn("assess_search_coverage", exposed_names)
        self.assertIn("select_comment_evidence", exposed_names)
        retrieval_actions = [
            item["action"]
            for item in result["trace"]
            if item.get("action") in {
                "search_facilities",
                "search_multilingual_comments",
                "select_comment_evidence",
                "assess_search_coverage",
                "finish_search",
            }
        ]
        self.assertEqual(retrieval_actions, [
            "search_facilities",
            "search_multilingual_comments",
            "select_comment_evidence",
            "assess_search_coverage",
            "finish_search",
        ])
        forced_names = [
            request["tool_choice"]["function"]["name"]
            for request in completions.requests
        ]
        self.assertEqual(forced_names, [
            "search_facilities",
            "search_multilingual_comments",
            "select_comment_evidence",
            "assess_search_coverage",
            "finish_search",
        ])

    def test_finish_is_rejected_until_coverage_is_assessed(self):
        class FakeCompletions:
            def __init__(self):
                self.sequence = [
                    ("call-1", "search_facilities", {
                        "query": "친절한 치과 kind dentist",
                        "limit": 10,
                    }),
                    ("call-2", "finish_search", {
                        "reason": "Stopping too early.",
                    }),
                    ("call-3", "assess_search_coverage", {
                        "satisfied_constraints": ["dentist", "kind"],
                        "missing_constraints": [],
                        "evidence_sufficient": True,
                        "reason": "Semantic and BM25 agree.",
                    }),
                    ("call-4", "finish_search", {
                        "reason": "Coverage is now assessed.",
                    }),
                ]

            def create(self, **kwargs):
                call_id, name, arguments = self.sequence.pop(0)
                call = types.SimpleNamespace(
                    id=call_id,
                    function=types.SimpleNamespace(
                        name=name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    ),
                )
                return types.SimpleNamespace(choices=[
                    types.SimpleNamespace(message=types.SimpleNamespace(
                        content=None,
                        tool_calls=[call],
                    ))
                ])

        pipeline = object.__new__(RAGPipeline)
        pipeline.groq_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=FakeCompletions())
        )
        pipeline.raw_review_store = None
        pipeline.vector_search = lambda query, n_results: {
            "ids": [["clinic-1"]],
            "documents": [["친절한 치과 / kind dentist"]],
            "distances": [[0.1]],
        }
        pipeline.bm25_search = lambda query, n_results: {
            "ids": ["clinic-1"],
            "scores": [1.0],
        }
        pipeline.specific_bm25 = None
        pipeline.specific_evidence_records = []

        result = pipeline.agentic_search(
            "kind dentist",
            candidate_ids=["clinic-1"],
            n_results=10,
        )

        finish_observations = [
            item for item in result["observations"]
            if item.get("tool") == "finish_search"
        ]
        self.assertEqual(len(finish_observations), 2)
        self.assertFalse(finish_observations[0]["finished"])
        self.assertIn("error", finish_observations[0])
        self.assertTrue(finish_observations[1]["finished"])
        self.assertIn("facility_semantic", result["methods"]["clinic-1"])
        self.assertIn("facility_bm25", result["methods"]["clinic-1"])

    def test_missing_constraints_override_a_model_success_claim(self):
        class FakeCompletions:
            def __init__(self):
                self.sequence = [
                    ("call-1", "search_facilities", {
                        "query": "kind dentist",
                        "limit": 10,
                    }),
                    ("call-2", "assess_search_coverage", {
                        "satisfied_constraints": ["dentist"],
                        "missing_constraints": ["kind patient comments"],
                        "evidence_sufficient": True,
                        "reason": "Contradictory model output.",
                    }),
                ]

            def create(self, **kwargs):
                del kwargs
                call_id, name, arguments = self.sequence.pop(0)
                call = types.SimpleNamespace(
                    id=call_id,
                    function=types.SimpleNamespace(
                        name=name,
                        arguments=json.dumps(arguments),
                    ),
                )
                return types.SimpleNamespace(choices=[
                    types.SimpleNamespace(message=types.SimpleNamespace(
                        content=None,
                        tool_calls=[call],
                    ))
                ])

        pipeline = object.__new__(RAGPipeline)
        pipeline.groq_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=FakeCompletions())
        )
        pipeline.raw_review_store = None
        pipeline.vector_search = lambda query, n_results: {
            "ids": [["clinic-1"]],
            "documents": [["A dentist"]],
            "distances": [[0.1]],
        }
        pipeline.bm25_search = lambda query, n_results: {
            "ids": ["clinic-1"],
            "scores": [1.0],
        }
        pipeline.specific_bm25 = None
        pipeline.specific_evidence_records = []

        result = pipeline.agentic_search(
            "kind dentist",
            candidate_ids=["clinic-1"],
            n_results=10,
            max_iterations=2,
        )

        assessment = result["observations"][-1]
        self.assertEqual(assessment["tool"], "assess_search_coverage")
        self.assertFalse(assessment["evidence_sufficient"])
        self.assertFalse(result["coverage_sufficient"])

    def test_literal_evidence_search_must_precede_successful_coverage(self):
        class FakeCompletions:
            def __init__(self):
                self.sequence = [
                    ("call-1", "search_facilities", {
                        "query": "dentist with Tuesday evening hours",
                        "limit": 10,
                    }),
                    ("call-2", "assess_search_coverage", {
                        "satisfied_constraints": ["Tuesday evening hours"],
                        "missing_constraints": [],
                        "evidence_sufficient": True,
                        "reason": "The semantic result looks relevant.",
                    }),
                    ("call-3", "finish_search", {
                        "reason": "Stopping before literal evidence.",
                    }),
                    ("call-4", "search_indexed_evidence", {
                        "query": "Tuesday evening hours 화요일 야간 진료",
                        "exact_terms": ["Tuesday evening hours"],
                    }),
                    ("call-5", "assess_search_coverage", {
                        "satisfied_constraints": ["Tuesday evening hours"],
                        "missing_constraints": [],
                        "evidence_sufficient": True,
                        "reason": "Literal evidence now supports the requirement.",
                    }),
                    ("call-6", "finish_search", {
                        "reason": "Coverage includes literal evidence.",
                    }),
                ]

            def create(self, **kwargs):
                del kwargs
                call_id, name, arguments = self.sequence.pop(0)
                call = types.SimpleNamespace(
                    id=call_id,
                    function=types.SimpleNamespace(
                        name=name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    ),
                )
                return types.SimpleNamespace(choices=[
                    types.SimpleNamespace(message=types.SimpleNamespace(
                        content=None,
                        tool_calls=[call],
                    ))
                ])

        pipeline = object.__new__(RAGPipeline)
        pipeline.groq_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=FakeCompletions())
        )
        pipeline.raw_review_store = None
        pipeline.vector_search = lambda query, n_results: {
            "ids": [["clinic-1"]],
            "documents": [["A dentist"]],
            "distances": [[0.1]],
        }
        pipeline.bm25_search = lambda query, n_results: {
            "ids": ["clinic-1"],
            "scores": [1.0],
        }
        pipeline.specific_bm25_search = (
            lambda query, exact_terms, n_results, candidate_ids=None: {
                "matches": [{
                    "place_id": "clinic-1",
                    "text": "Open Tuesday evening",
                    "source_type": "business_hours",
                    "source_field": "business_hours",
                    "matched_terms": ["Tuesday evening hours"],
                    "score": 1.0,
                    "is_verbatim": False,
                }],
                "facility_ids": ["clinic-1"],
            }
        )

        result = pipeline.agentic_search(
            "dentist with Tuesday evening hours",
            exact_terms=["Tuesday evening hours"],
            candidate_ids=["clinic-1"],
            n_results=10,
        )

        assessments = [
            item for item in result["observations"]
            if item.get("tool") == "assess_search_coverage"
        ]
        self.assertFalse(assessments[0]["evidence_sufficient"])
        self.assertTrue(assessments[1]["evidence_sufficient"])
        self.assertEqual(result["retrieval_status"], "complete")
        self.assertEqual(result["trace"][-1]["action"], "finish_search")

    def test_model_comment_ids_cannot_shrink_the_candidate_scope(self):
        class FakeCompletions:
            def __init__(self):
                self.sequence = [
                    ("call-1", "search_facilities", {
                        "query": "kind doctor with patient comments",
                        "limit": 10,
                    }),
                    ("call-2", "search_multilingual_comments", {
                        "facility_ids": ["clinic-0"],
                        "query_terms": ["kind", "친절"],
                        "limit": 10,
                    }),
                    ("call-3", "select_comment_evidence", {
                        "selections": [{
                            "evidence_id": "review:last",
                            "translated_text": "Kind care",
                            "relevance_reason": "Direct review support",
                        }],
                    }),
                    ("call-4", "assess_search_coverage", {
                        "satisfied_constraints": ["kind"],
                        "missing_constraints": [],
                        "evidence_sufficient": True,
                        "reason": "A scoped review supports the requirement.",
                    }),
                    ("call-5", "finish_search", {
                        "reason": "Coverage is complete.",
                    }),
                ]

            def create(self, **kwargs):
                call_id, name, arguments = self.sequence.pop(0)
                call = types.SimpleNamespace(
                    id=call_id,
                    function=types.SimpleNamespace(
                        name=name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    ),
                )
                return types.SimpleNamespace(choices=[
                    types.SimpleNamespace(message=types.SimpleNamespace(
                        content=None,
                        tool_calls=[call],
                    ))
                ])

        class CapturingRawReviewStore:
            def __init__(self):
                self.facility_ids = []

            def search_comments(self, facility_ids, query_terms=None, limit=30):
                self.facility_ids = list(facility_ids)
                place_id = self.facility_ids[-1]
                return [{
                    "evidence_id": "review:last",
                    "place_id": place_id,
                    "facility_name": "Last clinic",
                    "text": "친절한 진료",
                    "language": "Korean",
                    "source_type": "verbatim_review",
                    "source_field": "review_text",
                    "source_index": 1,
                    "visit_date": None,
                    "scraped_at": None,
                    "is_verbatim": True,
                    "matched_terms": ["친절"],
                    "score": 1.0,
                }]

        candidate_ids = [f"clinic-{index}" for index in range(700)]
        raw_store = CapturingRawReviewStore()
        pipeline = object.__new__(RAGPipeline)
        pipeline.groq_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=FakeCompletions())
        )
        pipeline.raw_review_store = raw_store
        pipeline.vector_search = lambda query, n_results: {
            "ids": [["clinic-0"]],
            "documents": [["Kind doctor"]],
            "distances": [[0.1]],
        }
        pipeline.bm25_search = lambda query, n_results: {
            "ids": ["clinic-0"],
            "scores": [1.0],
        }
        pipeline.specific_bm25 = None
        pipeline.specific_evidence_records = []

        result = pipeline.agentic_search(
            "kind doctor with patient comments",
            candidate_ids=candidate_ids,
            n_results=10,
        )

        self.assertEqual(raw_store.facility_ids, candidate_ids)
        self.assertIn(candidate_ids[-1], result["ids"][0])

    def test_no_tool_response_cannot_bypass_coverage_assessment(self):
        class FakeCompletions:
            def __init__(self):
                self.requests = []
                self.sequence = [
                    ("call-1", "search_facilities", {
                        "query": "kind dentist 친절한 치과",
                        "limit": 10,
                    }),
                    None,
                    ("call-3", "assess_search_coverage", {
                        "satisfied_constraints": ["kind dentist"],
                        "missing_constraints": [],
                        "evidence_sufficient": True,
                        "reason": "The result has direct support.",
                    }),
                    ("call-4", "finish_search", {
                        "reason": "Coverage is complete.",
                    }),
                ]

            def create(self, **kwargs):
                self.requests.append(kwargs)
                next_response = self.sequence.pop(0)
                if next_response is None:
                    message = types.SimpleNamespace(
                        content="I have enough information.",
                        tool_calls=[],
                    )
                else:
                    call_id, name, arguments = next_response
                    call = types.SimpleNamespace(
                        id=call_id,
                        function=types.SimpleNamespace(
                            name=name,
                            arguments=json.dumps(arguments, ensure_ascii=False),
                        ),
                    )
                    message = types.SimpleNamespace(content=None, tool_calls=[call])
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=message)]
                )

        completions = FakeCompletions()
        pipeline = object.__new__(RAGPipeline)
        pipeline.groq_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=completions)
        )
        pipeline.raw_review_store = None
        pipeline.vector_search = lambda query, n_results: {
            "ids": [["clinic-1"]],
            "documents": [["Kind dentist"]],
            "distances": [[0.1]],
        }
        pipeline.bm25_search = lambda query, n_results: {
            "ids": ["clinic-1"],
            "scores": [1.0],
        }
        pipeline.specific_bm25 = None
        pipeline.specific_evidence_records = []

        result = pipeline.agentic_search(
            "kind dentist",
            candidate_ids=["clinic-1"],
            n_results=10,
        )

        self.assertEqual(len(completions.requests), 4)
        self.assertIn(
            "assistant_continue",
            [entry["action"] for entry in result["trace"]],
        )
        self.assertEqual(result["trace"][-1]["action"], "finish_search")

    def test_iteration_limit_is_reported_as_incomplete(self):
        class FakeCompletions:
            def create(self, **kwargs):
                del kwargs
                message = types.SimpleNamespace(
                    content="I am still thinking.",
                    tool_calls=[],
                )
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=message)]
                )

        pipeline = object.__new__(RAGPipeline)
        pipeline.groq_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=FakeCompletions())
        )
        pipeline.raw_review_store = None
        pipeline.vector_search = lambda query, n_results: {
            "ids": [["clinic-1"]],
            "documents": [["Kind dentist"]],
            "distances": [[0.1]],
        }
        pipeline.bm25_search = lambda query, n_results: {
            "ids": ["clinic-1"],
            "scores": [1.0],
        }
        pipeline.specific_bm25 = None
        pipeline.specific_evidence_records = []

        result = pipeline.agentic_search(
            "kind dentist",
            candidate_ids=["clinic-1"],
            n_results=10,
            max_iterations=2,
        )

        self.assertEqual(result["retrieval_status"], "incomplete")
        self.assertFalse(result["coverage_assessed"])
        self.assertFalse(result["coverage_sufficient"])
        self.assertEqual(result["termination_reason"], "iteration_limit")
        self.assertEqual(result["candidate_scope_count"], 1)
        self.assertEqual(result["trace"][-1]["action"], "retrieval_incomplete")

    def test_candidate_scope_count_records_the_enforced_cap(self):
        pipeline = object.__new__(RAGPipeline)
        pipeline.groq_client = None
        pipeline.raw_review_store = None
        pipeline.vector_search = lambda query, n_results: {
            "ids": [["clinic-1"]],
            "documents": [["Kind dentist"]],
            "distances": [[0.1]],
        }
        pipeline.bm25_search = lambda query, n_results: {
            "ids": ["clinic-1"],
            "scores": [1.0],
        }
        pipeline.specific_bm25 = None
        pipeline.specific_evidence_records = []

        result = pipeline.agentic_search(
            "kind dentist",
            candidate_ids=[f"clinic-{index}" for index in range(10_001)],
            n_results=10,
        )

        self.assertEqual(result["candidate_scope_count"], 10_000)
        self.assertEqual(result["retrieval_status"], "deterministic_fallback")

if __name__ == "__main__":
    unittest.main()
