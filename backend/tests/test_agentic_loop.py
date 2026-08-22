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
        pipeline.specific_bm25_search = lambda query, exact_terms, n_results: {
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
            ["bm25_specific", "dense_general"],
        )
        self.assertEqual(
            result["evidence"]["clinic-1"][0]["matched_terms"],
            ["parking"],
        )
        self.assertEqual(result["matched_exact_terms"]["clinic-1"], ["parking"])
        self.assertEqual(result["trace"][0]["action"], "hybrid")
        self.assertEqual(result["trace"][1]["action"], "finish")


if __name__ == "__main__":
    unittest.main()
