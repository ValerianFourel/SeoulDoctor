import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path

import requests


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from patient_journey import create_journey, finish_journey, run_journey, say  # noqa: E402


class Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.headers = {
            "Content-Type": "application/json",
            "Set-Cookie": "private-cookie",
        }

    def json(self):
        return self._payload


class Session:
    def __init__(self):
        self.posts = []

    def get(self, *args, **kwargs):
        del args, kwargs
        return Response({
            "status": "ok",
            "facilities": 8_484,
            "vector_documents": 8_484,
            "raw_reviews": 1_791_749,
            "model_provider": "openrouter",
            "model": "openai/gpt-oss-120b",
            "agent_model": "openai/gpt-oss-120b",
        })
    def post(self, endpoint, **kwargs):
        self.posts.append((endpoint, kwargs))
        body = kwargs["json"]
        next_state = dict(body["current_state"])
        next_state["turn_count"] += 1
        return Response({
            "response": "Here are two facilities.",
            "state": next_state,
            "results": [{"place_id": "clinic-1", "name": "Clinic"}],
        })


class TimeoutSession(Session):
    def post(self, endpoint, **kwargs):
        del endpoint, kwargs
        raise requests.Timeout("synthetic timeout")


class PatientJourneyTests(unittest.TestCase):
    def test_state_is_custodied_and_full_turn_is_saved_owner_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visit.json"
            session = Session()
            create_journey(
                path,
                "http://127.0.0.1:17860",
                language="English",
                session=session,
            )
            visible = say(path, "I need a dentist.", session=session)
            finish_journey(path, "I have a useful shortlist.")

            artifact = json.loads(path.read_text(encoding="utf-8"))
            mode = stat.S_IMODE(path.stat().st_mode)

        self.assertEqual(visible["response"], "Here are two facilities.")
        self.assertEqual(visible["results"][0]["place_id"], "clinic-1")
        self.assertEqual(session.posts[0][0], "http://127.0.0.1:17860/chat")
        self.assertNotIn("state", visible)
        self.assertEqual(session.posts[0][1]["json"]["current_state"]["turn_count"], 0)
        self.assertEqual(artifact["current_state"]["turn_count"], 1)
        self.assertEqual(
            artifact["turns"][0]["request"]["message"],
            "I need a dentist.",
        )
        self.assertEqual(
            artifact["turns"][0]["response"]["body"]["results"][0]["name"],
            "Clinic",
        )
        self.assertNotIn("set-cookie", artifact["turns"][0]["response"]["headers"])
        self.assertEqual(artifact["status"], "finished")
        self.assertEqual(mode, 0o600)

    def test_run_journey_serializes_turns_in_one_process(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visit.json"
            session = Session()
            artifact = run_journey(
                path,
                "http://127.0.0.1:17860",
                language="Korean",
                messages=("첫 번째 요청", "두 번째 요청"),
                session=session,
            )

        self.assertEqual(artifact["status"], "finished")
        self.assertEqual(len(artifact["turns"]), 2)
        self.assertEqual(session.posts[0][1]["json"]["current_state"]["turn_count"], 0)
        self.assertEqual(session.posts[1][1]["json"]["current_state"]["turn_count"], 1)

    def test_transport_failure_is_saved_with_the_attempted_message(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visit.json"
            create_journey(
                path,
                "http://127.0.0.1:17860",
                language="English",
                session=TimeoutSession(),
            )
            with self.assertRaises(RuntimeError):
                say(
                    path,
                    "Keep this attempted request.",
                    session=TimeoutSession(),
                )
            artifact = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(artifact["status"], "transport_error")
        self.assertEqual(len(artifact["turns"]), 1)
        self.assertEqual(
            artifact["turns"][0]["request"]["message"],
            "Keep this attempted request.",
        )
        self.assertEqual(
            artifact["turns"][0]["response"]["transport_error"]["type"],
            "Timeout",
        )


if __name__ == "__main__":
    unittest.main()
