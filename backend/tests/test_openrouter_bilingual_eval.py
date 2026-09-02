import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from run_openrouter_bilingual_eval import (  # noqa: E402
    SimulatorRequestError,
    atomic_write_json,
    load_scenarios,
    parse_decision,
    post_chat,
    request_simulator_decision,
    run_scenario,
    select_scenarios,
)


def decision(
    action="continue",
    message="next",
    goal_satisfied=False,
):
    return {
        "action": action,
        "message": message,
        "goal_satisfied": goal_satisfied,
        "criteria_met": [],
        "unmet_constraints": ["more evidence"] if not goal_satisfied else [],
        "observed_evidence": [],
        "rationale": "keep looking" if not goal_satisfied else "grounded match",
    }


def scenario():
    return {
        "id": "en-test-01",
        "source_language": "English",
        "user_persona": "A patient",
        "opening_query": "Find a dentist in Mapo.",
        "hidden_constraints": ["Ask for comments next."],
        "expected_retrieval_behaviors": ["Use semantic and BM25 retrieval."],
        "success_criteria": ["Return a grounded doctor."],
        "counterpart_id": "ko-test-01",
    }


class FakeProviderError(Exception):
    status_code = 400


class FakeCompletionResponse:
    def __init__(self, payload):
        self.payload = payload
        self.choices = [
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(payload, ensure_ascii=False)
                )
            )
        ]

    def model_dump(self, mode="json"):
        del mode
        return {
            "id": "generation-1",
            "choices": [{"message": {"content": json.dumps(self.payload)}}],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
            },
        }


class BilingualEvalTests(unittest.TestCase):
    def test_load_and_select_scenarios(self):
        korean = scenario()
        korean["id"] = "ko-test-01"
        korean["source_language"] = "Korean"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenarios.json"
            path.write_text(
                json.dumps({"scenarios": [scenario(), korean]}),
                encoding="utf-8",
            )
            loaded = load_scenarios(path)

        selected = select_scenarios(loaded, ["ko-test-01", "en-test-01"])

        self.assertEqual(
            [item["id"] for item in selected],
            ["ko-test-01", "en-test-01"],
        )

    def test_parse_decision_accepts_fenced_json(self):
        parsed = parse_decision(
            "```json\n" + json.dumps(decision()) + "\n```"
        )

        self.assertEqual(parsed["action"], "continue")
        self.assertEqual(parsed["message"], "next")

    def test_structured_output_falls_back_to_json_object(self):
        calls = []

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    raise FakeProviderError("json_schema is unsupported")
                return FakeCompletionResponse(decision())

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        )
        result = request_simulator_decision(
            client,
            "qwen/qwen3.8-max",
            [{"role": "user", "content": "decide"}],
        )

        self.assertEqual(result["response_mode"], "json_object")
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(calls[0]["response_format"]["type"], "json_schema")
        self.assertEqual(calls[1]["response_format"]["type"], "json_object")
        self.assertEqual(result["usage"]["total_tokens"], 30)

    def test_non_compatibility_provider_error_does_not_retry(self):
        calls = []

        class AuthError(Exception):
            status_code = 401

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                raise AuthError("unauthorized")

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        )
        with self.assertRaises(SimulatorRequestError) as raised:
            request_simulator_decision(
                client,
                "qwen/qwen3.8-max",
                [{"role": "user", "content": "decide"}],
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(raised.exception.attempts), 1)

    def test_post_chat_preserves_raw_json_and_only_safe_headers(self):
        returned_state = {"turn_count": 1, "language_pref": "English"}

        class Response:
            status_code = 200
            headers = {
                "Content-Type": "application/json",
                "X-Request-Id": "request-1",
                "Set-Cookie": "session=secret",
                "Authorization": "Bearer secret-secret-secret",
            }
            text = ""

            def json(self):
                return {
                    "response": "Which neighborhood?",
                    "state": returned_state,
                    "results": [],
                }

        class Session:
            def __init__(self):
                self.calls = []

            def post(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return Response()

        session = Session()
        result = post_chat(
            session,
            "https://example.test/chat",
            "Find a dentist",
            {"turn_count": 0},
            5,
        )

        self.assertEqual(result["response"]["raw_json"]["state"], returned_state)
        self.assertEqual(
            result["response"]["headers"],
            {
                "content-type": "application/json",
                "x-request-id": "request-1",
            },
        )
        self.assertNotIn("headers", result["request"])

    def test_post_chat_honors_one_retry_after_response(self):
        class Response:
            text = ""

            def __init__(self, status_code, headers, payload):
                self.status_code = status_code
                self.headers = headers
                self.payload = payload

            def json(self):
                return self.payload

        class Session:
            def __init__(self):
                self.responses = iter(
                    [
                        Response(429, {"Retry-After": "2"}, {"detail": "slow down"}),
                        Response(
                            200,
                            {"Content-Type": "application/json"},
                            {"response": "ready", "state": {}, "results": []},
                        ),
                    ]
                )
                self.calls = 0

            def post(self, *args, **kwargs):
                del args, kwargs
                self.calls += 1
                return next(self.responses)

        session = Session()
        waits = []
        result = post_chat(
            session,
            "https://example.test/chat",
            "Find a dentist",
            {"turn_count": 0},
            5,
            sleep=waits.append,
        )

        self.assertEqual(session.calls, 2)
        self.assertEqual(waits, [2.0])
        self.assertEqual(result["response"]["status_code"], 200)

    def test_run_scenario_carries_app_state_between_turns(self):
        simulator_decisions = iter(
            [
                decision(message="ignored first draft"),
                decision(message="Patient comments too, please."),
                decision(action="stop", message="", goal_satisfied=True),
            ]
        )
        observed_states = []
        checkpoints = []

        def fake_decision_requester(
            client,
            model,
            messages,
            first_turn_opening=None,
        ):
            del client, model, messages
            item = next(simulator_decisions)
            if first_turn_opening is not None:
                item["message"] = first_turn_opening
            return {
                "model": "qwen/qwen3.8-max",
                "decision": item,
                "request_messages": [],
                "attempts": [],
                "usage": {"total_tokens": 1},
            }

        def fake_chat_poster(session, endpoint, message, state, timeout):
            del session, endpoint, timeout
            observed_states.append(dict(state))
            next_state = dict(state)
            next_state["turn_count"] = len(observed_states)
            return {
                "request": {
                    "endpoint": "https://example.test/chat",
                    "body": {"message": message, "current_state": state},
                },
                "response": {
                    "status_code": 200,
                    "headers": {},
                    "raw_json": {
                        "response": f"reply {len(observed_states)}",
                        "state": next_state,
                        "results": [{"name": "Clinic"}],
                    },
                    "raw_text": None,
                },
                "started_at": "2026-01-01T00:00:00+00:00",
                "elapsed_seconds": 0.01,
                "error": None,
            }

        result = run_scenario(
            scenario(),
            client=object(),
            model="qwen/qwen3.8-max",
            session=object(),
            endpoint="https://example.test/chat",
            max_turns=3,
            timeout=5,
            on_progress=lambda partial: checkpoints.append(
                json.loads(json.dumps(partial))
            ),
            decision_requester=fake_decision_requester,
            chat_poster=fake_chat_poster,
        )

        self.assertEqual(len(result["turns"]), 2)
        self.assertEqual(observed_states[0]["turn_count"], 0)
        self.assertEqual(observed_states[1]["turn_count"], 1)
        self.assertEqual(
            result["turns"][0]["patient_message"],
            scenario()["opening_query"],
        )
        self.assertEqual(
            result["turns"][0]["app"]["response"]["raw_json"]["results"],
            [{"name": "Clinic"}],
        )
        self.assertEqual(result["status"], "satisfied")
        self.assertTrue(result["goal_satisfied"])
        self.assertGreaterEqual(len(checkpoints), 4)

    def test_atomic_checkpoint_redacts_secret_material(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            atomic_write_json(
                path,
                {
                    "Authorization": "Bearer abcdefghijklmnop",
                    "nested": {
                        "api_key": "sk-or-v1-abcdefghijklmnop",
                        "message": "token sk-or-v1-qrstuvwxyz123456",
                    },
                },
            )
            raw = path.read_text(encoding="utf-8")
            parsed = json.loads(raw)
            mode = stat.S_IMODE(path.stat().st_mode)

        self.assertNotIn("abcdefghijklmnop", raw)
        self.assertNotIn("qrstuvwxyz123456", raw)
        self.assertEqual(parsed["Authorization"], "[REDACTED]")
        self.assertEqual(parsed["nested"]["api_key"], "[REDACTED]")
        self.assertEqual(mode, 0o600)


if __name__ == "__main__":
    unittest.main()
