import os
from unittest import TestCase
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx

from backend.inference_gateway import router


class InferenceGatewayTests(TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)
        self.environment = patch.dict(os.environ, {"HF_TOKEN": "fixture-secret"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_unauthorized_requests_never_reach_gpu(self):
        with patch("backend.inference_gateway.httpx.AsyncClient") as transport:
            result = self.client.post("/internal/inference/rerank", json={})
        self.assertEqual(result.status_code, 403)
        transport.assert_not_called()

    def test_authenticated_request_uses_fixed_loopback_path(self):
        payload = {"query": "발목", "texts": ["발목 치료 후기"]}
        response = httpx.Response(200, json=[{"index": 0, "score": 0.8}],
                                  request=httpx.Request("POST", "http://loopback/rerank"))
        with patch("backend.inference_gateway.httpx.AsyncClient") as transport:
            session = transport.return_value.__aenter__.return_value
            session.request = AsyncMock(return_value=response)
            result = self.client.post("/internal/inference/rerank", json=payload,
                                      headers={"Authorization": "Bearer fixture-secret"})
        self.assertEqual(result.json(), [{"index": 0, "score": 0.8}])
        self.assertEqual(session.request.call_args.args[:2], ("POST", "http://127.0.0.1:7861/rerank"))
        self.assertNotIn("Authorization", session.request.call_args.kwargs["headers"])

    def test_upstream_failure_does_not_expose_internal_details(self):
        with patch("backend.inference_gateway.httpx.AsyncClient") as transport:
            session = transport.return_value.__aenter__.return_value
            session.request = AsyncMock(side_effect=httpx.ConnectError("private diagnostic"))
            result = self.client.get("/internal/inference/health",
                                     headers={"Authorization": "Bearer fixture-secret"})
        self.assertEqual(result.status_code, 503)
        self.assertNotIn("private diagnostic", result.text)

    def test_oversized_body_is_rejected_before_forwarding(self):
        with patch("backend.inference_gateway.httpx.AsyncClient") as transport:
            result = self.client.post("/internal/inference/v1/retrieve", content=b"x" * 1_000_001,
                                      headers={"Authorization": "Bearer fixture-secret"})
        self.assertEqual(result.status_code, 413)
        transport.assert_not_called()
