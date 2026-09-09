import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
import requests

ROOT = Path(__file__).resolve().parents[2]


class SpaceDeploymentTests(unittest.TestCase):
    def test_container_has_a_start_command_separate_from_healthcheck(self):
        lines = (ROOT / 'Dockerfile').read_text().splitlines()
        commands = [line for line in lines if line.startswith('CMD ')]
        self.assertEqual(commands, ['CMD ["python", "space_runtime.py"]'])
        health = next(i for i, line in enumerate(lines) if line.startswith('HEALTHCHECK '))
        self.assertIn('urlopen', lines[health + 1])
        self.assertNotIn('uvicorn', lines[health + 1])

    def test_website_and_api_coexist_with_gpu_readiness(self):
        app = FastAPI()
        @app.get('/')
        def root_status():
            return {'status': 'api-only'}
        @app.get('/health')
        def health():
            return {'status': 'ok'}
        @app.post('/chat')
        def chat():
            return {'reply': 'existing API'}

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'index.html').write_text('<html>SeoulDoctor fixture</html>')
            source_file = Path(directory, 'ncs-source.json')
            source_file.write_text('{"branch":"ncs","commit":"fixture"}')
            env = {'FRONTEND_STATIC_DIR': directory, 'NCS_ENABLE_GPU': 'true',
                   'NCS_SOURCE_FILE': str(source_file)}
            with patch.dict(os.environ, env), patch.dict(sys.modules, {
                'main': SimpleNamespace(app=app, root_status=root_status),
                'mini_retrieval': SimpleNamespace(router=APIRouter()),
                'inference_gateway': SimpleNamespace(router=APIRouter()),
            }):
                spec = importlib.util.spec_from_file_location('isolated_space_app', ROOT / 'backend/space_app.py')
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            with TestClient(app) as client:
                response = client.get('/')
                self.assertEqual(response.status_code, 200)
                self.assertIn('text/html', response.headers['content-type'])
                self.assertIn('SeoulDoctor fixture', response.text)
                self.assertEqual(client.get('/health').json(), {'status': 'ok'})
                self.assertEqual(client.post('/chat').json(), {'reply': 'existing API'})
                self.assertEqual(client.get('/ncs-source.json').json(),
                                 {'branch': 'ncs', 'commit': 'fixture'})
                with patch.object(module.requests, 'get') as get:
                    get.return_value.json.return_value = {'ready': True, 'device': 'cuda:0'}
                    self.assertEqual(client.get('/ready/gpu').json()['device'], 'cuda:0')
                    get.assert_called_once_with('http://127.0.0.1:7861/ready/gpu', timeout=10)
                with patch.object(module.requests, 'get', side_effect=requests.ConnectionError):
                    self.assertEqual(client.get('/ready/gpu').status_code, 503)
                    self.assertEqual(client.get('/').status_code, 200)

    def test_gpu_service_is_internal_and_website_is_public(self):
        spec = importlib.util.spec_from_file_location('isolated_space_runtime', ROOT / 'backend/space_runtime.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'restore_release': SimpleNamespace(main=lambda: None)}):
            spec.loader.exec_module(module)
        gpu, web = module.service_commands(True)
        self.assertEqual(gpu[0][gpu[0].index('--host') + 1], '127.0.0.1')
        self.assertEqual(gpu[0][gpu[0].index('--port') + 1], '7861')
        self.assertEqual(web[0][web[0].index('--port') + 1], '7860')
        self.assertEqual(module.service_commands(False), [web])
        for command, _ in (gpu, web):
            self.assertIn('--no-access-log', command)
