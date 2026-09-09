"""Hugging Face Docker Space entry point.

The API routes remain on the FastAPI application. The exported Next.js site is
mounted last, so it handles only paths that were not claimed by the API.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles
import requests

from main import app, root_status
from mini_retrieval import router as mini_retrieval_router

app.include_router(mini_retrieval_router)


frontend_directory = Path(
    os.getenv(
        "FRONTEND_STATIC_DIR",
        Path(__file__).resolve().parent.parent / "frontend" / "out",
    )
).resolve()

index_file = frontend_directory / "index.html"
ncs_source_file = Path(os.getenv(
    "NCS_SOURCE_FILE", Path(__file__).resolve().parent.parent / "ncs-source.json",
)).resolve()
if not index_file.is_file():
    raise RuntimeError(
        f"Static frontend not found at {index_file}. Build it with npm run build."
    )

# The API's diagnostic root is replaced by the Space UI. All other registered
# routes, including /chat, /health, /docs, and /openapi.json, stay unchanged.
app.router.routes = [
    route
    for route in app.router.routes
    if getattr(route, "endpoint", None) is not root_status
]
if os.environ.get("NCS_ENABLE_GPU") == "true":
    from inference_gateway import router as inference_router
    app.include_router(inference_router)
    @app.get("/ready/gpu")
    def gpu_readiness():
        try:
            response = requests.get("http://127.0.0.1:7861/ready/gpu", timeout=10)
            response.raise_for_status()
        except requests.RequestException:
            raise HTTPException(503, "GPU retrieval service is not ready") from None
        return response.json()


@app.get("/ncs-source.json")
def ncs_source():
    if not ncs_source_file.is_file():
        raise HTTPException(503, "NCS source marker is unavailable")
    try:
        source = json.loads(ncs_source_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise HTTPException(503, "NCS source marker is invalid") from None
    if source.get("branch") != "ncs" or not isinstance(source.get("commit"), str):
        raise HTTPException(503, "NCS source marker is invalid")
    return source


app.mount(
    "/",
    StaticFiles(directory=str(frontend_directory), html=True),
    name="frontend",
)
