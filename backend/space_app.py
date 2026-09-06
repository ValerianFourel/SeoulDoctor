"""Hugging Face Docker Space entry point.

The API routes remain on the FastAPI application. The exported Next.js site is
mounted last, so it handles only paths that were not claimed by the API.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles
import requests

from main import app, root_status


frontend_directory = Path(
    os.getenv(
        "FRONTEND_STATIC_DIR",
        Path(__file__).resolve().parent.parent / "frontend" / "out",
    )
).resolve()

index_file = frontend_directory / "index.html"
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
    @app.get("/ready/gpu")
    def gpu_readiness():
        try:
            response = requests.get("http://127.0.0.1:7861/ready/gpu", timeout=10)
            response.raise_for_status()
        except requests.RequestException:
            raise HTTPException(503, "GPU retrieval service is not ready") from None
        return response.json()


app.mount(
    "/",
    StaticFiles(directory=str(frontend_directory), html=True),
    name="frontend",
)
