"""Hugging Face Docker Space entry point.

The API routes remain on the FastAPI application. The exported Next.js site is
mounted last, so it handles only paths that were not claimed by the API.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi.staticfiles import StaticFiles

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
app.mount(
    "/",
    StaticFiles(directory=str(frontend_directory), html=True),
    name="frontend",
)
