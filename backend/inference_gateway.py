"""Authenticated access to the Space's existing loopback inference service."""

import hmac
import os
from typing import Literal

import httpx
from fastapi import APIRouter, Header, HTTPException, Request

router = APIRouter(prefix="/internal/inference")


def authorize(authorization: str) -> None:
    expected = os.environ.get("SEOULDOC_INFERENCE_API_TOKEN") or os.environ.get("HF_TOKEN", "")
    if not expected or not hmac.compare_digest(authorization, "Bearer " + expected):
        raise HTTPException(403, "Inference credentials required")


async def forward(request: Request, path: Literal["/health", "/info", "/v1/retrieve", "/rerank"]):
    body = await request.body()
    if len(body) > 1_000_000:
        raise HTTPException(413, "Inference request is too large")
    try:
        async with httpx.AsyncClient(timeout=90, trust_env=False) as client:
            response = await client.request(request.method, "http://127.0.0.1:7861" + path,
                                            content=body, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError):
        raise HTTPException(503, "Inference service unavailable") from None


@router.get("/health")
async def health(request: Request, authorization: str = Header(default="")):
    authorize(authorization)
    return await forward(request, "/health")


@router.get("/info")
async def info(request: Request, authorization: str = Header(default="")):
    authorize(authorization)
    return await forward(request, "/info")


@router.post("/v1/retrieve")
async def retrieve(request: Request, authorization: str = Header(default="")):
    authorize(authorization)
    return await forward(request, "/v1/retrieve")


@router.post("/rerank")
async def rerank(request: Request, authorization: str = Header(default="")):
    authorize(authorization)
    return await forward(request, "/rerank")
