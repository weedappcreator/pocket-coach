"""Vercel serverless entry point.

Vercel's Python runtime picks up an ASGI callable exported as ``app`` from
``api/index.py``. Everything real lives in ``po_coach/server.py``; this file
only exists because the platform insists on that path.
"""
from po_coach.server import app  # noqa: F401

__all__ = ["app"]
