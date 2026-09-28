"""Vercel serverless entry point (catch-all).

A function at ``api/index.py`` only answers on ``/api/index``. The dashboard
calls ``/api/scan``, ``/api/backtest`` and friends, so the entry point has to
be a catch-all named ``[...path]`` -- that is what preserves the original path
and lets FastAPI route on it. The real application lives in
``po_coach/server.py``; this file only exists because the platform insists on
the ``api/`` convention.
"""
from po_coach.server import app  # noqa: F401

__all__ = ["app"]
