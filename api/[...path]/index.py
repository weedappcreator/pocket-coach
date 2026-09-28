"""Vercel serverless entry point.

Routing notes, learned the hard way:

* `api/index.py` only answers on `/api/index`, so a catch-all is required.
* The catch-all must be the **directory** form `api/[...path]/index.py`; the
  flat form matched only one path segment, so `/api/scan` resolved while
  `/api/journal/import` 404ed at the edge with a Vercel error page.
* `rewrites` is required in addition, and must catch `/api/` as well. Vercel
  gives registered function routes priority, so a rewrite that excludes
  `/api/` (the obvious-looking negative lookahead) leaves multi-segment API
  calls unrouted.

The real application lives in `po_coach/server.py`; this file exists only
because the platform insists on the `api/` convention.
"""
from po_coach.server import app  # noqa: F401

__all__ = ["app"]
