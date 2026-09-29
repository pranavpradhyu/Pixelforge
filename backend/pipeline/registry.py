"""
Model registry.

Restorers register under one or more route keys. Deep models can share a route
with a classical fallback (e.g. 'upscale_deep' and 'upscale_lanczos' both serve
the "needs upscaling" branch); the selector decides which output actually ships.
"""
from .models.classical import CLASSICAL
from .models.deep import DEEP


class Registry:
    def __init__(self):
        self._by_key = {}
        for r in CLASSICAL + DEEP:
            self._by_key[r.key] = r

    def get(self, key):
        return self._by_key.get(key)

    def available_for(self, routes):
        """Return the runnable restorers for a routing plan, in plan order."""
        out = []
        for key in routes:
            r = self._by_key.get(key)
            if r is not None and r.available:
                out.append(r)
        return out

    def catalogue(self):
        """All registered restorers with availability — for /api/models."""
        return [
            {"key": r.key, "label": r.label, "family": r.family, "available": r.available}
            for r in self._by_key.values()
        ]


registry = Registry()
