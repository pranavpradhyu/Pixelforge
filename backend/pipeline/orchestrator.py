"""
Orchestrator: the end-to-end restoration pipeline.

    analyze -> route -> run applicable candidates -> score -> guard -> pick best

Candidates run on a size-capped working copy for speed. Deep families that
support it (e.g. upscalers) are re-run on the native image only if they win, so
the user gets full resolution without paying for it on every branch.
"""
import time

import cv2
import numpy as np

from config import Config
from backend.pipeline.analyzer import analyze
from backend.pipeline.registry import registry
from backend.pipeline.selector import select


def _working_copy(rgb):
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest <= Config.WORKING_MAX_EDGE:
        return rgb, 1.0
    scale = Config.WORKING_MAX_EDGE / longest
    small = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return small, scale


def run(rgb, intent: str = "auto", progress=None) -> dict:
    """
    intent biases routing: 'auto' | 'oldphoto' | 'lowlight' | 'product' |
    'upscale' | 'cutout'. Returns a result dict; only 'output' + 'analysis'
    summary are meant to reach the user, the rest is for research/debug mode.
    """
    def tick(pct, msg):
        if progress:
            progress(pct, msg)

    t0 = time.time()
    tick(5, "Inspecting the photo")
    analysis = analyze(rgb)

    routes = _apply_intent(analysis.routes, intent)
    restorers = registry.available_for(routes)
    if not restorers:
        restorers = [registry.get("passthrough")]

    work, _ = _working_copy(rgb)

    candidates = []
    total = len(restorers)
    for i, r in enumerate(restorers):
        tick(10 + int(70 * (i / max(total, 1))), "Trying restoration paths")
        try:
            out = r.apply(work)
            candidates.append({"key": r.key, "label": r.label, "family": r.family, "image": out})
        except Exception as exc:  # a failing candidate never sinks the request
            candidates.append({"key": r.key, "label": r.label, "family": r.family,
                               "image": work.copy(), "error": str(exc)})

    tick(85, "Scoring results")
    decision = select(work, candidates)
    winner = decision["winner"]

    # Re-run the winner at native resolution when it makes sense.
    final = _finalize(rgb, work, winner)

    tick(100, "Done")
    return {
        "output": final,
        "winner_key": winner["key"],
        "winner_label": winner["label"],
        "elapsed_s": round(time.time() - t0, 2),
        "analysis": _analysis_summary(analysis),
        "ranking": [
            {"key": r["key"], "label": r["label"], "family": r["family"],
             "score": round(r["score"], 4), "fidelity": round(r["fidelity"], 4),
             "passes_guard": r["passes_guard"]}
            for r in decision["ranking"]
        ],
    }


def _finalize(native_rgb, work_rgb, winner):
    key = winner["key"]
    r = registry.get(key)
    # Upscalers / passthrough: re-apply to native for full quality.
    if r and winner["family"] in {"upscale", "identity", "matting"}:
        try:
            return r.apply(native_rgb)
        except Exception:
            pass
    # Otherwise upsample the winning working-size result back to native size so
    # the user keeps their original resolution.
    h, w = native_rgb.shape[:2]
    if winner["image"].shape[:2] != (h, w):
        return cv2.resize(winner["image"], (w, h), interpolation=cv2.INTER_LANCZOS4)
    return winner["image"]


def _apply_intent(routes, intent):
    intent = (intent or "auto").lower()
    if intent == "auto":
        return routes
    boosts = {
        "oldphoto": ["denoise_nlm", "denoise_deep", "sharpen_unsharp", "white_balance",
                     "face_restore", "colorize", "tone_recover"],
        "lowlight": ["lowlight_clahe", "lowlight_deep", "tone_recover", "denoise_nlm"],
        "product": ["white_balance", "sharpen_unsharp", "cutout", "tone_recover"],
        "upscale": ["upscale_deep", "upscale_lanczos", "sharpen_unsharp"],
        "cutout": ["cutout"],
    }
    forced = boosts.get(intent, [])
    merged = ["passthrough"] + forced + [r for r in routes if r not in forced]
    seen, out = set(), []
    for r in merged:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _analysis_summary(a):
    tags = []
    if a.is_low_light:
        tags.append("low light")
    if a.is_overexposed:
        tags.append("overexposed")
    if a.is_noisy:
        tags.append("noisy")
    if a.is_blurry:
        tags.append("soft focus")
    if a.is_low_res:
        tags.append("low resolution")
    if a.face_count:
        tags.append(f"{a.face_count} face(s)")
    if a.is_grayscale:
        tags.append("black & white")
    return {"dimensions": f"{a.width}x{a.height}", "detected": tags or ["looks clean"]}
