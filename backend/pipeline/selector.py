"""
Selection: given the input and a set of candidate outputs, choose the one to
ship. Winner = highest no-reference quality score, subject to a fidelity guard
that rejects candidates which no longer structurally resemble the input.

The guard is what makes automatic model selection safe for real use: without it
an aggressive sharpener or a hallucinating GAN could "win" on a raw quality
metric while destroying the actual content. For forensic / archival use this
guard is the feature, not a footnote.
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from config import Config
from backend.iqa.metrics import quality


def _fidelity(input_rgb, candidate_rgb) -> float:
    """Structural similarity of a candidate to the input, size-invariant."""
    h, w = input_rgb.shape[:2]
    cand = candidate_rgb
    if cand.shape[:2] != (h, w):
        cand = cv2.resize(cand, (w, h), interpolation=cv2.INTER_AREA)
    gi = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2GRAY)
    gc = cv2.cvtColor(cand, cv2.COLOR_RGB2GRAY)
    return float(ssim(gi, gc))


def select(input_rgb, candidates: list[dict]) -> dict:
    """
    candidates: [{'key','label','family','image'}]
    Returns the full ranking plus the chosen candidate.
    """
    scored = []
    for c in candidates:
        q = quality(c["image"])
        fid = _fidelity(input_rgb, c["image"])
        scored.append({
            "key": c["key"],
            "label": c["label"],
            "family": c["family"],
            "image": c["image"],
            "score": q["score"],
            "components": q["components"],
            "fidelity": fid,
            "passes_guard": fid >= Config.FIDELITY_SSIM_FLOOR,
        })

    eligible = [s for s in scored if s["passes_guard"]] or scored
    eligible.sort(key=lambda s: s["score"], reverse=True)
    winner = eligible[0]

    ranking = sorted(scored, key=lambda s: s["score"], reverse=True)
    return {"winner": winner, "ranking": ranking}
