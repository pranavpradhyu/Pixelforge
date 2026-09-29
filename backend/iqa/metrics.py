"""
No-reference image quality assessment (NR-IQA).

This is how the pipeline chooses a winner without the user ever seeing a
comparison. Each candidate output is scored on a 0..1 perceptual-quality scale;
the orchestrator returns the highest scorer that also passes the fidelity guard.

Two backends:
  * composite  — always available, no weights. A weighted blend of classical
                 no-reference cues (sharpness, local contrast, colourfulness,
                 exposure balance, blockiness penalty, noise penalty).
  * pyiqa      — optional. If ENABLE_PYIQA and the `pyiqa` package are present,
                 we fold in a learned metric (CLIP-IQA / BRISQUE) that
                 correlates far better with human preference. This is the
                 recommended production path and the main research lever.
"""
import cv2
import numpy as np

from config import Config

_PYIQA_MODEL = None


def _normalize(value, lo, hi):
    return float(np.clip((value - lo) / (hi - lo + 1e-9), 0.0, 1.0))


def _sharpness(gray):
    return _normalize(cv2.Laplacian(gray, cv2.CV_64F).var(), 20, 1200)


def _contrast(gray):
    return _normalize(gray.std(), 20, 75)


def _colorfulness(rgb):
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    val = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return _normalize(val, 5, 60)


def _exposure(gray):
    mean = gray.mean()
    # best near mid-grey, penalise crushed / blown frames
    return float(np.exp(-((mean - 118.0) ** 2) / (2 * 55.0 ** 2)))


def _blockiness_penalty(gray):
    # crude JPEG/upsampling artefact proxy: energy at 8-px block boundaries
    g = gray.astype(np.float32)
    v = h = 0.0
    if g.shape[1] > 8:
        a, b = g[:, 7:-1:8], g[:, 8::8]
        n = min(a.shape[1], b.shape[1])
        v = np.abs(a[:, :n] - b[:, :n]).mean()
    if g.shape[0] > 8:
        a, b = g[7:-1:8, :], g[8::8, :]
        n = min(a.shape[0], b.shape[0])
        h = np.abs(a[:n, :] - b[:n, :]).mean()
    return _normalize((v + h) / 2.0, 0, 12)  # higher = worse


def _noise_penalty(gray):
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    sigma = np.abs(cv2.filter2D(gray.astype(np.float32), -1, kernel)).mean() / 6.0
    return _normalize(sigma, 1.5, 10)  # higher = worse


def composite_score(rgb) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    quality = (
        0.34 * _sharpness(gray)
        + 0.20 * _contrast(gray)
        + 0.16 * _colorfulness(rgb)
        + 0.18 * _exposure(gray)
    )
    penalty = 0.06 * _blockiness_penalty(gray) + 0.06 * _noise_penalty(gray)
    return float(np.clip(quality - penalty, 0.0, 1.0))


def _pyiqa_score(rgb) -> float | None:
    global _PYIQA_MODEL
    if not Config.ENABLE_PYIQA:
        return None
    try:
        import pyiqa
        import torch
        if _PYIQA_MODEL is None:
            _PYIQA_MODEL = pyiqa.create_metric("clipiqa")
        t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        return float(_PYIQA_MODEL(t).item())
    except Exception:
        return None


def quality(rgb) -> dict:
    """Return {'score': float, 'components': {...}} for one image."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    comp = composite_score(rgb)
    learned = _pyiqa_score(rgb)
    if learned is not None:
        score = 0.5 * comp + 0.5 * learned
    else:
        score = comp
    return {
        "score": score,
        "components": {
            "sharpness": _sharpness(gray),
            "contrast": _contrast(gray),
            "colorfulness": _colorfulness(rgb),
            "exposure": _exposure(gray),
            "learned": learned,
        },
    }
