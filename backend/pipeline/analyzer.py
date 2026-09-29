"""
Degradation analysis.

Before any restoration runs, we measure the image so the orchestrator only
spends compute on candidates that are actually relevant. Everything here is
classical CV (fast, CPU, no weights) and feeds the routing decision.

The measurements are also reused by the selector as light priors, but the final
choice of output is made by no-reference quality scoring, not by these numbers.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

def _load_face_cascade():
    """Haar face detector, if this OpenCV build provides it.

    OpenCV 4.x ships it in the main package. Some builds (e.g. 5.x) do not, so we
    treat face detection as optional: without it, face_count is simply 0 and the
    face-restoration branch is skipped. Nothing else in the pipeline depends on it.
    """
    try:
        cls = getattr(cv2, "CascadeClassifier", None)
        data = getattr(cv2, "data", None)
        if cls is None or data is None:
            return None
        cascade = cls(data.haarcascades + "haarcascade_frontalface_default.xml")
        return None if cascade.empty() else cascade
    except Exception:
        return None


_FACE_CASCADE = _load_face_cascade()


@dataclass
class Analysis:
    width: int
    height: int
    is_low_res: bool
    blur_score: float          # variance of Laplacian; low => blurry
    is_blurry: bool
    noise_sigma: float         # estimated Gaussian noise std
    is_noisy: bool
    mean_luma: float           # 0..255
    is_low_light: bool
    is_overexposed: bool
    colorfulness: float        # Hasler-Susstrunk metric
    is_grayscale: bool
    face_count: int
    routes: list[str] = field(default_factory=list)


def _estimate_noise_sigma(gray: np.ndarray) -> float:
    """Robust noise std estimate (Immerkaer 1996) via a Laplacian-like kernel."""
    h, w = gray.shape
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    conv = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    sigma = np.abs(conv).mean() * np.sqrt(0.5 * np.pi) / (6.0)
    return float(sigma)


def _colorfulness(rgb: np.ndarray) -> float:
    r, g, b = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    rg = r - g
    yb = 0.5 * (r + g) - b
    std = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std + 0.3 * mean)


def analyze(rgb: np.ndarray) -> Analysis:
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    noise_sigma = _estimate_noise_sigma(gray)
    mean_luma = float(gray.mean())
    colorfulness = _colorfulness(rgb)

    # channel spread tells grayscale-stored-as-RGB apart from real colour
    chan_spread = float(np.mean(np.std(rgb.astype(np.float32), axis=2)))
    is_grayscale = chan_spread < 3.0

    face_count = 0
    if _FACE_CASCADE is not None:
        faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        face_count = 0 if faces is None else len(faces)

    a = Analysis(
        width=w,
        height=h,
        is_low_res=max(h, w) < 900,
        blur_score=blur_score,
        is_blurry=blur_score < 120.0,
        noise_sigma=noise_sigma,
        is_noisy=noise_sigma > 3.0,
        mean_luma=mean_luma,
        is_low_light=mean_luma < 80.0,
        is_overexposed=mean_luma > 210.0,
        colorfulness=colorfulness,
        is_grayscale=is_grayscale,
        face_count=face_count,
    )
    a.routes = _route(a)
    return a


def _route(a: Analysis) -> list[str]:
    """Map measurements to the candidate branches worth trying.

    Branch names correspond to model families registered in the registry. The
    orchestrator always keeps a 'passthrough' candidate so a clean image can win
    and the pipeline never degrades good input.
    """
    routes: list[str] = ["passthrough"]

    if a.is_low_light:
        routes += ["lowlight_clahe", "lowlight_deep"]
    if a.is_overexposed:
        routes += ["tone_recover"]
    if a.is_noisy:
        routes += ["denoise_nlm", "denoise_deep"]
    if a.is_blurry:
        routes += ["sharpen_unsharp", "deblur_wiener"]
    if a.is_low_res:
        routes += ["upscale_lanczos", "upscale_deep"]
    if a.face_count > 0:
        routes += ["face_restore"]
    if a.is_grayscale:
        routes += ["colorize"]

    # White-balance correction is cheap and broadly helpful on old prints.
    routes += ["white_balance"]

    # de-dup while preserving order
    seen, ordered = set(), []
    for r in routes:
        if r not in seen:
            seen.add(r)
            ordered.append(r)
    return ordered
