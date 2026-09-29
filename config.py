"""
Central configuration for PixelForge.

Every heavy capability is behind a feature flag so the app boots and runs on a
plain CPU box with only the base requirements installed. Turn flags on as you
install the corresponding model weights (see requirements-heavy.txt / README).
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


class Config:
    # --- Flask ---
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-change-me")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "tiff"}

    # --- Storage ---
    UPLOAD_DIR = BASE_DIR / "data" / "uploads"
    OUTPUT_DIR = BASE_DIR / "data" / "outputs"
    # How long to keep files (seconds) before the janitor removes them.
    RETENTION_SECONDS = int(os.getenv("RETENTION_SECONDS", str(60 * 60 * 6)))

    # --- Pipeline behaviour ---
    # Longest edge the working image is resized to before candidate models run.
    # Keeps CPU inference tractable; the winning result is re-applied to the
    # native-resolution image where the model supports it.
    WORKING_MAX_EDGE = int(os.getenv("WORKING_MAX_EDGE", "1600"))
    # Fidelity guard: reject a candidate whose structural similarity to the input
    # drops below this, so the pipeline never returns a hallucinated image that no
    # longer resembles what the user uploaded. Lower = more permissive.
    FIDELITY_SSIM_FLOOR = float(os.getenv("FIDELITY_SSIM_FLOOR", "0.35"))

    # --- Model feature flags (deep models are optional) ---
    ENABLE_REMBG = _flag("ENABLE_REMBG", "1")          # background removal (onnx, CPU-ok)
    ENABLE_REALESRGAN = _flag("ENABLE_REALESRGAN")     # super-resolution
    ENABLE_GFPGAN = _flag("ENABLE_GFPGAN")             # face restoration
    ENABLE_ZERODCE = _flag("ENABLE_ZERODCE")           # deep low-light
    ENABLE_DNCNN = _flag("ENABLE_DNCNN")               # deep denoise
    ENABLE_PYIQA = _flag("ENABLE_PYIQA")               # learned no-reference IQA (BRISQUE/CLIP-IQA)

    # --- Jobs ---
    JOB_WORKERS = int(os.getenv("JOB_WORKERS", "2"))
    JOB_TIMEOUT = int(os.getenv("JOB_TIMEOUT", "300"))
