"""Filesystem storage for uploads and results, plus a small retention janitor."""
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from config import Config


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def ext_ok(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def save_upload(file_storage) -> tuple[str, Path]:
    """Persist a Werkzeug FileStorage, return (job_id, path)."""
    Config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    job_id = new_id()
    suffix = file_storage.filename.rsplit(".", 1)[1].lower()
    path = Config.UPLOAD_DIR / f"{job_id}.{suffix}"
    file_storage.save(str(path))
    return job_id, path


def read_image(path: Path) -> np.ndarray:
    """Load an image as RGB uint8 (drops alpha; callers that need it re-read)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_image(img_rgb: np.ndarray, job_id: str, suffix: str = "png") -> Path:
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = Config.OUTPUT_DIR / f"{job_id}.{suffix}"
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(f".{suffix}", bgr)
    if not ok:
        raise ValueError("Could not encode image")
    buf.tofile(str(path))
    return path


def sweep_expired() -> int:
    """Delete files older than the retention window. Returns count removed."""
    cutoff = time.time() - Config.RETENTION_SECONDS
    removed = 0
    for folder in (Config.UPLOAD_DIR, Config.OUTPUT_DIR):
        if not folder.exists():
            continue
        for p in folder.iterdir():
            try:
                if p.is_file() and p.stat().st_mtime < cutoff:
                    p.unlink()
                    removed += 1
            except OSError:
                pass
    return removed
