"""
Adapters for heavy deep models. Each one imports its dependency lazily and
reports availability, so the app runs with none of them installed. Flip the
matching flag in config (env var) after installing the weights.

Keeping these behind a uniform Restorer interface is the whole point: the
orchestrator and selector treat a 30M-parameter network exactly like a 3-line
OpenCV call, and the best output wins on merit.
"""
import numpy as np

from config import Config
from .base import Restorer


class RembgCutout(Restorer):
    """Background removal via rembg (U2Net/IS-Net, ONNX runtime, CPU-friendly).

    Returns the subject composited on a neutral card so it can be scored like
    any other candidate; the API also exposes the true transparent cutout.
    """
    key, label, family = "cutout", "rembg U2Net", "matting"
    _session = None

    @property
    def available(self):
        if not Config.ENABLE_REMBG:
            return False
        try:
            import rembg  # noqa: F401
            return True
        except Exception:
            return False

    def _sess(self):
        if RembgCutout._session is None:
            from rembg import new_session
            RembgCutout._session = new_session("u2net")
        return RembgCutout._session

    def cutout_rgba(self, rgb) -> np.ndarray:
        from rembg import remove
        from PIL import Image
        out = remove(Image.fromarray(rgb), session=self._sess())
        return np.array(out.convert("RGBA"))

    def apply(self, rgb):
        rgba = self.cutout_rgba(rgb)
        alpha = rgba[..., 3:4].astype(np.float32) / 255.0
        bg = np.full_like(rgb, 245)
        comp = rgba[..., :3].astype(np.float32) * alpha + bg.astype(np.float32) * (1 - alpha)
        return comp.astype(np.uint8)


class RealEsrganUpscale(Restorer):
    key, label, family = "upscale_deep", "Real-ESRGAN x4", "upscale"
    _model = None

    @property
    def available(self):
        if not Config.ENABLE_REALESRGAN:
            return False
        try:
            import realesrgan  # noqa: F401
            import torch  # noqa: F401
            return True
        except Exception:
            return False

    def _load(self):
        if RealEsrganUpscale._model is None:
            import torch
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                           num_block=23, num_grow_ch=32, scale=4)
            RealEsrganUpscale._model = RealESRGANer(
                scale=4, model_path="weights/RealESRGAN_x4plus.pth",
                model=arch, half=torch.cuda.is_available(),
            )
        return RealEsrganUpscale._model

    def apply(self, rgb):
        out, _ = self._load().enhance(rgb, outscale=4)
        return out


class GfpganFace(Restorer):
    key, label, family = "face_restore", "GFPGAN", "face"
    _model = None

    @property
    def available(self):
        if not Config.ENABLE_GFPGAN:
            return False
        try:
            import gfpgan  # noqa: F401
            return True
        except Exception:
            return False

    def _load(self):
        if GfpganFace._model is None:
            from gfpgan import GFPGANer
            GfpganFace._model = GFPGANer(
                model_path="weights/GFPGANv1.4.pth",
                upscale=1, arch="clean", channel_multiplier=2,
            )
        return GfpganFace._model

    def apply(self, rgb):
        import cv2
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        _, _, restored = self._load().enhance(
            bgr, has_aligned=False, only_center_face=False, paste_back=True)
        return cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)


class ZeroDceLowLight(Restorer):
    key, label, family = "lowlight_deep", "Zero-DCE", "lowlight"

    @property
    def available(self):
        # Wire up your Zero-DCE checkpoint here; disabled unless flag + weights.
        return bool(Config.ENABLE_ZERODCE)

    def apply(self, rgb):  # pragma: no cover - stub for the enabled path
        raise RuntimeError("Zero-DCE weights not wired in this build")


class DnCnnDenoise(Restorer):
    key, label, family = "denoise_deep", "DnCNN", "denoise"

    @property
    def available(self):
        return bool(Config.ENABLE_DNCNN)

    def apply(self, rgb):  # pragma: no cover - stub for the enabled path
        raise RuntimeError("DnCNN weights not wired in this build")


DEEP = [
    RembgCutout(),
    RealEsrganUpscale(),
    GfpganFace(),
    ZeroDceLowLight(),
    DnCnnDenoise(),
]
