"""
Classical, dependency-light restorers. These always run (OpenCV + NumPy only)
and give the pipeline a competitive floor: on many real photos a well-tuned
classical method beats a mis-applied heavy model, and the selector will pick it.
"""
import cv2
import numpy as np

from .base import Restorer


class Passthrough(Restorer):
    key, label, family = "passthrough", "Original", "identity"

    def apply(self, rgb):
        return rgb.copy()


class ClaheLowLight(Restorer):
    key, label, family = "lowlight_clahe", "CLAHE + gamma", "lowlight"

    def apply(self, rgb):
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        # gentle gamma lift
        g = 0.85
        lut = np.array([((i / 255.0) ** g) * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(out, lut)


class ToneRecover(Restorer):
    key, label, family = "tone_recover", "Highlight recovery", "tone"

    def apply(self, rgb):
        # compress highlights, lift shadows via an S-curve in luma
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        l = lab[..., 0] / 255.0
        l = np.clip(0.5 + 1.15 * (l - 0.5), 0, 1)          # contrast
        l = l - 0.12 * np.clip(l - 0.75, 0, 1) / 0.25      # tame blown highlights
        lab[..., 0] = np.clip(l * 255.0, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


class NlmDenoise(Restorer):
    key, label, family = "denoise_nlm", "Non-local means", "denoise"

    def apply(self, rgb):
        return cv2.fastNlMeansDenoisingColored(rgb, None, 7, 7, 7, 21)


class UnsharpSharpen(Restorer):
    key, label, family = "sharpen_unsharp", "Unsharp mask", "sharpen"

    def apply(self, rgb):
        blur = cv2.GaussianBlur(rgb, (0, 0), 2.0)
        return cv2.addWeighted(rgb, 1.5, blur, -0.5, 0)


class WienerDeblur(Restorer):
    key, label, family = "deblur_wiener", "Richardson-Lucy", "deblur"

    def apply(self, rgb):
        # light Richardson-Lucy per channel with a small Gaussian PSF
        psf = cv2.getGaussianKernel(5, 1.2)
        psf = psf @ psf.T
        out = np.zeros_like(rgb, dtype=np.float32)
        for c in range(3):
            ch = rgb[..., c].astype(np.float32) / 255.0 + 1e-6
            est = ch.copy()
            for _ in range(8):
                conv = cv2.filter2D(est, -1, psf) + 1e-6
                relative = ch / conv
                est *= cv2.filter2D(relative, -1, psf[::-1, ::-1])
            out[..., c] = np.clip(est, 0, 1) * 255.0
        return out.astype(np.uint8)


class LanczosUpscale(Restorer):
    key, label, family = "upscale_lanczos", "Lanczos x2", "upscale"

    def apply(self, rgb):
        h, w = rgb.shape[:2]
        return cv2.resize(rgb, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)


class GrayWorldWB(Restorer):
    key, label, family = "white_balance", "Gray-world WB", "color"

    def apply(self, rgb):
        result = rgb.astype(np.float32)
        avg = result.reshape(-1, 3).mean(axis=0)
        gray = avg.mean()
        for c in range(3):
            if avg[c] > 1e-3:
                result[..., c] *= gray / avg[c]
        return np.clip(result, 0, 255).astype(np.uint8)


class SimpleColorize(Restorer):
    """Placeholder colouriser: applies a learned-looking warm tint mapping.

    This is intentionally conservative — it will usually lose to 'passthrough'
    on the quality score unless a real colouriser (DeOldify) is enabled, at
    which point that deep model registers under the same 'colorize' route.
    """
    key, label, family = "colorize", "Sepia toning", "color"

    def apply(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        toned = cv2.applyColorMap(gray, cv2.COLORMAP_BONE)
        toned = cv2.cvtColor(toned, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(rgb, 0.4, toned, 0.6, 0)


CLASSICAL = [
    Passthrough(),
    ClaheLowLight(),
    ToneRecover(),
    NlmDenoise(),
    UnsharpSharpen(),
    WienerDeblur(),
    LanczosUpscale(),
    GrayWorldWB(),
    SimpleColorize(),
]
