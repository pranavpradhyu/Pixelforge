"""Common interface every restoration candidate implements."""
from abc import ABC, abstractmethod

import numpy as np


class Restorer(ABC):
    #: unique branch key, matches analyzer route names
    key: str = "base"
    #: short human label (never shown to end users; used in logs / research mode)
    label: str = "Base"
    #: family, e.g. 'lowlight', 'denoise', 'upscale' — used for grouping
    family: str = "generic"

    @property
    def available(self) -> bool:
        """Whether this candidate can run in the current environment."""
        return True

    @abstractmethod
    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """Return a restored RGB uint8 image (may differ in size, e.g. upscalers)."""
        raise NotImplementedError
