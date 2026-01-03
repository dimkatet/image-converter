from image_pipeline.core.image_data import ImageData

import numpy as np
import warnings

from image_pipeline.types import TransferFunction
from .base import ImageFilter


class PQDecodeFilter(ImageFilter):
    """
    Filter for PQ decoding (inverse transformation)
    Converts PQ-encoded values back to linear HDR values
    """
    
    M1 = 2610.0 / 16384.0
    M2 = 2523.0 / 4096.0 * 128.0
    C1 = 3424.0 / 4096.0
    C2 = 2413.0 / 4096.0 * 32.0
    C3 = 2392.0 / 4096.0 * 32.0
    
    def __init__(self, peak_luminance: float = 10000.0):
        """
        Args:
            peak_luminance: Peak luminance in nits (should match encoding)
        """
        self.peak_luminance = peak_luminance
        super().__init__()
    
    def apply(self, pixels: np.ndarray) -> np.ndarray:
        self.validate(pixels)
        
        # Validate dtype
        self._check_dtype(pixels, [np.float32, np.float64])
        
        # Strict range check [0, 1] for PQ-encoded values
        self._check_range(pixels, 0.0, 1.0)
        
        # Inverse PQ transformation
        # L = ((max(Y^(1/m2) - c1, 0)) / (c2 - c3 * Y^(1/m2)))^(1/m1)
        Y_pow = np.power(pixels, 1.0 / self.M2)
        
        numerator = np.maximum(Y_pow - self.C1, 0.0)
        denominator = self.C2 - self.C3 * Y_pow
        
        # Avoid division by zero
        denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
        
        L = np.power(numerator / denominator, 1.0 / self.M1)
        
        # Scale back to the original luminance range
        linear = L * self.peak_luminance
        
        return linear.astype(np.float32)
    
    def validate_params(self) -> None:
        if not isinstance(self.peak_luminance, (int, float)):
            raise TypeError(
                f"peak_luminance must be numeric, got {type(self.peak_luminance).__name__}"
            )
        
        if self.peak_luminance <= 0:
            raise ValueError(
                f"peak_luminance must be positive, got {self.peak_luminance}"
            )
        
        # Soft warning for non-standard values
        if self.peak_luminance > 10000:
            warnings.warn(
                f"peak_luminance={self.peak_luminance} exceeds ST.2084 standard (10000 nits). "
                f"PQ decoding may be suboptimal for values above 10000.",
                UserWarning
            )
    
    def update_metadata(self, img_data: ImageData) -> None:
        super().update_metadata(img_data)
        img_data.metadata['transfer_function'] = TransferFunction.LINEAR
    
    def __repr__(self) -> str:
        return f"PQDecodeFilter(peak_luminance={self.peak_luminance})"
