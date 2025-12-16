import numpy as np
import warnings

from image_pipeline.core.image_data import ImageData
from image_pipeline.types import ColorSpace, TransferFunction
from .base import ImageFilter


class PQEncodeFilter(ImageFilter):
    """
    Filter for applying PQ (Perceptual Quantizer) gamma curve according to ST.2084
    Converts linear HDR values to PQ-encoded values
    """
    
    # PQ constants according to ITU-R BT.2100
    M1 = 2610.0 / 16384.0  # 0.1593017578125
    M2 = 2523.0 / 4096.0 * 128.0  # 78.84375
    C1 = 3424.0 / 4096.0  # 0.8359375
    C2 = 2413.0 / 4096.0 * 32.0  # 18.8515625
    C3 = 2392.0 / 4096.0 * 32.0  # 18.6875
    
    def __init__(self, peak_luminance: float = 10000.0):
        """
        Args:
            peak_luminance: Peak luminance in nits (cd/m²)
                           Usually 10000 for ST.2084, but can be adjusted
        """
        self.peak_luminance = peak_luminance
        super().__init__()
    
    def apply(self, pixels: np.ndarray) -> np.ndarray:
        self.validate(pixels)
        
        # Validate dtype
        self._check_dtype(pixels, [np.float32, np.float64])
        
        # Clip to [0, peak_luminance]
        clipped = np.clip(pixels, 0.0, self.peak_luminance)
        
        # Normalize to [0, 1] relative to peak luminance
        normalized = clipped / self.peak_luminance
        
        # Apply PQ EOTF
        # Y = ((c1 + c2 * L^m1) / (1 + c3 * L^m1))^m2
        L_m1 = np.power(normalized, self.M1)
        
        numerator = self.C1 + self.C2 * L_m1
        denominator = 1.0 + self.C3 * L_m1
        
        # Avoid division by zero
        denominator = np.maximum(denominator, 1e-10)
        
        pq_encoded = np.power(numerator / denominator, self.M2)
        
        return pq_encoded.astype(np.float32)
    
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
                f"PQ encoding may be suboptimal for values above 10000.",
                UserWarning
            )
    
    def update_metadata(self, img_data: ImageData) -> None:
        super().update_metadata(img_data)
        img_data.metadata['transfer_function'] = TransferFunction.PQ
        img_data.metadata['peak_luminance'] = self.peak_luminance
        
    def __repr__(self) -> str:
        return f"PQEncodeFilter(peak_luminance={self.peak_luminance})"
