import numpy as np
import warnings

from image_pipeline.core.image_data import ImageData
from image_pipeline.types import TransferFunction
from .base import ImageFilter


class PQEncodeFilter(ImageFilter):
    """
    Filter for applying PQ (Perceptual Quantizer) gamma curve according to ST.2084
    Converts linear HDR values (in absolute nits) to PQ-encoded values.

    Input data must be display-referred (absolute luminance in nits).
    For scene-referred data, use AbsoluteLuminanceFilter first.
    """

    # PQ constants according to ITU-R BT.2100
    M1 = 2610.0 / 16384.0  # 0.1593017578125
    M2 = 2523.0 / 4096.0 * 128.0  # 78.84375
    C1 = 3424.0 / 4096.0  # 0.8359375
    C2 = 2413.0 / 4096.0 * 32.0  # 18.8515625
    C3 = 2392.0 / 4096.0 * 32.0  # 18.6875

    def __init__(self, reference_peak: float = 10000.0):
        """
        Args:
            reference_peak: Reference peak luminance for PQ normalization, in nits (cd/m²).
                           Standard value is 10000 nits (ST.2084 specification).
                           Input values are normalized to [0, 1] relative to this peak.
        """
        self.reference_peak = reference_peak
        super().__init__()
    
    def apply(self, pixels: np.ndarray) -> np.ndarray:
        self.validate(pixels)

        # Validate dtype
        self._check_dtype(pixels, [np.float16, np.float32, np.float64])

        # Clip to [0, reference_peak]
        clipped = np.clip(pixels, 0.0, self.reference_peak)

        # Normalize to [0, 1] relative to reference peak
        normalized = clipped / self.reference_peak
        
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
        if not isinstance(self.reference_peak, (int, float)):
            raise TypeError(
                f"reference_peak must be numeric, got {type(self.reference_peak).__name__}"
            )

        if self.reference_peak <= 0:
            raise ValueError(
                f"reference_peak must be positive, got {self.reference_peak}"
            )

        # Soft warning for non-standard values
        if self.reference_peak > 10000:
            warnings.warn(
                f"reference_peak={self.reference_peak} exceeds ST.2084 standard (10000 nits). "
                f"PQ encoding may be suboptimal for values above 10000.",
                UserWarning
            )
    
    def update_metadata(self, img_data: ImageData) -> None:
        super().update_metadata(img_data)
        img_data.metadata['transfer_function'] = TransferFunction.PQ
        # Note: We don't store reference_peak in metadata as it's an encoding parameter,
        # not a content property. Use mastering_display_max_luminance for display metadata.

    def __repr__(self) -> str:
        return f"PQEncodeFilter(reference_peak={self.reference_peak})"
