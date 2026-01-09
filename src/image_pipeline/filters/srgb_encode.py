import numpy as np

from image_pipeline.core.image_data import ImageData
from image_pipeline.types import TransferFunction
from .base import ImageFilter


class SRGBEncodeFilter(ImageFilter):
    """
    Filter for applying sRGB transfer function (gamma encoding).
    Converts linear RGB values to sRGB-encoded values.

    Uses the standard sRGB OETF (Opto-Electronic Transfer Function):
    - Linear segment for dark values (linear <= 0.0031308)
    - Power function for bright values (gamma 2.4)

    This is the correct transfer function for standard SDR images (PNG, JPEG, etc.)
    """

    # sRGB constants from IEC 61966-2-1 specification
    LINEAR_THRESHOLD = 0.0031308
    SRGB_THRESHOLD = 0.04045
    LINEAR_SCALE = 12.92
    GAMMA = 2.4
    A = 0.055

    def __init__(self):
        """
        Initialize sRGB encoder.
        No parameters needed - sRGB spec is fixed.
        """
        super().__init__()

    def apply(self, pixels: np.ndarray) -> np.ndarray:
        self.validate(pixels)

        # Validate dtype
        self._check_dtype(pixels, [np.float16, np.float32, np.float64])

        # Clip to [0, 1] for SDR range
        clipped = np.clip(pixels, 0.0, 1.0)

        # Apply sRGB OETF
        # For linear <= 0.0031308: srgb = 12.92 * linear
        # For linear > 0.0031308:  srgb = 1.055 * linear^(1/2.4) - 0.055
        srgb_encoded = np.where(
            clipped <= self.LINEAR_THRESHOLD,
            self.LINEAR_SCALE * clipped,
            (1.0 + self.A) * np.power(clipped, 1.0 / self.GAMMA) - self.A
        )

        return srgb_encoded.astype(np.float32)

    def validate_params(self) -> None:
        # No parameters to validate
        pass

    def update_metadata(self, img_data: ImageData) -> None:
        super().update_metadata(img_data)
        img_data.metadata['transfer_function'] = TransferFunction.SRGB

    def __repr__(self) -> str:
        return "SRGBEncodeFilter()"
