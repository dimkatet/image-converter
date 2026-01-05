import numpy as np

from image_pipeline.core.image_data import ImageData
from image_pipeline.types import TransferFunction
from .base import ImageFilter


class SRGBDecodeFilter(ImageFilter):
    """
    Filter for sRGB decoding (inverse transformation).
    Converts sRGB-encoded values back to linear RGB values.

    Uses the inverse sRGB EOTF (Electro-Optical Transfer Function):
    - Linear segment for dark values (srgb <= 0.04045)
    - Power function for bright values (gamma 2.4)

    This is useful when reading SDR images that need to be processed in linear space.
    """

    # sRGB constants from IEC 61966-2-1 specification
    LINEAR_THRESHOLD = 0.0031308
    SRGB_THRESHOLD = 0.04045
    LINEAR_SCALE = 12.92
    GAMMA = 2.4
    A = 0.055

    def __init__(self):
        """
        Initialize sRGB decoder.
        No parameters needed - sRGB spec is fixed.
        """
        super().__init__()

    def apply(self, pixels: np.ndarray) -> np.ndarray:
        self.validate(pixels)

        # Validate dtype
        self._check_dtype(pixels, [np.float32, np.float64])

        # Range check [0, 1] for sRGB-encoded values
        self._check_range(pixels, 0.0, 1.0)

        # Apply inverse sRGB EOTF
        # For srgb <= 0.04045: linear = srgb / 12.92
        # For srgb > 0.04045:  linear = ((srgb + 0.055) / 1.055)^2.4
        linear = np.where(
            pixels <= self.SRGB_THRESHOLD,
            pixels / self.LINEAR_SCALE,
            np.power((pixels + self.A) / (1.0 + self.A), self.GAMMA)
        )

        return linear.astype(np.float32)

    def validate_params(self) -> None:
        # No parameters to validate
        pass

    def update_metadata(self, img_data: ImageData) -> None:
        super().update_metadata(img_data)
        img_data.metadata['transfer_function'] = TransferFunction.LINEAR

    def __repr__(self) -> str:
        return "SRGBDecodeFilter()"
