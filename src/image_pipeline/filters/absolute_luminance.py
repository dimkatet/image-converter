from image_pipeline.core.image_data import ImageData

import numpy as np
from .base import ImageFilter


class AbsoluteLuminanceFilter(ImageFilter):
    """
    Convert scene-referred linear HDR values to display-referred absolute luminance (nits).

    Scene-referred values are relative to a paper white reference:
        pixel_value * paper_white = luminance_in_nits

    This filter scales scene-referred values to display-referred absolute
    luminance values suitable for PQ encoding or other display transforms.

    Example:
        Input: pixel = 0.2, paper_white = 100 nits
        Output: pixel = 20.0 (absolute nits)
    """

    def __init__(self, paper_white: float = 100.0):
        """
        Args:
            paper_white: Reference white luminance in nits (cd/m²).
                        Typical values: 100 nits (SDR standard), 203 nits (HDR reference).
        """
        self.paper_white = paper_white
        super().__init__()

    def validate_params(self) -> None:
        if self.paper_white <= 0:
            raise ValueError(
                f"paper_white must be positive, got {self.paper_white}"
            )

    def apply(self, pixels: np.ndarray) -> np.ndarray:
        self.validate(pixels)

        # Validate dtype - should be float for HDR data
        self._check_dtype(pixels, [np.float16, np.float32, np.float64])

        # Scale scene-referred values to absolute nits
        result = pixels * self.paper_white

        return result.astype(np.float32)

    def update_metadata(self, img_data: ImageData) -> None:
        """Update metadata to reflect display-referred absolute luminance."""
        super().update_metadata(img_data)

        # Store paper_white to indicate this was scene-referred data
        img_data.metadata['paper_white'] = self.paper_white

        # Calculate and store MaxCLL and MaxFALL (HDR10 metadata)
        # TODO: Update MaxCLL/MaxFALL in brightness/contrast filters when implemented
        img_data.metadata['max_cll'] = int(round(img_data.pixels.max()))
        img_data.metadata['max_fall'] = int(round(img_data.pixels.mean()))

    def __repr__(self) -> str:
        return f"AbsoluteLuminanceFilter(paper_white={self.paper_white})"
