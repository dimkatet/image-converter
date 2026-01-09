from image_pipeline.core.image_data import ImageData

import numpy as np
from .base import ImageFilter


class RelativeLuminanceFilter(ImageFilter):
    """
    Convert display-referred absolute luminance (nits) to scene-referred relative values.

    This is the inverse operation of AbsoluteLuminanceFilter:
        luminance_in_nits / paper_white = pixel_value

    Converts display-referred absolute luminance values back to scene-referred
    relative values normalized to a paper white reference.

    Example:
        Input: pixel = 20.0 (absolute nits), paper_white = 100 nits
        Output: pixel = 0.2 (relative to paper white)
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

        # Scale absolute nits to scene-referred relative values
        result = pixels / self.paper_white

        return result.astype(np.float32)

    def update_metadata(self, img_data: ImageData) -> None:
        """Update metadata to reflect scene-referred relative luminance."""
        super().update_metadata(img_data)

        # Keep paper_white - it describes the scene reference, not encoding state
        # Store it if not already present (indicates the scene's reference white level)
        if 'paper_white' not in img_data.metadata:
            img_data.metadata['paper_white'] = self.paper_white

        # Remove MaxCLL/MaxFALL (only valid for display-referred data)
        if 'max_cll' in img_data.metadata:
            del img_data.metadata['max_cll']
        if 'max_fall' in img_data.metadata:
            del img_data.metadata['max_fall']

    def __repr__(self) -> str:
        return f"RelativeLuminanceFilter(paper_white={self.paper_white})"
