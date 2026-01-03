from typing import Optional

import numpy as np
from .base import ImageFilter


class NormalizeFilter(ImageFilter):
    """Filter for normalizing pixel values to a given range"""
    
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0, min_in: Optional[float] = None, max_in: Optional[float] = None):
        """
        Args:
            min_val: Minimum value of the output range
            max_val: Maximum value of the output range
        """
        self.min_val = min_val
        self.max_val = max_val
        self.min_in = min_in
        self.max_in = max_in
        super().__init__()
    
    def validate_params(self) -> None:
        if not isinstance(self.min_val, (int, float)):
            raise TypeError(
                f"min_val must be numeric, got {type(self.min_val).__name__}"
            )
        
        if not isinstance(self.max_val, (int, float)):
            raise TypeError(
                f"max_val must be numeric, got {type(self.max_val).__name__}"
            )
        
        if self.min_val >= self.max_val:
            raise ValueError(
                f"min_val must be less than max_val, "
                f"got min_val={self.min_val}, max_val={self.max_val}"
            )
    
    def apply(self, pixels: np.ndarray) -> np.ndarray:
        self.validate(pixels)
        pix_min = self.min_in if self.min_in is not None else pixels.min()
        pix_max = self.max_in if self.max_in is not None else pixels.max()
        
        if pix_max == pix_min:
            return np.full_like(pixels, self.min_val, dtype=np.float32)
        
        # Normalization
        normalized = (pixels - pix_min) / (pix_max - pix_min)
        result = normalized * (self.max_val - self.min_val) + self.min_val
        
        return result.astype(np.float32)
    
    # todo
    # def update_metadata(self, img_data: ImageData) -> None:
    #     super().update_metadata(img_data)
    #     img_data.metadata['normalized'] = True
    #     img_data.metadata['normalize_range'] = (self.min_val, self.max_val)
    
    def __repr__(self) -> str:
        return f"NormalizeFilter(min={self.min_val}, max={self.max_val})"
