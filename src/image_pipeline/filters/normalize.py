from image_pipeline.core.image_data import ImageData

import numpy as np
from .base import ImageFilter


class NormalizeFilter(ImageFilter):
    """Filter for normalizing pixel values to a given range"""
    
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        """
        Args:
            min_val: Minimum value of the output range
            max_val: Maximum value of the output range
        """
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
    
    def apply(self, pixels: np.ndarray) -> np.ndarray:
        self.validate(pixels)
        
        pix_min = pixels.min()
        pix_max = pixels.max()
        
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
