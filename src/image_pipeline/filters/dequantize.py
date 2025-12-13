from image_pipeline.core.image_data import ImageData
from typing import Optional

import numpy as np
from .base import ImageFilter


class DequantizeFilter(ImageFilter):
    """
    Filter for dequantizing integer values back to float
    Converts [0, 2^bit_depth - 1] → [0, 1]
    """
    
    def __init__(self, bit_depth: int = 8):
        """
        Args:
            bit_depth: Bit depth of the source data (8, 10, 12, 16, 32)
        """
        super().__init__()
        self.bit_depth = bit_depth
    
    def apply(self, pixels: np.ndarray) -> np.ndarray:
        self.validate(pixels)
        
        bit_depth = self.bit_depth
        max_value = (2 ** bit_depth) - 1
        
        # Normalize to [0, 1]
        normalized = pixels.astype(np.float32) / max_value
        
        return normalized
    
    def update_metadata(self, img_data: ImageData) -> None:
        super().update_metadata(img_data)
        img_data.metadata['bit_depth'] = self.bit_depth
    
    def __repr__(self) -> str:
        return f"DequantizeFilter(bit_depth={self.bit_depth})"
