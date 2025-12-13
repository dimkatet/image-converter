from typing import Optional

import numpy as np
from .base import ImageFilter


class DequantizeFilter(ImageFilter):
    """
    Filter for dequantizing integer values back to float
    Converts [0, 2^bit_depth - 1] → [0, 1]
    """
    
    def __init__(self, bit_depth: Optional[int] = None):
        """
        Args:
            bit_depth: Bit depth of the source data (8, 10, 12, 16, 32)
                      If None, determined automatically from dtype
        """
        super().__init__()
        self.bit_depth = bit_depth
    
    def apply(self, pixels: np.ndarray) -> np.ndarray:
        self.validate(pixels)
        
        # Determine bit depth if not specified
        if self.bit_depth is None:
            if pixels.dtype == np.uint8:
                bit_depth = 8
            elif pixels.dtype == np.uint16:
                bit_depth = 16
            elif pixels.dtype == np.uint32:
                bit_depth = 32
            else:
                raise ValueError(f"Could not determine bit depth for type {pixels.dtype}")
        else:
            bit_depth = self.bit_depth
        
        max_value = (2 ** bit_depth) - 1
        
        # Normalize to [0, 1]
        normalized = pixels.astype(np.float32) / max_value
        
        return normalized
    
    def __repr__(self) -> str:
        return f"DequantizeFilter(bit_depth={self.bit_depth})"
