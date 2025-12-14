from image_pipeline.core.image_data import ImageData

import numpy as np
import warnings
from .base import ImageFilter


class QuantizeFilter(ImageFilter):
    """
    Filter for quantizing float values to integer types
    Converts [0, 1] → [0, 2^bit_depth - 1]
    """
    
    def __init__(self, bit_depth: int = 8):
        """
        Args:
            bit_depth: Target bit depth (8, 10, 12, 16, 32)
        """
        super().__init__()
        
        if bit_depth not in [8, 10, 12, 16, 32]:
            raise ValueError(f"Unsupported bit depth: {bit_depth}. "
                           f"Supported: 8, 10, 12, 16, 32")
        
        self.bit_depth = bit_depth
        
        # Determine target data type
        if bit_depth == 8:
            self.target_dtype = np.uint8
        elif bit_depth in [10, 12]:
            self.target_dtype = np.uint16  # 10/12-bit stored in uint16
        elif bit_depth == 16:
            self.target_dtype = np.uint16
        elif bit_depth == 32:
            self.target_dtype = np.uint32
        
        self.max_value = (2 ** bit_depth) - 1
    
    def apply(self, pixels: np.ndarray) -> np.ndarray:
        self.validate(pixels)
        
        # Validate dtype
        self._check_dtype(pixels, [np.float32, np.float64])
        
        # Clip values to [0, 1] with warning
        if pixels.min() < 0.0 or pixels.max() > 1.0:
            warnings.warn(
                f"{self.name}: input values [{pixels.min():.6f}, {pixels.max():.6f}] "
                f"were clipped to [0.0, 1.0]",
                UserWarning
            )
        
        clipped = np.clip(pixels, 0.0, 1.0)
        
        # Quantize: [0, 1] → [0, max_value]
        quantized = clipped * self.max_value
        
        # Round and convert to integer type
        result = np.round(quantized).astype(self.target_dtype)
        
        return result

    def update_metadata(self, img_data: ImageData) -> None:
        super().update_metadata(img_data)
        img_data.metadata['bit_depth'] = self.bit_depth
        img_data.metadata['dtype'] = str(self.target_dtype)
    
    def __repr__(self) -> str:
        return f"QuantizeFilter(bit_depth={self.bit_depth})"
