from image_pipeline.core.image_data import ImageData

import numpy as np
from .base import ImageFilter


class GrayscaleFilter(ImageFilter):
    """Filter for converting to grayscale"""
    
    def __init__(self, method: str = 'luminosity'):
        """
        Args:
            method: Conversion method ('luminosity', 'average', 'lightness')
        """
        super().__init__()
        self.method = method
    
    def apply(self, pixels: np.ndarray) -> np.ndarray:
        self.validate(pixels)
        
        # If already grayscale, return as is
        if len(pixels.shape) == 2:
            return pixels
        
        # If only one channel
        if pixels.shape[-1] == 1:
            return pixels.squeeze()
        
        # Convert RGB to grayscale
        if pixels.shape[-1] >= 3:
            if self.method == 'luminosity':
                # Standard luminosity formula
                weights = np.array([0.299, 0.587, 0.114])
                gray = np.dot(pixels[..., :3], weights)
            elif self.method == 'average':
                gray = np.mean(pixels[..., :3], axis=-1)
            elif self.method == 'lightness':
                gray = (np.max(pixels[..., :3], axis=-1) + 
                       np.min(pixels[..., :3], axis=-1)) / 2
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            return gray.astype(pixels.dtype)
        
        return pixels
    
    def update_metadata(self, img_data: ImageData) -> None:
        super().update_metadata(img_data)
        img_data.metadata['color_mode'] = 'grayscale'
        img_data.metadata['grayscale_method'] = self.method
    
    def __repr__(self) -> str:
        return f"GrayscaleFilter(method='{self.method}')"
