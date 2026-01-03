import numpy as np
from scipy import ndimage

from image_pipeline.filters.base import ImageFilter


class BlurFilter(ImageFilter):
    """Blur filter (Gaussian blur)"""
    
    def __init__(self, sigma: float = 1.0):
        """
        Args:
            sigma: Standard deviation for Gaussian blur
        """
        self.sigma = sigma
        super().__init__()
    
    def validate_params(self) -> None:
        if not isinstance(self.sigma, (int, float)):
            raise TypeError(
                f"sigma must be numeric (int or float), got {type(self.sigma).__name__}"
            )
        
        if self.sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {self.sigma}")
    
    def apply(self, pixels: np.ndarray) -> np.ndarray:
        self.validate(pixels)
        
        if self.sigma <= 0:
            return pixels
        
        # Apply blur to each channel separately
        if len(pixels.shape) == 2:
            return ndimage.gaussian_filter(pixels, sigma=self.sigma)
        else:
            result = np.zeros_like(pixels)
            for i in range(pixels.shape[-1]):
                result[..., i] = ndimage.gaussian_filter(pixels[..., i], sigma=self.sigma)
            return result
    
    def __repr__(self) -> str:
        return f"BlurFilter(sigma={self.sigma})"
