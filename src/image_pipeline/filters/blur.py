from image_pipeline.core.image_data import ImageData
    
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
        super().__init__()
        self.sigma = sigma
    
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
    
    def update_metadata(self, img_data: ImageData) -> None:
        super().update_metadata(img_data)
        img_data.metadata['filter_blur'] = {'sigma': self.sigma}
    
    def __repr__(self) -> str:
        return f"BlurFilter(sigma={self.sigma})"
