import numpy as np
from scipy import ndimage

from image_pipeline.filters.base import ImageFilter


class BlurFilter(ImageFilter):
    """Фильтр размытия (Gaussian blur)"""
    
    def __init__(self, sigma: float = 1.0):
        """
        Args:
            sigma: Стандартное отклонение для Gaussian blur
        """
        super().__init__()
        self.sigma = sigma
    
    def apply(self, pixels: np.ndarray) -> np.ndarray:
        self.validate(pixels)
        
        if self.sigma <= 0:
            return pixels
        
        # Применяем размытие к каждому каналу отдельно
        if len(pixels.shape) == 2:
            return ndimage.gaussian_filter(pixels, sigma=self.sigma)
        else:
            result = np.zeros_like(pixels)
            for i in range(pixels.shape[-1]):
                result[..., i] = ndimage.gaussian_filter(pixels[..., i], sigma=self.sigma)
            return result
    
    def __repr__(self) -> str:
        return f"BlurFilter(sigma={self.sigma})"
