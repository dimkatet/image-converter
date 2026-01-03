import numpy as np
from scipy.ndimage import convolve
from .base import ImageFilter


class SharpenFilter(ImageFilter):
    """Sharpening filter"""
    
    def __init__(self, strength: float = 1.0):
        """
        Args:
            strength: Effect strength (0.0 - no change, higher - stronger)
        """
        self.strength = strength
        super().__init__()
    
    def validate_params(self) -> None:
        if not isinstance(self.strength, (int, float)):
            raise TypeError(
                f"strength must be numeric (int or float), got {type(self.strength).__name__}"
            )
        
        if self.strength < 0:
            raise ValueError(f"strength must be non-negative, got {self.strength}")
    
    def apply(self, pixels: np.ndarray) -> np.ndarray:
        self.validate(pixels)
        
        # Sharpen kernel
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ]) * self.strength
        
        # Normalize the central element
        kernel[1, 1] = 1 + 4 * self.strength
        
        if len(pixels.shape) == 2:
            return convolve(pixels, kernel, mode='reflect')
        else:
            result = np.zeros_like(pixels)
            for i in range(pixels.shape[-1]):
                result[..., i] = convolve(pixels[..., i], kernel, mode='reflect')
            return result
        
    # todo
    # def update_metadata(self, img_data: ImageData) -> None:
    #     super().update_metadata(img_data)
    #     img_data.metadata['filter_sharpen'] = {'strength': self.strength}
    
    def __repr__(self) -> str:
        return f"SharpenFilter(strength={self.strength})"
