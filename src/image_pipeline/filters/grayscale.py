import numpy as np
from .base import ImageFilter


class GrayscaleFilter(ImageFilter):
    """Фильтр для преобразования в оттенки серого"""
    
    def __init__(self, method: str = 'luminosity'):
        """
        Args:
            method: Метод преобразования ('luminosity', 'average', 'lightness')
        """
        super().__init__()
        self.method = method
    
    def apply(self, pixels: np.ndarray) -> np.ndarray:
        self.validate(pixels)
        
        # Если уже grayscale, возвращаем как есть
        if len(pixels.shape) == 2:
            return pixels
        
        # Если есть только один канал
        if pixels.shape[-1] == 1:
            return pixels.squeeze()
        
        # Преобразование RGB в grayscale
        if pixels.shape[-1] >= 3:
            if self.method == 'luminosity':
                # Стандартная формула luminosity
                weights = np.array([0.299, 0.587, 0.114])
                gray = np.dot(pixels[..., :3], weights)
            elif self.method == 'average':
                gray = np.mean(pixels[..., :3], axis=-1)
            elif self.method == 'lightness':
                gray = (np.max(pixels[..., :3], axis=-1) + 
                       np.min(pixels[..., :3], axis=-1)) / 2
            else:
                raise ValueError(f"Неизвестный метод: {self.method}")
            
            return gray.astype(pixels.dtype)
        
        return pixels
    
    def __repr__(self) -> str:
        return f"GrayscaleFilter(method='{self.method}')"
