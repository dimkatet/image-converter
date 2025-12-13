from typing import Optional

import numpy as np
from .base import ImageFilter


class DequantizeFilter(ImageFilter):
    """
    Фильтр для деквантизации integer значений обратно в float
    Преобразует [0, 2^bit_depth - 1] → [0, 1]
    """
    
    def __init__(self, bit_depth: Optional[int] = None):
        """
        Args:
            bit_depth: Битность исходных данных (8, 10, 12, 16, 32)
                      Если None, определяется автоматически из dtype
        """
        super().__init__()
        self.bit_depth = bit_depth
    
    def apply(self, pixels: np.ndarray) -> np.ndarray:
        self.validate(pixels)
        
        # Определяем битность если не указана
        if self.bit_depth is None:
            if pixels.dtype == np.uint8:
                bit_depth = 8
            elif pixels.dtype == np.uint16:
                bit_depth = 16
            elif pixels.dtype == np.uint32:
                bit_depth = 32
            else:
                raise ValueError(f"Не удалось определить битность для типа {pixels.dtype}")
        else:
            bit_depth = self.bit_depth
        
        max_value = (2 ** bit_depth) - 1
        
        # Нормализуем в [0, 1]
        normalized = pixels.astype(np.float32) / max_value
        
        return normalized
    
    def __repr__(self) -> str:
        return f"DequantizeFilter(bit_depth={self.bit_depth})"
