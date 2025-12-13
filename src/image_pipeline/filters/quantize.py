import numpy as np
from .base import ImageFilter


class QuantizeFilter(ImageFilter):
    """
    Фильтр для квантизации float значений в integer типы
    Преобразует [0, 1] → [0, 2^bit_depth - 1]
    """
    
    def __init__(self, bit_depth: int = 8):
        """
        Args:
            bit_depth: Целевая битность (8, 10, 12, 16, 32)
        """
        super().__init__()
        
        if bit_depth not in [8, 10, 12, 16, 32]:
            raise ValueError(f"Неподдерживаемая битность: {bit_depth}. "
                           f"Поддерживается: 8, 10, 12, 16, 32")
        
        self.bit_depth = bit_depth
        
        # Определяем целевой тип данных
        if bit_depth == 8:
            self.target_dtype = np.uint8
        elif bit_depth in [10, 12]:
            self.target_dtype = np.uint16  # 10/12-bit хранятся в uint16
        elif bit_depth == 16:
            self.target_dtype = np.uint16
        elif bit_depth == 32:
            self.target_dtype = np.uint32
        
        self.max_value = (2 ** bit_depth) - 1
    
    def apply(self, pixels: np.ndarray) -> np.ndarray:
        self.validate(pixels)
        
        # Обрезаем значения в диапазон [0, 1]
        clipped = np.clip(pixels, 0.0, 1.0)
        
        # Квантизуем: [0, 1] → [0, max_value]
        quantized = clipped * self.max_value
        
        # Округляем и конвертируем в целочисленный тип
        result = np.round(quantized).astype(self.target_dtype)
        
        return result
    
    def __repr__(self) -> str:
        return f"QuantizeFilter(bit_depth={self.bit_depth})"
