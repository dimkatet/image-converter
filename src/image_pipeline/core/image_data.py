"""
Модуль для чтения изображений различных форматов
"""
import numpy as np
from typing import Dict, Any


class ImageData:
    """Класс для хранения данных изображения и метаданных"""
    
    def __init__(self, pixels: np.ndarray, metadata: Dict[str, Any]):
        """
        Инициализация данных изображения
        
        Args:
            pixels: Массив numpy с пикселями изображения
            metadata: Словарь с метаданными
        """
        self.pixels = pixels
        self.metadata = metadata
    
    @property
    def shape(self) -> tuple:
        """Форма массива пикселей (height, width, channels)"""
        return self.pixels.shape
    
    @property
    def dtype(self) -> np.dtype:
        """Тип данных пикселей"""
        return self.pixels.dtype
    
    @property
    def width(self) -> int:
        """Ширина изображения"""
        return self.pixels.shape[1]
    
    @property
    def height(self) -> int:
        """Высота изображения"""
        return self.pixels.shape[0]
    
    @property
    def channels(self) -> int:
        """Количество каналов"""
        return self.pixels.shape[2] if len(self.pixels.shape) > 2 else 1
    
    @property
    def format(self) -> str:
        """Формат исходного файла"""
        return self.metadata.get('format', 'unknown')
    
    def __repr__(self) -> str:
        return (f"ImageData(shape={self.shape}, dtype={self.dtype}, "
                f"format={self.format})")
