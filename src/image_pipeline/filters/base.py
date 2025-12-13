from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class ImageFilter(ABC):
    """Базовый абстрактный класс для всех фильтров"""
    
    def __init__(self, name: Optional[str] = None):
        """
        Инициализация фильтра
        
        Args:
            name: Название фильтра (опционально)
        """
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def apply(self, pixels: np.ndarray) -> np.ndarray:
        """
        Применение фильтра к массиву пикселей
        
        Args:
            pixels: Входной массив пикселей
            
        Returns:
            Обработанный массив пикселей
        """
        pass
    
    def validate(self, pixels: np.ndarray) -> None:
        """
        Валидация входных данных
        
        Args:
            pixels: Массив пикселей для проверки
            
        Raises:
            ValueError: Если данные невалидны
        """
        if not isinstance(pixels, np.ndarray):
            raise ValueError(f"{self.name}: входные данные должны быть numpy array")
        
        if pixels.size == 0:
            raise ValueError(f"{self.name}: пустой массив пикселей")
    
    def __repr__(self) -> str:
        return f"{self.name}()"
