from typing import List
from typing import Optional
import numpy as np

from image_pipeline.filters.base import ImageFilter


class FilterPipeline:
    """Класс для последовательного применения нескольких фильтров"""
    
    def __init__(self, filters: Optional[List[ImageFilter]] = None):
        """
        Args:
            filters: Список фильтров для применения
        """
        self.filters: List[ImageFilter] = filters or []
    
    def add(self, filter: ImageFilter) -> 'FilterPipeline':
        """
        Добавление фильтра в pipeline
        
        Args:
            filter: Фильтр для добавления
            
        Returns:
            self для chaining
        """
        self.filters.append(filter)
        return self
    
    def remove(self, index: int) -> 'FilterPipeline':
        """
        Удаление фильтра по индексу
        
        Args:
            index: Индекс фильтра
            
        Returns:
            self для chaining
        """
        if 0 <= index < len(self.filters):
            self.filters.pop(index)
        return self
    
    def clear(self) -> 'FilterPipeline':
        """Очистка всех фильтров"""
        self.filters.clear()
        return self
    
    def apply(self, pixels: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Применение всех фильтров последовательно
        
        Args:
            pixels: Входной массив пикселей
            verbose: Выводить информацию о каждом шаге
            
        Returns:
            Обработанный массив пикселей
        """
        result = pixels.copy()
        
        for i, filter in enumerate(self.filters):
            if verbose:
                print(f"Шаг {i+1}/{len(self.filters)}: Применение {filter}")
            
            result = filter.apply(result)
            
            if verbose:
                print(f"  Форма: {result.shape}, Тип: {result.dtype}, "
                      f"Диапазон: [{result.min():.3f}, {result.max():.3f}]")
        
        return result
    
    def __len__(self) -> int:
        return len(self.filters)
    
    def __repr__(self) -> str:
        filters_str = ", ".join(str(f) for f in self.filters)
        return f"FilterPipeline([{filters_str}])"
