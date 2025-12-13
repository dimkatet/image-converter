from typing import Optional, Union
import numpy as np

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.writer import ImageWriter

class ImageSaver:
    """Вспомогательный класс для групповых операций сохранения"""
    
    @staticmethod
    def save_with_format_conversion(data: Union['ImageData', np.ndarray],
                                    output_path: str,
                                    target_dtype: Optional[np.dtype] = None,
                                    **save_options) -> None:
        """
        Сохранение с конвертацией типа данных
        
        Args:
            data: ImageData или numpy array
            output_path: Путь для сохранения
            target_dtype: Целевой тип данных (uint8, uint16, float32 и т.д.)
            **save_options: Дополнительные параметры для сохранения
        """
        # Извлекаем пиксели
        if hasattr(data, 'pixels'):
            pixels = data.pixels
            metadata = data.metadata
        else:
            pixels = data
            metadata = {}
        
        # Конвертируем тип данных если нужно
        if target_dtype and pixels.dtype != target_dtype:
            pixels = ImageSaver._convert_dtype(pixels, target_dtype)
        
        # Сохраняем
        writer = ImageWriter(output_path)
        writer.write(pixels, metadata=metadata, **save_options)
    
    @staticmethod
    def _convert_dtype(pixels: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
        """Конвертация типа данных с правильным масштабированием"""
        source_dtype = pixels.dtype
        
        # Если типы одинаковые, возвращаем как есть
        if source_dtype == target_dtype:
            return pixels
        
        # float -> uint
        if np.issubdtype(source_dtype, np.floating) and \
           np.issubdtype(target_dtype, np.unsignedinteger):
            pix_min = pixels.min()
            pix_max = pixels.max()
            
            if pix_max > pix_min:
                normalized = (pixels - pix_min) / (pix_max - pix_min)
            else:
                normalized = np.zeros_like(pixels)
            
            max_val = np.iinfo(target_dtype).max
            return (normalized * max_val).astype(target_dtype)
        
        # uint -> float
        elif np.issubdtype(source_dtype, np.unsignedinteger) and \
             np.issubdtype(target_dtype, np.floating):
            max_val = np.iinfo(source_dtype).max
            return (pixels.astype(target_dtype) / max_val)
        
        # uint -> uint другой разрядности
        elif np.issubdtype(source_dtype, np.unsignedinteger) and \
             np.issubdtype(target_dtype, np.unsignedinteger):
            source_max = np.iinfo(source_dtype).max
            target_max = np.iinfo(target_dtype).max
            return (pixels.astype(np.float64) / source_max * target_max).astype(target_dtype)
        
        # Для остальных случаев просто приводим тип
        return pixels.astype(target_dtype)
