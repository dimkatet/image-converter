"""
Модуль для чтения изображений различных форматов
"""
import numpy as np
import imageio.v3 as iio
import os
from pathlib import Path
import tifffile

from image_pipeline.core.image_data import ImageData


class ImageReader:
    """Класс для чтения изображений различных форматов"""
    
    TIFF_FORMATS = {'.tiff', '.tif'}
    IMAGEIO_FORMATS = {
        '.jpg', '.jpeg', '.png', '.bmp', '.gif', 
        '.webp', '.ico', '.exr', '.hdr', '.pfm'
    }
    SUPPORTED_FORMATS = TIFF_FORMATS | IMAGEIO_FORMATS
    
    def __init__(self, filepath: str):
        """
        Инициализация читателя изображений
        
        Args:
            filepath: Путь к файлу изображения
        """
        self.filepath = Path(filepath)
        self._validate_file()
    
    def _validate_file(self) -> None:
        """Проверка существования и формата файла"""
        if not self.filepath.exists():
            raise FileNotFoundError(f"Файл не найден: {self.filepath}")
        
        if not self.filepath.is_file():
            raise ValueError(f"Путь не является файлом: {self.filepath}")
        
        ext = self.filepath.suffix.lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Неподдерживаемый формат: {ext}. "
                f"Поддерживаемые форматы: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )
    
    def read(self) -> ImageData:
        """
        Чтение изображения из файла
        
        Returns:
            ImageData объект с пикселями и метаданными
        """
        ext = self.filepath.suffix.lower()
        
        if ext in self.TIFF_FORMATS:
            return self._read_tiff()
        else:
            return self._read_imageio()
    
    def _read_tiff(self) -> ImageData:
        """Чтение TIFF файлов через tifffile"""
        try:
            pixels = tifffile.imread(self.filepath)
            
            with tifffile.TiffFile(self.filepath) as tif:
                metadata = {
                    'format': 'TIFF',
                    'filename': self.filepath.name,
                    'file_size': os.path.getsize(self.filepath),
                    'bit_depth': pixels.dtype.itemsize * 8,
                    'is_float': np.issubdtype(pixels.dtype, np.floating),
                    'pages': len(tif.pages),
                    'shape': pixels.shape,
                    'dtype': str(pixels.dtype),
                }
                
                if tif.pages:
                    page = tif.pages[0]
                    metadata['compression'] = page.compression.name if hasattr(page, 'compression') else 'unknown'
                    metadata['photometric'] = page.photometric.name if hasattr(page, 'photometric') else 'unknown'
                    
                    if hasattr(page, 'description') and page.description:
                        metadata['description'] = page.description
            
            return ImageData(pixels, metadata)
            
        except Exception as e:
            raise IOError(f"Ошибка при чтении TIFF файла: {e}")
    
    def _read_imageio(self) -> ImageData:
        """Чтение изображений через imageio"""
        try:
            # Читаем изображение
            pixels = iio.imread(self.filepath)
            
            # Получаем метаданные
            props = iio.improps(self.filepath)
            
            metadata = {
                'format': self.filepath.suffix.upper().lstrip('.'),
                'filename': self.filepath.name,
                'file_size': os.path.getsize(self.filepath),
                'shape': pixels.shape,
                'dtype': str(pixels.dtype),
                'bit_depth': pixels.dtype.itemsize * 8,
                'is_float': np.issubdtype(pixels.dtype, np.floating),
            }
            
            # Добавляем свойства из imageio
            if props.shape:
                metadata['original_shape'] = props.shape
            if props.n_images:
                metadata['n_images'] = props.n_images
            if props.is_batch:
                metadata['is_batch'] = props.is_batch
            
            # Проверяем прозрачность
            if len(pixels.shape) == 3:
                channels = pixels.shape[2]
                metadata['channels'] = channels
                metadata['has_transparency'] = channels in (2, 4)  # LA или RGBA
            elif len(pixels.shape) == 2:
                metadata['channels'] = 1
                metadata['has_transparency'] = False
            
            # Пытаемся получить EXIF из метаданных imageio
            try:
                meta = iio.immeta(self.filepath)
                if meta:
                    metadata['imageio_meta'] = meta
            except:
                pass
            
            return ImageData(pixels, metadata)
            
        except Exception as e:
            raise IOError(f"Ошибка при чтении изображения через imageio: {e}")