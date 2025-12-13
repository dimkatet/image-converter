"""
Модуль для сохранения изображений в различные форматы
"""
import numpy as np
import imageio.v3 as iio
import tifffile
from pathlib import Path
from typing import Optional, Union, Dict, Any
import warnings

try:
    import png as pypng
    HAS_PYPNG = True
except ImportError:
    HAS_PYPNG = False

from image_pipeline.core.image_data import ImageData


class ImageWriter:
    """Класс для сохранения изображений в файл"""
    
    TIFF_FORMATS = {'.tiff', '.tif'}
    HDR_FORMATS = {'.exr', '.hdr', '.pfm'}
    PNG_FORMAT = {'.png'}
    UINT8_FORMATS = {'.jpg', '.jpeg', '.bmp', '.webp'}
    SUPPORTED_FORMATS = TIFF_FORMATS | HDR_FORMATS | PNG_FORMAT | UINT8_FORMATS
    
    # Доступные методы сжатия для TIFF
    TIFF_COMPRESSIONS = {
        'none': 0,
        'lzw': 5,
        'jpeg': 7,
        'deflate': 8,
        'zstd': 50000,
    }
    
    def __init__(self, filepath: str):
        """
        Инициализация writer'а
        
        Args:
            filepath: Путь для сохранения файла
        """
        self.filepath = Path(filepath)
        self._validate_format()
    
    def _validate_format(self) -> None:
        """Проверка поддерживаемости формата"""
        ext = self.filepath.suffix.lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Неподдерживаемый формат: {ext}. "
                f"Поддерживаемые форматы: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )
    
    def write(self, 
              data: Union['ImageData', np.ndarray],
              quality: int = 95,
              compression: str = 'lzw',
              compression_level: int = 6,
              metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Сохранение изображения в файл
        
        Args:
            data: ImageData объект или numpy array с пикселями
            quality: Качество для JPEG/WebP (1-100), по умолчанию 95
            compression: Тип сжатия для TIFF ('none', 'lzw', 'jpeg', 'deflate', 'zstd')
            compression_level: Уровень сжатия для PNG (0-9), по умолчанию 6
            metadata: Дополнительные метаданные для сохранения
        """
        # Извлекаем пиксели и метаданные
        if hasattr(data, 'pixels'):
            pixels = data.pixels
            saved_metadata = metadata or data.metadata
        else:
            pixels = data
            saved_metadata = metadata or {}
        
        # Валидируем данные
        self._validate_data(pixels)
        
        # Создаём директорию если не существует
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        ext = self.filepath.suffix.lower()
        
        if ext in self.TIFF_FORMATS:
            self._write_tiff(pixels, compression, saved_metadata)
        elif ext in self.PNG_FORMAT:
            self._write_png(pixels, compression_level, saved_metadata)
        elif ext in self.HDR_FORMATS:
            self._write_hdr(pixels, saved_metadata)
        else:
            self._write_uint8(pixels, quality, saved_metadata)
    
    def _validate_data(self, pixels: np.ndarray) -> None:
        """
        Валидация данных перед сохранением
        
        Args:
            pixels: Массив пикселей
            
        Raises:
            ValueError: Если данные невалидны
        """
        if not isinstance(pixels, np.ndarray):
            raise ValueError("Данные должны быть numpy array")
        
        if pixels.size == 0:
            raise ValueError("Пустой массив пикселей")
        
        ext = self.filepath.suffix.lower()
        
        # PNG поддерживает uint8 и uint16
        if ext == '.png':
            if pixels.dtype not in (np.uint8, np.uint16):
                raise ValueError(
                    f"PNG поддерживает только uint8 и uint16. "
                    f"Получен: {pixels.dtype}.\n"
                    f"Решения:\n"
                    f"  1. Для float: используйте TIFF, EXR или HDR\n"
                    f"  2. Для uint32: конвертируйте в uint16 или используйте TIFF"
                )
        
        # Остальные uint8 форматы
        elif ext in self.UINT8_FORMATS:
            if pixels.dtype != np.uint8:
                raise ValueError(
                    f"{ext.upper()} поддерживает только uint8. "
                    f"Получен: {pixels.dtype}.\n"
                    f"Решения:\n"
                    f"  1. Используйте QuantizeFilter(bit_depth=8) для конвертации\n"
                    f"  2. Для uint16: сохраняйте в PNG или TIFF\n"
                    f"  3. Для float32: сохраняйте в TIFF, EXR или HDR"
                )
            
            # JPEG не поддерживает прозрачность
            if ext in {'.jpg', '.jpeg'}:
                if len(pixels.shape) == 3 and pixels.shape[2] == 4:
                    raise ValueError(
                        "JPEG не поддерживает прозрачность (RGBA). "
                        "Используйте PNG или конвертируйте в RGB."
                    )
        
        # HDR форматы требуют float
        elif ext in self.HDR_FORMATS:
            if not np.issubdtype(pixels.dtype, np.floating):
                raise ValueError(
                    f"{ext.upper()} требует float данные. "
                    f"Получен: {pixels.dtype}. "
                    f"Конвертируйте в float32 перед сохранением."
                )
        
        # Проверка диапазона для integer типов
        if np.issubdtype(pixels.dtype, np.integer):
            pix_min = pixels.min()
            pix_max = pixels.max()
            
            if pix_min < 0:
                raise ValueError(
                    f"Отрицательные значения пикселей ({pix_min}) недопустимы"
                )
            
            dtype_max = np.iinfo(pixels.dtype).max
            if pix_max > dtype_max:
                raise ValueError(
                    f"Значения пикселей ({pix_max}) превышают максимум для {pixels.dtype} ({dtype_max})"
                )
    
    def _write_tiff(self, 
                    pixels: np.ndarray, 
                    compression: str,
                    metadata: Dict[str, Any]) -> None:
        """
        Сохранение в TIFF через tifffile
        Поддерживает: uint8, uint16, uint32, float32, float64
        """
        try:
            if compression not in self.TIFF_COMPRESSIONS:
                warnings.warn(
                    f"Сжатие '{compression}' не поддерживается, используется 'lzw'",
                    UserWarning
                )
                compression = 'lzw'
            
            tiff_metadata = {}
            if 'description' in metadata:
                tiff_metadata['description'] = str(metadata['description'])
            
            tifffile.imwrite(
                self.filepath,
                pixels,
                compression=compression,
                metadata=tiff_metadata
            )
            
        except Exception as e:
            raise IOError(f"Ошибка при сохранении TIFF: {e}")
    
    def _write_png(self,
                   pixels: np.ndarray,
                   compression_level: int,
                   metadata: Dict[str, Any]) -> None:
        """
        Сохранение PNG с поддержкой uint16
        Использует pypng для uint16, imageio для uint8
        """
        try:
            # uint8 - используем imageio (быстрее)
            if pixels.dtype == np.uint8:
                self._write_png_imageio(pixels, compression_level)
            
            # uint16 - используем pypng (единственная либа с нормальной поддержкой)
            elif pixels.dtype == np.uint16:
                if not HAS_PYPNG:
                    raise ImportError(
                        "Для сохранения uint16 PNG требуется библиотека pypng.\n"
                        "Установите: pip install pypng"
                    )
                self._write_png_pypng(pixels)
            
            else:
                raise ValueError(f"PNG не поддерживает тип {pixels.dtype}")
                
        except Exception as e:
            raise IOError(f"Ошибка при сохранении PNG: {e}")
    
    def _write_png_imageio(self, pixels: np.ndarray, compression_level: int) -> None:
        """Сохранение uint8 PNG через imageio"""
        iio.imwrite(
            self.filepath,
            pixels,
            compress_level=compression_level,
            optimize=True
        )
    
    def _write_png_pypng(self, pixels: np.ndarray) -> None:
        """Сохранение uint16 PNG через pypng"""
        height, width = pixels.shape[:2]
        
        # Определяем тип изображения
        if len(pixels.shape) == 2:
            # Grayscale
            greyscale = True
            alpha = False
            planes = 1
            # pypng требует 2D массив для grayscale
            img_data = pixels
        
        elif len(pixels.shape) == 3:
            channels = pixels.shape[2]
            
            if channels == 1:
                # Grayscale с одним каналом
                greyscale = True
                alpha = False
                planes = 1
                img_data = pixels.squeeze(-1)
            
            elif channels == 2:
                # Grayscale + Alpha
                greyscale = True
                alpha = True
                planes = 2
                # Преобразуем в 2D массив строк
                img_data = pixels.reshape(height, width * 2)
            
            elif channels == 3:
                # RGB
                greyscale = False
                alpha = False
                planes = 3
                # Преобразуем в 2D массив строк
                img_data = pixels.reshape(height, width * 3)
            
            elif channels == 4:
                # RGBA
                greyscale = False
                alpha = True
                planes = 4
                # Преобразуем в 2D массив строк
                img_data = pixels.reshape(height, width * 4)
            
            else:
                raise ValueError(f"Неподдерживаемое количество каналов: {channels}")
        
        else:
            raise ValueError(f"Неподдерживаемая форма массива: {pixels.shape}")
        
        # Создаём PNG writer
        writer = pypng.Writer(
            width=width,
            height=height,
            greyscale=greyscale,
            alpha=alpha,
            bitdepth=16,  # uint16 = 16 bit
            compression=9  # максимальное сжатие
        )
        
        # Сохраняем
        with open(self.filepath, 'wb') as f:
            # pypng требует список строк (каждая строка - 1D массив)
            if len(img_data.shape) == 2 and planes > 1:
                # Уже в формате (height, width*channels)
                writer.write(f, img_data)
            else:
                # Grayscale 2D
                writer.write(f, img_data)
    
    def _write_hdr(self,
                   pixels: np.ndarray,
                   metadata: Dict[str, Any]) -> None:
        """
        Сохранение HDR форматов через imageio
        Поддерживает: float32, float64
        """
        try:
            kwargs = {}
            
            # Для EXR можно указать компрессию
            if self.filepath.suffix.lower() == '.exr':
                kwargs['compression'] = 'ZIP_COMPRESSION'
            
            iio.imwrite(self.filepath, pixels, **kwargs)
            
        except Exception as e:
            raise IOError(f"Ошибка при сохранении HDR формата: {e}")
    
    def _write_uint8(self, 
                     pixels: np.ndarray,
                     quality: int,
                     metadata: Dict[str, Any]) -> None:
        """
        Сохранение uint8 форматов через imageio
        Поддерживает: только uint8
        """
        try:
            ext = self.filepath.suffix.lower()
            kwargs = {}
            
            if ext in {'.jpg', '.jpeg'}:
                kwargs['quality'] = quality
                kwargs['optimize'] = True
            
            elif ext == '.webp':
                kwargs['quality'] = quality
                kwargs['method'] = 6
            
            iio.imwrite(self.filepath, pixels, **kwargs)
            
        except Exception as e:
            raise IOError(f"Ошибка при сохранении {ext.upper()}: {e}")