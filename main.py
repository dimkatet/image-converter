import argparse
from pathlib import Path
from typing import List, Optional, Union

from image_pipeline.core.filter_pipeline import FilterPipeline
from image_pipeline import RemoveAlphaFilter, PQEncodeFilter, QuantizeFilter
from image_pipeline.core.image_data import ImageData
from image_pipeline.filters.base import ImageFilter
from image_pipeline.io.reader import ImageReader
from image_pipeline.io.saver import ImageSaver

def process_image(input_path: str,
                 output_path: str,
                 filters: Union[List['ImageFilter'], 'FilterPipeline'],
                 save_options: Optional[dict] = None,
                 verbose: bool = False) -> 'ImageData':
    """
    Обработка изображения: чтение -> применение фильтров -> сохранение
    
    Args:
        input_path: Путь к входному файлу
        output_path: Путь для сохранения результата
        filters: Список фильтров или FilterPipeline для применения
        save_options: Параметры сохранения (quality, compression и т.д.)
        verbose: Выводить детальную информацию о процессе
        
    Returns:
        ImageData с обработанным изображением
        
    Example:
        >>> from image_filters import GrayscaleFilter, BlurFilter
        >>> filters = [GrayscaleFilter(), BlurFilter(sigma=2.0)]
        >>> result = process_image('input.tiff', 'output.png', filters)
    """
    save_options = save_options or {}
    
    try:
        # 1. Чтение изображения
        if verbose:
            print(f"Чтение файла: {input_path}")
        
        reader = ImageReader(input_path)
        img_data = reader.read()
        
        if verbose:
            print(f"  Загружено: {img_data.shape}, {img_data.dtype}")
            print(f"  Формат: {img_data.format}")
        
        # 2. Применение фильтров
        if verbose:
            print(f"\nПрименение фильтров...")
        
        # Если передан список фильтров, создаём pipeline
        if isinstance(filters, list):
            pipeline = FilterPipeline(filters)
        else:
            pipeline = filters
        
        # Применяем фильтры
        processed_pixels = pipeline.apply(img_data.pixels, verbose=verbose)
        
        # Создаём новый ImageData с обработанными пикселями
        processed_data = ImageData(processed_pixels, img_data.metadata.copy())
        
        # 3. Сохранение результата
        if verbose:
            print(f"\nСохранение в: {output_path}")
        
        ImageSaver.save_with_format_conversion(processed_data, output_path, **save_options)
        
        if verbose:
            print(f"  Успешно сохранено!")
            output_size = Path(output_path).stat().st_size / 1024  # KB
            print(f"  Размер файла: {output_size:.2f} KB")
        
        return processed_data
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Входной файл не найден: {input_path}") from e
    except Exception as e:
        raise RuntimeError(f"Ошибка при обработке изображения: {e}") from e



def main():
    parser = argparse.ArgumentParser(description="Image Converter")
    parser.add_argument("input_file", help="Path to the input image file")
    parser.add_argument("output_file", help="Path to the output image file")
    parser.add_argument("--bit-depth", type=int, choices=[8, 12, 16, 32], default=8,
                        help="Bit depth for output image (default: 8)")
    parser.add_argument("--quality", type=int, choices=range(1, 101), metavar="[1-100]",
                        help="Quality for output image (for lossy formats)")
    parser.add_argument("--format", type=str, help="Output image format (e.g., png, jpg, tiff)")
    # Add more optional arguments as needed

    args = parser.parse_args()
    

    process_image(
        input_path=args.input_file,
        output_path=args.output_file,
        filters=[RemoveAlphaFilter(), PQEncodeFilter(peak_luminance=10000), QuantizeFilter(bit_depth=16)],
        save_options={
            # "bit_depth": args.bit_depth,
            "quality": args.quality,
            # "format": args.format,
        },
        verbose=True
    )

if __name__ == "__main__":
    main()