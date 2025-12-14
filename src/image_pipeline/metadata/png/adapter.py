"""
PNG Metadata Adapter - converts ImageMetadata to PNG-specific chunks
"""
from typing import Optional

from image_pipeline.types import ImageMetadata
from image_pipeline.constants import (
    STANDARD_COLOR_PRIMARIES,
    TRANSFER_TO_CICP,
    COLORSPACE_TO_CICP,
)
from .codec import CICPData, MDCVData, CLLIData


class PNGMetadataAdapter:
    """Converts generic ImageMetadata to PNG-specific chunk structures"""
    
    @staticmethod
    def convert(metadata: ImageMetadata) -> dict:
        """
        Convert ImageMetadata to PNG chunks
        
        Args:
            metadata: ImageMetadata dictionary
            
        Returns:
            Dictionary with chunk structures: {'cicp': CICPData, 'mdcv': MDCVData, ...}
            Only includes chunks for which sufficient data is available
        """
        chunks = {}
        
        PNGMetadataAdapter._maybe_add_chunk(chunks, 'cicp', 
            PNGMetadataAdapter._to_cicp(metadata))
        
        PNGMetadataAdapter._maybe_add_chunk(chunks, 'mdcv', 
            PNGMetadataAdapter._to_mdcv(metadata))
        
        PNGMetadataAdapter._maybe_add_chunk(chunks, 'clli', 
            PNGMetadataAdapter._to_clli(metadata))
        
        # Text metadata (если есть)
        if 'text' in metadata and metadata['text']:
            chunks['text'] = metadata['text']
        
        return chunks
    
    @staticmethod
    def _maybe_add_chunk(chunks: dict, key: str, value):
        """Add chunk to dictionary only if value is not None"""
        if value is not None:
            chunks[key] = value
    
    @staticmethod
    def _to_cicp(metadata: ImageMetadata) -> Optional[CICPData]:
        """
        Generate cICP chunk from metadata
        
        Requires: transfer_function OR color_space (at least one)
        """
        transfer_function = metadata.get('transfer_function')
        color_space = metadata.get('color_space')
        
        # Нужна хотя бы transfer_function или color_space
        if not transfer_function and not color_space:
            return None
        
        # Transfer characteristics
        transfer_code = TRANSFER_TO_CICP.get(transfer_function, 2) if transfer_function else 2  # 2 = unspecified
        
        # Color primaries - приоритет color_primaries, потом color_space
        color_primaries_code = 2  # Default: unspecified
        if metadata.get('color_primaries'):
            # Если указаны кастомные primaries, ставим 2 (unspecified/custom)
            color_primaries_code = 2
        elif color_space:
            color_primaries_code = COLORSPACE_TO_CICP.get(color_space, 2)
        
        # Matrix coefficients - для RGB всегда 0 (Identity)
        matrix_coefficients = 0
        
        # Full range flag - всегда 1 (full range) для наших данных
        video_full_range_flag = 1
        
        return CICPData(
            color_primaries=color_primaries_code,
            transfer_characteristics=transfer_code,
            matrix_coefficients=matrix_coefficients,
            video_full_range_flag=video_full_range_flag
        )
    
    @staticmethod
    def _to_mdcv(metadata: ImageMetadata) -> Optional[MDCVData]:
        """
        Generate mDCv chunk from metadata
        
        Requires: peak_luminance AND (color_space OR color_primaries)
        """
        peak_luminance = metadata.get('peak_luminance')
        min_luminance = metadata.get('min_luminance', 0.0001)
        
        # Валидация: нужен peak_luminance
        if not peak_luminance:
            return None
        
        # Получаем primaries - приоритет custom, иначе standard
        primaries = None
        primaries = metadata.get('color_primaries')
        color_space = metadata.get('color_space')
        if not primaries:
          if not color_space:
            return None
          primaries = STANDARD_COLOR_PRIMARIES.get(color_space)
            
        
        # Если primaries не определены - не создаём chunk
        if not primaries:
            return None
        
        # Конвертируем координаты в формат mDCv (умножаем на 50000 для 0.00002 units)
        def to_mdcv_coord(value: float) -> int:
            result = int(value * 50000)
            return max(0, min(65535, result)) 
        
        display_primaries_x = (
            to_mdcv_coord(primaries['red'][0]),
            to_mdcv_coord(primaries['green'][0]),
            to_mdcv_coord(primaries['blue'][0])
        )
        
        display_primaries_y = (
            to_mdcv_coord(primaries['red'][1]),
            to_mdcv_coord(primaries['green'][1]),
            to_mdcv_coord(primaries['blue'][1])
        )
        
        white_point_x = to_mdcv_coord(primaries['white'][0])
        white_point_y = to_mdcv_coord(primaries['white'][1])
        
        # todo
        # Luminance в формат mDCv (умножаем на 10000 для 0.0001 nits units)
        max_luminance_mdcv = max(0, min(4294967295, int(peak_luminance)))
        min_luminance_mdcv = max(0, min(4294967295, int(min_luminance)))
        
        return MDCVData(
            display_primaries_x=display_primaries_x,
            display_primaries_y=display_primaries_y,
            white_point_x=white_point_x,
            white_point_y=white_point_y,
            max_display_mastering_luminance=max_luminance_mdcv,
            min_display_mastering_luminance=min_luminance_mdcv
        )
    
    @staticmethod
    def _to_clli(metadata: ImageMetadata) -> Optional[CLLIData]:
        """
        Generate cLLi chunk from metadata
        
        Requires: max_cll AND max_fall
        """
        max_cll = metadata.get('max_cll')
        max_fall = metadata.get('max_fall')
        
        # Оба значения должны быть указаны
        if max_cll is None or max_fall is None:
            return None
        
        return CLLIData(
            max_content_light_level=max_cll,
            max_frame_average_light_level=max_fall
        )